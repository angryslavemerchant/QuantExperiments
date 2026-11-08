"""Turn a self-contained shapequant packed file into a directory vLLM serves with the
"shapequant" quantization method (shapequant/vllm_ext; install this repo so its entry point
registers the method).

Output (text-only Qwen3.5 architecture, Qwen3_5ForCausalLM):
  config.json          the packed file's text config + quantization_config
  model.safetensors    the non-block tensors (norms, DeltaNet gates / conv / A_log / dt_bias);
                       with --embed-head bf16 also the embedding (4-bit) and head (8-bit) at their
                       dequantized values (bf16). Default --embed-head packed: the plugin reads both
                       from the packed file (~3 GB less GPU memory on 27B, same values)
  shapequant.packed    the packed file (block linears; read by the plugin, not by vLLM's loader)
  tokenizer + chat template + generation config from the base repo

Run: python scripts/export_vllm_checkpoint.py --packed-hf luckymushy/Qwen3.8-27B-shapequant-2.5bpw:s16_25_ft.safetensors \
        --out /workspace/vllm_ckpt/ours
then: vllm serve /workspace/vllm_ckpt/ours   (or LLM(model=...))
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
              "chat_template.jinja", "chat_template.json", "generation_config.json")
PACKED_NAME = "shapequant.packed"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def vllm_config(aux_config: dict, embed_head: str = "packed") -> dict:
    cfg = dict(aux_config)
    cfg.update(architectures=["Qwen3_5ForCausalLM"], tie_word_embeddings=False, torch_dtype="bfloat16",
               quantization_config={"quant_method": "shapequant", "packed_file": PACKED_NAME,
                                    # PRO 6000 s36: shared-memory batched kernel to a per-shape crossover (<= 64 rows, tuned
                                    # at load), fused GEMM above
                                    "xs": True, "bmax": 64, "large": "fused", "tune": True,
                                    "embed_head": embed_head})
    cfg.pop("dtype", None)
    cfg.pop("_name_or_path", None)
    return cfg


def dense_tensors(pf, chunk: int = 16384, embed_head: str = "packed") -> dict[str, torch.Tensor]:
    from shapequant.packed import q4_embed_rows, q8_rows
    aux = pf.aux
    if aux.get("embed") != "q4_b32" or aux.get("lm_head") != "q8_b32":
        raise ValueError(f"unexpected aux formats: embed {aux.get('embed')}, head {aux.get('lm_head')}")
    out = {n: pf.aux_tensor(f"dense.{n}").to(torch.bfloat16).contiguous() for n in aux["dense"]}
    if embed_head == "packed":
        return out
    V = aux["vocab"]
    eq, es = pf.aux_tensor("embed.q"), pf.aux_tensor("embed.scale")
    hq, hs = pf.aux_tensor("lm_head.q"), pf.aux_tensor("lm_head.scale")
    emb = torch.empty(V, aux["hidden"], dtype=torch.bfloat16)
    head = torch.empty(V, aux["hidden"], dtype=torch.bfloat16)
    for s in range(0, V, chunk):
        e = min(V, s + chunk)
        emb[s:e] = q4_embed_rows(eq, es, torch.arange(s, e)).to(torch.bfloat16)
        head[s:e] = q8_rows(hq, hs, s, e).to(torch.bfloat16)
    out["model.embed_tokens.weight"] = emb
    out["lm_head.weight"] = head
    return out


def mtp_tensors(base: str) -> dict[str, torch.Tensor]:
    """The base checkpoint's MTP head (mtp.* tensors, bf16) for vLLM's qwen3_5_mtp speculative
    decoding: only the shards holding them are downloaded."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    idx = json.loads(Path(hf_hub_download(base, "model.safetensors.index.json")).read_text())["weight_map"]
    names = [n for n in idx if n.startswith("mtp.") or ".mtp." in n]
    out = {}
    for shard in sorted({idx[n] for n in names}):
        with safe_open(hf_hub_download(base, shard), "pt") as f:
            for n in names:
                if idx[n] == shard:
                    key = n[n.index("mtp."):]
                    out[key] = f.get_tensor(n).to(torch.bfloat16).contiguous()
    return out


def export(packed: str, out: Path, base: str | None = None, embed_head: str = "packed", mtp: bool = False) -> dict:
    from safetensors.torch import save_file
    from shapequant.packed import PackedFile
    pf = PackedFile(packed)
    if pf.aux is None:
        raise ValueError("packed file has no aux section (not self-contained)")
    base = base or pf.summary["base_model"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    (out / "config.json").write_text(json.dumps(vllm_config(pf.aux["config"], embed_head), indent=2))
    tensors = dense_tensors(pf, embed_head=embed_head)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    log(f"wrote {len(tensors)} dense tensors ({sum(t.numel() * 2 for t in tensors.values()) / 1e9:.2f} GB)")
    if mtp:
        mt = mtp_tensors(base)
        if not mt:
            raise ValueError(f"{base} has no mtp.* tensors")
        save_file(mt, str(out / "mtp.safetensors"), metadata={"format": "pt"})
        log(f"wrote {len(mt)} MTP tensors ({sum(t.numel() * 2 for t in mt.values()) / 1e9:.2f} GB)")
    dst = out / PACKED_NAME
    if dst.exists():
        dst.unlink()
    packed = os.path.realpath(packed)       # the HF cache entry is a (relative) symlink to its blob
    try:
        os.link(packed, dst)
    except OSError:
        shutil.copyfile(packed, dst)
    from huggingface_hub import hf_hub_download
    got = []
    for f in BASE_FILES:
        try:
            shutil.copyfile(hf_hub_download(base, f), out / f)
            got.append(f)
        except Exception:
            pass
    rec = {"packed": str(packed), "base": base, "summary": pf.summary, "base_files": got,
           "seconds": round(time.time() - t0, 1)}
    (out / "shapequant_export.json").write_text(json.dumps(rec, indent=2))
    log(f"exported to {out} in {rec['seconds']}s (base files: {', '.join(got)})")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--packed")
    src.add_argument("--packed-hf", help="REPO:FILE")
    ap.add_argument("--base", help="tokenizer / chat template source (default: the packed file's base model)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mtp", action="store_true", help="also the base model's MTP head (bf16), for MTP speculative decoding")
    ap.add_argument("--embed-head", choices=["packed", "bf16"], default="packed",
                    help="packed: plugin serves the q4 embedding / q8 head; bf16: dequantized into model.safetensors")
    a = ap.parse_args()
    packed = a.packed
    if a.packed_hf:
        from huggingface_hub import hf_hub_download
        repo, fname = a.packed_hf.split(":", 1)
        packed = hf_hub_download(repo, fname)
    export(packed, Path(a.out), a.base, a.embed_head, a.mtp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
