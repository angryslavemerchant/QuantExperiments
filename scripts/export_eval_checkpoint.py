"""Write a quantized model as a standard HF checkpoint (bf16) for engine-neutral benchmarks.

The output is the base repo's own checkpoint (same files, config, tokenizer, chat template,
shard layout and tensor names) with the quantized tensors replaced by their dequantized
values, so any engine that runs the base model (vLLM, SGLang, transformers) runs the
quantized weights unchanged. Quality benchmarks then measure the weights, not the engine
(our runtime reproduces the dequantized weights' logits: scripts/check_qwen35_runtime.py).

Sources:
  --packed FILE | --packed-hf REPO:FILE    a shapequant packed file: the block linears
        (reference decode), the 8-bit lm_head and the 4-bit embedding (its deploy formats).
  --gguf FILE | --gguf-hf REPO:FILE        a GGUF: the same block linears plus the file's own
        output / token_embd tensors (its deploy formats), dequantized with gguf-py.
Everything else (norms, convolutions, DeltaNet gates, vision tower, MTP head) stays at the
base checkpoint's 16-bit values for both, the protocol of the earlier KL comparisons.

Run: python scripts/export_eval_checkpoint.py --base Qwen/Qwen3.8-27B \
        --packed-hf luckymushy/Qwen3.8-27B-shapequant-2.5bpw:s16_25_ft.safetensors --out /workspace/eval_ckpt/ours
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shapequant.gguf_ref import _GGUF_NAMES  # noqa: E402

TEXT_PREFIXES = ("model.language_model.", "model.")   # multimodal repos first, then text-only


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hf_file(spec: str) -> str:
    from huggingface_hub import hf_hub_download
    repo, fname = spec.split(":", 1)
    return hf_hub_download(repo, fname)


def text_prefix(names) -> str:
    for p in TEXT_PREFIXES:
        if any(n.startswith(p + "layers.") for n in names):
            return p
    raise ValueError("no decoder layers found in the base checkpoint")


def block_names(ckpt_names, prefix: str) -> dict[str, str]:
    """{shapequant name ('model.layers.3.mlp.down_proj'): checkpoint tensor name} for every
    block linear of the decoder (the matrices both the packed files and GGUF comparisons cover)."""
    out = {}
    for n in ckpt_names:
        if not n.startswith(prefix + "layers.") or not n.endswith(".weight"):
            continue
        rest = n[len(prefix):-len(".weight")]           # layers.3.mlp.down_proj
        parts = rest.split(".")
        if len(parts) >= 4 and ".".join(parts[2:]) in _GGUF_NAMES:
            out["model." + rest] = n
    return out


# ---------------------------------------------------------------- sources

def packed_replacements(path: str, names: dict[str, str], prefix: str, dev: str):
    """Yields (checkpoint name, float32 tensor) for the packed file's matrices, head, embed."""
    from shapequant.packed import PackedFile, q4_embed_rows, q8_rows
    pf = PackedFile(path)
    missing = set(names) - set(pf.meta)
    if missing:
        raise ValueError(f"packed file lacks {len(missing)} block linears, e.g. {sorted(missing)[:3]}")
    for n, ck in names.items():
        yield ck, pf.decode(n, device=dev).cpu()
    if pf.aux is None:
        raise ValueError("packed file has no aux section (lm_head / embedding)")
    hq, hs = pf.aux_tensor("lm_head.q").to(dev), pf.aux_tensor("lm_head.scale").to(dev)
    yield "lm_head.weight", torch.cat([q8_rows(hq, hs, s, s + 8192).cpu() for s in range(0, hq.shape[0], 8192)])
    eq, es = pf.aux_tensor("embed.q").to(dev), pf.aux_tensor("embed.scale").to(dev)
    V = pf.aux["vocab"]
    yield prefix + "embed_tokens.weight", torch.cat(
        [q4_embed_rows(eq, es, torch.arange(s, min(s + 8192, V), device=dev)).float().cpu()
         for s in range(0, V, 8192)])


_GG = None


def _gguf_job(args):
    global _GG
    from shapequant.gguf_ref import GGUFSource
    path, hf_name, shape = args
    if _GG is None or _GG.path != path:
        _GG = GGUFSource(path)
    w, _, typ = _GG.matrix(hf_name, shape)
    return hf_name, w.to(torch.bfloat16), typ


def gguf_replacements(path: str, names: dict[str, str], shapes: dict[str, tuple], prefix: str, workers: int):
    from shapequant.gguf_ref import GGUFSource
    jobs = [(path, n, shapes[ck]) for n, ck in names.items()]
    types: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for hf_name, w, typ in ex.map(_gguf_job, jobs, chunksize=1):
            types[typ] = types.get(typ, 0) + 1
            yield names[hf_name], w
    log(f"  GGUF block linear types: {types}")
    src = GGUFSource(path)
    V, H = shapes.get("lm_head.weight") or shapes[prefix + "embed_tokens.weight"]
    he = src.head_and_embed(V, H)
    log(f"  GGUF head {he['head_type']}, embed {he['embed_type']}")
    yield "lm_head.weight", he["head"]
    yield prefix + "embed_tokens.weight", he["embed"]


# ---------------------------------------------------------------- writer

def export(base: str, out: Path, repl_iter_fn, *, info: dict) -> dict:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    t0 = time.time()
    src = Path(snapshot_download(base))
    log(f"base checkpoint at {src} ({time.time() - t0:.0f}s)")
    idx_path = src / "model.safetensors.index.json"
    if idx_path.exists():
        weight_map = json.loads(idx_path.read_text())["weight_map"]
    else:
        with safe_open(str(src / "model.safetensors"), "pt") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}
    shapes = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(src / shard), "pt") as f:
            for k in f.keys():
                shapes[k] = tuple(f.get_slice(k).get_shape())
    prefix = text_prefix(weight_map)
    names = block_names(weight_map, prefix)
    # shard order, so each shard is written (and its replacements freed) as early as possible
    names = dict(sorted(names.items(), key=lambda kv: (weight_map[kv[1]], kv[1])))
    log(f"{len(names)} block linears (prefix {prefix!r})")

    out.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():                       # config, tokenizer, chat template, index, ...
        if p.is_file() and not p.name.endswith(".safetensors"):
            shutil.copy2(p, out / p.name)

    # replacements, grouped by shard as they arrive (bounded memory: one shard at a time is
    # written once all of its replacements are in)
    pending: dict[str, dict] = {}
    need: dict[str, set] = {}
    # tied-embedding checkpoints have no lm_head tensor: only the embedding is replaced there
    targets = set(names.values()) | {prefix + "embed_tokens.weight"} | (
        {"lm_head.weight"} if "lm_head.weight" in weight_map else set())
    for k in targets:
        if k not in weight_map:
            raise ValueError(f"base checkpoint has no tensor {k}")
        need.setdefault(weight_map[k], set()).add(k)
    stats = {"replaced": 0, "rel_err_max": 0.0}
    done_shards = set()

    def write_shard(shard):
        with safe_open(str(src / shard), "pt") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
            meta = f.metadata()
        for k, v in pending.pop(shard, {}).items():
            orig = tensors[k]
            if tuple(v.shape) != tuple(orig.shape):
                raise ValueError(f"{k}: replacement {tuple(v.shape)} != base {tuple(orig.shape)}")
            o = orig.float()
            rel = float((v.float() - o).pow(2).mean() / o.pow(2).mean().clamp_min(1e-30))
            stats["rel_err_max"] = max(stats["rel_err_max"], rel)
            if rel > 0.5:                         # a mis-mapped (transposed / permuted) matrix lands near 2
                raise ValueError(f"{k}: relative error {rel:.3f} vs the base weight: mapping is wrong")
            tensors[k] = v.to(orig.dtype).contiguous()
            stats["replaced"] += 1
        save_file(tensors, str(out / shard), metadata=meta or {"format": "pt"})
        done_shards.add(shard)

    for ck, w in repl_iter_fn(names, shapes, prefix):
        if ck not in targets:
            continue
        shard = weight_map[ck]
        pending.setdefault(shard, {})[ck] = w.to(torch.bfloat16)
        need[shard].discard(ck)
        if not need[shard]:
            write_shard(shard)
    for shard in sorted(set(weight_map.values())):
        if shard not in done_shards:
            if need.get(shard):
                raise RuntimeError(f"{shard}: {len(need[shard])} replacements never arrived")
            write_shard(shard)
    if stats["replaced"] != len(targets):
        raise RuntimeError(f"replaced {stats['replaced']} tensors, expected {len(targets)}")
    rec = {"base": base, "out": str(out), "block_linears": len(names), **stats, **info,
           "seconds": time.time() - t0}
    (out / "shapequant_export.json").write_text(json.dumps(rec, indent=2))
    log(f"exported {stats['replaced']} tensors to {out} in {rec['seconds']:.0f}s "
        f"(max rel err vs base {stats['rel_err_max']:.4f})")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--packed")
    g.add_argument("--packed-hf")
    g.add_argument("--gguf")
    g.add_argument("--gguf-hf")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--record", help="also write the export record (json) here")
    a = ap.parse_args()
    if a.packed or a.packed_hf:
        path = a.packed or hf_file(a.packed_hf)
        info = {"source": "packed", "path": a.packed or a.packed_hf}
        fn = lambda names, shapes, prefix: packed_replacements(path, names, prefix, a.device)  # noqa: E731
    else:
        path = a.gguf or hf_file(a.gguf_hf)
        info = {"source": "gguf", "path": a.gguf or a.gguf_hf}
        fn = lambda names, shapes, prefix: gguf_replacements(path, names, shapes, prefix, a.workers)  # noqa: E731
    rec = export(a.base, Path(a.out), fn, info=info)
    if a.record:
        Path(a.record).parent.mkdir(parents=True, exist_ok=True)
        Path(a.record).write_text(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
