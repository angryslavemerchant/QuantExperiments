"""Runtime speed with a real model's shapes and random codes (speed does not depend on the
values): writes a self-contained packed file (random tags, G=4 books, the deployed 4-bit
embedding / 8-bit head formats) from the model's config alone (no weights downloaded), builds
the engine (shapequant.runtime for Qwen2-style models, shapequant.qwen35_runtime for qwen3_5)
and times decode (tg, CUDA graph) and batched prefill (pp).

    python scripts/bench_synthetic.py --model Qwen/Qwen2.5-7B-Instruct --bits 10 12 \\
        --out artifacts/bench/synth_7b.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shapequant import masks as mk, packed  # noqa: E402
from shapequant.pipeline import compressible_linears  # noqa: E402


def write_synthetic(model_name: str, bits: int, path: Path) -> dict:
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_name)
    cfg = getattr(cfg, "text_config", cfg)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg)
    g = torch.Generator().manual_seed(0)
    lins = dict(compressible_linears(m))
    items = {}
    K = 2 ** bits
    for j, (n, mod) in enumerate(lins.items()):
        R, C = mod.weight.shape
        t = {"tags": packed.pack_bits(torch.randint(0, K, (R, C // 4), generator=g), bits),
             "table": (torch.randn(4 * K, 4, generator=g) * 0.5).half(),
             "row_scale": (torch.rand(R, generator=g) * 0.02 + 0.01).half(),
             "row_book": torch.randint(0, 4, (R,), generator=g).to(torch.uint8)}
        perms = mk.seeded_perms(C, 1, torch.Generator().manual_seed(j))
        items[n] = (t, {"R": R, "C": C, "d": 4, "K": K, "G": 4, "M": 1, "seed": j, "perm_crc": packed.perm_crc(perms),
                        "bits": R * C * bits / 4 + R * 16 + 4 * K * 4 * 16 + R * 2})
    V, H = cfg.vocab_size, cfg.hidden_size
    skip = {f"{n}.weight" for n in lins} | {"lm_head.weight", "model.embed_tokens.weight"}
    dense = {n: (torch.randn(p.shape, generator=g) * 0.05).to(torch.bfloat16) for n, p in m.named_parameters()
             if n not in skip}
    for n in dense:
        if n.endswith("A_log"):
            dense[n] = torch.zeros_like(dense[n])
    at = {"aux.embed.q": torch.randint(0, 255, (V, H // 2), dtype=torch.uint8, generator=g),
          "aux.embed.scale": torch.full((V, H // 32), 0.01).half(),
          "aux.lm_head.q": torch.randint(-127, 127, (V, H), dtype=torch.int8, generator=g),
          "aux.lm_head.scale": torch.full((V, H // 32), 0.001).half()}
    at.update({f"aux.dense.{n}": v for n, v in dense.items()})
    am = {"embed": "q4_b32", "lm_head": "q8_b32", "dense": sorted(dense), "config": cfg.to_dict(), "vocab": V,
          "hidden": H, "tied": bool(getattr(cfg, "tie_word_embeddings", False))}
    return packed.save(path, items, base_model=model_name, aux=(at, am))


@torch.no_grad()
def bench(path: Path, model_type: str, tg: int, pp: int, max_len: int) -> dict:
    if model_type == "qwen3_5" or model_type == "qwen3_5_text":
        from shapequant import qwen35_runtime as q35
        eng, info = q35.engine_from_packed(path, max_len=max_len)
        q35.attach_prefill(eng)
        run_pp = lambda ids: q35.prefill(eng, ids)          # noqa: E731
        tg_s = eng.time_tokens(tg, 3)
        embed_bytes = eng.embed.nbytes()
    else:
        from shapequant import runtime
        from shapequant.prefill import prefill
        eng, info = runtime.engine_from_packed(path, max_len=max_len, prefill=True)
        run_pp = lambda ids: prefill(eng, ids, logits="last")  # noqa: E731
        tg_s = eng.time_tokens(tg, 3)
        embed_bytes = info["embed_bytes"]
    ids = torch.randint(0, 1000, (pp,), device="cuda")
    ts = []
    for _ in range(4):
        eng.reset(0)
        torch.cuda.synchronize()
        t = time.time()
        run_pp(ids)
        torch.cuda.synchronize()
        ts.append(time.time() - t)
    return {"tg": tg, "tg_tok_s": 1 / tg_s, "pp": pp, "pp_tok_s": pp / sorted(ts[1:])[1],
            "weight_bytes": info["weight_bytes"], "embed_bytes": embed_bytes,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, nargs="+", default=[10, 12])
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--pp", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(a.model)
    mt = getattr(getattr(c, "text_config", c), "model_type", c.model_type)
    res = {"model": a.model, "model_type": mt, "gpu": torch.cuda.get_device_name(), "runs": {}}
    for bits in a.bits:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "synth.safetensors"
            s = write_synthetic(a.model, bits, p)
            torch.cuda.reset_peak_memory_stats()
            r = bench(p, mt, a.tg, a.pp, a.max_len)
            r["bpw_blocks"] = s["bpw"]
            r["bpw_all"] = s.get("bpw_all")
            res["runs"][f"K{2 ** bits}"] = r
            print(f"{a.model} K{2 ** bits}: tg{a.tg} {r['tg_tok_s']:.1f} tok/s, pp{a.pp} {r['pp_tok_s']:.0f} tok/s, "
                  f"weights {r['weight_bytes'] / 1e9:.2f} GB, peak {r['peak_gb']:.2f} GB", flush=True)
        torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
