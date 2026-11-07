"""Prefill (pp) speed of the runtime on a model's shapes with random codes, old vs new path.

Writes a self-contained synthetic packed file from the model's config alone (bench_synthetic:
random tags at the shipped format, G=4 books, 4-bit embedding / 8-bit head; no weights are
downloaded), builds the engine (shapequant.runtime for Qwen2-style models, qwen35_runtime for
Qwen3.5 / 3.8) and times one batched prefill of T tokens per path, the paths interleaved round
by round (so clock drift hits them alike), median over rounds. Logits for the last token only,
as llama-bench's pp.

Paths: ref (the original torch prefill: per-matrix dequant + cuBLAS, torch elementwise, HF's
chunked DeltaNet), fast (the default: fused prefill kernels, grouped GEMMs; GEMM = dequant +
cuBLAS), overlap (fast, the next group's dequant on a side stream during the current GEMM),
fused (fast with the fused dequant-GEMM kernel, kernels.runs_gemm), fp8 (EXPERIMENTAL, changes
numerics, cc >= 8.9 only: e4m3 weights / activations, torch._scaled_mm). A path that raises is
reported with its error and skipped. Codebooks: --bits mix (default) cycles K = 4096 / 1024 /
4096 / 1024 / 256 over the matrices like the shipped 27B allocation; --bits 10 = K 1024 everywhere.

    python scripts/bench_prefill.py --model Qwen/Qwen3.8-27B --out artifacts/bench/pp_27b.json
    python scripts/bench_prefill.py --model Qwen/Qwen2.5-7B-Instruct --bits mix 10 --T 512 2048 4096
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

PATHS = {"ref": {"impl": "ref"}, "fast": {"impl": "fast", "gemm": "dequant"},
         "overlap": {"impl": "fast", "gemm": "overlap"}, "fused": {"impl": "fast", "gemm": "fused"},
         "fp8": {"impl": "fast", "gemm": "fp8"}}     # fp8: EXPERIMENTAL (numerics change), cc >= 8.9 only


def model_type(model: str) -> str:
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model)
    return getattr(getattr(c, "text_config", c), "model_type", c.model_type)


def build(path: Path, mt: str, max_len: int):
    """-> (engine, run(ids, **path_kw), info)"""
    if mt in ("qwen3_5", "qwen3_5_text"):
        from shapequant import qwen35_runtime as q35
        eng, info = q35.engine_from_packed(path, max_len=max_len)
        q35.attach_prefill(eng)
        return eng, (lambda ids, **kw: q35.prefill(eng, ids, logits="last", **kw)), info
    from shapequant import runtime
    from shapequant.prefill import prefill
    eng, info = runtime.engine_from_packed(path, max_len=max_len, prefill=True)
    return eng, (lambda ids, **kw: prefill(eng, ids, logits="last", **kw)), info


def time_one(eng, run, ids, kw) -> float:
    eng.reset(0)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = time.perf_counter()
    a.record()
    run(ids, **kw)
    b.record()
    torch.cuda.synchronize()
    return max(a.elapsed_time(b) / 1e3, 0.0) or (time.perf_counter() - t)


def bench(eng, run, Ts, paths, rounds: int, vocab: int) -> list[dict]:
    out = []
    for T in Ts:
        ids = torch.randint(0, min(vocab, 100000), (T,), device="cuda")
        ok, errors = [], {}
        for p in paths:                       # warm-up: compile kernels, cuBLAS heuristics, allocator
            try:
                time_one(eng, run, ids, PATHS[p])
                ok.append(p)
            except Exception as e:            # noqa: BLE001 - e.g. fp8 on an unsupported GPU: skip the path
                errors[p] = f"{type(e).__name__}: {e}"[:500]
                print(f"  T={T}: path {p} failed, skipped: {errors[p]}", flush=True)
                torch.cuda.synchronize()
        if not ok:
            out += [{"T": T, "path": p, "error": e} for p, e in errors.items()]
            continue
        t0 = time.time()                      # >= 2 s of work so the GPU clocks settle
        while time.time() - t0 < 2.0:
            time_one(eng, run, ids, PATHS[ok[-1]])
        torch.cuda.reset_peak_memory_stats()
        times = {p: [] for p in ok}
        for _ in range(rounds):
            for p in ok:
                times[p].append(time_one(eng, run, ids, PATHS[p]))
        rows = []
        for p in ok:
            s = statistics.median(times[p])
            rows.append({"T": T, "path": p, "pp_tok_s": T / s, "ms": s * 1e3,
                         "ms_all": [round(x * 1e3, 2) for x in times[p]]})
        base = rows[0]["pp_tok_s"]
        print("  T=%d: " % T + ", ".join(f"{r['path']} {r['pp_tok_s']:.0f} tok/s ({r['pp_tok_s'] / base:.2f}x)"
                                          for r in rows), flush=True)
        rows[-1]["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        out += rows + [{"T": T, "path": p, "error": e} for p, e in errors.items()]
    return out


MIX = (4096, 1024, 4096, 1024, 256)       # ~ the shipped 27B allocation's K mix (40 / 35 / 25 %)


def write_synthetic_mixed(model_name: str, path: Path) -> dict:
    """bench_synthetic.write_synthetic with K cycling through MIX per matrix (the shipped files mix
    K = 256 / 1024 / 4096; speed depends on K through the codebook size)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from shapequant import masks as mk, packed
    from shapequant.pipeline import compressible_linears
    cfg = AutoConfig.from_pretrained(model_name)
    cfg = getattr(cfg, "text_config", cfg)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg)
    g = torch.Generator().manual_seed(0)
    lins = dict(compressible_linears(m))
    items = {}
    for j, (n, mod) in enumerate(lins.items()):
        R, C = mod.weight.shape
        K = MIX[j % len(MIX)]
        bits = K.bit_length() - 1
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id (only its config is read)")
    ap.add_argument("--bits", nargs="+", default=["mix"],
                    help="tag bits per run (K = 2^bits, every matrix), or 'mix' (K cycling 4096/1024/4096/1024/256)")
    ap.add_argument("--T", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--paths", nargs="+", default=["ref", "fast", "overlap", "fused", "fp8"], choices=list(PATHS),
                    help="fp8 is skipped below compute capability 8.9")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=0, help="KV cache length (default: max T)")
    ap.add_argument("--out", help="JSON output path")
    ap.add_argument("--tmp", help="directory for the synthetic packed file (default: system temp)")
    a = ap.parse_args()
    from bench_synthetic import write_synthetic
    mt = model_type(a.model)
    res = {"model": a.model, "model_type": mt, "gpu": torch.cuda.get_device_name(),
           "cc": "%d.%d" % torch.cuda.get_device_capability(), "torch": torch.__version__, "runs": {}}
    paths = [p for p in a.paths if p != "fp8" or torch.cuda.get_device_capability() >= (8, 9)]
    for bits in a.bits:
        with tempfile.TemporaryDirectory(dir=a.tmp) as d:
            p = Path(d) / "synth.safetensors"
            t0 = time.time()
            s = write_synthetic_mixed(a.model, p) if bits == "mix" else write_synthetic(a.model, int(bits), p)
            name = "Kmix" if bits == "mix" else f"K{2 ** int(bits)}"
            eng, run, info = build(p, mt, a.max_len or max(a.T))
            print(f"{a.model} {name}: synthetic engine in {time.time() - t0:.0f} s, "
                  f"weights {info['weight_bytes'] / 1e9:.2f} GB", flush=True)
            r = bench(eng, run, a.T, paths, a.rounds, eng.cfg.vocab)
            res["runs"][name] = {"bpw_blocks": s["bpw"], "weight_bytes": info["weight_bytes"], "results": r}
            del eng, run
        torch.cuda.empty_cache()
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
