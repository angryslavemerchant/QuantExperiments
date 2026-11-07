"""Prefill (pp) speed of the runtime on a model's shapes with random codes, old vs new path.

Writes a self-contained synthetic packed file from the model's config alone (bench_synthetic:
random tags at the shipped format, G=4 books, 4-bit embedding / 8-bit head; no weights are
downloaded), builds the engine (shapequant.runtime for Qwen2-style models, qwen35_runtime for
Qwen3.5 / 3.8) and times one batched prefill of T tokens per path, the paths interleaved round
by round (so clock drift hits them alike), median over rounds. Logits for the last token only,
as llama-bench's pp.

Paths: ref (the original torch prefill: per-matrix dequant + cuBLAS, torch elementwise, HF's
chunked DeltaNet), fast (fused prefill kernels, grouped GEMMs; GEMM = dequant + cuBLAS),
fused (fast with the fused dequant-GEMM kernel, kernels.runs_gemm).

    python scripts/bench_prefill.py --model Qwen/Qwen3.8-27B --T 512 2048 --out artifacts/bench/pp_27b.json
    python scripts/bench_prefill.py --model Qwen/Qwen2.5-7B-Instruct --bits 10 12 --paths ref fast fused
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
         "overlap": {"impl": "fast", "gemm": "overlap"}, "fused": {"impl": "fast", "gemm": "fused"}}


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
        times = {p: [] for p in paths}
        for p in paths:                       # warm-up: compile kernels, cuBLAS heuristics, allocator
            time_one(eng, run, ids, PATHS[p])
        t0 = time.time()                      # and >= 2 s of work so the GPU clocks settle
        while time.time() - t0 < 2.0:
            time_one(eng, run, ids, PATHS[paths[-1]])
        torch.cuda.reset_peak_memory_stats()
        for _ in range(rounds):
            for p in paths:
                times[p].append(time_one(eng, run, ids, PATHS[p]))
        for p in paths:
            s = statistics.median(times[p])
            out.append({"T": T, "path": p, "pp_tok_s": T / s, "ms": s * 1e3,
                        "ms_all": [round(x * 1e3, 2) for x in times[p]]})
        base = out[-len(paths)]["pp_tok_s"]
        print("  T=%d: " % T + ", ".join(f"{r['path']} {r['pp_tok_s']:.0f} tok/s ({r['pp_tok_s'] / base:.2f}x)"
                                          for r in out[-len(paths):]), flush=True)
        out[-1]["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id (only its config is read)")
    ap.add_argument("--bits", type=int, nargs="+", default=[10], help="tag bits (K = 2^bits)")
    ap.add_argument("--T", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--paths", nargs="+", default=["ref", "fast", "fused"], choices=list(PATHS))
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=0, help="KV cache length (default: max T)")
    ap.add_argument("--out", help="JSON output path")
    ap.add_argument("--tmp", help="directory for the synthetic packed file (default: system temp)")
    a = ap.parse_args()
    from bench_synthetic import write_synthetic
    mt = model_type(a.model)
    res = {"model": a.model, "model_type": mt, "gpu": torch.cuda.get_device_name(),
           "cc": "%d.%d" % torch.cuda.get_device_capability(), "torch": torch.__version__, "runs": {}}
    for bits in a.bits:
        with tempfile.TemporaryDirectory(dir=a.tmp) as d:
            p = Path(d) / "synth.safetensors"
            t0 = time.time()
            s = write_synthetic(a.model, bits, p)
            eng, run, info = build(p, mt, a.max_len or max(a.T))
            print(f"{a.model} K{2 ** bits}: synthetic engine in {time.time() - t0:.0f} s, "
                  f"weights {info['weight_bytes'] / 1e9:.2f} GB", flush=True)
            r = bench(eng, run, a.T, a.paths, a.rounds, eng.cfg.vocab)
            res["runs"][f"K{2 ** bits}"] = {"bpw_blocks": s["bpw"], "weight_bytes": info["weight_bytes"], "results": r}
            del eng, run
        torch.cuda.empty_cache()
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
