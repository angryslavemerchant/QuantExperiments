"""Where batched prefill time goes: torch.profiler (CUPTI kernel trace) over one prefill of a
synthetic engine (bench_synthetic.write_synthetic: a model's shapes, random codes), kernels
grouped into components by name.

    python scripts/profile_prefill.py --model Qwen/Qwen2.5-1.5B-Instruct --T 512 2048 [--path old|new]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# (component, substrings of the kernel name), first match wins
RULES = [
    ("dequant", ("dequant_runs",)),
    ("fused_gemm", ("runs_gemm",)),
    ("deltanet", ("chunk_gdr", "gdr_")),
    ("attention", ("flash", "fmha", "attention", "softmax")),
    ("gemm", ("gemm", "cutlass", "s1688", "s884", "h1688", "h884", "sm75_xmma", "sm80_xmma", "xmma", "cublas",
              "Kernel2", "splitKreduce", "ampere", "turing", "sm90", "sm100", "sm120")),
    ("copy/cat", ("copy", "cat", "Cat", "index", "gather", "scatter")),
]


def classify(name: str) -> str:
    for comp, keys in RULES:
        if any(k in name for k in keys):
            return comp
    return "elementwise/other"


def build(model: str, bits: int, max_len: int, d: Path):
    from bench_synthetic import write_synthetic
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model)
    mt = getattr(getattr(c, "text_config", c), "model_type", c.model_type)
    p = d / "synth.safetensors"
    write_synthetic(model, bits, p)
    if mt in ("qwen3_5", "qwen3_5_text"):
        from shapequant import qwen35_runtime as q35
        eng, _ = q35.engine_from_packed(p, max_len=max_len)
        q35.attach_prefill(eng)
        return eng, mt, (lambda ids, **kw: q35.prefill(eng, ids, **kw))
    from shapequant import runtime
    from shapequant import prefill as pf
    eng, _ = runtime.engine_from_packed(p, max_len=max_len, prefill=True)
    return eng, mt, (lambda ids, **kw: pf.prefill(eng, ids, logits="last", **kw))


def timeit(eng, run, ids, reps=3, **kw):
    ts = []
    for _ in range(reps + 1):
        eng.reset(0)
        torch.cuda.synchronize()
        t = time.perf_counter()
        run(ids, **kw)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return sorted(ts[1:])[len(ts[1:]) // 2]


def profile(eng, run, ids, **kw):
    from torch.profiler import ProfilerActivity, profile as tprof
    eng.reset(0)
    run(ids, **kw)
    torch.cuda.synchronize()
    eng.reset(0)
    with tprof(activities=[ProfilerActivity.CUDA]) as prof:
        run(ids, **kw)
        torch.cuda.synchronize()
    comp = defaultdict(float)
    top = defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            us = e.device_time if hasattr(e, "device_time") else e.cuda_time
            comp[classify(e.name)] += us
            top[e.name] += us
    return dict(comp), sorted(top.items(), key=lambda kv: -kv[1])[:15]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, default=10)
    ap.add_argument("--T", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--path", nargs="+", default=["default"], help="prefill gemm path(s): old, new or default")
    ap.add_argument("--json")
    ap.add_argument("--top", action="store_true")
    a = ap.parse_args()
    res = {"model": a.model, "gpu": torch.cuda.get_device_name(), "runs": []}
    with tempfile.TemporaryDirectory() as d:
        eng, mt, run = build(a.model, a.bits, max(a.T), Path(d))
        for T in a.T:
            ids = torch.randint(0, 1000, (T,), device="cuda")
            for path in a.path:
                kw = {} if path == "default" else ({"impl": "ref"} if path == "ref" else {"impl": "fast", "gemm": path})
                s = timeit(eng, run, ids, **kw)
                comp, top = profile(eng, run, ids, **kw)
                tot = sum(comp.values())
                r = {"T": T, "path": path, "tok_s": T / s, "wall_ms": s * 1e3, "kernel_ms": tot / 1e3,
                     "components_ms": {k: v / 1e3 for k, v in sorted(comp.items(), key=lambda kv: -kv[1])}}
                res["runs"].append(r)
                print(f"{a.model} T={T} path={path}: {T / s:.0f} tok/s (wall {s * 1e3:.1f} ms, kernels {tot / 1e3:.1f} ms)")
                for k, v in r["components_ms"].items():
                    print(f"    {k:18s} {v:8.2f} ms  {100 * v * 1e3 / tot:5.1f}%")
                if a.top:
                    for n, v in top:
                        print(f"      {v / 1e3:8.2f} ms  {n[:110]}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
