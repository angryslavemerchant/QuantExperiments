"""Prefill GEMM micro-benchmark: Y = X @ W^T for one run-format matrix, three ways:
cublas (W already fp16, the lower bound for the dequant path), dequant (cuda_dequant + cuBLAS,
the prefill default) and fused (kernels.runs_gemm). Prints TFLOPS and the fused / dequant ratio.

    python scripts/bench_runs_gemm.py --shapes 3584x3584 18944x3584 3584x18944 --T 64 512 2048 --bits 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shapequant.kernels import cuda_dequant, proto_runs, runs_gemm  # noqa: E402
from shapequant.kernels.gemv import PackedLinear  # noqa: E402
from shapequant.packed import pack_bits  # noqa: E402


def make_rl(R, C, bits, G=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    K = 2 ** bits
    nch = C // 4
    tags = torch.randint(0, K, (R, nch), generator=g)
    lin = PackedLinear(pack_bits(tags, bits).cuda(), (torch.randn(G * K, 4, generator=g) * 0.5).half().cuda(),
                       (torch.rand(R, generator=g) * 0.02 + 0.01).half().cuda(),
                       torch.randint(0, G, (R,), generator=g).to(torch.uint8).cuda(), None,
                       torch.arange(C)[None].cuda(), K, 4)
    return proto_runs.RunsLinear(lin, tags.cuda())


def timeit(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(3):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / reps)
    return sorted(ts)[1]


def time_interleaved(fns: dict, rounds: int = 7, reps: int = 5) -> dict:
    """min over rounds of the mean of `reps` calls, methods interleaved round by round (robust to
    clock changes and to other work sharing the GPU)."""
    for f in fns.values():
        f()
    torch.cuda.synchronize()
    best = {k: float("inf") for k in fns}
    for _ in range(rounds):
        for k, f in fns.items():
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(reps):
                f()
            b.record()
            torch.cuda.synchronize()
            best[k] = min(best[k], a.elapsed_time(b) / reps)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["2048x2048", "12288x2048", "2048x6144"], help="RxC")
    ap.add_argument("--T", type=int, nargs="+", default=[64, 512, 2048])
    ap.add_argument("--bits", type=int, nargs="+", default=[10])
    ap.add_argument("--json")
    a = ap.parse_args()
    res = []
    for bits in a.bits:
        for s in a.shapes:
            R, C = map(int, s.split("x"))
            rl = make_rl(R, C, bits)
            w = cuda_dequant.dequant(rl).clone()
            scratch = torch.empty(R * C, dtype=torch.float16, device="cuda")
            plan = runs_gemm.plan(rl)
            for T in a.T:
                x = torch.randn(T, C, device="cuda").half()
                t = time_interleaved({"cb": lambda: F.linear(x, w),
                                      "dq": lambda: F.linear(x, cuda_dequant.dequant(rl, scratch)),
                                      "fu": lambda: runs_gemm.gemm(x, [rl])})
                t_cb, t_dq, t_fu = t["cb"], t["dq"], t["fu"]
                fl = 2 * T * R * C / 1e9
                r = {"bits": bits, "R": R, "C": C, "T": T, "cublas_ms": t_cb, "dequant_ms": t_dq, "fused_ms": t_fu,
                     "plan": plan[:2]}
                res.append(r)
                print(f"K{2 ** bits} {R}x{C} T={T:5d}: cublas {t_cb:7.3f} ms ({fl / t_cb:5.1f} TF)  "
                      f"dequant {t_dq:7.3f} ms ({fl / t_dq:5.1f} TF)  fused {t_fu:7.3f} ms ({fl / t_fu:5.1f} TF)  "
                      f"fused/dequant {t_dq / t_fu:4.2f}x  plan bn={plan[0]} tsmem={plan[1]}", flush=True)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "runs": res}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
