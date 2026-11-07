"""Per-matrix time (L2-cold, CUDA-graph replay) of the ways to multiply M activation rows by a packed
matrix: batched run kernel (32 rows per launch), the same with X staged in shared memory (xs; up
to 128 rows per launch, 4 or 8 warps per block), fused dequant-GEMM (kernels.runs_gemm), optionally
GPU dequant + fp16 GEMM; plus the bf16 dense GEMM a stock engine does.

  python scripts/bench_mid_batch.py --packed FILE --names model.layers.0.mlp.gate_proj,model.layers.0.mlp.down_proj
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


_FLUSH = None
_FLUSH_US = None


def flush():
    """Evict L2 (a 512 MB write): in a model every matrix is read from DRAM once per step."""
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    _FLUSH.zero_()


def timeit(fn, n=20, reps=3, cold=True) -> float:
    """GPU time per call (us), CUDA-graph replay of n calls (as vLLM's captured decode steps; no
    host launch overhead), min of reps. cold: L2 flushed before each call (the flush's own time,
    measured the same way, is subtracted)."""
    if cold:
        global _FLUSH_US
        if _FLUSH_US is None:
            _FLUSH_US = timeit(flush, n, reps, cold=False)
        return timeit(lambda: (flush(), fn()), n, reps, cold=False) - _FLUSH_US
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / n)
    del g
    return min(ts) * 1e3          # us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--Ms", default="1,8,32,48,64,96,128,192,256,512,1024")
    ap.add_argument("--slow", action="store_true", help="also dequant + GEMM")
    ap.add_argument("--bm-sweep", action="store_true", help="also the fused kernel at other token tiles")
    ap.add_argument("--out")
    a = ap.parse_args()
    from shapequant.kernels import batched_runs as br, cuda_dequant, runs_gemm
    from shapequant.packed import PackedFile
    from shapequant.runtime import runs_linear
    pf = PackedFile(a.packed)
    rows = []
    for name in a.names.split(","):
        rl = runs_linear(pf, name).rl
        R, C = rl.R, rl.nch * 4
        lb = br.RunsBatch(rl, bmax=32, layout="tile")
        lbx = {w: br.RunsBatch(rl, bmax=128, layout="tile", xs=True, warps=w) for w in (4, 8)}
        Wd = cuda_dequant.dequant(rl).clone().view(R, C).to(torch.bfloat16)
        for M in map(int, a.Ms.split(",")):
            x = torch.randn(M, C, device="cuda").half()
            xb = x.to(torch.bfloat16)
            y = torch.empty(M, R, dtype=torch.float16, device="cuda")

            def chunks(launcher, n):
                xc = [x[s:s + n].contiguous() for s in range(0, M, n)]
                yc = [torch.empty(xi.shape[0], R, dtype=torch.float16, device="cuda") for xi in xc]
                return lambda: [launcher.run(xi, yi, xi.shape[0]) for xi, yi in zip(xc, yc)]

            r = {"name": name, "R": R, "C": C, "bits": rl.bits, "M": M,
                 "batched_us": timeit(chunks(lb, 32)),
                 **{f"xs_w{w}_us": timeit(chunks(l, 128)) for w, l in lbx.items()},
                 "fused_us": timeit(lambda: runs_gemm.gemm(x, [rl], out=y)) if runs_gemm.supported(rl) else None,
                 **({f"fused_bm{bm}_us": timeit(lambda bm=bm: runs_gemm.gemm(x, [rl], out=y, bm=bm))
                     for bm in (32, 64, 128) if bm != runs_gemm.auto_bm(M) and bm * 4 >= M} if a.bm_sweep and M > 16 else {}),
                 "bf16_us": timeit(lambda: torch.matmul(xb, Wd.t()))}
            if a.slow:
                scratch = torch.empty(R * C, dtype=torch.float16, device="cuda")
                r["dequant_gemm_us"] = timeit(lambda: torch.matmul(x, cuda_dequant.dequant(rl, scratch).view(R, C).t(), out=y))
            rows.append(r)
            print(f"{name.split('layers.')[-1]:26s} R={R:6d} C={C:6d} b={rl.bits:2d} M={M:4d}  " +
                  "  ".join(f"{k[:-3]} {v:7.1f}" for k, v in r.items() if k.endswith("_us") and v is not None) + " us",
                  flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
