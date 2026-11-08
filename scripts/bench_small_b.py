"""Small-batch launch study for the batched run kernel (the vLLM plugin's decode path) on real
27B matrices: per matrix and batch B, the default RunsBatch config vs the best of a launch grid
(warps x blocks/SM x split-K), for the tile layout with and without X staged in shared memory, next
to the batch-1 lane kernel (proto_runs.gemv, the standalone runtime's B = 1 path) and bf16.
CUDA-graph timed, L2 not flushed (decode-like: weights streamed once per call anyway).

  python scripts/bench_small_b.py --packed PATH --names model.layers.0.mlp.gate_proj,... --out x.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def timeit(fn, n: int = 20, reps: int = 5) -> float:
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
    return min(ts) * 1e3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--Bs", default="1,2,4,8,16,32")
    ap.add_argument("--out")
    a = ap.parse_args()
    from shapequant.kernels import batched_runs as br, cuda_dequant, proto_runs
    from shapequant.packed import PackedFile
    from shapequant.runtime import runs_linear
    from shapequant.qwen35_runtime import default_large
    pf = PackedFile(a.packed)
    grid = list(itertools.product((4, 8), (1, 2, 4), (1, 2, 4, 8, 16)))
    rows = []
    for name in a.names.split(","):
        lin = runs_linear(pf, name, large=default_large())
        rl = lin.rl
        R, C = rl.R, rl.nch * 4
        Wd = cuda_dequant.dequant(rl).clone().view(R, C).to(torch.bfloat16)
        for B in map(int, a.Bs.split(",")):
            x = torch.randn(B, C, device="cuda").half()
            y = torch.empty(B, R, dtype=torch.float16, device="cuda")
            r = {"name": name, "R": R, "C": C, "bits": rl.bits, "B": B}
            if B == 1:
                xp = x[0].contiguous()
                y1 = torch.empty(R, dtype=torch.float16, device="cuda")
                r["batch1_us"] = timeit(lambda: lin.run(xp, y1))
            for xs in (False, True):
                bmax = max(B, 8)
                if not xs and bmax > br.BMAX:
                    continue
                d = br.RunsBatch(rl, bmax=bmax, layout="tile", xs=xs)
                r[f"default_xs{int(xs)}_us"] = timeit(lambda: d.run(x, y, B))
                best, seen = None, set()
                for w, bps, m in grid:
                    lb = br.RunsBatch(rl, bmax=bmax, warps=w, bps=bps, max_split=m, layout="tile", xs=xs)
                    eff = (w, lb.grid, lb.S)
                    if eff in seen:
                        continue
                    seen.add(eff)
                    t = timeit(lambda: lb.run(x, y, B), n=10, reps=3)
                    if best is None or t < best[0]:
                        best = (t, (w, bps, m))
                r[f"best_xs{int(xs)}_us"], r[f"best_xs{int(xs)}_cfg"] = best
            xb = x.to(torch.bfloat16)
            r["bf16_us"] = timeit(lambda: torch.matmul(xb, Wd.t()))
            rows.append(r)
            print(f"{name.split('layers.')[-1]:26s} B={B:3d} " + "  ".join(
                f"{k[:-3]} {v:6.1f}" for k, v in r.items() if k.endswith("_us")) +
                f"  cfg xs0 {r.get('best_xs0_cfg')} xs1 {r.get('best_xs1_cfg')}", flush=True)
            if a.out:
                Path(a.out).write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
