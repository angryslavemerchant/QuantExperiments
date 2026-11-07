"""Per-matrix time of the three ways to multiply M activation rows by a packed matrix, for the
vLLM plugin's large-M branch: chunked batched run kernel (32 rows per launch), GPU dequant + fp16
cuBLAS GEMM, fused dequant-GEMM (kernels.runs_gemm); plus the bf16 dense GEMM a stock engine does.

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


def timeit(fn, n=20, reps=3) -> float:
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(n):
            fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / n)
    return min(ts) * 1e3          # us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--Ms", default="1,8,32,48,64,96,128,192,256,512,1024")
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
        scratch = torch.empty(R * C, dtype=torch.float16, device="cuda")
        Wd = cuda_dequant.dequant(rl).clone().view(R, C).to(torch.bfloat16)
        for M in map(int, a.Ms.split(",")):
            x = torch.randn(M, C, device="cuda").half()
            xb = x.to(torch.bfloat16)
            y = torch.empty(M, R, dtype=torch.float16, device="cuda")
            yc = [torch.empty(min(32, M - s), R, dtype=torch.float16, device="cuda") for s in range(0, M, 32)]
            xc = [x[s:s + 32].contiguous() for s in range(0, M, 32)]

            def chunked():
                for xi, yi in zip(xc, yc):
                    lb.run(xi, yi, xi.shape[0])

            def deq():
                w = cuda_dequant.dequant(rl, scratch).view(R, C)
                torch.matmul(x, w.t(), out=y)

            def fused():
                runs_gemm.gemm(x, [rl], out=y)

            def bf16():
                torch.matmul(xb, Wd.t())

            r = {"name": name, "R": R, "C": C, "bits": rl.bits, "M": M,
                 "chunked_us": timeit(chunked), "dequant_gemm_us": timeit(deq),
                 "fused_us": timeit(fused) if runs_gemm.supported(rl) else None, "bf16_us": timeit(bf16)}
            rows.append(r)
            print(f"{name.split('layers.')[-1]:28s} R={R:6d} C={C:6d} b={rl.bits:2d} M={M:5d}  chunked {r['chunked_us']:8.1f}  "
                  f"deq+gemm {r['dequant_gemm_us']:8.1f}  fused {r['fused_us'] or float('nan'):8.1f}  bf16 {r['bf16_us']:8.1f} us",
                  flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
