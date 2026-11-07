"""Compile the prefill CUDA kernels for other GPU architectures with NVRTC (no GPU of that kind
needed), e.g. sm_120 (RTX PRO 6000 Blackwell) from a Turing laptop:

    python scripts/compile_check.py --arch sm_120 sm_89 sm_80 sm_75
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sources():
    from shapequant.kernels import cuda_dequant, prefill_kernels, runs_gemm
    out = {}
    for bits in (8, 10, 12):
        for ts in (0, 1):
            out[f"runs_gemm bits={bits} tsmem={ts}"] = f"#define BITS {bits}\n#define TSMEM {ts}\n" + runs_gemm._SRC
        out[f"dequant bits={bits}"] = f"#define BITS {bits}\n" + cuda_dequant._SRC
    for bv in (16, 32):
        out[f"prefill_kernels bv={bv}"] = f"#define SDK 128\n#define SDV 128\n#define SBV {bv}\n" + prefill_kernels._SRC
    try:
        from shapequant.kernels import fp8_gemm
        for bits in (8, 10, 12):
            out[f"fp8_gemm bits={bits}"] = fp8_gemm.source(bits)
    except ImportError:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", nargs="+", default=["sm_120", "sm_89", "sm_80", "sm_75"])
    a = ap.parse_args()
    import os

    import cupy
    from cupy.cuda import compiler
    inc = ("-I" + os.path.join(cupy.cuda.get_cuda_path(), "include"),)
    bad = 0
    for arch in a.arch:
        for name, src in sources().items():
            try:
                compiler.compile_using_nvrtc(src, options=(f"--gpu-architecture={arch}", "-std=c++17") + inc)
                print(f"ok    {arch} {name}")
            except Exception as e:                      # noqa: BLE001
                bad += 1
                print(f"FAIL  {arch} {name}: {str(e)[:2000]}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
