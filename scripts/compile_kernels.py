"""Compile every kernel variant with NVRTC, no launches (a CPU-side check before staging to a GPU box).

  python scripts/compile_kernels.py [--arch sm_120]
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=None, help="e.g. sm_120 (default: the local device's)")
    a = ap.parse_args()
    import cupy as cp
    from shapequant.kernels import batched_runs as br, runs_gemm as rg
    opts = ("-std=c++17",) + ((f"--gpu-architecture={a.arch.replace('sm_', 'compute_')}",) if a.arch else ())
    n = 0
    for bits, tsmem, bm, grouped in itertools.product((8, 10, 12), (0, 1), rg.BMS, (0, 1)):
        src = f"#define BITS {bits}\n#define TSMEM {tsmem}\n#define BM {bm}\n#define GROUPED {grouped}\n" + rg._SRC
        cp.RawModule(code=src, options=opts).compile()
        n += 1
    for bits, nb8, smem, books, tile, xs in itertools.product((8, 12), (1, 4, 16), (0, 1), (0, 1), (0, 1), (0, 1)):
        src = (f"#define BITS {bits}\n#define NFULL 2\n#define JT 4\n#define NB8 {nb8}\n#define SMEM {smem}\n"
               f"#define BOOKS {books}\n#define THREADS 256\n#define ACC16 0\n#define TILE {tile}\n#define XS {xs}\n") + br._COMMON + br._SRC_RUNS
        cp.RawModule(code=src, options=opts + ("--use_fast_math",)).compile()
        n += 1
    for bits, (nfull, jt) in itertools.product((8, 10, 12), ((2, 4), (0, 4), (2, 0))):
        src = f"#define BITS {bits}
#define NFULL {nfull}
#define JT {jt}
" + br._SRC_DEQ_TILE
        cp.RawModule(code=src, options=opts).compile()
        n += 1
    print(f"compiled {n} variants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
