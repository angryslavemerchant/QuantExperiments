"""Workload for Nsight Compute: runs one GEMV case 5 times on distinct weight
copies (synthetic tags, Qwen2.5-7B shapes, M = 16). Profile with e.g.
  ncu -k regex:gemv_runs --launch-skip 3 --launch-count 1 --set full python scripts/prof_driver.py runs up --bits 10

Kernels: q4 (proto_formats.gemv_q4), cur (cuda_gemv), runs (proto_runs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_bankfree import make_tags  # noqa: E402
from bench_runs import build, run_tags  # noqa: E402
from shapequant.kernels import cuda_gemv, proto_runs  # noqa: E402
from shapequant.kernels import proto_formats as pf  # noqa: E402

SHAPES = {"up": (18944, 3584), "down": (3584, 18944), "qo": (3584, 3584), "kv": (512, 3584)}
KERNEL_REGEX = {"q4": "regex:gemv_q4", "cur": "regex:gemv_packed", "runs": "regex:gemv_runs"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel", choices=list(KERNEL_REGEX))
    ap.add_argument("shape", choices=list(SHAPES))
    ap.add_argument("--bits", type=int, default=10)
    ap.add_argument("--pattern", default="random")
    ap.add_argument("--ru", type=int, default=2)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--bps", type=int, default=4)
    a = ap.parse_args()
    R, C = SHAPES[a.shape]
    x = torch.randn(C, generator=torch.Generator().manual_seed(0)).half().cuda()
    y = torch.empty(R, device="cuda", dtype=torch.float16)
    fns = []
    for i in range(5):
        g = torch.Generator().manual_seed(i)
        if a.kernel == "q4":
            q, sc, _ = pf.q4_pack(torch.randn(R, C, generator=g) * 0.02)
            q, sc = q.cuda(), sc.cuda()
            fns.append(lambda q=q, sc=sc: pf.gemv_q4(q, sc, x, y, R, C))
        elif a.kernel == "cur":
            tags = make_tags(R, C // 4, 1 << a.bits, a.pattern, g)
            lin = build(R, C, a.bits, tags, seed=i)
            fns.append(lambda lin=lin: cuda_gemv.gemv(lin, x, y))
        else:
            tags = run_tags(R, C // 4, 1 << a.bits, a.pattern, g)
            rl = proto_runs.RunsLinear(build(R, C, a.bits, tags, seed=i), tags, a.ru)
            xp = proto_runs.prep_x(rl, x)
            fns.append(lambda rl=rl, xp=xp: proto_runs.gemv(rl, x, y, xp=xp, threads=a.threads,
                                                            blocks_per_sm=a.bps))
    for f in fns:
        f()
    torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
