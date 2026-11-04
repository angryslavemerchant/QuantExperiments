"""Speed payoff of bank-conflict-free tag encoding for the current-format kernel
(shapequant/kernels/cuda_gemv.py), synthetic tags, Qwen2.5-7B shapes, M = 16.

The kernel reads codeword t as one 8-byte shared-memory load per lane; a 64-bit
warp load is served per half-warp (16 lanes x 8 B = 128 B), and codeword k sits
in bank pair (book_off + k) mod 16. Random tags collide (~6 wavefronts per
lookup measured); if the encoder makes the 16 tags of each half-warp hit 16
distinct bank pairs, every lookup is the 2-wavefront minimum.

Tag patterns (per 16 consecutive chunks of a row, in stored order):
  random  : unconstrained (today)
  perm    : classes t mod 16 form a permutation (conflict-free)
  cap2    : each class used exactly twice (8 classes): <= 2-way conflicts
The tag value within its class is random, so only the bank pattern changes.

Run: python scripts/bench_bankfree.py [--bw 253]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_gemv import SHAPES, clone_lin, copies_for, l2_bytes, timeit  # noqa: E402
from shapequant import masks as mk  # noqa: E402
from shapequant.kernels import cuda_gemv  # noqa: E402
from shapequant.kernels.gemv import PackedLinear  # noqa: E402
from shapequant.packed import pack_bits  # noqa: E402

M = 16


def make_tags(R, nch, K, pattern, g):
    hi = torch.randint(0, K // 16, (R, nch), generator=g)
    if pattern == "random":
        return torch.randint(0, K, (R, nch), generator=g)
    ng = nch // 16
    if pattern == "perm":
        cls = torch.argsort(torch.rand(R * ng, 16, generator=g), dim=1)
    elif pattern == "cap2":
        base = torch.argsort(torch.rand(R * ng, 16, generator=g), dim=1)[:, :8].repeat(1, 2)
        cls = torch.gather(base, 1, torch.argsort(torch.rand(R * ng, 16, generator=g), dim=1))
    else:
        raise ValueError(pattern)
    return hi * 16 + cls.reshape(R, nch)


def build(R, C, bits, G, pattern, seed=0, dev="cuda"):
    g = torch.Generator().manual_seed(seed)
    K = 1 << bits
    tags = make_tags(R, C // 4, K, pattern, g)
    table = (torch.randn(G * K, 4, generator=g) * 0.5).half()
    rs = (torch.rand(R, generator=g) * 0.02 + 0.01).half()
    rb = torch.randint(0, G, (R,), generator=g) if G > 1 else None
    rm = torch.randint(0, M, (R,), generator=g)
    perms = mk.seeded_perms(C, M, torch.Generator().manual_seed(seed + 1))
    to = (lambda t: None if t is None else t.to(dev))
    return PackedLinear(pack_bits(tags, bits).to(dev), table.to(dev), rs.to(dev), to(rb), to(rm),
                        perms.to(dev), K, 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw", type=float, default=253.0)
    a = ap.parse_args()
    print(f"{torch.cuda.get_device_name(0)}; ceiling {a.bw:.0f} GB/s; L2 {l2_bytes() >> 20} MB "
          f"(weights rotated over >= 4x L2); M={M}; table in shared memory when <= 48 KB")
    print(f"{'matrix':13s} {'format':22s} {'tags':7s} {'bpw':>5s} {'us':>8s} {'GB/s':>6s} {'%ceil':>6s} "
          f"{'vs fp16':>7s} {'vs random':>9s} {'rel err':>8s}")
    cases = [(10, 1), (10, 4), (12, 1), (14, 1)]          # (bits, G); 14-bit table = 128 KB -> global/L1 path
    for label, (R, C) in SHAPES.items():
        g = torch.Generator().manual_seed(0)
        x = torch.randn(C, generator=g).half().cuda()
        wds = [torch.randn(R, C, device="cuda", dtype=torch.float16) for _ in range(copies_for(R * C * 2))]
        t16 = timeit([lambda w=w: torch.mv(w, x) for w in wds])
        del wds
        y = torch.empty(R, device="cuda", dtype=torch.float16)
        for bits, G in cases:
            t_rand = None
            for pattern in ("random", "perm", "cap2"):
                lin = build(R, C, bits, G, pattern)
                ref = lin.dense() @ x.float()
                lins = [lin] + [clone_lin(lin) for _ in range(copies_for(lin.stored_bytes()) - 1)]
                got = cuda_gemv.gemv(lin, x, y).float()
                err = float((got - ref).abs().max() / ref.abs().max())
                t = timeit([lambda l=l: cuda_gemv.gemv(l, x, y) for l in lins])
                if pattern == "random":
                    t_rand = t
                nbytes = lin.stored_bytes()
                gbs = nbytes / t / 1e9
                fmt = f"K={1 << bits} G={G}" + (" (L1)" if G * (1 << bits) * 8 > 48 * 1024 else "")
                print(f"{label:13s} {fmt:22s} {pattern:7s} {bits / 4:5.2f} {t * 1e6:8.1f} {gbs:6.0f} "
                      f"{100 * gbs / a.bw:5.0f}% {t16 / t:6.2f}x {t_rand / t:8.2f}x {err:8.1e}", flush=True)
                del lin, lins
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
