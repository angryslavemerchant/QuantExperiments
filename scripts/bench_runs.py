"""Lane-owned-run GEMV (shapequant/kernels/proto_runs.py) vs Q4 and the current
kernel (cuda_gemv.py), Qwen2.5-7B shapes, synthetic tags, M = 16 column orders.

Tag patterns: random (today's encoder), and the bank-conflict-free patterns
'perm' / 'cap2' laid out for each kernel's lookup grouping (cuda_gemv: 16
consecutive chunks; runs: chunk j of 16 consecutive 16-chunk runs).

'runs+xp' times the run kernel with the M permuted copies of x prebuilt (they
can be shared by every matrix reading the same x: q/k/v, gate/up).

Run: python scripts/bench_runs.py [--bits 10,12]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_bankfree import make_tags  # noqa: E402
from bench_gemv import SHAPES, clone_lin, copies_for, timeit  # noqa: E402
from shapequant import masks as mk  # noqa: E402
from shapequant.kernels import cuda_gemv, proto_runs  # noqa: E402
from shapequant.kernels import proto_formats as pf  # noqa: E402
from shapequant.kernels.gemv import PackedLinear  # noqa: E402
from shapequant.packed import pack_bits  # noqa: E402

M = 16


def run_tags(R, nch, K, pattern, g):
    if pattern == "random":
        return torch.randint(0, K, (R, nch), generator=g)
    # chunk c = 64*j + 32*h + 2*l + e (lane 16h+l, pair j, element e); a half-warp looks up
    # (j, h, e) over l = 0..15 together -> classes form a permutation along l
    npl = nch // 64
    hi = torch.randint(0, K // 16, (R, npl, 2, 2, 16), generator=g)
    cls = torch.argsort(torch.rand(R, npl, 2, 2, 16, generator=g), dim=-1)    # [R, j, h, e, l]
    if pattern == "cap2":
        base = cls[..., :8].repeat(1, 1, 1, 1, 2)
        cls = torch.gather(base, -1, torch.argsort(torch.rand(R, npl, 2, 2, 16, generator=g), dim=-1))
    return (hi * 16 + cls).transpose(-1, -2).reshape(R, nch)                  # [R, j, h, l, e]


def build(R, C, bits, tags, seed=0, dev="cuda"):
    g = torch.Generator().manual_seed(seed + 7)
    K = 1 << bits
    table = (torch.randn(K, 4, generator=g) * 0.5).half()
    rs = (torch.rand(R, generator=g) * 0.02 + 0.01).half()
    rm = torch.randint(0, M, (R,), generator=g)
    perms = mk.seeded_perms(C, M, torch.Generator().manual_seed(seed + 1))
    return PackedLinear(pack_bits(tags, bits).to(dev), table.to(dev), rs.to(dev), None, rm.to(dev),
                        perms.to(dev), K, 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", default="10,12")
    ap.add_argument("--ru", default="1,2,4")
    a = ap.parse_args()
    print(f"{torch.cuda.get_device_name(0)}; M={M}; times are per call (CUDA graph, weights rotated over >= 4x L2)")
    print(f"{'matrix':13s} {'kernel':26s} {'us':>8s} {'vs Q4':>7s} {'vs fp16':>8s} {'rel err':>8s}")
    totals: dict = {}
    for label, (R, C) in SHAPES.items():
        g = torch.Generator().manual_seed(0)
        x = torch.randn(C, generator=g).half().cuda()
        y = torch.empty(R, device="cuda", dtype=torch.float16)
        wds = [torch.randn(R, C, device="cuda", dtype=torch.float16) for _ in range(copies_for(R * C * 2))]
        t16 = timeit([lambda w=w: torch.mv(w, x, out=y) for w in wds])
        del wds
        w0 = torch.randn(R, C, generator=g) * 0.02
        q, sc, deq = pf.q4_pack(w0)
        qs = [(q.cuda().clone(), sc.cuda()) for _ in range(copies_for(q.numel() * 4 + sc.numel() * 2))]
        tq4 = timeit([lambda c=c: pf.gemv_q4(c[0], c[1], x, y, R, C) for c in qs])
        del qs

        def report(name, t, err=float("nan")):
            totals[name] = totals.get(name, 0.0) + t * (1 if label == "down_proj" else 2)
            print(f"{label:13s} {name:26s} {t * 1e6:8.1f} {tq4 / t:6.2f}x {t16 / t:7.2f}x {err:8.1e}", flush=True)

        report("fp16", t16)
        report("Q4", tq4)
        for bits in [int(v) for v in a.bits.split(",")]:
            K = 1 << bits
            for kern, pats in (("cur", ("random",)), ("runs", ("random", "perm", "cap2"))):
                for pat in pats:
                    gg = torch.Generator().manual_seed(1)
                    tags = make_tags(R, C // 4, K, pat, gg) if kern == "cur" else run_tags(R, C // 4, K, pat, gg)
                    lin = build(R, C, bits, tags)
                    ref = lin.dense() @ x.float()
                    if kern == "cur":
                        lins = [lin] + [clone_lin(lin) for _ in range(copies_for(lin.stored_bytes()) - 1)]
                        got = cuda_gemv.gemv(lins[0], x, y).float()
                        err = float((got - ref).abs().max() / ref.abs().max())
                        report(f"K{K} {kern} {pat}", timeit([lambda l=l: cuda_gemv.gemv(l, x, y) for l in lins]), err)
                        del lin, lins
                        continue
                    xp = x[lin.perms_l].contiguous()
                    for ru in [int(v) for v in a.ru.split(",")]:
                        rl = proto_runs.RunsLinear(lin, tags, ru)
                        lins = [rl] + [rl.clone() for _ in range(copies_for(rl.stored_bytes()) - 1)]
                        y.zero_()
                        got = proto_runs.gemv(lins[0], x, y, xp=xp).float()
                        err = float((got - ref).abs().max() / ref.abs().max())
                        t = timeit([lambda l=l: proto_runs.gemv(l, x, y, xp=xp) for l in lins])
                        report(f"K{K} runs+xp RU{ru} {pat}", t, err)
                        del rl, lins
                    del lin
        torch.cuda.empty_cache()
    print("\nper decoder layer (q, k, v, o, gate, up, down):")
    for name, t in totals.items():
        print(f"  {name:26s} {t * 1e6:8.1f} us  {totals['Q4'] / t:5.2f}x vs Q4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
