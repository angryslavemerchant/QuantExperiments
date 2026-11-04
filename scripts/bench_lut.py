"""Lookup-table (activation-side decode) GEMV vs the existing kernels.
Synthetic weights, Qwen2.5-7B shapes, M = 16 masks, G = 1.
See shapequant/kernels/proto_lut.py.

Run: python scripts/bench_lut.py [--bw 253] [--quick]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_formats import M, ref_y, setup_common  # noqa: E402
from bench_gemv import SHAPES, clone_lin, copies_for, l2_bytes, synthetic, timeit  # noqa: E402
from shapequant.kernels import cuda_gemv, proto_formats as pf, proto_lut as pl  # noqa: E402
from shapequant.packed import pack_bits  # noqa: E402

# (label, KA, KB, [(KT, half, RU), ...]) -- all tile-major; best of the 2060 sweep
LUT_FORMATS = [
    ("A 32+32", 32, 32, [(4, False, 8), (2, False, 8)]),
    ("A 64+64", 64, 64, [(2, False, 4), (2, True, 8)]),
    ("A 128+128", 128, 128, [(2, True, 8)]),
    ("free K=1024", 1024, 0, [(1, True, 8)]),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw", type=float, default=253.0)
    ap.add_argument("--lut-only", action="store_true")
    ap.add_argument("--quick", action="store_true", help="only gate/up and down (the big matrices)")
    a = ap.parse_args()
    smem_max = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    print(f"{torch.cuda.get_device_name(0)}; ceiling {a.bw:.0f} GB/s; L2 {l2_bytes() >> 20} MB "
          f"(weights rotated over >= 4x L2); M={M}; smem opt-in {smem_max >> 10} KB")
    print(f"{'matrix':13s} {'format':40s} {'bpw':>5s} {'us':>8s} {'GB/s':>6s} {'%ceil':>6s} {'vs fp16':>7s} {'rel err':>8s}")
    shapes = {k: v for k, v in SHAPES.items() if not a.quick or k in ("gate/up_proj", "down_proj")}
    for label, (R, C) in shapes.items():
        g = torch.Generator().manual_seed(0)
        perms, rm, rs, x, xp = setup_common(R, C, g)
        nch = C // 4
        wds = [torch.randn(R, C, device="cuda", dtype=torch.float16) for _ in range(copies_for(R * C * 2))]
        t16 = timeit([lambda w=w: torch.mv(w, x) for w in wds])
        del wds
        y = torch.empty(R, device="cuda", dtype=torch.float16)
        y32 = torch.empty(R, device="cuda", dtype=torch.float32)

        def report(name, bpw, nbytes, fns, ref):
            got = fns[0]().float()
            err = float((got - ref).abs().max() / ref.abs().max())
            t = timeit(fns)
            gbs = nbytes / t / 1e9
            print(f"{label:13s} {name:40s} {bpw:5.2f} {t * 1e6:8.1f} {gbs:6.0f} {100 * gbs / a.bw:5.0f}% "
                  f"{t16 / t:6.2f}x {err:8.1e}", flush=True)

        # Q4 reference
        w0 = torch.randn(R, C, generator=g) * 0.02
        q, sc, deq = pf.q4_pack(w0)
        q, sc = q.cuda(), sc.cuda()
        ref = deq.cuda() @ x.float()
        nb = q.numel() * 4 + sc.numel() * 2
        qs = [(q, sc)] + [(q.clone(), sc.clone()) for _ in range(copies_for(nb) - 1)]
        report("Q4 reference (4.5 bpw)", 4.5, nb, [lambda c=c: pf.gemv_q4(c[0], c[1], x, y, R, C) for c in qs], ref)
        del w0, deq, q, sc, qs

        # current kernel, free K=1024, G=1
        lin = synthetic(R, C, 10, 1, M)
        ref = lin.dense() @ x.float()
        lins = [lin] + [clone_lin(lin) for _ in range(copies_for(lin.stored_bytes()) - 1)]
        report("current kernel: free K=1024", 2.5, lin.stored_bytes(),
               [lambda l=l: cuda_gemv.gemv(l, x, y) for l in lins], ref)
        del lin, lins

        groups = pl.mask_groups(rm, M)
        for name, KA, KB, variants in LUT_FORMATS:
            bits = (KA.bit_length() - 1) + ((KB.bit_length() - 1) if KB else 0)
            tA = torch.randint(0, KA, (R, nch), generator=g)
            books = (torch.randn(KA + KB, 4, generator=g) * 0.5).half()
            if KB:
                tB = torch.randint(0, KB, (R, nch), generator=g)
                vals = books.float()[tA] + books.float()[KA + tB]
                tags = tA + KA * tB
                del tB
            else:
                vals = books.float()[tA]
                tags = tA
            ref = ref_y(vals.cuda(), perms, rm, rs, x, 4)
            del vals
            words = pack_bits(tags, bits).reshape(-1)
            words = torch.cat([words, torch.zeros(8, dtype=words.dtype)]).cuda()
            books_c = books.cuda()
            books_f = books.float().cuda().contiguous()
            nbytes = R * nch * bits // 8
            wcopies = [words] + [words.clone() for _ in range(copies_for(nbytes) - 1)]
            bpw = bits / 4
            if KB == 32 and not a.lut_only:   # existing best format-A kernels, for comparison
                for rows in (2,):
                    tiles = pf.row_tiles(rm, rows)
                    report(f"{name} existing wide kernel ({rows} rows)", bpw, nbytes,
                           [lambda w=w: pf.gemv_add_wide(w, books_c, rs, tiles, xp, nch, y, rows) for w in wcopies], ref)
                rm32 = rm.to(torch.int32)
                report(f"{name} existing smem x32 kernel", bpw, nbytes,
                       [lambda w=w: pf.gemv_add(w, books_c, rs, rm32, xp, R, nch // 32, y, True) for w in wcopies], ref)
            mask_rows = groups[0]
            for KT, half, ru in variants:
                red32, tm = False, True
                plan_shared = KT * (KA + KB) * 32 * (2 if half else 4)
                tag = (f"{name} LUT KT{KT}{' fp16' if half else ''}{' red32' if red32 else ''}"
                       f" RU{ru}")
                if plan_shared > smem_max:
                    print(f"{label:13s} {tag:40s} skipped: needs {plan_shared >> 10} KB shared")
                    continue
                plan = pl.LutPlan(KA=KA, KB=KB, KT=KT, half=half, red32=red32, R=R, nch=nch, M=M, tilemajor=tm, ru=ru)
                if tm:
                    w0t = pl.tile_major(tags, mask_rows, bits, KT * 32).cuda()
                    wsrc = [w0t] + [w0t.clone() for _ in range(copies_for(nbytes) - 1)]
                else:
                    wsrc = wcopies
                report(f"{tag} (split {plan.rsplit})", bpw, nbytes,
                       [lambda w=w: plan(w, books_f, rs, groups, xp, y32) for w in wsrc], ref)
                if tm:
                    del wsrc, w0t
            del wcopies, words, tags, tA
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
