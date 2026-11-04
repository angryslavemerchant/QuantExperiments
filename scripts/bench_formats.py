"""Speed of kernel-friendly codebook formats vs the current kernel (synthetic
weights, Qwen2.5-7B shapes, M = 16 masks). See shapequant/kernels/proto_formats.py.

Run: python scripts/bench_formats.py [--bw 253]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_gemv import SHAPES, clone_lin, copies_for, l2_bytes, synthetic, timeit  # noqa: E402
from shapequant import masks as mk  # noqa: E402
from shapequant.kernels import cuda_gemv, proto_formats as pf  # noqa: E402
from shapequant.packed import pack_bits  # noqa: E402

M = 16


def setup_common(R, C, g):
    perms = mk.seeded_perms(C, M, torch.Generator().manual_seed(1)).cuda()
    rm = torch.randint(0, M, (R,), generator=g).cuda()
    rs = (torch.rand(R, generator=g) * 0.02 + 0.01).half().cuda()
    x = torch.randn(C, generator=g).half().cuda()
    xp = x[perms].contiguous()
    return perms, rm, rs, x, xp


def ref_y(chunks_vals, perms, rm, rs, x, d):
    """chunks_vals [R, nch, d] fp32 in each row's permuted order -> y fp32."""
    R = chunks_vals.shape[0]
    u = chunks_vals.reshape(R, -1)
    w = torch.empty_like(u)
    w.scatter_(1, perms[rm], u)
    return (w * rs.float()[:, None]) @ x.float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw", type=float, default=253.0)
    a = ap.parse_args()
    print(f"{torch.cuda.get_device_name(0)}; ceiling {a.bw:.0f} GB/s; L2 {l2_bytes() >> 20} MB "
          f"(weights rotated over >= 4x L2); M={M}")
    print(f"{'matrix':13s} {'format':28s} {'bpw':>5s} {'us':>8s} {'GB/s':>6s} {'%ceil':>6s} {'vs fp16':>7s} {'rel err':>8s}")
    for label, (R, C) in SHAPES.items():
        g = torch.Generator().manual_seed(0)
        perms, rm, rs, x, xp = setup_common(R, C, g)
        wds = [torch.randn(R, C, device="cuda", dtype=torch.float16) for _ in range(copies_for(R * C * 2))]
        t16 = timeit([lambda w=w: torch.mv(w, x) for w in wds])
        del wds
        y = torch.empty(R, device="cuda", dtype=torch.float16)

        def report(name, bpw, nbytes, fn, ref):
            """fn: list of callables over distinct weight copies (first one is checked)."""
            got = fn[0]().float()
            err = float((got - ref).abs().max() / ref.abs().max())
            t = timeit(fn)
            gbs = nbytes / t / 1e9
            print(f"{label:13s} {name:28s} {bpw:5.2f} {t * 1e6:8.1f} {gbs:6.0f} {100 * gbs / a.bw:5.0f}% "
                  f"{t16 / t:6.2f}x {err:8.1e}")

        # Q4 reference (llama.cpp Q4_0-like, 4.5 bpw)
        w0 = torch.randn(R, C, generator=g) * 0.02
        q, sc, deq = pf.q4_pack(w0)
        q, sc = q.cuda(), sc.cuda()
        ref = deq.cuda() @ x.float()
        nb = q.numel() * 4 + sc.numel() * 2
        qs = [(q, sc)] + [(q.clone(), sc.clone()) for _ in range(copies_for(nb) - 1)]
        report("Q4 reference (4.5 bpw)", 4.5, nb,
               [lambda c=c: pf.gemv_q4(c[0], c[1], x, y, R, C) for c in qs], ref)
        del w0, deq, q, sc, qs

        # current format, 10-bit free codebook, G=1 (best existing kernel)
        lin = synthetic(R, C, 10, 1, M)
        ref = lin.dense() @ x.float()
        lins = [lin] + [clone_lin(lin) for _ in range(copies_for(lin.stored_bytes()) - 1)]
        report("current: free K=1024", 2.5, lin.stored_bytes(),
               [lambda l=l: cuda_gemv.gemv(l, x, y) for l in lins], ref)
        del lin, lins

        # A: additive d4, 2 x 32
        nch = C // 4
        tA = torch.randint(0, 32, (R, nch), generator=g)
        tB = torch.randint(0, 32, (R, nch), generator=g)
        books = (torch.randn(64, 4, generator=g) * 0.5).half()
        vals = books.float()[tA] + books.float()[32 + tB]                      # [R, nch, 4]
        ref = ref_y(vals.cuda(), perms, rm, rs, x, 4)
        words = pack_bits(tA + 32 * tB, 10).reshape(-1)
        words = torch.cat([words, torch.zeros(8, dtype=words.dtype)]).cuda()
        books_c = books.cuda()
        rm32 = rm.to(torch.int32)
        nbytes = R * nch * 10 // 8
        wcopies = [words] + [words.clone() for _ in range(copies_for(nbytes) - 1)]
        for smem in (True, False):
            report(f"A: add 32+32 d4 ({'smem x32' if smem else 'shfl'})", 2.5, nbytes,
                   [lambda w=w: pf.gemv_add(w, books_c, rs, rm32, xp, R, nch // 32, y, smem) for w in wcopies], ref)
        for rows in (2, 4, 8):
            tiles = pf.row_tiles(rm, rows)
            report(f"A: add 32+32 d4 ({rows} rows/warp)", 2.5, nbytes,
                   [lambda w=w: pf.gemv_add_rows(w, books_c, rs, tiles, xp, nch // 32, y, rows) for w in wcopies], ref)
        for rows in (1, 2, 4):
            tiles = pf.row_tiles(rm, rows)
            report(f"A: add 32+32 d4 (wide, {rows} rows)", 2.5, nbytes,
                   [lambda w=w: pf.gemv_add_wide(w, books_c, rs, tiles, xp, nch, y, rows) for w in wcopies], ref)
        del vals, tA, tB, wcopies

        # B: d8, 2 x 256
        nch8 = C // 8
        t8 = torch.randint(0, 65536, (R, nch8), generator=g)
        books8 = (torch.randn(512, 8, generator=g) * 0.5).half()
        vals = books8.float()[t8 & 255] + books8.float()[256 + (t8 >> 8)]
        ref = ref_y(vals.cuda(), perms, rm, rs, x, 8)
        tags16 = torch.where(t8 >= 32768, t8 - 65536, t8).to(torch.int16).cuda()
        books8_c = books8.cuda()
        nbytes = R * nch8 * 2
        tcopies = [tags16] + [tags16.clone() for _ in range(copies_for(nbytes) - 1)]
        for rep in (1, 2, 4):
            report(f"B: 256+256 d8 (rep {rep})", 2.0, nbytes,
                   [lambda t=t: pf.gemv_d8(t, books8_c, rs, rm32, xp, R, nch8, y, rep) for t in tcopies], ref)
        del vals, t8, tcopies
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
