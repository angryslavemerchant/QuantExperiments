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
from bench_gemv import SHAPES, synthetic, timeit  # noqa: E402
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
    print(f"{torch.cuda.get_device_name(0)}; ceiling {a.bw:.0f} GB/s; M={M}")
    print(f"{'matrix':13s} {'format':28s} {'bpw':>5s} {'us':>8s} {'GB/s':>6s} {'%ceil':>6s} {'vs fp16':>7s} {'rel err':>8s}")
    for label, (R, C) in SHAPES.items():
        g = torch.Generator().manual_seed(0)
        perms, rm, rs, x, xp = setup_common(R, C, g)
        wd = torch.randn(R, C, device="cuda", dtype=torch.float16)
        t16 = timeit(lambda: torch.mv(wd, x))
        del wd
        y = torch.empty(R, device="cuda", dtype=torch.float16)

        def report(name, bpw, nbytes, fn, ref):
            got = fn().float()
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
        report("Q4 reference (4.5 bpw)", 4.5, q.numel() * 4 + sc.numel() * 2,
               lambda: pf.gemv_q4(q, sc, x, y, R, C), ref)
        del w0, deq, q, sc

        # current format, 10-bit free codebook, G=1 (best existing kernel)
        lin = synthetic(R, C, 10, 1, M)
        ref = lin.dense() @ x.float()
        report("current: free K=1024", 2.5, lin.stored_bytes(), lambda: cuda_gemv.gemv(lin, x, y), ref)
        del lin

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
        for smem in (True, False):
            report(f"A: add 32+32 d4 ({'smem x32' if smem else 'shfl'})", 2.5, nbytes,
                   lambda: pf.gemv_add(words, books_c, rs, rm32, xp, R, nch // 32, y, smem), ref)
        for rows in (2, 4, 8):
            tiles = pf.row_tiles(rm, rows)
            report(f"A: add 32+32 d4 ({rows} rows/warp)", 2.5, nbytes,
                   lambda: pf.gemv_add_rows(words, books_c, rs, tiles, xp, nch // 32, y, rows), ref)
        for rows in (1, 2, 4):
            tiles = pf.row_tiles(rm, rows)
            report(f"A: add 32+32 d4 (wide, {rows} rows)", 2.5, nbytes,
                   lambda: pf.gemv_add_wide(words, books_c, rs, tiles, xp, nch, y, rows), ref)
        del vals, tA, tB

        # B: d8, 2 x 256
        nch8 = C // 8
        t8 = torch.randint(0, 65536, (R, nch8), generator=g)
        books8 = (torch.randn(512, 8, generator=g) * 0.5).half()
        vals = books8.float()[t8 & 255] + books8.float()[256 + (t8 >> 8)]
        ref = ref_y(vals.cuda(), perms, rm, rs, x, 8)
        tags16 = torch.where(t8 >= 32768, t8 - 65536, t8).to(torch.int16).cuda()
        books8_c = books8.cuda()
        nbytes = R * nch8 * 2
        for rep in (1, 2, 4):
            report(f"B: 256+256 d8 (rep {rep})", 2.0, nbytes,
                   lambda: pf.gemv_d8(tags16, books8_c, rs, rm32, xp, R, nch8, y, rep), ref)
        del vals, t8
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
