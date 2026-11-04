"""Benchmark the packed GEMV on Qwen2.5-7B-shaped matrices (synthetic tags).

Speed does not depend on the tag values, so random tags with the real shapes,
tag widths, mixture and mask settings measure the real kernel. Each case is
checked against the reference decode first.

Run: python scripts/bench_gemv.py [--bits 10 12 14] [--G 4] [--M 16]
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shapequant import masks as mk  # noqa: E402
from shapequant.kernels.gemv import PackedLinear  # noqa: E402
from shapequant.packed import pack_bits  # noqa: E402

SHAPES = {"q/o_proj": (3584, 3584), "k/v_proj": (512, 3584), "gate/up_proj": (18944, 3584),
          "down_proj": (3584, 18944)}


def synthetic(R, C, bits, G, M, d=4, seed=0, dev="cuda"):
    g = torch.Generator().manual_seed(seed)
    K = 1 << bits
    tags = torch.randint(0, K, (R, C // d), generator=g)
    table = (torch.randn(G * K, d, generator=g) * 0.5).half()
    rs = (torch.rand(R, generator=g) * 0.02 + 0.01).half()
    rb = torch.randint(0, G, (R,), generator=g) if G > 1 else None
    rm = torch.randint(0, M, (R,), generator=g) if M > 1 else None
    perms = mk.seeded_perms(C, M, torch.Generator().manual_seed(seed + 1))
    to = (lambda t: None if t is None else t.to(dev))
    return PackedLinear(pack_bits(tags, bits).to(dev), table.to(dev), rs.to(dev), to(rb), to(rm),
                        perms.to(dev), K, d)


def l2_bytes() -> int:
    return int(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 6 << 20))


def copies_for(nbytes: int, factor: float = 4.0, cap: int = 64) -> int:
    """How many distinct weight copies to rotate through so the working set is
    `factor` x the L2 cache: every call then streams its weights from DRAM."""
    return max(2, min(cap, math.ceil(factor * l2_bytes() / max(1, nbytes))))


def clone_lin(lin):
    """Shallow copy of a PackedLinear with its own tag-word buffer (the only
    large per-matrix tensor); caches built on the original are dropped."""
    new = copy.copy(lin)
    new._buffers = dict(lin._buffers)
    new._buffers["words"] = lin.words.clone()
    for k in ("_slab_words", "_l1_layout"):
        new.__dict__.pop(k, None)
    return new


def timeit(fn, reps=200):
    """Per-call GPU time with CPU launch overhead removed: calls are captured
    into one CUDA graph and replayed. `fn` may be a list of callables on
    distinct weight copies, called round-robin, so the timing reflects DRAM
    rather than L2 (large-L2 GPUs otherwise keep a replayed matrix cached)."""
    fns = fn if isinstance(fn, (list, tuple)) else [fn]
    reps = max(reps, 4 * len(fns))
    reps -= reps % len(fns)
    for f in fns:
        f()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fns[0]()
        with torch.cuda.graph(graph, stream=s):
            for i in range(reps):
                fns[i % len(fns)]()
    torch.cuda.current_stream().wait_stream(s)
    graph.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(5):
        t = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / reps)
    return best


def timeit_eager(fn, reps=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / reps)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[10, 12, 14])
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--BR", type=int, default=16)
    ap.add_argument("--BJ", type=int, default=64)
    ap.add_argument("--bw", type=float, default=253.0, help="measured DRAM read GB/s (ceiling)")
    ap.add_argument("--impl", choices=["triton", "cuda", "tiles", "slab", "l1"], default="l1")
    ap.add_argument("--tile", type=int, default=64, help="tiles: rows per tile")
    ap.add_argument("--unroll", type=int, default=4)
    ap.add_argument("--smem-cap", type=int, default=None, help="tiles: shared-memory budget (bytes)")
    ap.add_argument("--bps", type=int, default=4, help="CUDA: blocks per SM")
    ap.add_argument("--threads", type=int, default=256)
    a = ap.parse_args()
    dev = "cuda"
    print(f"{torch.cuda.get_device_name(0)}; ceiling {a.bw:.0f} GB/s; L2 {l2_bytes() >> 20} MB "
          f"(weights rotated over >= 4x L2); G={a.G} M={a.M} impl={a.impl}")
    if a.impl == "cuda":
        from shapequant.kernels import cuda_gemv
        run = lambda lin, x, out=None: cuda_gemv.gemv(lin, x, out, blocks_per_sm=a.bps, threads=a.threads, unroll=a.unroll)
    elif a.impl == "l1":
        from shapequant.kernels import cuda_l1
        run = lambda lin, x, out=None: cuda_l1.gemv(lin, x, out, threads=a.threads, unroll=a.unroll)
    elif a.impl == "slab":
        from shapequant.kernels import cuda_slab
        run = lambda lin, x, out=None: cuda_slab.gemv(lin, x, out, threads=a.threads, blocks_per_sm=a.bps)
    elif a.impl == "tiles":
        from shapequant.kernels import cuda_tiles
        layouts = {}

        def run(lin, x, out=None):
            if id(lin) not in layouts:
                layouts[id(lin)] = cuda_tiles.TiledLayout(lin, a.tile)
            return cuda_tiles.gemv(layouts[id(lin)], x, out, threads=a.threads, unroll=a.unroll,
                                   smem_cap=a.smem_cap)
    else:
        run = lambda lin, x, out=None: lin.gemv(x, out)
    print(f"{'matrix':14s} {'bits':>4s} {'MB':>7s} {'us':>8s} {'GB/s':>6s} {'%ceil':>6s} "
          f"{'fp16 us':>8s} {'speedup':>7s} {'max rel err':>11s}")
    for label, (R, C) in SHAPES.items():
        x = torch.randn(C, device=dev, dtype=torch.float16)
        wds = [torch.randn(R, C, device=dev, dtype=torch.float16) for _ in range(copies_for(R * C * 2))]
        t16 = timeit([lambda w=w: torch.mv(w, x) for w in wds])
        del wds
        for bits in a.bits:
            lin = synthetic(R, C, bits, a.G, a.M)
            lin.BR, lin.BJ = a.BR, a.BJ
            ref = lin.dense() @ x.float()
            got = run(lin, x).float()
            err = float((got - ref).abs().max() / ref.abs().max())
            out = torch.empty(R, device=dev, dtype=torch.float16)
            nbytes = lin.stored_bytes()
            lins = [lin] + [clone_lin(lin) for _ in range(copies_for(nbytes) - 1)]
            t = timeit([lambda l=l: run(l, x, out) for l in lins])
            del lins
            gbs = nbytes / t / 1e9
            print(f"{label:14s} {bits:4d} {nbytes / 1e6:7.2f} {t * 1e6:8.1f} {gbs:6.0f} {100 * gbs / a.bw:5.0f}% "
                  f"{t16 * 1e6:8.1f} {t16 / t:6.2f}x {err:11.2e}")
            del lin
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
