"""Speed: d = 8 additive run kernel vs the d = 4 run kernel at matched tag bits, batch-1 GEMV.

  2.0 bpw: d4 K256  vs d8 256+256      2.5 bpw: d4 K1024 vs d8 1024+1024
Random tags/books (speed only), Qwen2.5-7B shapes, G1, M1. Times are medians of CUDA-event
measurements over `--iters` launches, reported per matrix and as effective GB/s of weight bytes.

Run: python scripts/bench_additive.py [--threads 256,512] [--bps 1,2,4] [--ru 1,2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shapequant.kernels import proto_runs as pr  # noqa: E402
from shapequant.kernels.decode_e2e import RunsLin  # noqa: E402

SHAPES = {"qkv-like 3584x3584": (3584, 3584), "gate/up 18944x3584": (18944, 3584),
          "down 3584x18944": (3584, 18944)}


def timeit(fn, iters, per_graph=20):
    """Median per-launch time of `per_graph` launches captured in one CUDA graph (no Python
    launch overhead)."""
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(per_graph):
            fn()
    for _ in range(3):
        gr.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(max(5, iters // per_graph)):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        gr.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / per_graph)
    ts.sort()
    return ts[len(ts) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="256,512")
    ap.add_argument("--bps", default="1,2,4")
    ap.add_argument("--ru", default="1,2")
    ap.add_argument("--bits", default="8,10")
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    dev = "cuda"
    print(torch.cuda.get_device_name(), flush=True)
    g = torch.Generator(device=dev).manual_seed(0)
    for name, (R, C) in SHAPES.items():
        x = torch.randn(C, device=dev).half()
        y = torch.empty(R, device=dev, dtype=torch.float16)
        for bits in [int(v) for v in a.bits.split(",")]:
            K = 1 << bits
            wbytes = R * C * bits / 4 / 8
            best = {}
            for ru in [int(v) for v in a.ru.split(",")]:
                d4 = RunsLin.random(R, C, bits, torch.arange(C, device=dev)[None], g, ru=ru).rl
                A = torch.randn(K, 8, device=dev).half()
                ta = torch.randint(0, K, (R, C // 8), device=dev, generator=g)
                tb = torch.randint(0, K, (R, C // 8), device=dev, generator=g)
                d8 = pr.AdditiveRunsLinear(A, A * 0.5, ta, tb, torch.ones(R, device=dev), ru)
                xp = x[None].contiguous()
                for th in [int(v) for v in a.threads.split(",")]:
                    for bps in [int(v) for v in a.bps.split(",")]:
                        t4 = timeit(lambda: pr.gemv(d4, x, y, bps, th, xp=xp), a.iters)
                        if d8.table_bytes() <= pr.smem_optin_limit():
                            t8 = timeit(lambda: pr.gemv_additive(d8, x, y, bps, th), a.iters)
                        else:
                            t8 = float("inf")   # both books over this device's shared memory
                        for k, t in (("d4", t4), ("d8", t8)):
                            if k not in best or t < best[k][0]:
                                best[k] = (t, f"RU{ru} {th}x{bps}")
            t4, c4 = best["d4"]
            t8, c8 = best["d8"]
            print(f"{name:20s} {bits * 0.25:.2f} bpw  d4 K{K}: {t4:7.1f} us ({wbytes / t4 / 1e3:6.0f} GB/s, {c4})   "
                  f"d8 {K}+{K}: {t8:7.1f} us ({wbytes / t8 / 1e3:6.0f} GB/s, {c8})   d8/d4 {t8 / t4:.2f}x", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
