"""Prefill GEMM: bf16 (today) vs W8A8 int8 (vLLM's cutlass_scaled_mm) on the 27B's linear shapes,
plus the per-chunk overheads the int8 path adds (block-128 Hadamard + int8 quantization of the
activations, and of the freshly dequantized weights).

The GPU's speed drifts between runs on this machine (up to ~3x on identical work), so every
variant is timed interleaved within one process, round-robin, and the medians are reported.

  python scripts/bench_int8_prefill.py --m 4096
"""

from __future__ import annotations

import argparse
import statistics

import torch

# (name, N, K) of Qwen3.8-27B's prefill GEMMs (per layer); counts per model in the second field
SHAPES = [
    ("mlp gate_up", 34816, 5120, 64),
    ("mlp down", 5120, 17408, 64),
    ("gdn in_proj", 16384, 5120, 48),
    ("gdn out_proj", 5120, 6144, 48),
    ("attn qkv", 14336, 5120, 16),
    ("attn o_proj", 5120, 6144, 16),
]


def hadamard(b: int, dev) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < b:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / b ** 0.5).to(dev, torch.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    from vllm import _custom_ops as ops
    dev = torch.device("cuda")
    H = hadamard(128, dev)
    M = a.m

    def rot(t):        # [N, K] @ blockdiag(H128) along K
        n, k = t.shape
        return (t.view(n * (k // 128), 128) @ H).view(n, k)

    def q8(t):          # dynamic per-row symmetric int8
        q, s, _ = ops.scaled_int8_quant(t.contiguous())
        return q, s

    tot = {}
    print(f"M = {M}; medians of {a.reps} interleaved reps, ms")
    print(f"{'shape':14s} {'N':>6s} {'K':>6s} | {'bf16':>7s} {'int8':>7s} {'x prep':>7s} {'w prep':>7s} | speedup gemm / all")
    for name, N, K, cnt in SHAPES:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        xq, sx = q8(rot(x))
        wq, sw = q8(rot(w))
        sw = sw.view(1, N)
        fns = {
            "bf16": lambda: x @ w.t(),
            "int8": lambda: ops.cutlass_scaled_mm(xq, wq.t(), sx, sw, torch.bfloat16),
            "x prep": lambda: q8(rot(x)),
            "w prep": lambda: q8(rot(w)),
        }
        # correctness of the int8 GEMM against bf16 on rotated inputs
        ref = (x.float() @ w.float().t())
        got = ops.cutlass_scaled_mm(xq, wq.t(), sx, sw, torch.bfloat16).float()
        rel = ((got - ref).norm() / ref.norm()).item()
        times = {k: [] for k in fns}
        for f in fns.values():
            f()
        torch.cuda.synchronize()
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in fns]
        for _ in range(a.reps):
            for (k, f), (e0, e1) in zip(fns.items(), ev):
                e0.record()
                f()
                e1.record()
                e1.synchronize()
                times[k].append(e0.elapsed_time(e1))
        med = {k: statistics.median(v) for k, v in times.items()}
        for k, v in med.items():
            tot[k] = tot.get(k, 0.0) + v * cnt
        allint8 = med["int8"] + med["x prep"] + med["w prep"]
        print(f"{name:14s} {N:6d} {K:6d} | {med['bf16']:7.2f} {med['int8']:7.2f} {med['x prep']:7.2f} {med['w prep']:7.2f} | "
              f"{med['bf16'] / med['int8']:.2f}x / {med['bf16'] / allint8:.2f}x   (int8 rel err {rel:.4f})")
        del x, w, xq, wq
    allint8 = tot["int8"] + tot["x prep"] + tot["w prep"]
    print(f"\nwhole model per {M}-token chunk: bf16 {tot['bf16']:.0f} ms, int8 GEMM {tot['int8']:.0f} ms "
          f"+ x prep {tot['x prep']:.0f} + w prep {tot['w prep']:.0f} = {allint8:.0f} ms  ->  "
          f"{tot['bf16'] / tot['int8']:.2f}x GEMM only, {tot['bf16'] / allint8:.2f}x with prep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
