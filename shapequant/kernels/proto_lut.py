"""Lookup-table (activation-side decode) GEMV prototype, batch 1.

Instead of decoding each weight group to its 4-value codeword and multiplying
by 4 inputs, precompute per token the dot product of every input chunk with
every codebook entry:  T[chunk][e] = <book[e], x_chunk>.  Each weight group
then costs one table read (two for additive books: T[a] + T[KA + b]) and one
add.  y[r] = scale[r] * sum_chunks T[chunk][tag[r, chunk]].

Block = (mask m, tile of KT*32 chunks, row split).  The block builds its table
tile in shared memory with layout [k][e][lane] (chunk = j0 + 32k + lane), so a
lane only ever reads its own bank: conflict-free for any random tags.  Rows are
pre-sorted by mask (row order is free), partial sums go to fp32 y by atomicAdd.

RED32 = 1: a warp handles 32 rows at once and reduces with a transpose
(reduce-scatter, 31 shuffles per 32 rows) instead of 5 shuffles per row.

Formats: additive KA+KB books (format A, KB > 0) or one free book of KA
entries (current format, KB = 0; HALF = 1 stores the table in fp16 so K = 1024
fits in 64 KB).  No mixture (G = 1).
"""

from __future__ import annotations

import math

import torch

_SRC = r"""
#include <cuda_fp16.h>
typedef unsigned int u32;
#define KE (KA + KB)
#define TILE (KT * 32)
#if HALF
typedef __half tval;
#define TV(x) __float2half(x)
#define TF(x) __half2float(x)
#else
typedef float tval;
#define TV(x) (x)
#define TF(x) (x)
#endif

__device__ __forceinline__ float lookup(const tval* T, const u32* wr, int j0, int k, int lane, int nch) {
    const int c = j0 + k * 32 + lane;
    if (c >= nch) return 0.f;
#if TILEMAJOR
    const int bit = (k * 32 + lane) * BITS;     // wr points at this row's tile segment
#else
    const int bit = c * BITS;
#endif
    const u32 v = __funnelshift_r(__ldg(wr + (bit >> 5)), __ldg(wr + (bit >> 5) + 1), bit & 31);
#if KB
    const int ta = v & (KA - 1), tb = (v >> LOGKA) & (KB - 1);
    return TF(T[(k * KE + ta) * 32 + lane]) + TF(T[(k * KE + KA + tb) * 32 + lane]);
#else
    return TF(T[(k * KE + (v & (KA - 1))) * 32 + lane]);
#endif
}

extern "C" __global__ void __launch_bounds__(256)
gemv_lut(const u32* __restrict__ words, const float4* __restrict__ books,
         const __half* __restrict__ row_scale, const int* __restrict__ mask_rows,
         const int* __restrict__ mask_off, const uint2* __restrict__ xp,
         float* __restrict__ y, int nch, int ntiles, int rsplit, int R)
{
    extern __shared__ __align__(16) unsigned char smraw[];
    tval* T = (tval*)smraw;
    int b = blockIdx.x;
    const int rsi = b % rsplit; b /= rsplit;
    const int tile = b % ntiles;
    const int m = b / ntiles;
    const int j0 = tile * TILE;
    const uint2* xm = xp + (size_t)m * nch;
    {   // build: lane = chunk (x held in registers), warp-uniform codeword (broadcast load)
        const int ln = threadIdx.x & 31, wp = threadIdx.x >> 5;
        #pragma unroll
        for (int k = 0; k < KT; ++k) {
            const int c = j0 + k * 32 + ln;
            float4 xf = make_float4(0.f, 0.f, 0.f, 0.f);
            if (c < nch) {
                const uint2 xv = __ldg(xm + c);
                const float2 x0 = __half22float2(*(const __half2*)&xv.x), x1 = __half22float2(*(const __half2*)&xv.y);
                xf = make_float4(x0.x, x0.y, x1.x, x1.y);
            }
            #pragma unroll 4
            for (int e = wp; e < KE; e += 8) {
                const float4 bv = __ldg(books + e);
                T[(k * KE + e) * 32 + ln] = TV(bv.x * xf.x + bv.y * xf.y + bv.z * xf.z + bv.w * xf.w);
            }
        }
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r0 = mask_off[m], r1 = mask_off[m + 1];
#if TILEMAJOR
    const int S = TILE * BITS / 32;                // words per row per tile; rows in mask-sorted order
    const u32* wt = words + (size_t)tile * R * S;
#else
    const int W = nch * BITS / 32;
#endif
#if RED32
    for (int base = r0 + (rsi * 8 + warp) * 32; base < r1; base += 8 * rsplit * 32) {
        float acc[32];
        #pragma unroll
        for (int r = 0; r < 32; ++r) {
            const int ri = min(base + r, r1 - 1);
#if TILEMAJOR
            const u32* wr = wt + (size_t)ri * S;
#else
            const u32* wr = words + (size_t)mask_rows[ri] * W;
#endif
            float a = 0.f;
            #pragma unroll
            for (int k = 0; k < KT; ++k) a += lookup(T, wr, j0, k, lane, nch);
            acc[r] = a;
        }
        #pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            const bool up = lane & off;
            #pragma unroll
            for (int i = 0; i < off; ++i) {
                const float send = up ? acc[i] : acc[i + off];
                const float keep = up ? acc[i + off] : acc[i];
                acc[i] = keep + __shfl_xor_sync(0xffffffffu, send, off);
            }
        }
        if (base + lane < r1) {
            const int row = mask_rows[base + lane];
            atomicAdd(y + row, acc[0] * __half2float(row_scale[row]));
        }
    }
#else
    // RU rows per warp iteration: all tag loads issued first (memory-level parallelism),
    // then table lookups, then RU interleaved reductions.
    for (int ri0 = r0 + (rsi * 8 + warp) * RU; ri0 < r1; ri0 += 8 * rsplit * RU) {
        u32 lo[RU][KT], hi[RU][KT];
        #pragma unroll
        for (int r = 0; r < RU; ++r) {
            const int ri = min(ri0 + r, r1 - 1);
#if TILEMAJOR
            const u32* wr = wt + (size_t)ri * S;
#else
            const u32* wr = words + (size_t)mask_rows[ri] * W;
#endif
            #pragma unroll
            for (int k = 0; k < KT; ++k) {
#if TILEMAJOR
                const int bit = (k * 32 + lane) * BITS;
#else
                const int bit = (j0 + k * 32 + lane) * BITS;
#endif
                lo[r][k] = __ldg(wr + (bit >> 5));
                hi[r][k] = __ldg(wr + (bit >> 5) + 1);
            }
        }
        float acc[RU];
        #pragma unroll
        for (int r = 0; r < RU; ++r) {
            float a = 0.f;
            #pragma unroll
            for (int k = 0; k < KT; ++k) {
                const int c = j0 + k * 32 + lane;
#if TILEMAJOR
                const int bit = (k * 32 + lane) * BITS;
#else
                const int bit = c * BITS;
#endif
                const u32 v = __funnelshift_r(lo[r][k], hi[r][k], bit & 31);
                if (c < nch) {
#if KB
                    a += TF(T[(k * KE + (v & (KA - 1))) * 32 + lane]) + TF(T[(k * KE + KA + ((v >> LOGKA) & (KB - 1))) * 32 + lane]);
#else
                    a += TF(T[(k * KE + (v & (KA - 1))) * 32 + lane]);
#endif
                }
            }
            acc[r] = a;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            #pragma unroll
            for (int r = 0; r < RU; ++r) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
        }
        if (lane < RU) {
            float mine = acc[0];
            #pragma unroll
            for (int r = 1; r < RU; ++r) if (lane == r) mine = acc[r];
            if (ri0 + lane < r1) {
                const int row = mask_rows[ri0 + lane];
                atomicAdd(y + row, mine * __half2float(row_scale[row]));
            }
        }
    }
#endif
}
"""

_mods: dict = {}


def _fn(defs: dict):
    import cupy as cp
    key = tuple(sorted(defs.items()))
    if key not in _mods:
        code = "".join(f"#define {k} {v}\n" for k, v in defs.items()) + _SRC
        _mods[key] = cp.RawModule(code=code, options=("-std=c++17", "--use_fast_math")).get_function("gemv_lut")
    return _mods[key]


def mask_groups(row_mask: torch.Tensor, M: int):
    """(mask_rows int32 [R] sorted by mask, mask_off int32 [M+1])."""
    rm = row_mask.to(torch.int64)
    order = torch.argsort(rm, stable=True)
    counts = torch.bincount(rm, minlength=M)
    off = torch.zeros(M + 1, dtype=torch.int64, device=rm.device)
    off[1:] = torch.cumsum(counts, 0)
    return order.to(torch.int32), off.to(torch.int32)


class LutPlan:
    def __init__(self, *, KA: int, KB: int, KT: int, half: bool, red32: bool, R: int, nch: int, M: int,
                 rsplit: int | None = None, tilemajor: bool = False, ru: int = 1):
        self.KA, self.KB, self.KT, self.M, self.nch = KA, KB, KT, M, nch
        self.bits = int(math.log2(KA)) + (int(math.log2(KB)) if KB else 0)
        assert (nch * self.bits) % 32 == 0
        self.ntiles = -(-nch // (KT * 32))
        self.shared = KT * (KA + KB) * 32 * (2 if half else 4)
        self.defs = {"KA": KA, "KB": KB, "LOGKA": int(math.log2(KA)), "BITS": self.bits, "KT": KT,
                     "HALF": int(half), "RED32": int(red32),
                     "TILEMAJOR": int(tilemajor), "RU": ru}
        self.R = R
        if rsplit is None:
            sms = torch.cuda.get_device_properties(0).multi_processor_count
            per_warp = 32 if red32 else ru
            max_split = max(1, (R // M) // (8 * per_warp))
            rsplit = max(1, min(max_split, -(-4 * sms // (M * self.ntiles))))
        self.rsplit = rsplit
        self.grid = M * self.ntiles * rsplit
        self.fn = _fn(self.defs)
        if self.shared > 48 * 1024:
            self.fn.max_dynamic_shared_size_bytes = self.shared

    def __call__(self, words, books, row_scale, groups, xp, y32):
        import cupy as cp
        mask_rows, mask_off = groups
        y32.zero_()
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            self.fn((self.grid,), (256,),
                    (cp.uint64(words.data_ptr()), cp.uint64(books.data_ptr()), cp.uint64(row_scale.data_ptr()),
                     cp.uint64(mask_rows.data_ptr()), cp.uint64(mask_off.data_ptr()), cp.uint64(xp.data_ptr()),
                     cp.uint64(y32.data_ptr()), cp.int32(self.nch), cp.int32(self.ntiles), cp.int32(self.rsplit), cp.int32(self.R)),
                    shared_mem=self.shared)
        return y32


def tile_major(tags: torch.Tensor, mask_rows: torch.Tensor, bits: int, tile_chunks: int) -> torch.Tensor:
    """tags [R, nch] -> packed words laid out [tile][row in mask-sorted order][tile words]
    (last tile zero-padded), so a block streams one contiguous region."""
    R, nch = tags.shape
    nt = -(-nch // tile_chunks)
    t = torch.zeros(R, nt * tile_chunks, dtype=tags.dtype)
    t[:, :nch] = tags
    t = t[mask_rows.long().cpu()].reshape(R, nt, tile_chunks).transpose(0, 1).reshape(nt * R, tile_chunks)
    from shapequant.packed import pack_bits
    w = pack_bits(t, bits).reshape(-1)
    return torch.cat([w, torch.zeros(8, dtype=w.dtype)])
