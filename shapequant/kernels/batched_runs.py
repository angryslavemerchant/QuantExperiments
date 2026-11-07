"""Batched decode GEMMs: Y[B, R] = X[B, C] . W^T for B = 1..32 activation vectors, decoding every
weight code once per step (tensor-core mma over the batch).

Codebook runs (d = 4, the lane-major words of proto_runs.RunsLinear, read in place):
- a warp owns a tile of 16 rows (one book: tiles are grouped by book) and does
  mma(16 rows x 8 batch x 4 codes) steps: thread (g, t) = (lane / 4, lane % 4) decodes the code
  of rows g and g + 8 at code slot t (table entry = 4 fp16 weights = its A fragment, with the
  mma's k index permuted so slot t's 4 k positions are one code's 4 columns) and loads the
  matching 4 x values of batch row g (8 bytes) per 8-row batch tile;
- slot t walks the codes of run-kernel lanes L = 4s + t (s = 0..7) of a 512-chunk batch, so the
  words can come straight from proto_runs' layout (same tail realignment), or from a tile-major
  copy (layout="tile") where each warp load is 128 contiguous bytes (DRAM-friendlier: the
  in-place layout scatters a warp's loads over 16 rows, 80-byte pieces each);
- a block = W warps on W tiles, all stepping through the same column units (x reuse via L1);
  work items = (tile group, column slice): split-K over slices for parallelism on wide GPUs, the
  last slice to finish for a tile sums the fp32 partials in slice order (deterministic);
- codebook in shared memory (whole table if it fits, else book-major blocks, else global).
Accumulation is fp32 (the batch-1 kernel sums groups of 8 products in fp16 first), so results
match the batch-1 kernel to fp16 rounding, not bitwise.

q8 head (Q8_0-like int8 + fp16 scale per 32): the same warp tiling, int8 -> fp16 exactly
(byte_perm magic), one mma pair per 32-column block into a temporary, times the row scales.

sm_80+: mma.m16n8k16; sm_75: two m16n8k8 (same fragments).
"""

from __future__ import annotations

import math

import torch

_COMMON = r"""
#include <cuda_fp16.h>
typedef unsigned int u32;

// c[0..3] += A(16x16: rows g, g+8 at k slots) * B(16x8): a_lo* = k {2t, 2t+1}, a_hi* = k {2t+8, 2t+9}
// (row g: *0, row g+8: *1); xv.x / xv.y = the B fragment's k {2t, 2t+1} / {2t+8, 2t+9} for column g
// ACC16: each k8 half summed in fp16 by the tensor core (8 products, the batch-1 kernel's bound
// under TABLE_MAX), then added in fp32: twice the mma rate on GeForce parts (fp32 accumulate is
// half rate there), a few more ALU ops.
#ifndef ACC16
#define ACC16 0
#endif
__device__ __forceinline__ void mma_k8_h(float* c, u32 a0, u32 a1, u32 b) {
    u32 d0, d1;
    asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%5,%5};\n"
        : "=r"(d0), "=r"(d1) : "r"(a0), "r"(a1), "r"(b), "r"(0u));
    const float2 f0 = __half22float2(*(const __half2*)&d0), f1 = __half22float2(*(const __half2*)&d1);
    c[0] += f0.x; c[1] += f0.y; c[2] += f1.x; c[3] += f1.y;
}
__device__ __forceinline__ void mma_code(float* c, u32 a_lo0, u32 a_lo1, u32 a_hi0, u32 a_hi1, u32 b0, u32 b1) {
#if ACC16
    mma_k8_h(c, a_lo0, a_lo1, b0);
    mma_k8_h(c, a_hi0, a_hi1, b1);
#elif __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a_lo0), "r"(a_lo1), "r"(a_hi0), "r"(a_hi1), "r"(b0), "r"(b1));
#else
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a_lo0), "r"(a_lo1), "r"(b0));
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a_hi0), "r"(a_hi1), "r"(b1));
#endif
}
"""

_SRC_RUNS = r"""
#define NWB (BITS / 2)
#define NWT ((JT * 2 * BITS + 31) / 32)
#define WROW (NFULL * 32 * NWB + JT * 2 * BITS)
#define NCH ((NFULL * 8 + JT) * 64)
#define NUNITS (NFULL * 8 + (JT > 0 ? 8 : 0))
#ifndef TILE
#define TILE 0
#endif
// TILE layout (RunsBatch(layout="tile")): per 16-row tile, unit by unit, each thread's words for rows
// g / g + 8 (h = 0 / 1) at [unit][h][word i][lane]: full units NWB words, tail units NWT (realigned,
// zero-padded), so every warp load is 128 contiguous bytes and a unit's words are one contiguous block.
#define TWORDS (2 * 32 * (NFULL * 8 * NWB + (JT > 0 ? 8 * NWT : 0)))

#ifndef XS
#define XS 0
#endif
// XS: the block stages each column unit's X slice in shared memory once (all warps step through
// the same units): xs[n][q], n < NB8*8 batch rows, q = 8m + 2t + e the unit's 64 chunks (code j of
// slot t: m = j >> 1, e = j & 1), XOR-swizzled by the row so a warp's 8-byte reads (rows g, slots
// t) hit distinct bank pairs. Rows >= B are zero.
__device__ __forceinline__ int xs_swz(int n) { return (n & 1) | ((n & 2) << 2); }

// the NT tags held in a (row g) and b (row g + 8), NWD words each (+1 zero word)
template <int NT, int NWD>
__device__ __forceinline__ void codes(const u32* a, const u32* b, int cbase, const uint2* tab0, const uint2* tab1,
                                      const uint2* __restrict__ X, int g, int B, float (*acc)[4],
                                      const uint2* xs, int t)
{
    const u32 MASK = (1u << BITS) - 1u;
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int bit = j * BITS, wi = bit >> 5, sh = bit & 31;
        const u32 va = ((sh + BITS <= 32) ? (a[wi] >> sh) : __funnelshift_r(a[wi], a[wi + 1], sh)) & MASK;
        const u32 vb = ((sh + BITS <= 32) ? (b[wi] >> sh) : __funnelshift_r(b[wi], b[wi + 1], sh)) & MASK;
        const uint2 ta = tab0[va], tb = tab1[vb];
#if XS
        const int q = (j >> 1) * 8 + 2 * t + (j & 1);
        #pragma unroll
        for (int nb = 0; nb < NB8; ++nb) {
            const int n = nb * 8 + g;
            const uint2 xv = xs[n * 64 + (q ^ xs_swz(n))];
            mma_code(acc[nb], ta.x, tb.x, ta.y, tb.y, xv.x, xv.y);
        }
#else
        const int c = cbase + (j >> 1) * 64 + (j & 1);
        #pragma unroll
        for (int nb = 0; nb < NB8; ++nb) {
            const int n = nb * 8 + g;
            const uint2 xv = n < B ? __ldg(X + (size_t)n * NCH + c) : make_uint2(0u, 0u);
            mma_code(acc[nb], ta.x, tb.x, ta.y, tb.y, xv.x, xv.y);
        }
#endif
    }
}

// XS: stage unit u's X slice (NB8*8 rows x 64 chunks) into xs
__device__ __forceinline__ void load_xs(const uint2* __restrict__ X, uint2* xs, int u, int B)
{
    int bb, ug, nm;
    if (u < NFULL * 8) { bb = u >> 3; ug = u & 7; nm = 8; } else { bb = NFULL; ug = u - NFULL * 8; nm = JT; }
    const int base = 512 * bb + 8 * ug;
    for (int idx = threadIdx.x; idx < NB8 * 8 * 64; idx += THREADS) {
        const int n = idx >> 6, q = idx & 63, m = q >> 3;
        const uint2 v = (n < B && m < nm) ? __ldg(X + (size_t)n * NCH + base + 64 * m + (q & 7)) : make_uint2(0u, 0u);
        xs[n * 64 + (q ^ xs_swz(n))] = v;
    }
}

// TILE: one unit from the tile layout (base = this unit's block + lane)
template <int NT, int NWD>
__device__ __forceinline__ void unit_t(const u32* __restrict__ base, int cbase, const uint2* tab0, const uint2* tab1,
                                       const uint2* __restrict__ X, int g, int B, float (*acc)[4],
                                       const uint2* xs, int t)
{
    u32 a[NWD + 1], b[NWD + 1];
    #pragma unroll
    for (int i = 0; i < NWD; ++i) { a[i] = __ldg(base + i * 32); b[i] = __ldg(base + (NWD + i) * 32); }
    a[NWD] = 0u; b[NWD] = 0u;
    codes<NT, NWD>(a, b, cbase, tab0, tab1, X, g, B, acc, xs, t);
}

// One column unit for this thread: the NT tags of run lane L for rows g (w0) and g + 8 (w1).
// cbase = chunk of tag 0; tag j sits at chunk cbase + (j >> 1) * 64 + (j & 1).
template <int NT, int NWD, bool TAIL>
__device__ __forceinline__ void unit(const u32* __restrict__ w0, const u32* __restrict__ w1, int L, int cbase,
                                     const uint2* tab0, const uint2* tab1, const uint2* __restrict__ X, int g,
                                     int B, float (*acc)[4], const uint2* xs, int t)
{
    u32 a[NWD + 1], b[NWD + 1];
    if constexpr (TAIL) {
        const int start = L * NT * BITS, i0 = start >> 5, sh0 = start & 31;
        u32 ra[NWD + 1], rb[NWD + 1];
        #pragma unroll
        for (int i = 0; i <= NWD; ++i) {
            const bool ok = i0 + i < NT * BITS;
            ra[i] = (w0 && ok) ? __ldg(w0 + i0 + i) : 0u;
            rb[i] = (w1 && ok) ? __ldg(w1 + i0 + i) : 0u;
        }
        #pragma unroll
        for (int i = 0; i < NWD; ++i) { a[i] = __funnelshift_r(ra[i], ra[i + 1], sh0); b[i] = __funnelshift_r(rb[i], rb[i + 1], sh0); }
    } else {
        #pragma unroll
        for (int i = 0; i < NWD; ++i) {
            a[i] = w0 ? __ldg(w0 + L * NWD + i) : 0u;
            b[i] = w1 ? __ldg(w1 + L * NWD + i) : 0u;
        }
    }
    a[NWD] = 0u; b[NWD] = 0u;
    codes<NT, NWD>(a, b, cbase, tab0, tab1, X, g, B, acc, xs, t);
}

// grid: persistent blocks of W = THREADS / 32 warps. Item it = (group, slice): the group's W tiles
// (one per warp) over the slice's column units. S == 1: Y written directly; else fp32 partials to
// ws[S][B][R] and the last slice of a tile (per-tile ticket in cnt, reset after use) finalizes.
extern "C" __global__ void __launch_bounds__(THREADS)
gemv_runs_b(const u32* __restrict__ words, const uint2* __restrict__ table, const __half* __restrict__ row_scale,
            const int* __restrict__ book_off, const uint2* __restrict__ X, __half* __restrict__ Y,
            const int* __restrict__ tiles, int nitems, int S, int R, int B, int table_n,
            const float* __restrict__ yscale, const int4* __restrict__ blk, float* __restrict__ ws,
            unsigned int* __restrict__ cnt)
{
    constexpr int W = THREADS / 32;
    extern __shared__ __align__(16) uint2 stab[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
#if BOOKS
    const int4 bk = blk[blockIdx.x];
    table += bk.x;
    int it = bk.y + (bk.w & 0xffff);
    const int istride = bk.w >> 16;
    nitems = bk.z;
#else
    int it = blockIdx.x;
    const int istride = gridDim.x;
#endif
#if SMEM
    {
        const uint4* src = (const uint4*)table;
        uint4* dst = (uint4*)stab;
        const int n4 = table_n >> 1;
        int i = threadIdx.x;
        for (; i + 3 * THREADS < n4; i += 4 * THREADS) {
            const uint4 p = __ldg(src + i), q = __ldg(src + i + THREADS), r = __ldg(src + i + 2 * THREADS),
                        s = __ldg(src + i + 3 * THREADS);
            dst[i] = p; dst[i + THREADS] = q; dst[i + 2 * THREADS] = r; dst[i + 3 * THREADS] = s;
        }
        for (; i < n4; i += THREADS) dst[i] = __ldg(src + i);
        if ((table_n & 1) && threadIdx.x == 0) stab[table_n - 1] = table[table_n - 1];
    }
    __syncthreads();
    const uint2* tab = stab;
#else
    const uint2* tab = table;
#endif
    const float ys = yscale[0];
#if XS
    uint2* xs = stab + (SMEM ? table_n : 0);
#else
    const uint2* xs = nullptr;
#endif
    for (; it < nitems; it += istride) {
        const int grp = it / S, sl = it - grp * S;
        const int tile = grp * W + warp;
#if XS
        const bool active = tiles[tile * 16] >= 0;   // padding warps still take part in the X staging
#else
        if (tiles[tile * 16] < 0) continue;     // all-padding tile (book padding): no work, no output
        const bool active = true;
#endif
        const int r0 = tiles[tile * 16 + g], r1 = tiles[tile * 16 + g + 8];
#if TILE
        // tiles are single-book: book_off holds each tile's table offset
        const uint2* tab0 = tab + (BOOKS ? 0 : book_off[tile]);
        const uint2* tab1 = tab0;
        const u32* wtl = words + (size_t)tile * TWORDS + lane;
#else
        const uint2* tab0 = tab + ((BOOKS || r0 < 0) ? 0 : book_off[r0]);
        const uint2* tab1 = tab + ((BOOKS || r1 < 0) ? 0 : book_off[r1]);
        const u32* wb0 = r0 >= 0 ? words + (size_t)r0 * WROW : nullptr;
        const u32* wb1 = r1 >= 0 ? words + (size_t)r1 * WROW : nullptr;
#endif
        float acc[NB8][4];
        #pragma unroll
        for (int nb = 0; nb < NB8; ++nb) { acc[nb][0] = acc[nb][1] = acc[nb][2] = acc[nb][3] = 0.f; }
        const int u0 = sl * NUNITS / S, u1 = (sl + 1) * NUNITS / S;
        #pragma unroll 1
        for (int u = u0; u < u1; ++u) {
#if XS
            __syncthreads();                     // previous unit's slice fully read
            load_xs(X, xs, u, B);
            __syncthreads();
#endif
            if (!active) continue;
            if (u < NFULL * 8) {
                const int bb = u >> 3, L = 4 * (u & 7) + t;
#if TILE
                unit_t<16, NWB>(wtl + u * 2 * NWB * 32, 2 * (bb * 256 + L), tab0, tab1, X, g, B, acc, xs, t);
#else
                const int off = bb * 32 * NWB;
                unit<16, NWB, false>(wb0 ? wb0 + off : nullptr, wb1 ? wb1 + off : nullptr, L, 2 * (bb * 256 + L),
                                     tab0, tab1, X, g, B, acc, xs, t);
#endif
            }
#if JT > 0
            else {
                const int L = 4 * (u - NFULL * 8) + t;
#if TILE
                unit_t<JT * 2, NWT>(wtl + (NFULL * 8 * 2 * NWB + (u - NFULL * 8) * 2 * NWT) * 32, 2 * (NFULL * 256 + L),
                                    tab0, tab1, X, g, B, acc, xs, t);
#else
                const int off = NFULL * 32 * NWB;
                unit<JT * 2, NWT, true>(wb0 ? wb0 + off : nullptr, wb1 ? wb1 + off : nullptr, L,
                                        2 * (NFULL * 256 + L), tab0, tab1, X, g, B, acc, xs, t);
#endif
            }
#endif
        }
        if (!active) continue;
        if (S > 1) {
            #pragma unroll
            for (int nb = 0; nb < NB8; ++nb) {
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int n = nb * 8 + 2 * t + (e & 1), row = e < 2 ? r0 : r1;
                    if (n < B && row >= 0) ws[((size_t)sl * B + n) * R + row] = acc[nb][e];
                }
            }
            __threadfence();
            __syncwarp();
            unsigned int tk = 0;
            if (lane == 0) tk = atomicAdd(&cnt[tile], 1u);
            tk = __shfl_sync(0xffffffffu, tk, 0);
            if (tk != (unsigned)(S - 1)) continue;
            __threadfence();
            #pragma unroll
            for (int nb = 0; nb < NB8; ++nb) {
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int n = nb * 8 + 2 * t + (e & 1), row = e < 2 ? r0 : r1;
                    float v = 0.f;
                    if (n < B && row >= 0)
                        for (int k = 0; k < S; ++k) v += k == sl ? acc[nb][e] : __ldcg(ws + ((size_t)k * B + n) * R + row);
                    acc[nb][e] = v;
                }
            }
            if (lane == 0) cnt[tile] = 0u;
        }
        #pragma unroll
        for (int nb = 0; nb < NB8; ++nb) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int n = nb * 8 + 2 * t + (e & 1), row = e < 2 ? r0 : r1;
                if (n < B && row >= 0) Y[(size_t)n * R + row] = __float2half(acc[nb][e] * ys * __half2float(row_scale[row]));
            }
        }
    }
}
"""

_SRC_Q8 = r"""
// int8 x4 -> two half2 (exact): bytes biased to uint8, placed under exponent 0x64 (1024 + u), minus 1152
__device__ __forceinline__ void i8x4_h2(u32 w, u32& lo, u32& hi) {
    const u32 u = w ^ 0x80808080u;
    const __half2 bias = __halves2half2(__ushort_as_half((unsigned short)0x6480), __ushort_as_half((unsigned short)0x6480));
    u32 l = __byte_perm(u, 0x64646464u, 0x4140), h = __byte_perm(u, 0x64646464u, 0x4342);
    __half2 lh = __hsub2(*(__half2*)&l, bias), hh = __hsub2(*(__half2*)&h, bias);
    lo = *(u32*)&lh; hi = *(u32*)&hh;
}

// warp = 16-row tile; thread (g, t) takes columns 8t..8t+7 of each 32-column block for rows g, g + 8
extern "C" __global__ void __launch_bounds__(256)
gemv_q8_b(const uint2* __restrict__ q, const __half* __restrict__ scales, const uint4* __restrict__ X,
          __half* __restrict__ Y, int R, int NB, int B)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int ntiles = (R + 15) >> 4;
    for (int tile = blockIdx.x * 8 + warp; tile < ntiles; tile += gridDim.x * 8) {
        const int r0 = tile * 16 + g, r1 = r0 + 8;
        const size_t ra = r0 < R ? r0 : R - 1, rb = r1 < R ? r1 : R - 1;
        float acc[NB8][4];
        #pragma unroll
        for (int nb = 0; nb < NB8; ++nb) { acc[nb][0] = acc[nb][1] = acc[nb][2] = acc[nb][3] = 0.f; }
        // KU consecutive 32-column blocks per iteration: 32 * KU contiguous bytes per row in flight
        #pragma unroll 1
        for (int k0 = 0; k0 < NB; k0 += KU) {
            uint2 qa[KU], qb[KU];
            float sa[KU], sb[KU];
            #pragma unroll
            for (int u = 0; u < KU; ++u) {
                const int kb = min(k0 + u, NB - 1);
                qa[u] = __ldg(q + (ra * NB + kb) * 4 + t);
                qb[u] = __ldg(q + (rb * NB + kb) * 4 + t);
                sa[u] = k0 + u < NB ? __half2float(scales[ra * NB + kb]) : 0.f;
                sb[u] = k0 + u < NB ? __half2float(scales[rb * NB + kb]) : 0.f;
            }
            #pragma unroll
            for (int u = 0; u < KU; ++u) {
                const int kb = min(k0 + u, NB - 1);
                u32 a0l, a0h, a1l, a1h, b0l, b0h, b1l, b1h;
                i8x4_h2(qa[u].x, a0l, a0h); i8x4_h2(qa[u].y, a1l, a1h);
                i8x4_h2(qb[u].x, b0l, b0h); i8x4_h2(qb[u].y, b1l, b1h);
                #pragma unroll
                for (int nb = 0; nb < NB8; ++nb) {
                    const int n = nb * 8 + g;
                    const uint4 xv = n < B ? __ldg(X + ((size_t)n * NB + kb) * 4 + t) : make_uint4(0u, 0u, 0u, 0u);
                    float c[4] = {0.f, 0.f, 0.f, 0.f};
                    mma_code(c, a0l, b0l, a0h, b0h, xv.x, xv.y);
                    mma_code(c, a1l, b1l, a1h, b1h, xv.z, xv.w);
                    acc[nb][0] += c[0] * sa[u]; acc[nb][1] += c[1] * sa[u];
                    acc[nb][2] += c[2] * sb[u]; acc[nb][3] += c[3] * sb[u];
                }
            }
        }
        #pragma unroll
        for (int nb = 0; nb < NB8; ++nb) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int n = nb * 8 + 2 * t + (e & 1), row = e < 2 ? r0 : r1;
                if (n < B && row < R) Y[(size_t)n * R + row] = __float2half(acc[nb][e]);
            }
        }
    }
}

// Tile-major variant (Q8Batch's repacked copy): per 16-row tile and 64-column superblock, 1 KB laid
// out in thread order, qt[((tile * NSB + sb) * 2 + h) * 32 + lane] (uint4; h = 0: row g, 1: row g + 8),
// so every warp load is 512 contiguous bytes; scales st[(tile * NB + kb) * 16 + row in tile].
// Otherwise as gemv_q8_bw (scale folded into the fp16 weights).
extern "C" __global__ void __launch_bounds__(256)
gemv_q8_bt(const uint4* __restrict__ qt, const __half* __restrict__ st, const uint4* __restrict__ X,
           __half* __restrict__ Y, int R, int NSB, int B)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int ntiles = (R + 15) >> 4, NB = NSB * 2;
    for (int tile = blockIdx.x * 8 + warp; tile < ntiles; tile += gridDim.x * 8) {
        const int r0 = tile * 16 + g, r1 = r0 + 8;
        const uint4* qtile = qt + (size_t)tile * NSB * 64 + lane;
        const __half* stile = st + (size_t)tile * NB * 16 + g + (t >> 1) * 16;
        float acc[NB8][4];
        #pragma unroll
        for (int nb = 0; nb < NB8; ++nb) { acc[nb][0] = acc[nb][1] = acc[nb][2] = acc[nb][3] = 0.f; }
        #pragma unroll 1
        for (int k0 = 0; k0 < NSB; k0 += KU) {
            uint4 qa[KU], qb[KU];
            __half sa[KU], sb[KU];
            #pragma unroll
            for (int u = 0; u < KU; ++u) {
                const int kb = min(k0 + u, NSB - 1);
                qa[u] = __ldg(qtile + kb * 64);
                qb[u] = __ldg(qtile + kb * 64 + 32);
                sa[u] = k0 + u < NSB ? stile[kb * 32] : __float2half(0.f);
                sb[u] = k0 + u < NSB ? stile[kb * 32 + 8] : __float2half(0.f);
            }
            #pragma unroll
            for (int u = 0; u < KU; ++u) {
                const int kb = min(k0 + u, NSB - 1);
                u32 a[8], b[8];
                const __half2 s2a = __half2half2(sa[u]), s2b = __half2half2(sb[u]);
                i8x4_h2(qa[u].x, a[0], a[1]); i8x4_h2(qa[u].y, a[2], a[3]);
                i8x4_h2(qa[u].z, a[4], a[5]); i8x4_h2(qa[u].w, a[6], a[7]);
                i8x4_h2(qb[u].x, b[0], b[1]); i8x4_h2(qb[u].y, b[2], b[3]);
                i8x4_h2(qb[u].z, b[4], b[5]); i8x4_h2(qb[u].w, b[6], b[7]);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    __half2 ha = __hmul2(*(__half2*)&a[i], s2a), hb = __hmul2(*(__half2*)&b[i], s2b);
                    a[i] = *(u32*)&ha; b[i] = *(u32*)&hb;
                }
                #pragma unroll
                for (int nb = 0; nb < NB8; ++nb) {
                    const int n = nb * 8 + g;
                    uint4 x0 = make_uint4(0u, 0u, 0u, 0u), x1 = x0;
                    if (n < B) { x0 = __ldg(X + ((size_t)n * NSB + kb) * 8 + 2 * t); x1 = __ldg(X + ((size_t)n * NSB + kb) * 8 + 2 * t + 1); }
                    mma_code(acc[nb], a[0], b[0], a[1], b[1], x0.x, x0.y);
                    mma_code(acc[nb], a[2], b[2], a[3], b[3], x0.z, x0.w);
                    mma_code(acc[nb], a[4], b[4], a[5], b[5], x1.x, x1.y);
                    mma_code(acc[nb], a[6], b[6], a[7], b[7], x1.z, x1.w);
                }
            }
        }
        #pragma unroll
        for (int nb = 0; nb < NB8; ++nb) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int n = nb * 8 + 2 * t + (e & 1), row = e < 2 ? r0 : r1;
                if (n < B && row < R) Y[(size_t)n * R + row] = __float2half(acc[nb][e]);
            }
        }
    }
}

// Wide variant: thread (g, t) takes columns 16t..16t+15 of each 64-column superblock (one 16-byte
// load per row: 64 contiguous bytes per row per warp load); its 32-column block's scale is folded
// into the fp16 weights (fp16(q * s): one rounding per weight), so the mmas accumulate directly.
extern "C" __global__ void __launch_bounds__(256)
gemv_q8_bw(const uint4* __restrict__ q, const __half* __restrict__ scales, const uint4* __restrict__ X,
           __half* __restrict__ Y, int R, int NSB, int B)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int ntiles = (R + 15) >> 4, NB = NSB * 2;
    for (int tile = blockIdx.x * 8 + warp; tile < ntiles; tile += gridDim.x * 8) {
        const int r0 = tile * 16 + g, r1 = r0 + 8;
        const size_t ra = r0 < R ? r0 : R - 1, rb = r1 < R ? r1 : R - 1;
        float acc[NB8][4];
        #pragma unroll
        for (int nb = 0; nb < NB8; ++nb) { acc[nb][0] = acc[nb][1] = acc[nb][2] = acc[nb][3] = 0.f; }
        #pragma unroll 1
        for (int k0 = 0; k0 < NSB; k0 += KU) {
            uint4 qa[KU], qb[KU];
            __half sa[KU], sb[KU];
            #pragma unroll
            for (int u = 0; u < KU; ++u) {
                const int kb = min(k0 + u, NSB - 1);
                qa[u] = __ldg(q + (ra * NSB + kb) * 4 + t);
                qb[u] = __ldg(q + (rb * NSB + kb) * 4 + t);
                sa[u] = k0 + u < NSB ? scales[ra * NB + kb * 2 + (t >> 1)] : __float2half(0.f);
                sb[u] = k0 + u < NSB ? scales[rb * NB + kb * 2 + (t >> 1)] : __float2half(0.f);
            }
            #pragma unroll
            for (int u = 0; u < KU; ++u) {
                const int kb = min(k0 + u, NSB - 1);
                u32 a[8], b[8];
                const __half2 s2a = __half2half2(sa[u]), s2b = __half2half2(sb[u]);
                i8x4_h2(qa[u].x, a[0], a[1]); i8x4_h2(qa[u].y, a[2], a[3]);
                i8x4_h2(qa[u].z, a[4], a[5]); i8x4_h2(qa[u].w, a[6], a[7]);
                i8x4_h2(qb[u].x, b[0], b[1]); i8x4_h2(qb[u].y, b[2], b[3]);
                i8x4_h2(qb[u].z, b[4], b[5]); i8x4_h2(qb[u].w, b[6], b[7]);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    __half2 ha = __hmul2(*(__half2*)&a[i], s2a), hb = __hmul2(*(__half2*)&b[i], s2b);
                    a[i] = *(u32*)&ha; b[i] = *(u32*)&hb;
                }
                #pragma unroll
                for (int nb = 0; nb < NB8; ++nb) {
                    const int n = nb * 8 + g;
                    uint4 x0 = make_uint4(0u, 0u, 0u, 0u), x1 = x0;
                    if (n < B) { x0 = __ldg(X + ((size_t)n * NSB + kb) * 8 + 2 * t); x1 = __ldg(X + ((size_t)n * NSB + kb) * 8 + 2 * t + 1); }
                    mma_code(acc[nb], a[0], b[0], a[1], b[1], x0.x, x0.y);
                    mma_code(acc[nb], a[2], b[2], a[3], b[3], x0.z, x0.w);
                    mma_code(acc[nb], a[4], b[4], a[5], b[5], x1.x, x1.y);
                    mma_code(acc[nb], a[6], b[6], a[7], b[7], x1.z, x1.w);
                }
            }
        }
        #pragma unroll
        for (int nb = 0; nb < NB8; ++nb) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int n = nb * 8 + 2 * t + (e & 1), row = e < 2 ? r0 : r1;
                if (n < B && row < R) Y[(size_t)n * R + row] = __float2half(acc[nb][e]);
            }
        }
    }
}
"""

_mods: dict = {}
SMEM_LIMIT = 48 * 1024
BMAX = 32
TILE_PAD = 8        # tiles per book padded to a multiple of this (RunsBatch warps must divide it)


BMAX_XS = 128      # X staged in shared memory (xs=True): larger batches fit


def _nb8(B: int) -> int:
    assert 1 <= B <= BMAX_XS, f"batch {B} outside 1..{BMAX_XS}"
    return (B + 7) // 8


def _runs_fn(bits, nch, nb8, smem, books, threads, acc16=False, tile=False, xs=False):
    import cupy as cp
    from .proto_runs import geometry
    nfull, jt = geometry(nch)
    key = ("runs", bits, nfull, jt, nb8, smem, books, threads, acc16, tile, xs)
    if key not in _mods:
        src = (f"#define BITS {bits}\n#define NFULL {nfull}\n#define JT {jt}\n#define NB8 {nb8}\n"
               f"#define SMEM {int(smem)}\n#define BOOKS {int(books)}\n#define THREADS {threads}\n"
               f"#define ACC16 {int(acc16)}\n#define TILE {int(tile)}\n#define XS {int(xs)}\n") + _COMMON + _SRC_RUNS
        _mods[key] = cp.RawModule(code=src, options=("-std=c++17", "--use_fast_math")).get_function("gemv_runs_b")
    return _mods[key]


_Q8_KERNELS = {"narrow": "gemv_q8_b", "wide": "gemv_q8_bw", "tile": "gemv_q8_bt"}


def _q8_fn(nb8, ku=4, kind="wide", acc16=False):
    """acc16 only for the scale-folded kinds (narrow sums raw int8 products: fp16 could overflow)."""
    import cupy as cp
    assert not (acc16 and kind == "narrow")
    key = ("q8", nb8, ku, kind, acc16)
    if key not in _mods:
        src = f"#define NB8 {nb8}\n#define KU {ku}\n#define ACC16 {int(acc16)}\n" + _COMMON + _SRC_Q8
        _mods[key] = cp.RawModule(code=src, options=("-std=c++17",)).get_function(_Q8_KERNELS[kind])
    return _mods[key]


def _sms(dev) -> int:
    return torch.cuda.get_device_properties(dev).multi_processor_count


class _Workspace:
    """fp32 split-K partials shared by every launcher on a device (kernels on one stream run in
    order, so one buffer serves them all); grown on demand before graph capture."""
    bufs: dict = {}

    @classmethod
    def get(cls, n: int, dev) -> torch.Tensor:
        key = str(torch.device(dev))
        b = cls.bufs.get(key)
        if b is None or b.numel() < n:
            b = torch.empty(max(n, 1), dtype=torch.float32, device=dev)
            cls.bufs[key] = b
        return b


def tile_words(rl, tiles: torch.Tensor):
    """The TILE layout of a RunsLinear's lane-major words for 16-row tiles `tiles` [nt, 16] (row
    ids, -1 = padding, every tile within one book): int32 [nt, TWORDS] and each tile's table
    offset (int32 [nt])."""
    from .proto_runs import geometry
    nfull, jt = geometry(rl.nch)
    bits = rl.bits
    nwb, nwt = bits // 2, (jt * 2 * bits + 31) // 32
    nt = tiles.shape[0]
    rows = tiles.reshape(-1).long()
    W = rl.words[rows.clamp_min(0)] * (rows >= 0)[:, None].to(rl.words.dtype)       # [nt * 16, WROW]
    parts = []
    if nfull:
        f = W[:, :nfull * 32 * nwb].reshape(nt, 2, 8, nfull, 8, 4, nwb)           # tile, h, g, b, s, t, i
        parts.append(f.permute(0, 3, 4, 1, 6, 2, 5).reshape(nt, -1))              # tile, b, s, h, i, g, t
    if jt:
        nt_tags = 2 * jt
        st = W[:, nfull * 32 * nwb:].to(torch.int64) & 0xFFFFFFFF                   # tail bit stream
        st = torch.cat([st, torch.zeros_like(st[:, :2])], 1)
        L = torch.arange(32, device=W.device)
        start = L[:, None] * nt_tags * bits + torch.arange(nwt, device=W.device)[None] * 32     # [32, nwt]
        wi, sh = start >> 5, start & 31
        lo = st[:, wi.reshape(-1)].view(-1, 32, nwt)
        hi = st[:, (wi + 1).reshape(-1)].view(-1, 32, nwt)
        v = ((lo >> sh) | (hi << (32 - sh))) & 0xFFFFFFFF
        end = (L[:, None] + 1) * nt_tags * bits - start                             # bits left in this lane
        keep = torch.where(end >= 32, torch.full_like(end, 0xFFFFFFFF), (1 << end.clamp(0, 31)) - 1)
        v = v & keep
        v = torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)              # [nt * 16, 32, nwt]
        tl = v.reshape(nt, 2, 8, 8, 4, nwt)                                         # tile, h, g, s, t, i
        parts.append(tl.permute(0, 3, 1, 5, 2, 4).reshape(nt, -1))                # tile, s, h, i, g, t
    wt = torch.cat(parts, 1).contiguous()
    first = tiles[:, 0].long()
    off = torch.where(first >= 0, rl.book_off[first.clamp_min(0)].long(), torch.zeros_like(first))
    return wt, off.to(torch.int32).contiguous()


class RunsBatch:
    """Batched launcher for one proto_runs.RunsLinear (d = 4, M = 1): run(X [B, C] fp16,
    Y [B, R] fp16) for any B <= bmax. warps: tiles per block; bps: blocks per SM;
    max_split: column slices (split-K) at most. layout: "lane" reads the RunsLinear's words in
    place; "tile" reads a repacked copy (tile_words: same size plus tail padding, cached on the
    RunsLinear) whose warp loads are contiguous: ~1.5-2x faster at small B on an RTX 2060."""

    def __init__(self, rl, bmax: int = BMAX, warps: int = 8, bps: int = 2, max_split: int = 8,
                 table: str = "auto", acc16: bool = False, layout: str = "lane", xs: bool = False):
        from . import proto_runs as pr
        assert not rl.has_mask, "batched runs kernel: one column order (M = 1) only"
        assert bmax <= (BMAX_XS if xs else BMAX), f"bmax {bmax} over {BMAX_XS if xs else BMAX} (xs={xs})"
        self.rl, self.R, self.C, self.bmax, self.xs = rl, rl.R, rl.nch * 4, bmax, xs
        dev = rl.words.device
        K = 1 << rl.bits
        G = rl.table.shape[0] // K
        tb = rl.table.numel() * 2
        self.books = table == "books" or (table == "auto" and tb > SMEM_LIMIT and K * 8 <= SMEM_LIMIT and G > 1)
        self.smem = self.books or (tb <= SMEM_LIMIT if table == "auto" else table == "smem")
        self.table_bytes = K * 8 if self.books else tb
        self.threads, self.W, self.acc16 = warps * 32, warps, acc16
        # 16-row tiles grouped by book, each book's tiles padded to whole groups of TILE_PAD (a
        # multiple of W: the tile layout is then the same for every W dividing it)
        assert TILE_PAD % warps == 0, f"warps must divide {TILE_PAD}"
        book = (rl.book_off.long() // K).cpu()
        tiles, gstart = [], [0]
        for b in range(G):
            rows = torch.nonzero(book == b).flatten()
            if rows.numel() == 0:
                gstart.append(gstart[-1])
                continue
            n_t = -(-rows.numel() // 16)
            n_t = -(-n_t // TILE_PAD) * TILE_PAD
            pad = n_t * 16 - rows.numel()
            tiles.append(torch.cat([rows, torch.full((pad,), -1, dtype=rows.dtype)]))
            gstart.append(gstart[-1] + n_t // warps)
        self.tiles = torch.cat(tiles).to(torch.int32).to(dev).contiguous()
        self.ngroups = gstart[-1]
        nunits = pr.geometry(rl.nch)[0] * 8 + (8 if pr.geometry(rl.nch)[1] else 0)
        sms = _sms(dev)
        self.grid_cap = sms * bps
        self.S = max(1, min(max_split, nunits // 2, math.ceil(self.grid_cap / self.ngroups)))
        self.nitems = self.ngroups * self.S
        if self.books:
            n = [gstart[b + 1] - gstart[b] for b in range(G)]
            alloc = [0 if c == 0 else max(1, min(c * self.S, round(self.grid_cap * c / max(1, self.ngroups))))
                     for c in n]
            rows = [(b * K, gstart[b] * self.S, gstart[b + 1] * self.S, r | (a << 16))
                    for b, a in enumerate(alloc) for r in range(a)]
            self.blk = torch.tensor(rows, dtype=torch.int32, device=dev).contiguous()
            self.grid = self.blk.shape[0]
        else:
            self.blk = self.tiles
            self.grid = min(self.nitems, self.grid_cap)
        self.cnt = torch.zeros(self.tiles.numel() // 16, dtype=torch.int32, device=dev)
        assert layout in ("lane", "tile")
        self.layout = layout
        if layout == "tile":         # one repacked copy per matrix, shared by every launcher
            if getattr(rl, "_tile_words", None) is None:
                rl._tile_words = tile_words(rl, self.tiles.view(-1, 16))
            self.wt, self.tile_off = rl._tile_words
        # the shared buffer as of now (large enough for this launcher); kept referenced, so a graph
        # captured with it stays valid when later launchers grow the shared one
        self.ws = _Workspace.get(self.S * bmax * self.R if self.S > 1 else 1, dev)

    def run(self, X: torch.Tensor, Y: torch.Tensor, B: int | None = None):
        import cupy as cp
        rl = self.rl
        B = X.shape[0] if B is None else B
        assert B <= self.bmax and X.is_contiguous() and X.dtype == torch.float16
        tile = self.layout == "tile"
        nb8 = _nb8(B)
        fn = _runs_fn(rl.bits, rl.nch, nb8, self.smem, self.books, self.threads, self.acc16, tile, self.xs)
        shm = (self.table_bytes if self.smem else 0) + (nb8 * 8 * 64 * 8 if self.xs else 0)
        if shm > 48 * 1024:
            fn.max_dynamic_shared_size_bytes = shm
        K = 1 << rl.bits
        words, boff = (self.wt, self.tile_off) if tile else (rl.words, rl.book_off)
        args = (cp.uint64(words.data_ptr()), cp.uint64(rl.table.data_ptr()), cp.uint64(rl.row_scale.data_ptr()),
                cp.uint64(boff.data_ptr()), cp.uint64(X.data_ptr()), cp.uint64(Y.data_ptr()),
                cp.uint64(self.tiles.data_ptr()), cp.int32(self.nitems), cp.int32(self.S), cp.int32(self.R),
                cp.int32(B), cp.int32(K if self.books else rl.table.shape[0]), cp.uint64(rl.yscale.data_ptr()),
                cp.uint64(self.blk.data_ptr()), cp.uint64(self.ws.data_ptr()), cp.uint64(self.cnt.data_ptr()))
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            fn((self.grid,), (self.threads,), args, shared_mem=shm)
        return Y


def gemv_runs_batched(rl, X: torch.Tensor, Y: torch.Tensor | None = None, **kw) -> torch.Tensor:
    """One-off batched run-kernel GEMM (builds the launcher; engines keep a RunsBatch)."""
    lb = getattr(rl, "_batch_launcher", None)
    if lb is None or kw:
        lb = RunsBatch(rl, **kw)
        if not kw:
            rl._batch_launcher = lb
    if Y is None:
        Y = torch.empty(X.shape[0], rl.R, dtype=torch.float16, device=X.device)
    return lb.run(X, Y)


def gemv_q8_batched(q: torch.Tensor, scales: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, R: int, C: int,
                    B: int | None = None, bps: int = 4, ku: int = 4, wide: bool | None = None) -> torch.Tensor:
    """Y[B, R] = X[B, C] . dequant(q, scales)^T for the Q8_0-like head (proto_formats.q8_pack), from
    its natural layout. wide (default when C % 64 == 0): scales folded into fp16 weights, 16-byte
    loads; else exact int8 products times the scale per 32-block. Engines use Q8Batch (repacked)."""
    import cupy as cp
    B = X.shape[0] if B is None else B
    assert C % 32 == 0 and X.is_contiguous() and X.dtype == torch.float16
    wide = C % 64 == 0 if wide is None else wide
    fn = _q8_fn(_nb8(B), ku, "wide" if wide else "narrow")
    grid = min(_sms(q.device) * bps, (R + 127) // 128)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        fn((grid,), (256,), (cp.uint64(q.data_ptr()), cp.uint64(scales.data_ptr()), cp.uint64(X.data_ptr()),
                             cp.uint64(Y.data_ptr()), cp.int32(R), cp.int32(C // (64 if wide else 32)), cp.int32(B)))
    return Y


class Q8Batch:
    """Batched launcher for the Q8_0-like head (decode_e2e.Q8Lin): a tile-major copy of its int8
    weights and scales (16-row tiles, each 64-column superblock's 1 KB in thread order: every warp
    load is 512 contiguous bytes), run(X [B, C], Y [B, R]). Costs a second copy of the head (the
    batch-1 kernel keeps reading the original). Needs C % 64 == 0."""

    def __init__(self, q: torch.Tensor, scales: torch.Tensor, R: int, C: int, bps: int = 4, ku: int = 4,
                 acc16: bool = False):
        assert C % 64 == 0, "Q8Batch needs C % 64 == 0"
        self.R, self.C, self.bps, self.ku, self.acc16 = R, C, bps, ku, acc16
        T, NSB = (R + 15) // 16, C // 64
        qp = torch.zeros(T * 16, C, dtype=torch.int8, device=q.device)
        qp[:R] = q
        # [tile, h, g, sb, t, 16 bytes] -> [tile, sb, h, g, t, 16]
        self.qt = qp.view(T, 2, 8, NSB, 4, 16).permute(0, 3, 1, 2, 4, 5).contiguous()
        sp = torch.zeros(T * 16, C // 32, dtype=torch.float16, device=q.device)
        sp[:R] = scales
        self.st = sp.view(T, 16, C // 32).transpose(1, 2).contiguous()          # [tile, block, row]
        del qp, sp

    @classmethod
    def from_lin(cls, lin, **kw):
        return cls(lin.q, lin.sc, lin.R, lin.C, **kw)

    def nbytes(self) -> int:
        return self.qt.numel() + self.st.numel() * 2

    def run(self, X: torch.Tensor, Y: torch.Tensor, B: int | None = None) -> torch.Tensor:
        import cupy as cp
        B = X.shape[0] if B is None else B
        assert X.is_contiguous() and X.dtype == torch.float16
        fn = _q8_fn(_nb8(B), self.ku, "tile", self.acc16)
        grid = min(_sms(X.device) * self.bps, (self.R + 127) // 128)
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            fn((grid,), (256,), (cp.uint64(self.qt.data_ptr()), cp.uint64(self.st.data_ptr()), cp.uint64(X.data_ptr()),
                                 cp.uint64(Y.data_ptr()), cp.int32(self.R), cp.int32(self.C // 64), cp.int32(B)))
        return Y


def runs_key(rl) -> tuple:
    """Launch-relevant shape of a RunsLinear: (R, C, bits, table rows)."""
    return (rl.R, rl.nch * 4, rl.bits, rl.table.shape[0])


TUNE_GRID = [(w, b, m) for w in (4, 8) for b in (1, 2, 4) for m in (1, 2, 4, 8)]


@torch.no_grad()
def tune_runs(rls: list, B: int, grid=TUNE_GRID, layout: str = "tile", acc16: bool = False, n: int = 10,
              reps: int = 3, log=None) -> dict:
    """Fastest (warps, bps, max_split) per distinct runs_key among `rls` at batch B (CUDA-graph
    timed, min of reps; configs with the same effective split / grid are timed once)."""
    reps_by_key = {}
    for rl in rls:
        reps_by_key.setdefault(runs_key(rl), rl)
    best = {}
    for key, rl in reps_by_key.items():
        X = torch.randn(B, key[1], device=rl.words.device).half()
        Y = torch.empty(B, key[0], dtype=torch.float16, device=rl.words.device)
        seen, res = set(), []
        for w, b, m in grid:
            lb = RunsBatch(rl, bmax=B, warps=w, bps=b, max_split=m, layout=layout, acc16=acc16)
            eff = (w, lb.grid, lb.S)
            if eff in seen:
                continue
            seen.add(eff)
            lb.run(X, Y, B)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(n):
                    lb.run(X, Y, B)
            ts = []
            for _ in range(reps):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                g.replay()
                e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1) / n)
            res.append((min(ts), (w, b, m)))
            del g
        res.sort()
        best[key] = res[0][1]
        if log:
            log(f"tune B={B} R={key[0]} C={key[1]} bits={key[2]} table={key[3]}: {res[0][1]} {res[0][0] * 1e3:.1f} us "
                f"(of {len(res)}; worst {res[-1][0] * 1e3:.1f} us)")
    return best
