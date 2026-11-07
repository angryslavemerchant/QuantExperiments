"""Batched decode GEMMs: Y[B, R] = X[B, C] . W^T for B = 1..32 activation vectors, decoding every
weight code once per step (tensor-core mma over the batch).

Codebook runs (d = 4, the lane-major words of proto_runs.RunsLinear, read in place):
- a warp owns a tile of 16 rows (one book: tiles are grouped by book) and does
  mma(16 rows x 8 batch x 4 codes) steps: thread (g, t) = (lane / 4, lane % 4) decodes the code
  of rows g and g + 8 at code slot t (table entry = 4 fp16 weights = its A fragment, with the
  mma's k index permuted so slot t's 4 k positions are one code's 4 columns) and loads the
  matching 4 x values of batch row g (8 bytes) per 8-row batch tile;
- slot t walks the codes of run-kernel lanes L = 4s + t (s = 0..7) of a 512-chunk batch, so the
  words come straight from proto_runs' layout (same tail realignment);
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
__device__ __forceinline__ void mma_code(float* c, u32 a_lo0, u32 a_lo1, u32 a_hi0, u32 a_hi1, u32 b0, u32 b1) {
#if __CUDA_ARCH__ >= 800
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

// One column unit for this thread: the NT tags of run lane L for rows g (w0) and g + 8 (w1).
// cbase = chunk of tag 0; tag j sits at chunk cbase + (j >> 1) * 64 + (j & 1).
template <int NT, int NWD, bool TAIL>
__device__ __forceinline__ void unit(const u32* __restrict__ w0, const u32* __restrict__ w1, int L, int cbase,
                                     const uint2* tab0, const uint2* tab1, const uint2* __restrict__ X, int g,
                                     int B, float (*acc)[4])
{
    const u32 MASK = (1u << BITS) - 1u;
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
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int bit = j * BITS, wi = bit >> 5, sh = bit & 31;
        const u32 va = ((sh + BITS <= 32) ? (a[wi] >> sh) : __funnelshift_r(a[wi], a[wi + 1], sh)) & MASK;
        const u32 vb = ((sh + BITS <= 32) ? (b[wi] >> sh) : __funnelshift_r(b[wi], b[wi + 1], sh)) & MASK;
        const uint2 ta = tab0[va], tb = tab1[vb];
        const int c = cbase + (j >> 1) * 64 + (j & 1);
        #pragma unroll
        for (int nb = 0; nb < NB8; ++nb) {
            const int n = nb * 8 + g;
            const uint2 xv = n < B ? __ldg(X + (size_t)n * NCH + c) : make_uint2(0u, 0u);
            mma_code(acc[nb], ta.x, tb.x, ta.y, tb.y, xv.x, xv.y);
        }
    }
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
    for (; it < nitems; it += istride) {
        const int grp = it / S, sl = it - grp * S;
        const int tile = grp * W + warp;
        const int r0 = tiles[tile * 16 + g], r1 = tiles[tile * 16 + g + 8];
        const uint2* tab0 = tab + ((BOOKS || r0 < 0) ? 0 : book_off[r0]);
        const uint2* tab1 = tab + ((BOOKS || r1 < 0) ? 0 : book_off[r1]);
        const u32* wb0 = r0 >= 0 ? words + (size_t)r0 * WROW : nullptr;
        const u32* wb1 = r1 >= 0 ? words + (size_t)r1 * WROW : nullptr;
        float acc[NB8][4];
        #pragma unroll
        for (int nb = 0; nb < NB8; ++nb) { acc[nb][0] = acc[nb][1] = acc[nb][2] = acc[nb][3] = 0.f; }
        const int u0 = sl * NUNITS / S, u1 = (sl + 1) * NUNITS / S;
        #pragma unroll 1
        for (int u = u0; u < u1; ++u) {
            if (u < NFULL * 8) {
                const int bb = u >> 3, L = 4 * (u & 7) + t, off = bb * 32 * NWB;
                unit<16, NWB, false>(wb0 ? wb0 + off : nullptr, wb1 ? wb1 + off : nullptr, L, 2 * (bb * 256 + L),
                                     tab0, tab1, X, g, B, acc);
            }
#if JT > 0
            else {
                const int L = 4 * (u - NFULL * 8) + t, off = NFULL * 32 * NWB;
                unit<JT * 2, NWT, true>(wb0 ? wb0 + off : nullptr, wb1 ? wb1 + off : nullptr, L,
                                        2 * (NFULL * 256 + L), tab0, tab1, X, g, B, acc);
            }
#endif
        }
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


def _nb8(B: int) -> int:
    assert 1 <= B <= BMAX, f"batch {B} outside 1..{BMAX}"
    return (B + 7) // 8


def _runs_fn(bits, nch, nb8, smem, books, threads):
    import cupy as cp
    from .proto_runs import geometry
    nfull, jt = geometry(nch)
    key = ("runs", bits, nfull, jt, nb8, smem, books, threads)
    if key not in _mods:
        src = (f"#define BITS {bits}\n#define NFULL {nfull}\n#define JT {jt}\n#define NB8 {nb8}\n"
               f"#define SMEM {int(smem)}\n#define BOOKS {int(books)}\n#define THREADS {threads}\n") + _COMMON + _SRC_RUNS
        _mods[key] = cp.RawModule(code=src, options=("-std=c++17", "--use_fast_math")).get_function("gemv_runs_b")
    return _mods[key]


def _q8_fn(nb8, ku=4, wide=True):
    import cupy as cp
    key = ("q8", nb8, ku, wide)
    if key not in _mods:
        src = f"#define NB8 {nb8}\n#define KU {ku}\n" + _COMMON + _SRC_Q8
        _mods[key] = cp.RawModule(code=src, options=("-std=c++17",)).get_function("gemv_q8_bw" if wide else "gemv_q8_b")
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


class RunsBatch:
    """Batched launcher for one proto_runs.RunsLinear (d = 4, M = 1): run(X [B, C] fp16,
    Y [B, R] fp16) for any B <= bmax. warps: tiles per block; bps: blocks per SM;
    max_split: column slices (split-K) at most."""

    def __init__(self, rl, bmax: int = BMAX, warps: int = 8, bps: int = 2, max_split: int = 8,
                 table: str = "auto"):
        from . import proto_runs as pr
        assert not rl.has_mask, "batched runs kernel: one column order (M = 1) only"
        self.rl, self.R, self.C, self.bmax = rl, rl.R, rl.nch * 4, bmax
        dev = rl.words.device
        K = 1 << rl.bits
        G = rl.table.shape[0] // K
        tb = rl.table.numel() * 2
        self.books = table == "books" or (table == "auto" and tb > SMEM_LIMIT and K * 8 <= SMEM_LIMIT and G > 1)
        self.smem = self.books or (tb <= SMEM_LIMIT if table == "auto" else table == "smem")
        self.table_bytes = K * 8 if self.books else tb
        self.threads, self.W = warps * 32, warps
        # 16-row tiles grouped by book, each book's tiles padded to whole groups of W
        book = (rl.book_off.long() // K).cpu()
        tiles, gstart = [], [0]
        for b in range(G):
            rows = torch.nonzero(book == b).flatten()
            if rows.numel() == 0:
                gstart.append(gstart[-1])
                continue
            n_t = -(-rows.numel() // 16)
            n_t = -(-n_t // warps) * warps
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
        self.ws = _Workspace.get(self.S * bmax * self.R if self.S > 1 else 1, dev)

    def run(self, X: torch.Tensor, Y: torch.Tensor, B: int | None = None):
        import cupy as cp
        rl = self.rl
        B = X.shape[0] if B is None else B
        assert B <= self.bmax and X.is_contiguous() and X.dtype == torch.float16
        if self.S > 1:      # the workspace may have grown (new launchers) since __init__
            self.ws = _Workspace.get(self.S * self.bmax * self.R, X.device)
        fn = _runs_fn(rl.bits, rl.nch, _nb8(B), self.smem, self.books, self.threads)
        shm = self.table_bytes if self.smem else 0
        K = 1 << rl.bits
        args = (cp.uint64(rl.words.data_ptr()), cp.uint64(rl.table.data_ptr()), cp.uint64(rl.row_scale.data_ptr()),
                cp.uint64(rl.book_off.data_ptr()), cp.uint64(X.data_ptr()), cp.uint64(Y.data_ptr()),
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
    """Y[B, R] = X[B, C] . dequant(q, scales)^T for the Q8_0-like head (proto_formats.q8_pack)."""
    import cupy as cp
    B = X.shape[0] if B is None else B
    assert C % 32 == 0 and X.is_contiguous() and X.dtype == torch.float16
    wide = C % 64 == 0 if wide is None else wide
    fn = _q8_fn(_nb8(B), ku, wide)
    grid = min(_sms(q.device) * bps, (R + 127) // 128)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        fn((grid,), (256,), (cp.uint64(q.data_ptr()), cp.uint64(scales.data_ptr()), cp.uint64(X.data_ptr()),
                             cp.uint64(Y.data_ptr()), cp.int32(R), cp.int32(C // (64 if wide else 32)), cp.int32(B)))
    return Y
