"""Fused dequant + GEMM for batched prefill: Y[T, R] = X[T, C] @ W^T with W decoded tile by tile
from a RunsLinear's lane-major words straight into shared memory (fp16) and multiplied on the
tensor cores (mma.sync, fp32 accumulate); no fp16 copy of W in global memory.

Layout (proto_runs.pack_lane_major): chunk c (4 columns) of a row sits at pair p = c >> 6, lane
l = (c >> 1) & 31, e = c & 1; full batches (p < 8*NFULL) keep each lane's 16 tags of pairs
8b .. 8b+7 in one segment of bits/2 words, the JT tail pairs of every lane form one bit stream.

K-tiles are 32 natural, contiguous columns p*256 + 32g .. +31 (chunks of lanes 4g .. 4g+3, pair
p), so X tiles load fully coalesced. They are visited group-major: a "super-tile" = the 8 pairs
of batch b (or the JT tail pairs) for one lane group g, whose tags all sit in 4 consecutive lane
segments (2*bits words per row, 16-byte aligned) or one span of the tail stream. Those words are
staged in shared memory once per super-tile (prefetched into registers during the previous one)
and every k-tile decodes its codewords from there. A GEMM may sum over k in any order.

The row scale multiplies the fp32 accumulator in the epilogue (the dequant path rounds
codeword * scale to fp16 first: results agree to fp16 rounding, ~1e-3 relative).

Tiles: BM = 128 tokens x BN = 128 rows x BK = 32, 8 warps of 64 x 32, double-buffered A / B
tiles (padded rows, conflict-free ldmatrix); the next k-tile's X is loaded into registers while
the current one is multiplied (sm_75 has no cp.async), one barrier per k-tile. MMA: m16n8k16 on
sm_80+ (sm_120 included), two m16n8k8 on sm_75. The codebook is read from shared memory when it
fits next to the tiles, else through L1.
"""

from __future__ import annotations

import torch

_SRC = r"""
#include <cuda_fp16.h>
typedef unsigned int u32;

// BM (tokens per block): 128 (warps 2 x 4, warp tile 64 x 32), 64 / 32 / 16 (warps 1 x 8, warp
// tile BM x 16): small batches skip the mma work on padding rows
#ifndef BM
#define BM 128
#endif
#if BM == 128
#define WN 4
#else
#define WN 8
#endif
#define WMN (8 / WN)                 // warps along tokens
#define TM (BM / WMN)                // warp tile: tokens
#define TN (128 / WN)                // warp tile: weight rows
#define MI (TM / 16)
#define NJ (TN / 8)
#define XU ((BM * 4 + 255) / 256)    // X tile uint4 loads per thread
#define BN 128
#define BK 32
#define LDS (BK + 8)                 // tile row stride in halves (80 bytes): conflict-free ldmatrix
#define NT 256
#define STAGE ((BM + BN) * LDS)      // halves per tile stage
#define WS (2 * BITS + 1)            // staged words per row (odd: conflict-free reads)
#define NWV (4 * ((BN * (BITS / 2) + NT - 1) / NT))   // word registers per thread: full super-tile (uint4s)
#define NWS ((BN * 2 * BITS + NT - 1) / NT)           // tail super-tile (scalars)
#define NWR (NWV > NWS ? NWV : NWS)

struct Ctx {
    const u32* words; const int* book_off; const __half* x;
    int T, R, W, nfull, jt, ldx;
    const int* rmap;
};

// k-tile kt -> super-tile s, pair p, lane group g, j (pair index within the super-tile)
struct KT { int s, p, g, j; };
__device__ __forceinline__ KT kt_info(const Ctx& c, int kt) {
    KT k;
    if (kt < c.nfull * 64) {
        k.s = kt >> 3; k.j = kt & 7; k.g = k.s & 7; k.p = (k.s >> 3) * 8 + k.j;
    } else {
        const int r = kt - c.nfull * 64;
        k.g = r / c.jt; k.j = r - k.g * c.jt; k.s = c.nfull * 8 + k.g; k.p = c.nfull * 8 + k.j;
    }
    return k;
}

// first k-tile of super-tile s (s = nsup: nk)
__device__ __forceinline__ int st_start(const Ctx& c, int s) {
    return s < c.nfull * 8 ? 8 * s : c.nfull * 64 + (s - c.nfull * 8) * c.jt;
}

// super-tile s: staged words are row words [w0, w0 + nw); tag (q, j, e) of lane 4g+q sits at
// staged bit off + (q * qs + 2j + e) * BITS
struct ST { int w0, nw, off, qs, vec; };
__device__ __forceinline__ ST st_info(const Ctx& c, int s) {
    ST t;
    if (s < c.nfull * 8) {
        const int b = s >> 3, g = s & 7;
        t.w0 = (b * 32 + 4 * g) * (BITS / 2); t.nw = 2 * BITS; t.off = 0; t.qs = 16; t.vec = 1;
    } else {
        const int g = s - c.nfull * 8;
        const int bit = g * 8 * c.jt * BITS;
        t.w0 = c.nfull * 32 * (BITS / 2) + (bit >> 5); t.off = bit & 31;
        t.nw = (t.off + 8 * c.jt * BITS + 31) >> 5; t.qs = 2 * c.jt; t.vec = 0;
    }
    return t;
}

// global -> registers: the super-tile's words for the block's BN rows
#ifndef GROUPED
#define GROUPED 0
#endif
// GROUPED: block row slot i holds matrix row rmap[i] (rows sorted by codebook, each book padded to
// whole 128-row tiles with -1), so a block needs one book's table; else slot i = row i.
__device__ __forceinline__ int mrow(const Ctx& c, int i) {
#if GROUPED
    return c.rmap[i];
#else
    return i < c.R ? i : -1;
#endif
}

__device__ __forceinline__ void load_words(const Ctx& c, int s, int n0, u32* wr) {
    const ST t = st_info(c, s);
    const int tid = threadIdx.x;
    if (t.vec) {                                       // BITS/2 aligned uint4 per row
        constexpr int V = BITS / 2;
        #pragma unroll
        for (int u = 0; u < NWV / 4; ++u) {
            const int idx = tid + NT * u, row = idx / V, v = idx - row * V;
            uint4 q = make_uint4(0, 0, 0, 0);
            const int mr = row < BN ? mrow(c, n0 + row) : -1;
            if (mr >= 0)
                q = __ldg((const uint4*)(c.words + (size_t)mr * c.W + t.w0) + v);
            wr[4 * u] = q.x; wr[4 * u + 1] = q.y; wr[4 * u + 2] = q.z; wr[4 * u + 3] = q.w;
        }
    } else {
        #pragma unroll
        for (int u = 0; u < NWS; ++u) {
            const int idx = tid + NT * u, row = idx / t.nw, i = idx - row * t.nw;
            const int mr = row < BN ? mrow(c, n0 + row) : -1;
            wr[u] = (mr >= 0 && t.w0 + i < c.W) ? __ldg(c.words + (size_t)mr * c.W + t.w0 + i) : 0u;
        }
    }
}

__device__ __forceinline__ void store_words(const Ctx& c, int s, const u32* wr, u32* ws) {
    const ST t = st_info(c, s);
    const int tid = threadIdx.x;
    if (t.vec) {
        constexpr int V = BITS / 2;
        #pragma unroll
        for (int u = 0; u < NWV / 4; ++u) {
            const int idx = tid + NT * u, row = idx / V, v = idx - row * V;
            if (row < BN) {
                #pragma unroll
                for (int q = 0; q < 4; ++q) ws[row * WS + 4 * v + q] = wr[4 * u + q];
            }
        }
    } else {
        #pragma unroll
        for (int u = 0; u < NWS; ++u) {
            const int idx = tid + NT * u, row = idx / t.nw, i = idx - row * t.nw;
            if (row < BN) ws[row * WS + i] = wr[u];
        }
    }
}

// X tile of k-tile kt: BM rows x 32 contiguous columns, 4 x 16 bytes per row
__device__ __forceinline__ void load_x(const Ctx& c, int kt, int m0, uint4* xa) {
    const KT k = kt_info(c, kt);
    const int col = k.p * 256 + 32 * k.g;
    #pragma unroll
    for (int u = 0; u < XU; ++u) {
        const int idx = threadIdx.x + NT * u, row = idx >> 2, part = idx & 3, t = m0 + row;
        xa[u] = (row < BM && t < c.T) ? __ldg((const uint4*)(c.x + (size_t)t * c.ldx + col + part * 8)) : make_uint4(0, 0, 0, 0);
    }
}

// decode k-tile kt from the staged words into the B tile; store the X registers into the A tile
__device__ __forceinline__ void store_tile(const Ctx& c, int kt, const uint4* xa, const u32* ws,
                                           const uint2* tb, __half* st) {
    const int tid = threadIdx.x;
    __half* As = st;
    __half* Bs = st + BM * LDS;
    #pragma unroll
    for (int u = 0; u < XU; ++u) {
        const int idx = tid + NT * u, row = idx >> 2, part = idx & 3;
        if (row < BM) *(uint4*)(As + row * LDS + part * 8) = xa[u];
    }
    const KT k = kt_info(c, kt);
    const ST t = st_info(c, k.s);
    const int n = tid % BN, s = tid / BN;             // lanes 4g + 2s, 4g + 2s + 1 of row n (tb: its book)
    const u32* wrow = ws + n * WS;
    uint2 v[4];
    #pragma unroll
    for (int qq = 0; qq < 2; ++qq) {
        const int bit = t.off + ((2 * s + qq) * t.qs + 2 * k.j) * BITS, wi = bit >> 5, sh = bit & 31;
        const u32 lo = wrow[wi], hi = wrow[wi + 1];
        const u32 two = __funnelshift_r(lo, hi, sh);  // 2 * BITS <= 24 bits: tags e = 0, 1
        v[2 * qq] = tb[two & ((1u << BITS) - 1u)];
        v[2 * qq + 1] = tb[(two >> BITS) & ((1u << BITS) - 1u)];
    }
    *(uint4*)(Bs + n * LDS + s * 16) = make_uint4(v[0].x, v[0].y, v[1].x, v[1].y);
    *(uint4*)(Bs + n * LDS + s * 16 + 8) = make_uint4(v[2].x, v[2].y, v[3].x, v[3].y);
}

__device__ __forceinline__ void ldsm4(u32* r, const __half* p) {
    const u32 a = (u32)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

// D += A (16x16, a[4]) * B (16x8, b[2]), fp32 accumulate
__device__ __forceinline__ void mma16816(float* d, const u32* a, const u32* b) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[2]), "r"(a[3]), "r"(b[1]));
#endif
}

extern "C" __global__ void __launch_bounds__(NT) runs_gemm(
        const u32* __restrict__ words, const uint2* __restrict__ table, const __half* __restrict__ row_scale,
        const int* __restrict__ book_off, const __half* __restrict__ x, __half* __restrict__ y,
        int T, int R, int W, int nfull, int jt, int ldx, int ldy, int table_n, int S, float* __restrict__ part,
        const int* __restrict__ rmap) {
    extern __shared__ __align__(16) unsigned char smem[];
    __half* st = (__half*)smem;                       // 2 tile stages
    u32* ws = (u32*)(st + 2 * STAGE);                 // staged words [BN][WS]
    const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
    Ctx c{words, book_off, x, T, R, W, nfull, jt, ldx, rmap};
#if GROUPED
    table += book_off[rmap[n0]];                      // the block's book (table_n = its K entries)
#endif
#if TSMEM
    uint2* stab = (uint2*)(ws + BN * WS + (BN * WS & 1));
    for (int i = threadIdx.x; i < table_n; i += NT) stab[i] = __ldg(table + i);
    const uint2* tab = stab;
#else
    const uint2* tab = table;
#endif
#if GROUPED
    const uint2* tb = tab;
#else
    const uint2* tb = tab + book_off[min(n0 + (int)threadIdx.x % BN, R - 1)];
#endif
    // split-K over super-tiles: slice blockIdx.z of S covers super-tiles [s_lo, s_hi)
    const int nsup = nfull * 8 + (jt > 0 ? 8 : 0);
    const int s_lo = (int)(((long long)nsup * blockIdx.z) / S), s_hi = (int)(((long long)nsup * (blockIdx.z + 1)) / S);
    const int k_lo = st_start(c, s_lo), nk = st_start(c, s_hi);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wm = warp / WN, wn = warp % WN;
    float acc[MI][NJ][4];
    #pragma unroll
    for (int i = 0; i < MI; ++i)
        #pragma unroll
        for (int j = 0; j < NJ; ++j)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
    uint4 xa[XU];
    u32 wr[NWR];
    // prologue: words of super-tile 0, tile 0, then the prefetch of super-tile 1's words
    if (s_lo < s_hi) {
        load_words(c, s_lo, n0, wr);
        load_x(c, k_lo, m0, xa);
        store_words(c, s_lo, wr, ws);
        __syncthreads();
        store_tile(c, k_lo, xa, ws, tb, st + (k_lo & 1) * STAGE);
        if (s_lo + 1 < s_hi) load_words(c, s_lo + 1, n0, wr);
    }
    __syncthreads();
    const int a_off = (wm * TM + (lane & 15)) * LDS + 8 * (lane >> 4);
    const int b_off = BM * LDS + (wn * TN + (lane & 7) + 8 * (lane >> 4)) * LDS + 8 * ((lane >> 3) & 1);
    for (int kt = k_lo; kt < nk; ++kt) {
        const __half* cur = st + (kt & 1) * STAGE;
        const bool more = kt + 1 < nk;
        if (more) load_x(c, kt + 1, m0, xa);
        #pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            u32 a[MI][4], b[NJ][2];
            #pragma unroll
            for (int i = 0; i < MI; ++i) ldsm4(a[i], cur + a_off + i * 16 * LDS + ks);
            #pragma unroll
            for (int j = 0; j < NJ / 2; ++j) {        // n16 groups -> pairs of n8 tiles
                u32 r[4];
                ldsm4(r, cur + b_off + j * 16 * LDS + ks);
                b[2 * j][0] = r[0]; b[2 * j][1] = r[1]; b[2 * j + 1][0] = r[2]; b[2 * j + 1][1] = r[3];
            }
            #pragma unroll
            for (int i = 0; i < MI; ++i)
                #pragma unroll
                for (int j = 0; j < NJ; ++j) mma16816(acc[i][j], a[i], b[j]);
        }
        if (more) {
            const int sn = kt_info(c, kt + 1).s;
            if (sn != kt_info(c, kt).s) {             // next tile starts a new super-tile: swap in its words
                __syncthreads();                      // everyone done decoding from the old words
                store_words(c, sn, wr, ws);
                __syncthreads();
                if (sn + 1 < s_hi) load_words(c, sn + 1, n0, wr);
            }
            store_tile(c, kt + 1, xa, ws, tb, st + ((kt + 1) & 1) * STAGE);
        }
        __syncthreads();
    }
    // epilogue: c0, c1 at (row g, cols 2q, 2q+1), c2, c3 at row g + 8; times the row scale
    const int gq = lane >> 2, q2 = (lane & 3) * 2;
    #pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const int n = n0 + wn * TN + j * 8 + q2;
#if GROUPED
        // rows of slots n, n + 1 are not adjacent: scalar stores to each (padding slots: -1)
        const int ra = rmap[n], rb = rmap[n + 1];
        const float sa = ra >= 0 ? __half2float(row_scale[ra]) : 0.f, sb = rb >= 0 ? __half2float(row_scale[rb]) : 0.f;
        #pragma unroll
        for (int i = 0; i < MI; ++i) {
            #pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int t = m0 + wm * TM + i * 16 + gq + 8 * hh;
                if (t >= T) continue;
                const float va = acc[i][j][2 * hh] * sa, vb = acc[i][j][2 * hh + 1] * sb;
                if (S > 1) {
                    float* pp = part + ((size_t)blockIdx.z * T + t) * R;
                    if (ra >= 0) pp[ra] = va;
                    if (rb >= 0) pp[rb] = vb;
                } else {
                    __half* yp = y + (size_t)t * ldy;
                    if (ra >= 0) yp[ra] = __float2half(va);
                    if (rb >= 0) yp[rb] = __float2half(vb);
                }
            }
        }
        continue;
#endif
        if (n >= R) continue;
        const float s0 = __half2float(row_scale[n]), s1 = n + 1 < R ? __half2float(row_scale[n + 1]) : 0.f;
        #pragma unroll
        for (int i = 0; i < MI; ++i) {
            #pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int t = m0 + wm * TM + i * 16 + gq + 8 * hh;
                if (t >= T) continue;
                if (S > 1) {                          // fp32 partial of this slice; split_reduce sums them
                    float* pp = part + ((size_t)blockIdx.z * T + t) * R + n;
                    pp[0] = acc[i][j][2 * hh] * s0;
                    if (n + 1 < R) pp[1] = acc[i][j][2 * hh + 1] * s1;
                    continue;
                }
                __half* yp = y + (size_t)t * ldy + n;
                if (n + 1 < R && ((size_t)yp & 3) == 0) {
                    *(__half2*)yp = __floats2half2_rn(acc[i][j][2 * hh] * s0, acc[i][j][2 * hh + 1] * s1);
                } else {
                    yp[0] = __float2half(acc[i][j][2 * hh] * s0);
                    if (n + 1 < R) yp[1] = __float2half(acc[i][j][2 * hh + 1] * s1);
                }
            }
        }
    }
}

// y[t, n] (fp16, row stride ldy) = sum over the S slices of part[z, t, n], in slice order
extern "C" __global__ void split_reduce(const float* __restrict__ part, __half* __restrict__ y, int T, int R, int S, int ldy) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, n = (size_t)T * R;
    if (i >= n) return;
    float a = 0.f;
    for (int z = 0; z < S; ++z) a += part[z * n + i];
    const size_t t = i / R, r = i - t * R;
    y[t * ldy + r] = __float2half(a);
}
"""

_fns: dict = {}


def _smem(bits: int, bm: int = 128) -> int:
    """Tiles (two stages of A and B) + staged words, bytes (8-byte aligned)."""
    return 2 * (bm + 128) * 40 * 2 + ((128 * (2 * bits + 1) + 1) // 2) * 8


def _smem_optin() -> int:
    import cupy as cp
    dev = torch.cuda.current_device()
    return int(cp.cuda.runtime.deviceGetAttribute(cp.cuda.runtime.cudaDevAttrMaxSharedMemoryPerBlockOptin, dev))


def _mod(bits: int, tsmem: bool, bm: int = 128, grouped: bool = False):
    import cupy as cp
    key = (bits, tsmem, bm, grouped)
    if key not in _fns:
        src = f"#define BITS {bits}\n#define TSMEM {int(tsmem)}\n#define BM {bm}\n#define GROUPED {int(grouped)}\n" + _SRC
        _fns[key] = cp.RawModule(code=src, options=("-std=c++17",))
    return _fns[key]


def _fn(bits: int, tsmem: bool):
    return _mod(bits, tsmem).get_function("runs_gemm")


_parts: dict = {}


def _part(n: int, dev) -> torch.Tensor:
    """fp32 split-K partials, one buffer per device (grown as needed; one stream)."""
    b = _parts.get(str(dev))
    if b is None or b.numel() < n:
        b = _parts[str(dev)] = torch.empty(n, dtype=torch.float32, device=dev)
    return b


BMS = (16, 32, 64, 128)


def auto_bm(T: int) -> int:
    """Smallest token tile covering T (up to 128; larger T: several 128-token tiles)."""
    return next((b for b in BMS if T <= b), 128)


def row_map(rl) -> tuple[torch.Tensor, int]:
    """(rmap int32 [ntiles * 128], ntiles): rows sorted by codebook, each book's rows padded to whole
    128-row tiles with -1, so every block of the GROUPED kernel reads a single book (cached on rl)."""
    cached = getattr(rl, "_gemm_rmap", None)
    if cached is not None:
        return cached
    K = 1 << rl.bits
    book = (rl.book_off.view(-1).long() // K).cpu()
    parts = []
    for b in torch.unique(book).tolist():
        rows = torch.nonzero(book == b).flatten().to(torch.int32)
        pad = (-rows.numel()) % 128
        parts.append(torch.cat([rows, torch.full((pad,), -1, dtype=torch.int32)]))
    rmap = torch.cat(parts).to(rl.words.device).contiguous()
    rl._gemm_rmap = (rmap, rmap.numel() // 128)
    return rl._gemm_rmap


def row_tiles(rl, grouped: bool) -> int:
    return row_map(rl)[1] if grouped else (rl.R + 127) // 128


def auto_split(rl, T: int, sms: int | None = None, max_split: int = 16, bm: int = 128, grouped: bool = False) -> int:
    """Split-K slices so the grid covers about two waves of the SMs (small T, few row tiles)."""
    from .proto_runs import geometry
    if sms is None:
        sms = torch.cuda.get_device_properties(rl.words.device).multi_processor_count
    nfull, jt = geometry(rl.nch)
    nsup = nfull * 8 + (8 if jt else 0)
    blocks = row_tiles(rl, grouped) * ((T + bm - 1) // bm)
    return max(1, min(max_split, nsup // 2, -(-2 * sms // blocks)))


def supported(rl) -> bool:
    return (not rl.has_mask and rl.nch % 64 == 0 and rl.bits % 2 == 0 and 4 <= rl.bits <= 12
            and getattr(rl, "table_raw", None) is not None)


def plan(rl, table: str = "auto", bm: int = 128):
    """(tsmem, shared memory bytes, grouped): the whole codebook goes to shared memory when it fits;
    else ("auto" / "grouped") rows are grouped by book and each block copies its one book; else
    ("global") the table is read through L1."""
    tb = rl.table_raw.numel() * 2
    book_b = (1 << rl.bits) * 8
    base = _smem(rl.bits, bm)
    optin = _smem_optin()
    fits = base + tb <= optin
    if table == "smem" and not fits:
        raise ValueError("codebook does not fit in shared memory")
    if table == "smem" or (table == "auto" and fits):
        return True, base + tb, False
    if table in ("auto", "grouped") and base + book_b <= optin and tb > book_b:
        return True, base + book_b, True
    return False, base, False


def gemm(x: torch.Tensor, rls: list, out: torch.Tensor | None = None, table: str = "auto",
         split: int | str = "auto", bm: int | str = "auto") -> torch.Tensor:
    """x fp16 [T, C] (row stride a multiple of 8) -> fp16 [T, sum R] = x @ [W_1; W_2; ...]^T for
    RunsLinears without column orders; out: optional [T, >= sum R] buffer. split: split-K slices
    per matrix ("auto": auto_split), summed in slice order by split_reduce (fp32 partials).
    bm: tokens per block (16 / 32 / 64 / 128; "auto": auto_bm)."""
    import cupy as cp
    from .prefill_kernels import stream
    from .proto_runs import geometry
    T, C = x.shape
    assert x.dtype == torch.float16 and x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
    Rt = sum(rl.R for rl in rls)
    y = out if out is not None else torch.empty(T, Rt, dtype=torch.float16, device=x.device)
    assert y.stride(1) == 1
    off = 0
    for rl in rls:
        assert supported(rl) and rl.nch * 4 == C
        assert rl.words.data_ptr() % 16 == 0 and rl.words.shape[1] % 4 == 0
        nfull, jt = geometry(rl.nch)
        BMv = auto_bm(T) if bm == "auto" else int(bm)
        assert BMv in BMS, BMv
        tsmem, shm, grouped = plan(rl, table, BMv)
        mod = _mod(rl.bits, tsmem, BMv, grouped)
        rmap = row_map(rl)[0] if grouped else None
        fn = mod.get_function("runs_gemm")
        if shm > 48 * 1024:
            fn.max_dynamic_shared_size_bytes = shm
        S = auto_split(rl, T, bm=BMv, grouped=grouped) if split == "auto" else int(split)
        part = _part(S * T * rl.R, x.device) if S > 1 else y
        grid = (row_tiles(rl, grouped), (T + BMv - 1) // BMv, S)
        fn(grid, (256,),
           (cp.uint64(rl.words.data_ptr()), cp.uint64(rl.table_raw.data_ptr()), cp.uint64(rl.row_scale.data_ptr()),
            cp.uint64(rl.book_off.data_ptr()), cp.uint64(x.data_ptr()), cp.uint64(y.data_ptr() + off * 2),
            cp.int32(T), cp.int32(rl.R), cp.int32(rl.words.shape[1]), cp.int32(nfull), cp.int32(jt),
            cp.int32(x.stride(0)), cp.int32(y.stride(0)), cp.int32(1 << rl.bits if grouped else rl.table_raw.shape[0]),
            cp.int32(S), cp.uint64(part.data_ptr()), cp.uint64(rmap.data_ptr() if grouped else 0)),
           shared_mem=shm, stream=stream())
        if S > 1:
            n = T * rl.R
            mod.get_function("split_reduce")(((n + 255) // 256,), (256,),
                (cp.uint64(part.data_ptr()), cp.uint64(y.data_ptr() + off * 2), cp.int32(T), cp.int32(rl.R),
                 cp.int32(S), cp.int32(y.stride(0))), stream=stream())
        off += rl.R
    return y
