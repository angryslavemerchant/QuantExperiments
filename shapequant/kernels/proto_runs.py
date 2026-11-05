"""Prototype: lane-interleaved GEMV for the packed stack format (d = 4).

Profiling cuda_gemv.py on the 2060 showed it stalls on the MIO queue: per
4-weight chunk a warp issues 2 shuffles (tag fetch), a tag word load, an
8-byte x load and the table read. A first attempt with lane-owned contiguous
runs removed the shuffles but made every x load touch 32 cache lines (L1-bound).

This layout keeps x coalesced and drops the shuffles:
- lane L owns chunk pairs p = L + 32*j (chunks 2p, 2p+1), so for each j the
  warp reads x as one contiguous 512-byte block (16-byte loads, 2 chunks each);
- each lane's tags are stored contiguously ("lane-major" storage, a pure
  reordering done at pack time): batches of 8 pairs (16 tags = BITS/2 words,
  even BITS), then a tail of JT pairs padded to whole words per lane;
- tags are unpacked in registers with compile-time shifts.

Bank pattern: the 16 lanes of a half-warp look up chunk 2p+e for 16
consecutive pairs p together, so the conflict-free groups are
{2p+e : p in 16 consecutive pairs} for each e.
"""

from __future__ import annotations

import math

import torch

from .pdl import PDL_DEFS

_SRC = r"""
#include <cuda_fp16.h>
typedef unsigned int u32;
#define NWB (BITS / 2)                       // words per lane per full batch (16 tags)
#define NWT ((JT * 2 * BITS + 31) / 32)      // aligned words per lane in the tail (after realignment)
#define WROW (NFULL * 32 * NWB + JT * 2 * BITS)   // tail: one unpadded bitstream, 32 lanes x JT*2*BITS bits

// one batch of NT tags per lane for RU rows sharing the same x (same column order).
// TAIL: the lanes' tags form one contiguous bitstream (no per-lane padding); each lane
// loads NWD+1 words and realigns them with a funnel shift by its bit offset.
template <int NT, int NWD, bool TAIL>
__device__ __forceinline__ void batch(const u32* const* __restrict__ wrow, int off, const uint4* __restrict__ x0,
                                      const uint2* const* __restrict__ tbr, int lane, float* acc,
                                      const u32 (*wpre)[NWB] = nullptr)
{
    const u32 MASK = (1u << BITS) - 1u;
    uint4 xv[NT / 2];
    #pragma unroll
    for (int i = 0; i < NT / 2; ++i) xv[i] = __ldg(x0 + i * 32 + lane);
    // full batches: all rows' words are loaded up front (memory-level parallelism);
    // tail: rows are loaded and decoded one at a time (the realignment needs extra registers)
    constexpr int RW = TAIL ? 1 : RU;
    u32 w[RW][NWD + 1];
    if constexpr (!TAIL) {
        #pragma unroll
        for (int r = 0; r < RU; ++r) {
            #pragma unroll
            for (int i = 0; i < NWD; ++i)
                w[r][i] = (PIPE && NFULL > 0) ? wpre[r][i] : (wrow[r] ? __ldg(wrow[r] + off + lane * NWD + i) : 0u);
            w[r][NWD] = 0u;
        }
    }
    #pragma unroll
    for (int r = 0; r < RU; ++r) {
        const int ri = TAIL ? 0 : r;
        if constexpr (TAIL) {
            const int start = lane * NT * BITS, w0 = start >> 5, sh0 = start & 31;
            u32 raw[NWD + 1];
            #pragma unroll
            for (int i = 0; i <= NWD; ++i)
                raw[i] = (wrow[r] && w0 + i < NT * BITS) ? __ldg(wrow[r] + off + w0 + i) : 0u;
            #pragma unroll
            for (int i = 0; i < NWD; ++i) w[0][i] = __funnelshift_r(raw[i], raw[i + 1], sh0);
            w[0][NWD] = 0u;
        }
        #pragma unroll
        for (int q = 0; q < (NT + 3) / 4; ++q) {
            __half2 a = __float2half2_rn(0.f);
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const int j = q * 4 + jj;
                if (j >= NT) break;
                const int bit = j * BITS, wi = bit >> 5, sh = bit & 31;
                const u32 v = ((sh + BITS <= 32) ? (w[ri][wi] >> sh) : __funnelshift_r(w[ri][wi], w[ri][wi + 1], sh)) & MASK;
                const uint2 t = tbr[r][v];
                const uint4 X = xv[j >> 1];
                const u32 x0v = (j & 1) ? X.z : X.x, x1v = (j & 1) ? X.w : X.y;
                a = __hfma2(*(const __half2*)&t.x, *(const __half2*)&x0v, a);
                a = __hfma2(*(const __half2*)&t.y, *(const __half2*)&x1v, a);
            }
            const float2 f = __half22float2(a);
            acc[r] += f.x + f.y;
        }
    }
}

extern "C" __global__ void __launch_bounds__(THREADS)
gemv_runs(const u32* __restrict__ words, const uint2* __restrict__ table,
          const __half* __restrict__ row_scale, const int* __restrict__ book_off,
          const int* __restrict__ row_mask, const uint4* __restrict__ xp,
          __half* __restrict__ y, const int* __restrict__ tiles, int ntiles, int table_n,
          const float* __restrict__ yscale, const int4* __restrict__ blk,
          unsigned int* ebar, const float* h_in, float* h_out, const __half* w, const __half* src,
          const unsigned short* eperms, int EM, int EC, float eps, __half* eout)
{
    PDL_TRIGGER();      // dependents may launch once every block is running (see pdl.py)
    const float ys = yscale[0];
    extern __shared__ uint2 stab[];
    const int lane = threadIdx.x & 31;
    const int wpb = blockDim.x >> 5;
#if BOOKS
    // book-major: this block serves one mixture book (tiles sorted by book) and copies only that
    // book. blk: x = first table row of the book, y/z = the book's tile range, w = rank | nblk << 16
    const int4 bk = blk[blockIdx.x];
    table += bk.x;
    int ti = bk.y + (bk.w & 0xffff) * wpb + (threadIdx.x >> 5);
    const int tstride = (bk.w >> 16) * wpb;
    ntiles = bk.z;
#else
    int ti = blockIdx.x * wpb + (threadIdx.x >> 5);
    const int tstride = gridDim.x * wpb;
#endif
    int rows[RU];
    const u32* wrow[RU];
    auto setup = [&](int t, int* rr, const u32** wr) {
        #pragma unroll
        for (int r = 0; r < RU; ++r) {
            rr[r] = t < ntiles ? tiles[t * RU + r] : -1;
            wr[r] = rr[r] >= 0 ? words + (size_t)rr[r] * WROW : nullptr;
        }
    };
    setup(ti, rows, wrow);
#if PIPE && NFULL > 0
    // software pipeline: the next full batch's words (and the first tile's, before the table
    // copy) are loaded while the current batch is decoded
    u32 wn[RU][NWB];
    auto load_words = [&](const u32* const* wr, int off) {
        #pragma unroll
        for (int r = 0; r < RU; ++r) {
            #pragma unroll
            for (int i = 0; i < NWB; ++i) wn[r][i] = wr[r] ? __ldg(wr[r] + off + lane * NWB + i) : 0u;
        }
    };
    load_words(wrow, 0);
#endif
#if SMEM
    {   // 16-byte copies, unrolled so each thread keeps several loads in flight
        const uint4* src = (const uint4*)table;
        uint4* dst = (uint4*)stab;
        const int n4 = table_n >> 1;
        int i = threadIdx.x;
        for (; i + 3 * blockDim.x < n4; i += 4 * blockDim.x) {
            const uint4 a = __ldg(src + i), b = __ldg(src + i + blockDim.x),
                        c = __ldg(src + i + 2 * blockDim.x), d = __ldg(src + i + 3 * blockDim.x);
            dst[i] = a; dst[i + blockDim.x] = b; dst[i + 2 * blockDim.x] = c; dst[i + 3 * blockDim.x] = d;
        }
        for (; i < n4; i += blockDim.x) dst[i] = __ldg(src + i);
        if ((table_n & 1) && threadIdx.x == 0) stab[table_n - 1] = table[table_n - 1];
    }
    __syncthreads();
    const uint2* tab = stab;
#else
    const uint2* tab = table;
#endif
    PDL_WAIT();         // x / y belong to the neighbouring kernels: everything above is weights only
    constexpr int NPAIR = (NFULL * 8 + JT) * 32;   // chunk pairs per row
    for (; ti < ntiles; ti += tstride) {
        const uint2* tbr[RU];
        #pragma unroll
        for (int r = 0; r < RU; ++r) tbr[r] = tab + ((BOOKS || rows[r] < 0) ? 0 : book_off[rows[r]]);
        int rows_n[RU];
        const u32* wrow_n[RU];
#if HAS_MASK
        const uint4* xr = xp + (size_t)row_mask[rows[0]] * NPAIR;
#else
        const uint4* xr = xp;
#endif
        float acc[RU];
        #pragma unroll
        for (int r = 0; r < RU; ++r) acc[r] = 0.f;
        #pragma unroll 1
        for (int b = 0; b < NFULL; ++b) {
#if PIPE && NFULL > 0
            u32 wc[RU][NWB];
            #pragma unroll
            for (int r = 0; r < RU; ++r) {
                #pragma unroll
                for (int i = 0; i < NWB; ++i) wc[r][i] = wn[r][i];
            }
            if (b + 1 < NFULL) load_words(wrow, (b + 1) * 32 * NWB);
            else { setup(ti + tstride, rows_n, wrow_n); load_words(wrow_n, 0); }
            batch<16, NWB, false>(wrow, b * 32 * NWB, xr + b * 8 * 32, tbr, lane, acc, wc);
#else
            batch<16, NWB, false>(wrow, b * 32 * NWB, xr + b * 8 * 32, tbr, lane, acc);
#endif
        }
#if !(PIPE && NFULL > 0)
        setup(ti + tstride, rows_n, wrow_n);
#endif
#if JT > 0
        batch<JT * 2, NWT, true>(wrow, NFULL * 32 * NWB, xr + NFULL * 8 * 32, tbr, lane, acc);
#endif
        #pragma unroll
        for (int r = 0; r < RU; ++r) {
            float s = acc[r];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            if (lane == 0 && rows[r] >= 0) y[rows[r]] = __float2half(s * ys * __half2float(row_scale[rows[r]]));
        }
        #pragma unroll
        for (int r = 0; r < RU; ++r) { rows[r] = rows_n[r]; wrow[r] = wrow_n[r]; }
    }
#if EPI
    // Epilogue run by the last NE blocks to finish, once every block has written its rows: it
    // stages the next matmul's input vector in shared memory (the table region, no longer needed)
    // and writes its M column-order copies contiguously, eout[m*EC + i] = act[eperms[m*EC + i]].
    //   EPI 1: act = rmsnorm(h_in + y) * w (residual add + norm; h_out = h_in + y)
    //   EPI 2: act = silu(g) * u with g = src[i], u = src[EC + i] (HF rounding: fp16 silu, fp16 mul)
    // Every block has finished reading x before any copy is written, so eout may be x's buffer.
    {
        __shared__ unsigned int e_ticket;
        __shared__ float e_red[32];
        __syncthreads();
        if (threadIdx.x == 0) { __threadfence(); e_ticket = atomicAdd(&ebar[0], 1u); }
        __syncthreads();
        const unsigned int nb = gridDim.x, ne = (unsigned)NE < nb ? (unsigned)NE : nb;
        if (e_ticket < nb - ne) return;
        const unsigned int part = e_ticket - (nb - ne);
        if (threadIdx.x == 0) {
            while (atomicAdd(&ebar[0], 0u) < nb) __nanosleep(32);
            __threadfence();
        }
        __syncthreads();
        unsigned short* xs = (unsigned short*)stab;
        const unsigned short* s16 = (const unsigned short*)src;
#if EPI == 1
        float ss = 0.f;
        for (int i = threadIdx.x; i < EC; i += blockDim.x) {
            const unsigned short a = __ldcg(s16 + i);
            const float v = h_in[i] + __half2float(*(const __half*)&a);
            ss += v * v;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane == 0) e_red[threadIdx.x >> 5] = ss;
        __syncthreads();
        if (threadIdx.x < 32) {
            float t = threadIdx.x < (blockDim.x >> 5) ? e_red[threadIdx.x] : 0.f;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
            if (threadIdx.x == 0) e_red[0] = t;
        }
        __syncthreads();
        const float rstd = rsqrtf(e_red[0] / EC + eps);
        for (int i = threadIdx.x; i < EC; i += blockDim.x) {
            const unsigned short a = __ldcg(s16 + i);
            const float v = h_in[i] + __half2float(*(const __half*)&a);
            if (part == 0) h_out[i] = v;
            const __half o = __float2half(v * rstd * __half2float(w[i]));
            xs[i] = *(const unsigned short*)&o;
        }
#else
        for (int i = threadIdx.x; i < EC; i += blockDim.x) {
            const unsigned short gs = __ldcg(s16 + i), us = __ldcg(s16 + EC + i);
            const float a = __half2float(*(const __half*)&gs);
            const __half o = __hmul(__float2half(a / (1.f + __expf(-a))), *(const __half*)&us);
            xs[i] = *(const unsigned short*)&o;
        }
#endif
        __syncthreads();
        unsigned short* o16 = (unsigned short*)eout;
        const int n = EM * EC;
        for (int kk = part * blockDim.x + threadIdx.x; kk < n; kk += ne * blockDim.x)
            o16[kk] = xs[eperms ? eperms[kk] : kk];
        if (threadIdx.x == 0 && atomicAdd(&ebar[1], 1u) == ne - 1) { ebar[1] = 0u; atomicExch(&ebar[0], 0u); }
    }
#endif
}
"""

_mods: dict = {}
SMEM_LIMIT = 48 * 1024


def geometry(nch: int):
    """(NFULL, JT): full 8-pair batches and tail pairs per lane."""
    assert nch % 64 == 0
    npl = nch // 64                       # pairs per lane
    return divmod(npl, 8)


def words_per_row(nch: int, bits: int) -> int:
    nfull, jt = geometry(nch)
    return nfull * 32 * (bits // 2) + jt * 2 * bits          # = nch * bits / 32: no padding


def _pack_segments(t: torch.Tensor, bits: int) -> torch.Tensor:
    """t [N, n] int64 tags -> [N, ceil(n*bits/32)] int32 words, little-endian bit stream."""
    N, n = t.shape
    nw = (n * bits + 31) // 32
    out = torch.zeros(N, nw + 1, dtype=torch.int64, device=t.device)
    for i in range(n):
        pos = i * bits
        wi, sh = pos >> 5, pos & 31
        v = t[:, i] << sh
        out[:, wi] |= v & 0xFFFFFFFF
        out[:, wi + 1] |= v >> 32
    out = out[:, :nw]
    return torch.where(out >= (1 << 31), out - (1 << 32), out).to(torch.int32)


def pack_lane_major(tags: torch.Tensor, bits: int) -> torch.Tensor:
    """tags [R, nch] (natural chunk order) -> int32 words [R, words_per_row]."""
    R, nch = tags.shape
    nfull, jt = geometry(nch)
    npl = nch // 64
    t = tags.to(torch.int64).view(R, npl, 32, 2).transpose(1, 2)        # [R, lane, j, e]
    parts = []
    if nfull:
        tf = t[:, :, :nfull * 8].reshape(R, 32, nfull, 16).transpose(1, 2)   # [R, b, lane, 16]
        parts.append(_pack_segments(tf.reshape(-1, 16), bits).view(R, -1))
    if jt:
        tt = t[:, :, nfull * 8:].reshape(R, 32 * jt * 2)                    # one lane-major stream per row
        parts.append(_pack_segments(tt, bits).view(R, -1))
    assert sum(p.shape[1] for p in parts) == words_per_row(nch, bits)
    return torch.cat(parts, 1).contiguous()


TABLE_MAX = 0.125   # |codeword| bound after scaling: 8 products of |x| <= 65504 sum to <= 65504 (half2 lane)


def scaled_table(table: torch.Tensor, limit: float):
    """Scale a codebook by a power of two so max|codeword| <= limit; the kernel multiplies
    y by the returned 1-element fp32 tensor. With the bound, the fp16 partial sums cannot
    overflow for any fp16 input (normalized codewords x outlier activations otherwise can)."""
    cmax = float(table.float().abs().max())
    k = 0 if cmax == 0 else int(torch.ceil(torch.log2(torch.tensor(cmax / limit))))
    t = (table.float() * 2.0 ** -k).half()
    return t, torch.tensor([2.0 ** k], dtype=torch.float32, device=table.device)


def book_tiles(rl, K: int):
    """Tiles grouped by (book, column order), book-major, plus each book's tile range starts."""
    if getattr(rl, "_book_tiles", None) is None:
        book = (rl.book_off // K).long()
        M = int(rl.row_mask.max()) + 1
        tiles = make_tiles(book * M + rl.row_mask.long(), rl.ru)
        tb = book.cpu()[tiles[:, 0].long().cpu()]           # each tile's book (first row is never padding)
        G = rl.table.shape[0] // K
        starts = [0]
        for c in torch.bincount(tb, minlength=G).tolist():
            starts.append(starts[-1] + c)
        rl._book_tiles, rl._blk = (tiles, starts), {}
    return rl._book_tiles


def book_blocks(rl, K: int, grid: int, wpb: int) -> torch.Tensor:
    """Per block (first table row, tile lo, tile hi, rank | nblk << 16): blocks split over books in
    proportion to their tiles (at least one per non-empty book, no more than its tiles need)."""
    _, starts = book_tiles(rl, K)
    if (grid, wpb) not in rl._blk:
        n = [starts[g + 1] - starts[g] for g in range(len(starts) - 1)]
        nt = max(1, sum(n))
        alloc = [0 if c == 0 else max(1, min(math.ceil(c / wpb), round(grid * c / nt))) for c in n]
        rows = [(g * K, starts[g], starts[g + 1], r | (a << 16)) for g, a in enumerate(alloc) for r in range(a)]
        rl._blk[(grid, wpb)] = torch.tensor(rows, dtype=torch.int32, device=rl.table.device).contiguous()
    return rl._blk[(grid, wpb)]


def make_tiles(row_mask: torch.Tensor, ru: int) -> torch.Tensor:
    """Group rows with the same column order into tiles of `ru` rows (-1 = padding)."""
    rm = row_mask.cpu()
    out = []
    for m in torch.unique(rm).tolist():
        rows = torch.nonzero(rm == m).flatten()
        pad = (-rows.numel()) % ru
        out.append(torch.cat([rows, torch.full((pad,), -1, dtype=rows.dtype)]).view(-1, ru))
    return torch.cat(out).to(torch.int32).to(row_mask.device).contiguous()


def _kernel(bits, nch, smem, has_mask, threads, ru, pipe=True, books=False, epi=0, ne=8):
    import cupy as cp
    nfull, jt = geometry(nch)
    key = (bits, nfull, jt, smem, has_mask, threads, ru, pipe, books, epi, ne)
    if key not in _mods:
        src = (f"#define BITS {bits}\n#define NFULL {nfull}\n#define JT {jt}\n#define SMEM {int(smem)}\n"
               f"#define HAS_MASK {int(has_mask)}\n#define THREADS {threads}\n#define RU {ru}\n"
               f"#define PIPE {int(pipe)}\n#define BOOKS {int(books)}\n#define EPI {epi}\n#define NE {ne}\n") + PDL_DEFS + _SRC
        _mods[key] = cp.RawModule(code=src, options=("-std=c++17", "--use_fast_math")).get_function("gemv_runs")
    return _mods[key]


class RunsLinear:
    """Lane-major repack of a PackedLinear (d = 4, single-bits)."""

    def __init__(self, lin, tags: torch.Tensor, ru: int = 1):
        assert lin.d == 4 and lin.bits % 2 == 0
        self.ru = ru
        self.tiles = make_tiles(lin.row_mask, ru)
        self.R, self.nch, self.bits = lin.R, lin.nch, lin.bits
        self.words = pack_lane_major(tags.to(lin.table.device), lin.bits)
        self.table, self.row_scale, self.book_off = lin.table, lin.row_scale, lin.book_off
        self.row_mask, self.perms_l, self.has_mask = lin.row_mask, lin.perms_l, lin.has_mask
        self.table, self.yscale = scaled_table(lin.table, TABLE_MAX)

    def clone(self):
        new = object.__new__(RunsLinear)
        new.__dict__.update(self.__dict__)
        new.words = self.words.clone()
        return new

    def stored_bytes(self) -> int:
        return self.words.numel() * 4


def prep_x(rl: RunsLinear, x: torch.Tensor) -> torch.Tensor:
    """The M permuted copies of x; shareable by every matrix reading the same x with the same perms."""
    return (x[rl.perms_l] if rl.has_mask else x[None]).contiguous()


def gemv(rl: RunsLinear, x: torch.Tensor, out: torch.Tensor | None = None, blocks_per_sm: int = 4,
         threads: int = 256, xp: torch.Tensor | None = None, table: str = "auto", pipe: bool = False,
         pdl: bool = False, epi: dict | None = None) -> torch.Tensor:
    """table: "books" (and "auto" when the G-book table is over the limit but one book fits):
    book-major blocks that each copy one book. "smem" copies the codebook into shared memory per block, "global" reads it through
    L1 (no per-block copy: cheaper for small matrices, where the copies dominate), "auto" = smem
    when it fits. threads=0: one block per SM sized to cover all tiles in one round (small
    matrices; capped at 1024 threads). pipe: load the next
    batch's weight words while decoding the current one (skipped where it would spill).
    pdl: programmatic dependent launch (sm_90+, see pdl.py): the codebook copy overlaps the
    previous kernel.
    epi: the next matmul's input written by the last `ne` blocks (see the kernel's EPI section):
    dict(mode=1 (add + rmsnorm) | 2 (silu * mul), ne, bar (int32[2], zeros), h_in, h_out, w (mode 1),
    src (mode 1: this launch's y; mode 2: the [2C] gate|up output), perms (uint16 [M, C] or None),
    C, eps, out)."""
    import cupy as cp
    y = out if out is not None else torch.empty(rl.R, dtype=torch.float16, device=x.device)
    if xp is None:
        xp = prep_x(rl, x)
    yscale = rl.yscale
    K = 1 << rl.bits
    table_bytes = rl.table.numel() * 2
    G = rl.table.shape[0] // K
    # book-major when the whole G-book table is over the limit but one book fits
    books = table == "books" or (table == "auto" and G > 1 and table_bytes > SMEM_LIMIT and K * 8 <= SMEM_LIMIT)
    if books:
        table_bytes = K * 8
    smem = books or (table_bytes <= SMEM_LIMIT if table == "auto" else table == "smem")
    assert not smem or table_bytes <= SMEM_LIMIT
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    tiles = book_tiles(rl, K)[0] if books else rl.tiles
    nt = tiles.shape[0]
    if threads == 0:   # auto: one block per SM, just enough warps to cover every tile in one round
        threads, blocks_per_sm = min(1024, max(64, math.ceil(nt / sms) * 32)), 1
    mode, ne = (epi["mode"], epi["ne"]) if epi else (0, 8)
    fn = _kernel(rl.bits, rl.nch, smem, rl.has_mask, threads, rl.ru, False, books, mode, ne)
    if pipe:   # the pipeline's extra registers spill under a tight per-thread cap (1024 threads, RU >= 2)
        fp = _kernel(rl.bits, rl.nch, smem, rl.has_mask, threads, rl.ru, True, books, mode, ne)
        fn = fp if fp.local_size_bytes <= fn.local_size_bytes else fn
    grid = min(sms * blocks_per_sm, math.ceil(nt / (threads // 32)))
    blk = tiles                               # unused unless book-major
    if books:
        blk = book_blocks(rl, K, grid, threads // 32)
        grid = blk.shape[0]
    args = (cp.uint64(rl.words.data_ptr()), cp.uint64(rl.table.data_ptr()),
            cp.uint64(rl.row_scale.data_ptr()), cp.uint64(rl.book_off.data_ptr()),
            cp.uint64(rl.row_mask.data_ptr()), cp.uint64(xp.data_ptr()), cp.uint64(y.data_ptr()),
            cp.uint64(tiles.data_ptr()), cp.int32(nt), cp.int32(K if books else rl.table.shape[0]),
            cp.uint64(yscale.data_ptr()), cp.uint64(blk.data_ptr()))
    shm = table_bytes if smem else 0
    if epi:
        p = lambda t: cp.uint64(0 if t is None else t.data_ptr())   # noqa: E731
        perms = epi.get("perms")
        args += (p(epi["bar"]), p(epi.get("h_in")), p(epi.get("h_out")), p(epi.get("w")), p(epi["src"]), p(perms),
                 cp.int32(1 if perms is None else perms.shape[0]), cp.int32(epi["C"]),
                 cp.float32(epi.get("eps", 0.0)), p(epi["out"]))
        shm = max(shm, epi["C"] * 2)
        assert shm <= SMEM_LIMIT
    else:
        args += (cp.uint64(0),) * 6 + (cp.int32(0), cp.int32(0), cp.float32(0.0), cp.uint64(0))
    if pdl:
        from . import pdl as pdl_mod
        pdl_mod.launch(fn, grid, threads, args, shm)
        return y
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        fn((grid,), (threads,), args, shared_mem=shm)
    return y
