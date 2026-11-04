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

_SRC = r"""
#include <cuda_fp16.h>
typedef unsigned int u32;
#define NWB (BITS / 2)                       // words per lane per full batch (16 tags)
#define NWT ((JT * 2 * BITS + 31) / 32)      // words per lane in the tail
#define WROW (NFULL * 32 * NWB + 32 * NWT)

// one batch of NT tags per lane for RU rows sharing the same x (same column order)
template <int NT, int NWD>
__device__ __forceinline__ void batch(const u32* const* __restrict__ wrow, int off, const uint4* __restrict__ x0,
                                      const uint2* const* __restrict__ tbr, int lane, float* acc)
{
    const u32 MASK = (1u << BITS) - 1u;
    uint4 xv[NT / 2];
    #pragma unroll
    for (int i = 0; i < NT / 2; ++i) xv[i] = __ldg(x0 + i * 32 + lane);
    u32 w[RU][NWD + 1];
    #pragma unroll
    for (int r = 0; r < RU; ++r) {
        #pragma unroll
        for (int i = 0; i < NWD; ++i) w[r][i] = wrow[r] ? __ldg(wrow[r] + off + lane * NWD + i) : 0u;
        w[r][NWD] = 0u;
    }
    #pragma unroll
    for (int r = 0; r < RU; ++r) {
        #pragma unroll
        for (int q = 0; q < (NT + 3) / 4; ++q) {
            __half2 a = __float2half2_rn(0.f);
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const int j = q * 4 + jj;
                if (j >= NT) break;
                const int bit = j * BITS, wi = bit >> 5, sh = bit & 31;
                const u32 v = ((sh + BITS <= 32) ? (w[r][wi] >> sh) : __funnelshift_r(w[r][wi], w[r][wi + 1], sh)) & MASK;
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
          __half* __restrict__ y, const int* __restrict__ tiles, int ntiles, int table_n)
{
    extern __shared__ uint2 stab[];
#if SMEM
    for (int i = threadIdx.x; i < table_n; i += blockDim.x) stab[i] = table[i];
    __syncthreads();
    const uint2* tab = stab;
#else
    const uint2* tab = table;
#endif
    const int lane = threadIdx.x & 31;
    const int wpb = blockDim.x >> 5;
    constexpr int NPAIR = (NFULL * 8 + JT) * 32;   // chunk pairs per row
    for (int ti = blockIdx.x * wpb + (threadIdx.x >> 5); ti < ntiles; ti += gridDim.x * wpb) {
        int rows[RU];
        const u32* wrow[RU];
        const uint2* tbr[RU];
        #pragma unroll
        for (int r = 0; r < RU; ++r) {
            rows[r] = tiles[ti * RU + r];
            wrow[r] = rows[r] >= 0 ? words + (size_t)rows[r] * WROW : nullptr;
            tbr[r] = tab + (rows[r] >= 0 ? book_off[rows[r]] : 0);
        }
#if HAS_MASK
        const uint4* xr = xp + (size_t)row_mask[rows[0]] * NPAIR;
#else
        const uint4* xr = xp;
#endif
        float acc[RU];
        #pragma unroll
        for (int r = 0; r < RU; ++r) acc[r] = 0.f;
        #pragma unroll 1
        for (int b = 0; b < NFULL; ++b)
            batch<16, NWB>(wrow, b * 32 * NWB, xr + b * 8 * 32, tbr, lane, acc);
#if JT > 0
        batch<JT * 2, NWT>(wrow, NFULL * 32 * NWB, xr + NFULL * 8 * 32, tbr, lane, acc);
#endif
        #pragma unroll
        for (int r = 0; r < RU; ++r) {
            float s = acc[r];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            if (lane == 0 && rows[r] >= 0) y[rows[r]] = __float2half(s * __half2float(row_scale[rows[r]]));
        }
    }
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
    return nfull * 32 * (bits // 2) + 32 * ((jt * 2 * bits + 31) // 32)


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
        tt = t[:, :, nfull * 8:].reshape(R * 32, jt * 2)
        parts.append(_pack_segments(tt, bits).view(R, -1))
    return torch.cat(parts, 1).contiguous()


def make_tiles(row_mask: torch.Tensor, ru: int) -> torch.Tensor:
    """Group rows with the same column order into tiles of `ru` rows (-1 = padding)."""
    rm = row_mask.cpu()
    out = []
    for m in torch.unique(rm).tolist():
        rows = torch.nonzero(rm == m).flatten()
        pad = (-rows.numel()) % ru
        out.append(torch.cat([rows, torch.full((pad,), -1, dtype=rows.dtype)]).view(-1, ru))
    return torch.cat(out).to(torch.int32).to(row_mask.device).contiguous()


def _kernel(bits, nch, smem, has_mask, threads, ru):
    import cupy as cp
    nfull, jt = geometry(nch)
    key = (bits, nfull, jt, smem, has_mask, threads, ru)
    if key not in _mods:
        src = (f"#define BITS {bits}\n#define NFULL {nfull}\n#define JT {jt}\n#define SMEM {int(smem)}\n"
               f"#define HAS_MASK {int(has_mask)}\n#define THREADS {threads}\n#define RU {ru}\n") + _SRC
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

    def clone(self):
        new = object.__new__(RunsLinear)
        new.__dict__.update(self.__dict__)
        new.words = self.words.clone()
        return new

    def stored_bytes(self) -> int:
        return self.words.numel() * 4


def gemv(rl: RunsLinear, x: torch.Tensor, out: torch.Tensor | None = None, blocks_per_sm: int = 4,
         threads: int = 256, xp: torch.Tensor | None = None) -> torch.Tensor:
    import cupy as cp
    y = out if out is not None else torch.empty(rl.R, dtype=torch.float16, device=x.device)
    if xp is None:
        xp = (x[rl.perms_l] if rl.has_mask else x[None]).contiguous()
    table_bytes = rl.table.numel() * 2
    smem = table_bytes <= SMEM_LIMIT
    fn = _kernel(rl.bits, rl.nch, smem, rl.has_mask, threads, rl.ru)
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    nt = rl.tiles.shape[0]
    grid = min(sms * blocks_per_sm, math.ceil(nt / (threads // 32)))
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        fn((grid,), (threads,),
           (cp.uint64(rl.words.data_ptr()), cp.uint64(rl.table.data_ptr()),
            cp.uint64(rl.row_scale.data_ptr()), cp.uint64(rl.book_off.data_ptr()),
            cp.uint64(rl.row_mask.data_ptr()), cp.uint64(xp.data_ptr()), cp.uint64(y.data_ptr()),
            cp.uint64(rl.tiles.data_ptr()), cp.int32(nt), cp.int32(rl.table.shape[0])),
           shared_mem=table_bytes if smem else 0)
    return y
