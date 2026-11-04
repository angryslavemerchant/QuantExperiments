"""Batch-1 GEMV v3 for the packed stack format: mask tiles (CUDA via NVRTC/CuPy).

Rows are sorted by column order at load time and cut into tiles whose rows all
share one order. A block owns one tile: it builds that order's permuted input
in shared memory once (straight through the permutation, no separate pass),
stages the codebook table in shared memory too when both fit, then its warps
stream the tile's rows (warp-cooperative: one 32-tag group per step, lane l
owns tag l; the group's BITS words arrive in one coalesced load and each lane
pulls its two words with shuffles). Results go to the rows' original positions.
"""

from __future__ import annotations

import torch

from .cuda_gemv import supported

_SRC = r"""
#include <cuda_fp16.h>
typedef unsigned int u32;

__device__ __forceinline__ float chunk_dot(uint2 t, uint2 xv) {
    __half2 p = __hmul2(*(const __half2*)&t.x, *(const __half2*)&xv.x);
    p = __hfma2(*(const __half2*)&t.y, *(const __half2*)&xv.y, p);
    const float2 f = __half22float2(p);
    return f.x + f.y;
}

extern "C" __global__ void __launch_bounds__(THREADS)
gemv_tiles(const u32* __restrict__ words, const uint2* __restrict__ table,
           const __half* __restrict__ row_scale, const int* __restrict__ book_off,
           const int* __restrict__ out_row, const int* __restrict__ tiles,
           const int* __restrict__ perms, const __half* __restrict__ x,
           __half* __restrict__ y, int NG, int C, int table_n)
{
    extern __shared__ uint2 smem[];
    uint2* xs = smem;
    const int r0 = tiles[3 * blockIdx.x], r1 = tiles[3 * blockIdx.x + 1], m = tiles[3 * blockIdx.x + 2];
    __half* xsh = (__half*)xs;
    const int* pm = perms + (size_t)m * C;
    for (int c = threadIdx.x; c < C; c += THREADS) xsh[c] = x[pm[c]];
#if SMEM_TABLE
    uint2* stab = smem + NG * 32;
    for (int i = threadIdx.x; i < table_n; i += THREADS) stab[i] = table[i];
    const uint2* tab = stab;
#else
    const uint2* tab = table;
#endif
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int bit = lane * BITS, wi = bit >> 5, sh = bit & 31, wi1 = (wi + 1) & 31;
    const int W = NG * BITS;
    for (int row = r0 + warp; row < r1; row += THREADS / 32) {
        const u32* wr = words + (size_t)row * W;
        const int boff = book_off[row];
        float acc = 0.f;
        int g = 0;
        for (; g + UNROLL <= NG; g += UNROLL) {
            u32 word[UNROLL];
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) word[u] = lane < BITS ? __ldg(wr + (g + u) * BITS + lane) : 0u;
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const u32 lo = __shfl_sync(0xffffffffu, word[u], wi);
                const u32 hi = __shfl_sync(0xffffffffu, word[u], wi1);
                const u32 v = sh ? ((lo >> sh) | (hi << (32 - sh))) : lo;
                acc += chunk_dot(tab[(int)(v & ((1u << BITS) - 1u)) + boff], xs[(g + u) * 32 + lane]);
            }
        }
        for (; g < NG; ++g) {
            const u32 word = lane < BITS ? __ldg(wr + g * BITS + lane) : 0u;
            const u32 lo = __shfl_sync(0xffffffffu, word, wi);
            const u32 hi = __shfl_sync(0xffffffffu, word, wi1);
            const u32 v = sh ? ((lo >> sh) | (hi << (32 - sh))) : lo;
            acc += chunk_dot(tab[(int)(v & ((1u << BITS) - 1u)) + boff], xs[g * 32 + lane]);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[out_row[row]] = __float2half(acc * __half2float(row_scale[row]));
    }
}
"""

_modules: dict = {}


def _kernel(bits: int, smem_table: bool, threads: int, unroll: int):
    import cupy as cp
    key = (bits, smem_table, threads, unroll)
    if key not in _modules:
        major, minor = torch.cuda.get_device_capability()
        src = (f"#define BITS {bits}\n#define SMEM_TABLE {int(smem_table)}\n#define THREADS {threads}\n"
               f"#define UNROLL {unroll}\n") + _SRC
        mod = cp.RawModule(code=src, options=(f"-arch=sm_{major}{minor}", "-std=c++17", "--use_fast_math"))
        _modules[key] = mod.get_function("gemv_tiles")
    return _modules[key]


class TiledLayout:
    """Load-time layout of a PackedLinear for gemv_tiles."""

    def __init__(self, lin, tile_rows: int = 64):
        if not supported(lin):
            raise ValueError("layout not supported by the CUDA GEMV")
        dev = lin.words.device
        rm = lin.row_mask.to(torch.int64) if lin.has_mask else torch.zeros(lin.R, dtype=torch.int64, device=dev)
        order = torch.argsort(rm, stable=True)
        self.words = lin.words[order].contiguous()
        self.row_scale = lin.row_scale[order].contiguous()
        self.book_off = lin.book_off[order].contiguous()
        self.out_row = order.to(torch.int32).contiguous()
        counts = torch.bincount(rm, minlength=lin.perms.shape[0]).tolist()
        tiles, start = [], 0
        for m, n in enumerate(counts):
            for a in range(start, start + n, tile_rows):
                tiles.append((a, min(a + tile_rows, start + n), m))
            start += n
        self.tiles = torch.tensor(tiles, dtype=torch.int32, device=dev).reshape(-1)
        self.n_tiles = len(tiles)
        self.perms = lin.perms.to(torch.int32).contiguous()
        self.lin = lin


def gemv(layout: TiledLayout, x: torch.Tensor, out: torch.Tensor | None = None, threads: int = 256,
         unroll: int = 4, smem_cap: int | None = None) -> torch.Tensor:
    """y = W x (x fp16 [C], contiguous)."""
    import cupy as cp
    lin = layout.lin
    y = out if out is not None else torch.empty(lin.R, dtype=torch.float16, device=x.device)
    x_bytes = lin.C * 2
    table_bytes = lin.table.numel() * 2
    if smem_cap is None:
        smem_cap = torch.cuda.get_device_properties(x.device).shared_memory_per_block_optin
    smem_table = x_bytes + table_bytes <= smem_cap
    shared = x_bytes + (table_bytes if smem_table else 0)
    fn = _kernel(lin.bits, smem_table, threads, unroll)
    if shared > 48 * 1024:
        fn.max_dynamic_shared_size_bytes = shared
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        fn((layout.n_tiles,), (threads,),
           (cp.uint64(layout.words.data_ptr()), cp.uint64(lin.table.data_ptr()),
            cp.uint64(layout.row_scale.data_ptr()), cp.uint64(layout.book_off.data_ptr()),
            cp.uint64(layout.out_row.data_ptr()), cp.uint64(layout.tiles.data_ptr()),
            cp.uint64(layout.perms.data_ptr()), cp.uint64(x.data_ptr()), cp.uint64(y.data_ptr()),
            cp.int32(lin.nch // 32), cp.int32(lin.C), cp.int32(lin.table.shape[0])),
           shared_mem=shared)
    return y
