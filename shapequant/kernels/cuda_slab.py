"""Batch-1 GEMV v4 for the packed stack format: slab-loaded tag words (CUDA via
NVRTC/CuPy, launched on PyTorch's stream).

v2 (cuda_gemv) was limited by memory-level parallelism, not math: each warp
loaded one 32-tag group's BITS words (40-56 bytes, 10-14 active lanes) per step,
far too few bytes in flight to keep DRAM busy. Here each warp pulls a slab of
U groups' words (U*BITS words, 1.3-1.8 KB) with 16-byte vector loads across all
32 lanes into a per-warp shared-memory buffer, then lane l reads its two words
for tag l of each group from there. Everything else is as in v2: one warp per
row, the d=4 codebook entry is one 8-byte load from a shared-memory table when
it fits, inputs come from the M permuted copies of x, half2 products, fp32 sums.

Alignment: needs W = NG*BITS divisible by 4 (16-byte row starts) and U*BITS
divisible by 4; the words buffer is padded by 8 words so slab over-reads at the
end of the last row stay in bounds.
"""

from __future__ import annotations

import torch

_SRC = r"""
#include <cuda_fp16.h>
typedef unsigned int u32;

extern "C" __global__ void __launch_bounds__(THREADS)
gemv_slab(const u32* __restrict__ words, const uint2* __restrict__ table,
          const __half* __restrict__ row_scale, const int* __restrict__ book_off,
          const int* __restrict__ row_mask, const uint2* __restrict__ xp,
          __half* __restrict__ y, int R, int NG, int table_n)
{
    extern __shared__ uint4 sm4[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#if SMEM
    uint2* stab = (uint2*)sm4;
    for (int i = threadIdx.x; i < table_n; i += THREADS) stab[i] = table[i];
    u32* wbuf = (u32*)(stab + TABLE_PAD) + warp * WBUF;
    __syncthreads();
    const uint2* tab = stab;
#else
    u32* wbuf = (u32*)sm4 + warp * WBUF;
    const uint2* tab = table;
#endif
    const int W = NG * BITS, nch = NG * 32;
    const int bit = lane * BITS, wi = bit >> 5, sh = bit & 31;
    for (int row = blockIdx.x * (THREADS / 32) + warp; row < R; row += gridDim.x * (THREADS / 32)) {
        const int boff = book_off[row];
#if HAS_MASK
        const uint2* xr = xp + (size_t)row_mask[row] * nch;
#else
        const uint2* xr = xp;
#endif
        const uint4* wr4 = (const uint4*)(words + (size_t)row * W);
        float acc = 0.f;
        for (int g0 = 0; g0 < NG; g0 += U) {
            const int ng = min(U, NG - g0);
            const int n4 = (ng * BITS + 1 + 3) >> 2;          // words through index ng*BITS
            const uint4* src = wr4 + ((g0 * BITS) >> 2);
            for (int i = lane; i < n4; i += 32) ((uint4*)wbuf)[i] = __ldg(src + i);
            __syncwarp();
            #pragma unroll 4
            for (int k = 0; k < ng; ++k) {
                const u32* wk = wbuf + k * BITS + wi;
                const u32 lo = wk[0], hi = wk[1];
                const u32 v = sh ? ((lo >> sh) | (hi << (32 - sh))) : lo;
                const uint2 t = tab[(int)(v & ((1u << BITS) - 1u)) + boff];
                const uint2 xv = __ldg(xr + (g0 + k) * 32 + lane);
                __half2 p = __hmul2(*(const __half2*)&t.x, *(const __half2*)&xv.x);
                p = __hfma2(*(const __half2*)&t.y, *(const __half2*)&xv.y, p);
                const float2 f = __half22float2(p);
                acc += f.x + f.y;
            }
            __syncwarp();
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[row] = __float2half(acc * __half2float(row_scale[row]));
    }
}
"""

_modules: dict = {}


def _wbuf_words(bits: int, U: int) -> int:
    n = U * bits + 8
    return (n + 3) // 4 * 4


def _kernel(bits: int, smem: bool, has_mask: bool, threads: int, U: int, table_pad: int):
    import cupy as cp
    key = (bits, smem, has_mask, threads, U, table_pad)
    if key not in _modules:
        defs = {"BITS": bits, "SMEM": int(smem), "HAS_MASK": int(has_mask), "THREADS": threads,
                "U": U, "WBUF": _wbuf_words(bits, U), "TABLE_PAD": table_pad}
        src = "".join(f"#define {k} {v}\n" for k, v in defs.items()) + _SRC
        mod = cp.RawModule(code=src, options=("-std=c++17", "--use_fast_math"))
        _modules[key] = mod.get_function("gemv_slab")
    return _modules[key]


def supported(lin, U: int = 32) -> bool:
    return (lin.d == 4 and lin.nch % 32 == 0 and lin.W == lin.nch * lin.bits // 32
            and lin.W % 4 == 0 and (U * lin.bits) % 4 == 0)


def _padded_words(lin) -> torch.Tensor:
    pw = getattr(lin, "_slab_words", None)
    if pw is None:
        flat = lin.words.reshape(-1)
        pw = torch.cat([flat, torch.zeros(8, dtype=flat.dtype, device=flat.device)])
        lin._slab_words = pw
    return pw


def gemv(lin, x: torch.Tensor, out: torch.Tensor | None = None, threads: int = 256, U: int = 32,
         blocks_per_sm: int = 4, smem_cap: int = 48 * 1024) -> torch.Tensor:
    """y = W x for a PackedLinear `lin`; x fp16 [C] contiguous."""
    import cupy as cp
    if not supported(lin, U):
        raise ValueError("layout not supported by the slab GEMV")
    y = out if out is not None else torch.empty(lin.R, dtype=torch.float16, device=x.device)
    xp = (x[lin.perms_l] if lin.has_mask else x[None]).contiguous()
    warps = threads // 32
    wbuf_bytes = warps * _wbuf_words(lin.bits, U) * 4
    table_n = lin.table.shape[0]
    table_pad = (table_n + 1) // 2 * 2               # keep the word buffers 16-byte aligned
    smem = table_pad * 8 + wbuf_bytes <= smem_cap
    shared = (table_pad * 8 if smem else 0) + wbuf_bytes
    fn = _kernel(lin.bits, smem, lin.has_mask, threads, U, table_pad)
    if shared > 48 * 1024:
        fn.max_dynamic_shared_size_bytes = shared
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    grid = min(sms * blocks_per_sm, (lin.R + warps - 1) // warps)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        fn((grid,), (threads,),
           (cp.uint64(_padded_words(lin).data_ptr()), cp.uint64(lin.table.data_ptr()),
            cp.uint64(lin.row_scale.data_ptr()), cp.uint64(lin.book_off.data_ptr()),
            cp.uint64(lin.row_mask.data_ptr()), cp.uint64(xp.data_ptr()), cp.uint64(y.data_ptr()),
            cp.int32(lin.R), cp.int32(lin.nch // 32), cp.int32(table_n)),
           shared_mem=shared)
    return y
