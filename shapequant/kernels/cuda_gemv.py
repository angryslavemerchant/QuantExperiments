"""Batch-1 GEMV for the packed stack format, CUDA C++ compiled at runtime (NVRTC
via CuPy, so no host C++ compiler is needed), launched on PyTorch's stream.

Layout facts the kernel relies on (d = 4, rows with nch % 32 == 0):
- 32 consecutive b-bit tags occupy exactly b 32-bit words, word-aligned, so a
  lane owning one such group loads b words and unpacks all 32 tags in
  registers with compile-time shifts (BITS is a compile-time constant);
- a codebook entry is 4 fp16 = 8 bytes (one uint2 load), from shared memory
  when the table fits (SMEM=1), else from global memory through L1/L2;
- the input chunk for tag j of a row with column order m is xp[m, j] (8 bytes),
  xp being the M permuted copies of x built once per call.

Work split: one warp per row, lanes stride over the row's 32-tag groups;
persistent blocks loop over rows so a shared-memory table is loaded once per
block. Products are formed in half2 (two FMAs per chunk) and accumulated in
fp32; the row scale is applied after the warp reduction.
"""

from __future__ import annotations

import math

import torch

_SRC = r"""
#include <cuda_fp16.h>
typedef unsigned int u32;

extern "C" __global__ void __launch_bounds__(THREADS)
gemv_packed(const u32* __restrict__ words, const uint2* __restrict__ table,
            const __half* __restrict__ row_scale, const int* __restrict__ book_off,
            const int* __restrict__ row_mask, const uint2* __restrict__ xp,
            __half* __restrict__ y, int R, int NG, int table_n)
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
    const int warps_per_block = blockDim.x >> 5;
    const int nch = NG * 32;
    const int W = NG * BITS;
    for (int row = blockIdx.x * warps_per_block + (threadIdx.x >> 5); row < R;
         row += gridDim.x * warps_per_block) {
        const uint2* tb = tab + book_off[row];
#if HAS_MASK
        const uint2* xr = xp + (size_t)row_mask[row] * nch;
#else
        const uint2* xr = xp;
#endif
        const u32* wr = words + (size_t)row * W;
        // warp-cooperative: one 32-tag group per step, lane l owns tag l.
        const int bit = lane * BITS, wi = bit >> 5, sh = bit & 31;
        const int wi1 = (wi + 1) & 31;
        float acc = 0.f;
        int g = 0;
        // software pipeline: batch b+1's words and inputs are in flight while batch b computes
        u32 word[UNROLL];
        uint2 xv[UNROLL];
        if (UNROLL <= NG) {
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                word[u] = lane < BITS ? __ldg(wr + u * BITS + lane) : 0u;
                xv[u] = __ldg(xr + u * 32 + lane);
            }
        }
        for (; g + UNROLL <= NG; g += UNROLL) {
            u32 nword[UNROLL];
            uint2 nxv[UNROLL];
            const bool more = g + 2 * UNROLL <= NG;
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                nword[u] = (more && lane < BITS) ? __ldg(wr + (g + UNROLL + u) * BITS + lane) : 0u;
                nxv[u] = more ? __ldg(xr + (g + UNROLL + u) * 32 + lane) : make_uint2(0u, 0u);
            }
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const u32 lo = __shfl_sync(0xffffffffu, word[u], wi);
                const u32 hi = __shfl_sync(0xffffffffu, word[u], wi1);
                const u32 v = __funnelshift_r(lo, hi, sh);
                const uint2 t = tb[v & ((1u << BITS) - 1u)];
                __half2 p = __hmul2(*(const __half2*)&t.x, *(const __half2*)&xv[u].x);
                p = __hfma2(*(const __half2*)&t.y, *(const __half2*)&xv[u].y, p);
                const float2 f = __half22float2(p);
                acc += f.x + f.y;
            }
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) { word[u] = nword[u]; xv[u] = nxv[u]; }
        }
        for (; g < NG; ++g) {
            const u32 word = lane < BITS ? __ldg(wr + g * BITS + lane) : 0u;
            const uint2 xg = __ldg(xr + g * 32 + lane);
            const u32 lo = __shfl_sync(0xffffffffu, word, wi);
            const u32 hi = __shfl_sync(0xffffffffu, word, wi1);
            const u32 v = __funnelshift_r(lo, hi, sh);
            const uint2 t = tb[v & ((1u << BITS) - 1u)];
            __half2 p = __hmul2(*(const __half2*)&t.x, *(const __half2*)&xg.x);
            p = __hfma2(*(const __half2*)&t.y, *(const __half2*)&xg.y, p);
            const float2 f = __half22float2(p);
            acc += f.x + f.y;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[row] = __float2half(acc * __half2float(row_scale[row]));
    }
}
"""

_modules: dict = {}
SMEM_LIMIT = 48 * 1024  # default dynamic shared memory without opt-in


def _kernel(bits: int, smem: bool, has_mask: bool, unroll: int = 4, threads: int = 256):
    import cupy as cp
    key = (bits, smem, has_mask, unroll, threads)
    if key not in _modules:
        major, minor = torch.cuda.get_device_capability()
        src = (f"#define BITS {bits}\n#define SMEM {int(smem)}\n#define HAS_MASK {int(has_mask)}\n"
               f"#define UNROLL {unroll}\n#define THREADS {threads}\n") + _SRC
        mod = cp.RawModule(code=src, options=(f"-arch=sm_{major}{minor}", "-std=c++17", "--use_fast_math"))
        _modules[key] = mod.get_function("gemv_packed")
    return _modules[key]


def supported(lin) -> bool:
    return lin.d == 4 and lin.nch % 32 == 0 and lin.W == lin.nch * lin.bits // 32


def gemv(lin, x: torch.Tensor, out: torch.Tensor | None = None, blocks_per_sm: int = 4, unroll: int = 4,
         threads: int = 256) -> torch.Tensor:
    """y = W x for a shapequant.kernels.gemv.PackedLinear `lin` (x fp16 [C])."""
    import cupy as cp
    if not supported(lin):
        raise ValueError("layout not supported by the CUDA GEMV")
    if x.dtype != torch.float16:
        raise ValueError("x must be fp16")
    y = out if out is not None else torch.empty(lin.R, dtype=torch.float16, device=x.device)
    xp = (x[lin.perms_l] if lin.has_mask else x[None]).contiguous()
    table_bytes = lin.table.numel() * 2
    smem = table_bytes <= SMEM_LIMIT
    fn = _kernel(lin.bits, smem, lin.has_mask, unroll, threads)
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    warps = threads // 32
    grid = min(sms * blocks_per_sm, math.ceil(lin.R / warps))
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        fn((grid,), (threads,),
           (cp.uint64(lin.words.data_ptr()), cp.uint64(lin.table.data_ptr()),
            cp.uint64(lin.row_scale.data_ptr()), cp.uint64(lin.book_off.data_ptr()),
            cp.uint64(lin.row_mask.data_ptr()), cp.uint64(xp.data_ptr()), cp.uint64(y.data_ptr()),
            cp.int32(lin.R), cp.int32(lin.nch // 32), cp.int32(lin.table.shape[0])),
           shared_mem=table_bytes if smem else 0)
    return y
