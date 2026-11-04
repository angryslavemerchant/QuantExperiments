"""Batch-1 GEMV v5 for the packed stack format: everything through L1, no shared
memory or shuffles (CUDA via NVRTC/CuPy, on PyTorch's stream).

Profiling v2 (cuda_gemv) at full occupancy showed the MIO pipe saturated:
random 8-byte codebook lookups from shared memory cost ~6 wavefronts each (bank
conflicts) and the two tag-word shuffles per tag share that pipe; separately
every row re-read its permuted input from L2 (131 MB of L2 traffic for 21.7 MB
of weights on a gate_proj). Here:
- the codebook entry is an __ldg (L1/TEX pipe, table cache-resident);
- each lane loads its two tag words directly (L1 hits, coalesced per warp);
- rows are sorted by column order at load time and each warp walks a
  contiguous run of rows, so consecutive rows share their permuted input and
  it stays in L1. Results are written to the rows' original positions.
"""

from __future__ import annotations

import torch

_SRC = r"""
#include <cuda_fp16.h>
typedef unsigned int u32;

extern "C" __global__ void __launch_bounds__(THREADS)
gemv_l1(const u32* __restrict__ words, const uint2* __restrict__ table,
        const __half* __restrict__ row_scale, const int* __restrict__ book_off,
        const int* __restrict__ row_mask, const int* __restrict__ out_row,
        const uint2* __restrict__ xp, __half* __restrict__ y, int R, int NG, int rows_per_warp)
{
    const int lane = threadIdx.x & 31;
    const int gw = blockIdx.x * (THREADS / 32) + (threadIdx.x >> 5);
    const int r0 = gw * rows_per_warp, r1 = min(R, r0 + rows_per_warp);
    const int nch = NG * 32, W = NG * BITS;
    const int bit = lane * BITS, wi = bit >> 5, sh = bit & 31;
    for (int row = r0; row < r1; ++row) {
        const uint2* tb = table + book_off[row];
#if HAS_MASK
        const uint2* xr = xp + (size_t)row_mask[row] * nch;
#else
        const uint2* xr = xp;
#endif
        const u32* wr = words + (size_t)row * W + wi;
        float acc = 0.f;
        #pragma unroll UNROLL
        for (int g = 0; g < NG; ++g) {
            const u32 lo = __ldg(wr + g * BITS), hi = __ldg(wr + g * BITS + 1);
            const u32 v = __funnelshift_r(lo, hi, sh);
            const uint2 t = __ldg(tb + (v & ((1u << BITS) - 1u)));
            const uint2 xv = __ldg(xr + g * 32 + lane);
            __half2 p = __hmul2(*(const __half2*)&t.x, *(const __half2*)&xv.x);
            p = __hfma2(*(const __half2*)&t.y, *(const __half2*)&xv.y, p);
            const float2 f = __half22float2(p);
            acc += f.x + f.y;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[out_row[row]] = __float2half(acc * __half2float(row_scale[row]));
    }
}
"""

_modules: dict = {}


def _kernel(bits: int, has_mask: bool, threads: int, unroll: int):
    import cupy as cp
    key = (bits, has_mask, threads, unroll)
    if key not in _modules:
        major, minor = torch.cuda.get_device_capability()
        defs = {"BITS": bits, "HAS_MASK": int(has_mask), "THREADS": threads, "UNROLL": unroll}
        src = "".join(f"#define {k} {v}\n" for k, v in defs.items()) + _SRC
        mod = cp.RawModule(code=src, options=(f"-arch=sm_{major}{minor}", "-std=c++17", "--use_fast_math"))
        _modules[key] = mod.get_function("gemv_l1")
    return _modules[key]


def supported(lin) -> bool:
    return lin.d == 4 and lin.nch % 32 == 0 and lin.W == lin.nch * lin.bits // 32


class SortedLayout:
    """Rows sorted by column order (load time); words padded by one row-end word."""

    def __init__(self, lin):
        if not supported(lin):
            raise ValueError("layout not supported by the L1 GEMV")
        dev = lin.words.device
        rm = lin.row_mask.to(torch.int64) if lin.has_mask else torch.zeros(lin.R, dtype=torch.int64, device=dev)
        order = torch.argsort(rm, stable=True)
        flat = lin.words[order].reshape(-1)
        self.words = torch.cat([flat, torch.zeros(8, dtype=flat.dtype, device=dev)])
        self.row_scale = lin.row_scale[order].contiguous()
        self.book_off = lin.book_off[order].contiguous()
        self.row_mask = lin.row_mask[order].contiguous()
        self.out_row = order.to(torch.int32).contiguous()
        self.lin = lin


def layout(lin) -> SortedLayout:
    lay = getattr(lin, "_l1_layout", None)
    if lay is None:
        lay = lin._l1_layout = SortedLayout(lin)
    return lay


def gemv(lin, x: torch.Tensor, out: torch.Tensor | None = None, threads: int = 256, unroll: int = 4,
         warps_per_sm: int = 32) -> torch.Tensor:
    """y = W x for a PackedLinear `lin`; x fp16 [C] contiguous."""
    import cupy as cp
    lay = layout(lin)
    y = out if out is not None else torch.empty(lin.R, dtype=torch.float16, device=x.device)
    xp = (x[lin.perms_l] if lin.has_mask else x[None]).contiguous()
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    total_warps = sms * warps_per_sm
    rows_per_warp = max(1, -(-lin.R // total_warps))
    warps_needed = -(-lin.R // rows_per_warp)
    wpb = threads // 32
    grid = -(-warps_needed // wpb)
    fn = _kernel(lin.bits, lin.has_mask, threads, unroll)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        fn((grid,), (threads,),
           (cp.uint64(lay.words.data_ptr()), cp.uint64(lin.table.data_ptr()),
            cp.uint64(lay.row_scale.data_ptr()), cp.uint64(lay.book_off.data_ptr()),
            cp.uint64(lay.row_mask.data_ptr()), cp.uint64(lay.out_row.data_ptr()),
            cp.uint64(xp.data_ptr()), cp.uint64(y.data_ptr()),
            cp.int32(lin.R), cp.int32(lin.nch // 32), cp.int32(rows_per_warp)))
    return y
