"""Dequantize a run-kernel matrix (proto_runs.RunsLinear) to a dense fp16 weight, on the GPU,
straight from the lane-major packed words the decode kernel reads, so batched prefill needs no
second copy of the codes.

Lane-major layout (proto_runs.pack_lane_major, nch % 64 == 0, even bits): chunk c of a row sits
at pair p = c >> 6, lane l = (c >> 1) & 31, element e = c & 1. Pairs p < NFULL*8 form 16-tag
segments (8 pairs x 2) per (batch b = p >> 3, lane), each exactly bits/2 words, stored batch-major
then lane; the remaining JT pairs per lane form one tail bit stream per row in (lane, pair, e)
order. One thread per chunk: locate its tag, look up the 4-entry codeword (fp16, the unscaled
table), multiply by the row scale in fp32 and store 4 fp16 (8 bytes, coalesced across the warp).

Numerics match prefill.PrefillLinear.dense: half(float(codeword) * float(row_scale)).
Column orders (M > 1) are not handled here; callers fall back to PrefillLinear for them.
"""

from __future__ import annotations

import torch

_SRC = r"""
#include <cuda_fp16.h>
typedef unsigned int u32;

extern "C" __global__ void dequant_runs(const u32* __restrict__ words, const uint2* __restrict__ table,
        const __half* __restrict__ row_scale, const int* __restrict__ book_off,
        uint2* __restrict__ out, int R, int nch, int W, int nfull, int jt)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)R * nch) return;
    const int r = (int)(idx / nch), c = (int)(idx - (long long)r * nch);
    const int p = c >> 6, l = (c >> 1) & 31, e = c & 1;
    const u32* wr = words + (size_t)r * W;
    int base, bit;
    if (p < nfull * 8) {
        base = ((p >> 3) * 32 + l) * (BITS / 2);
        bit = ((p & 7) * 2 + e) * BITS;
    } else {
        base = nfull * 32 * (BITS / 2);
        bit = (l * jt * 2 + (p - nfull * 8) * 2 + e) * BITS;
    }
    const int wi = bit >> 5, sh = bit & 31;
    const u32 lo = __ldg(wr + base + wi);
    const u32 hi = (sh + BITS > 32) ? __ldg(wr + base + wi + 1) : 0u;
    const u32 v = __funnelshift_r(lo, hi, sh) & ((1u << BITS) - 1u);
    const uint2 t = __ldg(table + book_off[r] + v);
    const float s = __half2float(row_scale[r]);
    const float2 a = __half22float2(*(const __half2*)&t.x);
    const float2 b = __half22float2(*(const __half2*)&t.y);
    __half2 o0 = __floats2half2_rn(a.x * s, a.y * s);
    __half2 o1 = __floats2half2_rn(b.x * s, b.y * s);
    uint2 o;
    o.x = *(u32*)&o0;
    o.y = *(u32*)&o1;
    out[idx] = o;
}
"""

_fns: dict = {}


def _fn(bits: int):
    import cupy as cp
    if bits not in _fns:
        mod = cp.RawModule(code=f"#define BITS {bits}\n" + _SRC, options=("-std=c++17",))
        _fns[bits] = mod.get_function("dequant_runs")
    return _fns[bits]


def supported(rl) -> bool:
    return not rl.has_mask and rl.nch % 64 == 0 and rl.bits % 2 == 0 and getattr(rl, "table_raw", None) is not None


def dequant(rl, out: torch.Tensor | None = None) -> torch.Tensor:
    """fp16 [R, nch*4] weight of a RunsLinear without column orders. out: optional buffer with at
    least R*nch*4 elements (a view of its front is returned)."""
    import cupy as cp
    from .proto_runs import geometry
    if not supported(rl):
        raise ValueError("dequant: needs M = 1, nch % 64 == 0, even bits and rl.table_raw")
    R, nch = rl.R, rl.nch
    n = R * nch * 4
    if out is None:
        out = torch.empty(n, dtype=torch.float16, device=rl.words.device)
    w = out.view(-1)[:n].view(R, nch * 4)
    nfull, jt = geometry(nch)
    threads = 256
    grid = (R * nch + threads - 1) // threads
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        _fn(rl.bits)((grid,), (threads,),
                     (cp.uint64(rl.words.data_ptr()), cp.uint64(rl.table_raw.data_ptr()),
                      cp.uint64(rl.row_scale.data_ptr()), cp.uint64(rl.book_off.data_ptr()),
                      cp.uint64(w.data_ptr()), cp.int32(R), cp.int32(nch), cp.int32(rl.words.shape[1]),
                      cp.int32(nfull), cp.int32(jt)))
    return w
