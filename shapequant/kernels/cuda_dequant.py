"""Dequantize a run-kernel matrix (proto_runs.RunsLinear) to a dense fp16 weight, on the GPU,
straight from the lane-major packed words the decode kernel reads, so batched prefill needs no
second copy of the codes.

Lane-major layout (proto_runs.pack_lane_major, nch % 64 == 0, even bits): chunk c of a row sits
at pair p = c >> 6, lane l = (c >> 1) & 31, element e = c & 1. Pairs p < NFULL*8 form 16-tag
segments (8 pairs x 2) per (batch b = p >> 3, lane), each exactly bits/2 words, stored batch-major
then lane; the remaining JT pairs per lane form one tail bit stream per row in (lane, pair, e)
order. One thread per chunk pair (2q, 2q+1: adjacent tags of one lane segment): look up the two
4-entry codewords (fp16, the unscaled table), multiply by the row scale in fp32 and store 8 fp16
(16 bytes, coalesced across the warp).

Numerics match prefill.PrefillLinear.dense: half(float(codeword) * float(row_scale)).
Column orders (M > 1) are not handled here; callers fall back to PrefillLinear for them.
"""

from __future__ import annotations

import torch

_SRC = r"""
#include <cuda_fp16.h>
typedef unsigned int u32;

// one thread per chunk pair (chunks 2q, 2q+1: same lane segment, adjacent tags): 8 weights in one
// 16-byte store; consecutive threads = consecutive pairs of a row (coalesced stores)
__device__ __forceinline__ u32 scale2(u32 t, float s) {
    const float2 a = __half22float2(*(const __half2*)&t);
    __half2 o = __floats2half2_rn(a.x * s, a.y * s);
    return *(u32*)&o;
}
extern "C" __global__ void dequant_runs(const u32* __restrict__ words, const uint2* __restrict__ table,
        const __half* __restrict__ row_scale, const int* __restrict__ book_off,
        uint4* __restrict__ out, int R, int nch, int W, int nfull, int jt)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int npr = nch >> 1;
    if (idx >= (long long)R * npr) return;
    const int r = (int)(idx / npr), q = (int)(idx - (long long)r * npr);
    const int p = q >> 5, l = q & 31;
    const u32* wr = words + (size_t)r * W;
    int base, bit;
    if (p < nfull * 8) {
        base = ((p >> 3) * 32 + l) * (BITS / 2);
        bit = ((p & 7) * 2) * BITS;
    } else {
        base = nfull * 32 * (BITS / 2);
        bit = (l * jt * 2 + (p - nfull * 8) * 2) * BITS;
    }
    const int wi = bit >> 5, sh = bit & 31;
    const u32 lo = __ldg(wr + base + wi);
    const u32 hi = (sh + 2 * BITS > 32) ? __ldg(wr + base + wi + 1) : 0u;
    const u32 v2 = __funnelshift_r(lo, hi, sh);          // 2 * BITS <= 24 bits: both tags
    const u32 m = (1u << BITS) - 1u;
    const uint2* tb = table + book_off[r];
    const uint2 t0 = __ldg(tb + (v2 & m)), t1 = __ldg(tb + ((v2 >> BITS) & m));
    const float s = __half2float(row_scale[r]);
    uint4 o;
    o.x = scale2(t0.x, s); o.y = scale2(t0.y, s); o.z = scale2(t1.x, s); o.w = scale2(t1.y, s);
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


def dequant(rl, out: torch.Tensor | None = None, rows: tuple[int, int] | None = None) -> torch.Tensor:
    """fp16 [R, nch*4] weight of a RunsLinear without column orders. out: optional buffer with at
    least R*nch*4 elements (a view of its front is returned). rows: (r0, r1) decodes only those
    rows ([r1 - r0, nch*4])."""
    import cupy as cp
    from .proto_runs import geometry
    if not supported(rl):
        raise ValueError("dequant: needs M = 1, nch % 64 == 0, even bits and rl.table_raw")
    r0, r1 = rows if rows is not None else (0, rl.R)
    R, nch = r1 - r0, rl.nch
    n = R * nch * 4
    if out is None:
        out = torch.empty(n, dtype=torch.float16, device=rl.words.device)
    w = out.view(-1)[:n].view(R, nch * 4)
    nfull, jt = geometry(nch)
    assert w.data_ptr() % 16 == 0, "dequant output must be 16-byte aligned"
    threads = 256
    grid = (R * nch // 2 + threads - 1) // threads
    from .prefill_kernels import stream
    _fn(rl.bits)((grid,), (threads,),
                 (cp.uint64(rl.words[r0:r1].data_ptr()), cp.uint64(rl.table_raw.data_ptr()),
                  cp.uint64(rl.row_scale[r0:r1].data_ptr()), cp.uint64(rl.book_off[r0:r1].data_ptr()),
                  cp.uint64(w.data_ptr()), cp.int32(R), cp.int32(nch), cp.int32(rl.words.shape[1]),
                  cp.int32(nfull), cp.int32(jt)), stream=stream())
    return w
