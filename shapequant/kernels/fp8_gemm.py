"""EXPERIMENTAL, UNTESTED ON HARDWARE: FP8 (e4m3) prefill GEMMs for sm_89+ (Ada, Hopper, Blackwell
incl. sm_120 RTX PRO 6000). Changes the model's numerics (weights and activations rounded to
e4m3, 3 mantissa bits): measure KL before using it for anything but speed.

Per GEMM group (prefill.Group):
  - weights: each member decodes from its run words straight to e4m3 bytes (kernel dequant_fp8:
    codeword * row scale / S_w, saturating), S_w = one per-tensor scale for the whole group
    (max |codeword * row scale| / 448, computed once from the codebooks and row scales); half the
    bytes of the fp16 dequant;
  - activations: per-tensor e4m3 with S_x = max|x| / 448 (torch ops on the device, no host sync);
  - y = torch._scaled_mm(x8, w8^T, S_x, S_w) in fp16 (cuBLASLt FP8 GEMM, ~2x the fp16 tensor rate).

What has been verified on an sm_75 card: dequant_fp8 against a torch reference and the scaled
math by emulation (fp8 -> fp32 matmul). torch._scaled_mm itself needs cc >= 8.9 and has NOT been
run; the kernels compile for sm_89 / sm_120 (scripts/compile_check.py).
"""

from __future__ import annotations

import torch

_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_fp8.h>
typedef unsigned int u32;

// one thread per chunk pair (as cuda_dequant): 8 weights -> 8 e4m3 bytes (one 8-byte store)
extern "C" __global__ void dequant_fp8(const u32* __restrict__ words, const uint2* __restrict__ table,
        const __half* __restrict__ row_scale, const int* __restrict__ book_off, const float* __restrict__ sw,
        uint2* __restrict__ out, int R, int nch, int W, int nfull, int jt)
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
    const u32 v2 = __funnelshift_r(lo, hi, sh);
    const u32 m = (1u << BITS) - 1u;
    const uint2* tb = table + book_off[r];
    const uint2 t0 = __ldg(tb + (v2 & m)), t1 = __ldg(tb + ((v2 >> BITS) & m));
    const float s = __half2float(row_scale[r]) / sw[0];
    const u32 tv[4] = {t0.x, t0.y, t1.x, t1.y};
    unsigned short b[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(*(const __half2*)&tv[i]);
        b[i] = __nv_cvt_float2_to_fp8x2(make_float2(f.x * s, f.y * s), __NV_SATFINITE, __NV_E4M3);
    }
    out[idx] = make_uint2((u32)b[0] | ((u32)b[1] << 16), (u32)b[2] | ((u32)b[3] << 16));
}
"""

_fns: dict = {}
E4M3_MAX = 448.0


def source(bits: int) -> str:
    return f"#define BITS {bits}\n" + _SRC


def _fn(bits: int):
    import cupy as cp
    if bits not in _fns:
        _fns[bits] = cp.RawModule(code=source(bits), options=("-std=c++17",)).get_function("dequant_fp8")
    return _fns[bits]


def device_ok(device=None) -> bool:
    """FP8 tensor-core GEMMs (torch._scaled_mm): compute capability >= 8.9."""
    return torch.cuda.get_device_capability(device) >= (8, 9)


def weight_absmax(rl) -> torch.Tensor:
    """max |codeword * row scale| over the matrix (fp32 device scalar, cached on rl)."""
    t = getattr(rl, "_fp8_absmax", None)
    if t is None:
        K = 1 << rl.bits
        bmax = rl.table_raw.float().abs().view(-1, K * 4).amax(1)          # per book
        rows = (rl.book_off.long() // K)
        t = rl._fp8_absmax = (rl.row_scale.float().abs() * bmax[rows]).amax().reshape(1)
    return t


def dequant_fp8(rl, sw: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """e4m3 [R, C] of a RunsLinear: codeword * row scale / sw (out: a byte buffer, >= R*C)."""
    import cupy as cp
    from .prefill_kernels import stream
    from .proto_runs import geometry
    R, nch = rl.R, rl.nch
    n = R * nch * 4
    w = out.view(torch.uint8).view(-1)[:n]
    assert w.data_ptr() % 8 == 0
    nfull, jt = geometry(nch)
    grid = (R * nch // 2 + 255) // 256
    _fn(rl.bits)((grid,), (256,),
                 (cp.uint64(rl.words.data_ptr()), cp.uint64(rl.table_raw.data_ptr()),
                  cp.uint64(rl.row_scale.data_ptr()), cp.uint64(rl.book_off.data_ptr()), cp.uint64(sw.data_ptr()),
                  cp.uint64(w.data_ptr()), cp.int32(R), cp.int32(nch), cp.int32(rl.words.shape[1]),
                  cp.int32(nfull), cp.int32(jt)), stream=stream())
    return w.view(torch.float8_e4m3fn).view(R, nch * 4)


def group_weights(rls: list, scratch: torch.Tensor):
    """All members into one e4m3 [sum R, C] with one shared scale -> (w8, sw)."""
    sw = (torch.stack([weight_absmax(rl) for rl in rls]).amax() / E4M3_MAX).clamp_min(1e-12).reshape(1)
    buf = scratch.view(torch.uint8).view(-1)
    off = 0
    for rl in rls:
        n = rl.R * rl.nch * 4
        dequant_fp8(rl, sw, buf[off:off + n])
        off += n
    C = rls[0].nch * 4
    return buf[:off].view(torch.float8_e4m3fn).view(-1, C), sw


def quant_act(x: torch.Tensor):
    """x fp16 [T, C] -> (e4m3 [T, C], per-tensor scale (fp32 [1]))."""
    sx = (x.abs().amax().float() / E4M3_MAX).clamp_min(1e-12).reshape(1)
    return (x.float() / sx).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), sx


def gemm(x: torch.Tensor, rls: list, scratch: torch.Tensor) -> torch.Tensor:
    """fp16 [T, sum R] ~= x @ W^T through FP8 (see the module doc). Needs cc >= 8.9."""
    if not device_ok(x.device):
        raise RuntimeError("FP8 prefill GEMM needs compute capability >= 8.9 (torch._scaled_mm)")
    w8, sw = group_weights(rls, scratch)
    x8, sx = quant_act(x.contiguous())
    return torch._scaled_mm(x8, w8.t(), scale_a=sx, scale_b=sw, out_dtype=torch.float16)


def gemm_emulated(x: torch.Tensor, rls: list, scratch: torch.Tensor) -> torch.Tensor:
    """gemm() with the FP8 GEMM replaced by fp32 math on the e4m3 values (any GPU; for tests)."""
    w8, sw = group_weights(rls, scratch)
    x8, sx = quant_act(x.contiguous())
    return ((x8.float() * sx) @ (w8.float() * sw).T).half()
