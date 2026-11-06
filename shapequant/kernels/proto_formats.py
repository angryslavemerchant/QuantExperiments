"""Speed prototypes for kernel-friendly codebook formats (batch-1 GEMV).

The current format (one free codebook of 1024-16384 entries per matrix) is
limited by random table lookups: shared-memory bank conflicts (~6 wavefronts per
8-byte lookup) on the MIO pipe. These prototypes test layouts whose lookups
avoid that:

A  d=4, codeword = A[t & 31] + B[t >> 5] (two 32-entry books, 10-bit tags).
   A_SMEM: each book replicated 32x in shared memory, entry e of lane l at
           index e*32 + l, so every lane reads its own bank pair (2 wavefronts
           per 8-byte lookup, the minimum).
   A_SHFL: books in registers, lane l holds entry l; lookup = warp shuffle.
B  d=8, codeword = A[t & 255] + B[t >> 8] (two 256-entry books, 16-bit tags,
   2 bits/weight), tables replicated REP times (entry e copy c at e*REP + c,
   lane l uses copy l % REP).

Masks (M permuted copies of x) as in the real kernels; no mixture (G = 1).
"""

from __future__ import annotations

import torch

_SRC_A = r"""
#include <cuda_fp16.h>
typedef unsigned int u32;
#define BITS 10
__device__ __forceinline__ float dot4(uint2 a, uint2 b, uint2 xv) {
    const __half2 w0 = __hadd2(*(const __half2*)&a.x, *(const __half2*)&b.x);
    const __half2 w1 = __hadd2(*(const __half2*)&a.y, *(const __half2*)&b.y);
    __half2 p = __hmul2(w0, *(const __half2*)&xv.x);
    p = __hfma2(w1, *(const __half2*)&xv.y, p);
    const float2 f = __half22float2(p);
    return f.x + f.y;
}
extern "C" __global__ void __launch_bounds__(256)
gemv_add(const u32* __restrict__ words, const uint2* __restrict__ books,
         const __half* __restrict__ row_scale, const int* __restrict__ row_mask,
         const uint2* __restrict__ xp, __half* __restrict__ y, int R, int NG)
{
    const int lane = threadIdx.x & 31;
#if A_SMEM
    __shared__ uint2 sb[2 * 32 * 32];
    for (int i = threadIdx.x; i < 2 * 32 * 32; i += 256) sb[i] = books[(i >> 10) * 32 + ((i >> 5) & 31)];
    __syncthreads();
    const uint2* sA = sb + lane;
    const uint2* sB = sb + 1024 + lane;
#else
    const uint2 rA = books[lane], rB = books[32 + lane];
#endif
    const int nch = NG * 32, W = NG * BITS;
    const int bit = lane * BITS, wi = bit >> 5, sh = bit & 31;
    for (int row = blockIdx.x * 8 + (threadIdx.x >> 5); row < R; row += gridDim.x * 8) {
        const uint2* xr = xp + (size_t)row_mask[row] * nch;
        const u32* wr = words + (size_t)row * W + wi;
        float acc = 0.f;
        #pragma unroll 4
        for (int g = 0; g < NG; ++g) {
            const u32 v = __funnelshift_r(__ldg(wr + g * BITS), __ldg(wr + g * BITS + 1), sh);
            const int ta = v & 31, tb = (v >> 5) & 31;
#if A_SMEM
            const uint2 a = sA[ta * 32], b = sB[tb * 32];
#else
            uint2 a, b;
            a.x = __shfl_sync(0xffffffffu, rA.x, ta); a.y = __shfl_sync(0xffffffffu, rA.y, ta);
            b.x = __shfl_sync(0xffffffffu, rB.x, tb); b.y = __shfl_sync(0xffffffffu, rB.y, tb);
#endif
            acc += dot4(a, b, __ldg(xr + g * 32 + lane));
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[row] = __float2half(acc * __half2float(row_scale[row]));
    }
}
"""

_SRC_B = r"""
#include <cuda_fp16.h>
__device__ __forceinline__ float dot8(uint4 a, uint4 b, uint4 xv) {
    const __half2* ah = (const __half2*)&a; const __half2* bh = (const __half2*)&b;
    const __half2* xh = (const __half2*)&xv;
    __half2 p = __hmul2(__hadd2(ah[0], bh[0]), xh[0]);
    p = __hfma2(__hadd2(ah[1], bh[1]), xh[1], p);
    __half2 q = __hmul2(__hadd2(ah[2], bh[2]), xh[2]);
    q = __hfma2(__hadd2(ah[3], bh[3]), xh[3], q);
    const float2 f = __half22float2(p), g = __half22float2(q);
    return (f.x + f.y) + (g.x + g.y);
}
extern "C" __global__ void __launch_bounds__(256)
gemv_d8(const unsigned short* __restrict__ tags, const uint4* __restrict__ books,
        const __half* __restrict__ row_scale, const int* __restrict__ row_mask,
        const uint4* __restrict__ xp, __half* __restrict__ y, int R, int nch)
{
    extern __shared__ uint4 sb[];                      // [2][256][REP]
    for (int i = threadIdx.x; i < 2 * 256 * REP; i += 256) sb[i] = books[i / REP];
    __syncthreads();
    const int lane = threadIdx.x & 31, c = lane % REP;
    const uint4* sA = sb + c;
    const uint4* sB = sb + 256 * REP + c;
    for (int row = blockIdx.x * 8 + (threadIdx.x >> 5); row < R; row += gridDim.x * 8) {
        const uint4* xr = xp + (size_t)row_mask[row] * nch;
        const unsigned short* tr = tags + (size_t)row * nch;
        float acc = 0.f;
        #pragma unroll 4
        for (int j = lane; j < nch; j += 32) {
            const unsigned t = __ldg(tr + j);
            acc += dot8(sA[(t & 255) * REP], sB[(t >> 8) * REP], __ldg(xr + j));
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[row] = __float2half(acc * __half2float(row_scale[row]));
    }
}
"""

_mods: dict = {}


def _fn(name: str, src: str, defs: dict):
    import cupy as cp
    key = (name, tuple(sorted(defs.items())))
    if key not in _mods:
        code = "".join(f"#define {k} {v}\n" for k, v in defs.items()) + src
        _mods[key] = cp.RawModule(code=code, options=("--use_fast_math",)).get_function(name)
    return _mods[key]


def _grid(R: int, bps: int = 4) -> int:
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    return min(sms * bps, (R + 7) // 8)


def gemv_add(words, books, row_scale, row_mask, xp, R, NG, y, smem: bool):
    """A: words int32 [R*NG*10 + pad], books fp16 [2*32, 4], xp fp16 [M, C]."""
    import cupy as cp
    fn = _fn("gemv_add", _SRC_A, {"A_SMEM": int(smem)})
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        fn((_grid(R),), (256,), (cp.uint64(words.data_ptr()), cp.uint64(books.data_ptr()),
                                 cp.uint64(row_scale.data_ptr()), cp.uint64(row_mask.data_ptr()),
                                 cp.uint64(xp.data_ptr()), cp.uint64(y.data_ptr()), cp.int32(R), cp.int32(NG)))
    return y


def gemv_d8(tags, books, row_scale, row_mask, xp, R, nch, y, rep: int):
    """B: tags int16 [R, nch], books fp16 [2*256, 8], xp fp16 [M, C]."""
    import cupy as cp
    fn = _fn("gemv_d8", _SRC_B, {"REP": rep})
    shared = 2 * 256 * rep * 16
    if shared > 48 * 1024:
        fn.max_dynamic_shared_size_bytes = shared
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        fn((_grid(R),), (256,), (cp.uint64(tags.data_ptr()), cp.uint64(books.data_ptr()),
                                 cp.uint64(row_scale.data_ptr()), cp.uint64(row_mask.data_ptr()),
                                 cp.uint64(xp.data_ptr()), cp.uint64(y.data_ptr()), cp.int32(R), cp.int32(nch)),
           shared_mem=shared)
    return y


# A, ROWS rows per warp sharing one column order: each input chunk is loaded
# once and used ROWS times (rows sorted by mask at load time; groups padded with
# dummy rows whose output index is -1).
_SRC_A4 = r"""
#include <cuda_fp16.h>
typedef unsigned int u32;
#define BITS 10
__device__ __forceinline__ float dot4(uint2 a, uint2 b, uint2 xv) {
    const __half2 w0 = __hadd2(*(const __half2*)&a.x, *(const __half2*)&b.x);
    const __half2 w1 = __hadd2(*(const __half2*)&a.y, *(const __half2*)&b.y);
    __half2 p = __hmul2(w0, *(const __half2*)&xv.x);
    p = __hfma2(w1, *(const __half2*)&xv.y, p);
    const float2 f = __half22float2(p);
    return f.x + f.y;
}
extern "C" __global__ void __launch_bounds__(256)
gemv_add_rows(const u32* __restrict__ words, const uint2* __restrict__ books,
              const __half* __restrict__ row_scale, const int* __restrict__ tile_mask,
              const int* __restrict__ tile_rows, const uint2* __restrict__ xp,
              __half* __restrict__ y, int ntiles, int NG)
{
    __shared__ uint2 sb[2 * 32 * 32];
    for (int i = threadIdx.x; i < 2 * 32 * 32; i += 256) sb[i] = books[(i >> 10) * 32 + ((i >> 5) & 31)];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const uint2* sA = sb + lane;
    const uint2* sB = sb + 1024 + lane;
    const int nch = NG * 32, W = NG * BITS;
    const int bit = lane * BITS, wi = bit >> 5, sh = bit & 31;
    for (int t = blockIdx.x * 8 + (threadIdx.x >> 5); t < ntiles; t += gridDim.x * 8) {
        const uint2* xr = xp + (size_t)tile_mask[t] * nch;
        const u32* wr[ROWS];
        int rid[ROWS];
        float acc[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            rid[r] = tile_rows[t * ROWS + r];
            const int src = rid[r] < 0 ? -rid[r] - 1 : rid[r];
            wr[r] = words + (size_t)src * W + wi;
            acc[r] = 0.f;
        }
        #pragma unroll 2
        for (int g = 0; g < NG; ++g) {
            const uint2 xv = __ldg(xr + g * 32 + lane);
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const u32 v = __funnelshift_r(__ldg(wr[r] + g * BITS), __ldg(wr[r] + g * BITS + 1), sh);
                acc[r] += dot4(sA[(v & 31) * 32], sB[((v >> 5) & 31) * 32], xv);
            }
        }
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            float a = acc[r];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0 && rid[r] >= 0) y[rid[r]] = __float2half(a * __half2float(row_scale[rid[r]]));
        }
    }
}
"""


def row_tiles(row_mask: torch.Tensor, rows: int):
    """Tiles of `rows` rows sharing one mask: (tile_mask [T], tile_rows [T*rows]);
    padding entries are -(row+1) (computed, not written)."""
    rm = row_mask.to(torch.int64).cpu()
    order = torch.argsort(rm, stable=True)
    tm, tr = [], []
    for m in torch.unique(rm).tolist():
        ids = order[rm[order] == m].tolist()
        while len(ids) % rows:
            ids.append(-(ids[-1] if ids[-1] >= 0 else -ids[-1] - 1) - 1)
        for k in range(0, len(ids), rows):
            tm.append(m)
            tr.extend(ids[k:k + rows])
    dev = row_mask.device
    return (torch.tensor(tm, dtype=torch.int32, device=dev), torch.tensor(tr, dtype=torch.int32, device=dev))


def gemv_add_rows(words, books, row_scale, tiles, xp, NG, y, rows: int):
    import cupy as cp
    tile_mask, tile_rows = tiles
    fn = _fn("gemv_add_rows", _SRC_A4, {"ROWS": rows})
    T = tile_mask.numel()
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        fn((_grid(T),), (256,), (cp.uint64(words.data_ptr()), cp.uint64(books.data_ptr()),
                                 cp.uint64(row_scale.data_ptr()), cp.uint64(tile_mask.data_ptr()),
                                 cp.uint64(tile_rows.data_ptr()), cp.uint64(xp.data_ptr()),
                                 cp.uint64(y.data_ptr()), cp.int32(T), cp.int32(NG)))
    return y


# A, wide lanes: each lane owns 16 consecutive tags of a row (16 x 10 bits =
# exactly 5 words, unpacked in registers with compile-time shifts) and the 16
# matching input chunks (8 x 16-byte loads), shared by ROWS same-mask rows.
_SRC_AW = r"""
#include <cuda_fp16.h>
typedef unsigned int u32;
__device__ __forceinline__ float dot4(uint2 a, uint2 b, uint2 xv) {
    const __half2 w0 = __hadd2(*(const __half2*)&a.x, *(const __half2*)&b.x);
    const __half2 w1 = __hadd2(*(const __half2*)&a.y, *(const __half2*)&b.y);
    __half2 p = __hmul2(w0, *(const __half2*)&xv.x);
    p = __hfma2(w1, *(const __half2*)&xv.y, p);
    const float2 f = __half22float2(p);
    return f.x + f.y;
}
extern "C" __global__ void __launch_bounds__(256)
gemv_add_wide(const u32* __restrict__ words, const uint2* __restrict__ books,
              const __half* __restrict__ row_scale, const int* __restrict__ tile_mask,
              const int* __restrict__ tile_rows, const uint4* __restrict__ xp,
              __half* __restrict__ y, int ntiles, int nch)
{
    __shared__ uint2 sb[2 * 32 * 32];
    for (int i = threadIdx.x; i < 2 * 32 * 32; i += 256) sb[i] = books[(i >> 10) * 32 + ((i >> 5) & 31)];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const uint2* sA = sb + lane;
    const uint2* sB = sb + 1024 + lane;
    const int W = nch * 10 / 32, n16 = nch / 16;
    for (int t = blockIdx.x * 8 + (threadIdx.x >> 5); t < ntiles; t += gridDim.x * 8) {
        const uint4* xr = xp + (size_t)tile_mask[t] * (nch / 2);
        int rid[ROWS];
        const u32* wr[ROWS];
        float acc[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            rid[r] = tile_rows[t * ROWS + r];
            wr[r] = words + (size_t)(rid[r] < 0 ? -rid[r] - 1 : rid[r]) * W;
            acc[r] = 0.f;
        }
        for (int s = lane; s < n16; s += 32) {          // s: this lane's 16-tag segment
            uint2 xv[16];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                const uint4 q = __ldg(xr + s * 8 + k);
                xv[2 * k] = make_uint2(q.x, q.y);
                xv[2 * k + 1] = make_uint2(q.z, q.w);
            }
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                u32 w[6];
                #pragma unroll
                for (int k = 0; k < 5; ++k) w[k] = __ldg(wr[r] + s * 5 + k);
                w[5] = 0u;
                float a = 0.f;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int bit = i * 10, wi = bit >> 5, sh = bit & 31;
                    const u32 v = sh + 10 > 32 ? __funnelshift_r(w[wi], w[wi + 1], sh) : (w[wi] >> sh);
                    a += dot4(sA[(v & 31) * 32], sB[((v >> 5) & 31) * 32], xv[i]);
                }
                acc[r] += a;
            }
        }
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            float a = acc[r];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0 && rid[r] >= 0) y[rid[r]] = __float2half(a * __half2float(row_scale[rid[r]]));
        }
    }
}
"""


def gemv_add_wide(words, books, row_scale, tiles, xp, nch, y, rows: int):
    import cupy as cp
    tile_mask, tile_rows = tiles
    fn = _fn("gemv_add_wide", _SRC_AW, {"ROWS": rows})
    T = tile_mask.numel()
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        fn((_grid(T),), (256,), (cp.uint64(words.data_ptr()), cp.uint64(books.data_ptr()),
                                 cp.uint64(row_scale.data_ptr()), cp.uint64(tile_mask.data_ptr()),
                                 cp.uint64(tile_rows.data_ptr()), cp.uint64(xp.data_ptr()),
                                 cp.uint64(y.data_ptr()), cp.int32(T), cp.int32(nch)))
    return y


# Q4 reference (llama.cpp Q4_0-like, 4.5 bpw): blocks of 32 weights = 16 bytes of
# nibbles + one fp16 scale; w = (nibble - 8) * scale. Nibble -> half via the
# 0x6400 exponent trick (one LOP3 per weight pair), so decode is cheap arithmetic.
# Packing per 32-bit word (8 weights): weight pair (2k, 2k+1) = nibble k of the
# low / high half-word.
_SRC_Q4 = r"""
#include <cuda_fp16.h>
typedef unsigned int u32;
extern "C" __global__ void __launch_bounds__(256)
gemv_q4(const uint4* __restrict__ q, const __half* __restrict__ scales, const uint4* __restrict__ x,
        __half* __restrict__ y, int R, int NB)
{
    const int lane = threadIdx.x & 31;
    const __half2 k1032 = __float2half2_rn(1032.f);
    for (int row = blockIdx.x * 8 + (threadIdx.x >> 5); row < R; row += gridDim.x * 8) {
        const uint4* qr = q + (size_t)row * NB;
        const __half* sr = scales + (size_t)row * NB;
        float acc = 0.f;
        #pragma unroll 2
        for (int b = lane; b < NB; b += 32) {
            const uint4 qq = __ldg(qr + b);
            const __half2 s = __half2half2(sr[b]);
            const u32 w[4] = {qq.x, qq.y, qq.z, qq.w};
            __half2 part = __float2half2_rn(0.f);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint4 xv = __ldg(x + b * 4 + j);       // 8 inputs for word j
                const __half2* xh = (const __half2*)&xv;
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    u32 v = ((w[j] >> (4 * k)) & 0x000F000Fu) | 0x64006400u;
                    __half2 wv = __hsub2(*(__half2*)&v, k1032);
                    part = __hfma2(wv, xh[k], part);
                }
            }
            const float2 f = __half22float2(__hmul2(part, s));
            acc += f.x + f.y;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[row] = __float2half(acc);
    }
}
"""


def q4_pack(w: torch.Tensor):
    """w [R, C] float -> (q int32 [R, C/8], scales fp16 [R, C/32], dequantized fp32)."""
    R, C = w.shape
    blocks = w.reshape(R, C // 32, 32).float()
    amax = blocks.abs().amax(-1).clamp_min(1e-8)
    scale = (amax / 7.0).half().float()
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)          # all-zero block: fp16 underflow
    n = (blocks / scale[..., None]).round().clamp(-8, 7) + 8               # 0..15
    deq = ((n - 8) * scale[..., None]).reshape(R, C)
    words8 = n.reshape(R, C // 8, 8).to(torch.int64)                       # 8 weights per word
    word = torch.zeros(R, C // 8, dtype=torch.int64)
    for k in range(4):
        word |= words8[..., 2 * k] << (4 * k)                              # low half-word, nibble k
        word |= words8[..., 2 * k + 1] << (16 + 4 * k)                     # high half-word, nibble k
    word = torch.where(word >= (1 << 31), word - (1 << 32), word).to(torch.int32)
    return word, scale.half(), deq


def gemv_q4(q, scales, x, y, R, C):
    import cupy as cp
    fn = _fn("gemv_q4", _SRC_Q4, {})
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        fn((_grid(R),), (256,), (cp.uint64(q.data_ptr()), cp.uint64(scales.data_ptr()),
                                 cp.uint64(x.data_ptr()), cp.uint64(y.data_ptr()), cp.int32(R), cp.int32(C // 32)))
    return y


_SRC_Q8 = r"""
#include <cuda_fp16.h>
typedef unsigned int u32;
__device__ __forceinline__ float dot4(u32 w, uint2 xv) {
    const float2 a = __half22float2(*(const __half2*)&xv.x), b = __half22float2(*(const __half2*)&xv.y);
    return (float)(signed char)(w & 0xff) * a.x + (float)(signed char)((w >> 8) & 0xff) * a.y
         + (float)(signed char)((w >> 16) & 0xff) * b.x + (float)(signed char)(w >> 24) * b.y;
}
extern "C" __global__ void __launch_bounds__(256)
gemv_q8(const uint4* __restrict__ q, const __half* __restrict__ scales, const uint2* __restrict__ x,
        __half* __restrict__ y, int R, int NB)
{
    const int lane = threadIdx.x & 31;
    for (int row = blockIdx.x * 8 + (threadIdx.x >> 5); row < R; row += gridDim.x * 8) {
        const uint4* qr = q + (size_t)row * NB * 2;
        const __half* sr = scales + (size_t)row * NB;
        float acc = 0.f;
        #pragma unroll 2
        for (int b = lane; b < NB; b += 32) {
            const uint4 q0 = __ldg(qr + 2 * b), q1 = __ldg(qr + 2 * b + 1);
            const u32 w[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
            float part = 0.f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) part += dot4(w[j], __ldg(x + b * 8 + j));
            acc += part * __half2float(sr[b]);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[row] = __float2half(acc);
    }
}
"""


def q8_pack(w: torch.Tensor):
    """w [R, C] float -> (q int8 [R, C], scales fp16 [R, C/32], dequantized fp32): Q8_0-style blocks of 32."""
    R, C = w.shape
    blocks = w.reshape(R, C // 32, 32).float()
    scale = (blocks.abs().amax(-1).clamp_min(1e-8) / 127.0).half().float()
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)          # all-zero block: fp16 underflow
    n = (blocks / scale[..., None]).round().clamp(-127, 127)
    return n.reshape(R, C).to(torch.int8), scale.half(), (n * scale[..., None]).reshape(R, C)


def gemv_q8(q, scales, x, y, R, C):
    import cupy as cp
    fn = _fn("gemv_q8", _SRC_Q8, {})
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        fn((_grid(R),), (256,), (cp.uint64(q.data_ptr()), cp.uint64(scales.data_ptr()),
                                 cp.uint64(x.data_ptr()), cp.uint64(y.data_ptr()), cp.int32(R), cp.int32(C // 32)))
    return y
