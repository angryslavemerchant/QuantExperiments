"""Batched decode-step kernels for the Qwen2-architecture engine (shapequant.runtime_batch): one
sequence per blockIdx.y (addnorm16_b: per blockIdx.x), per-sequence positions. Numerics follow
kernels.decode_e2e (fp16 norm weights, rope with bias added and cos / sin rounded to fp16, HF's
fp16 silu * up); attention is the GQA-sharing flash decoding of kernels.qwen35_batch_kernels
(fp16 queries, no gate).

  addnorm16_b    grid B                h [B, C] (+= add [B, C] fp16); out [B, C] = fp16(rmsnorm(h) * w)
  rope_cache_b   grid (NH + NKV, B)    q / k / v (+ bias) -> qo [B, NH*HD] fp16, k / v into the caches
                                       kc / vc [B, NKV, maxlen, HD] at pos[b]
  attn_gqa_b / attn_comb_b             see qwen35_batch_kernels.GQA_SRC
  silu_mul16     elementwise           fp16(silu(g)) * u (fp16 multiply)
"""

from __future__ import annotations

_SRC = r"""
#include <cuda_fp16.h>

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float t = threadIdx.x < nw ? red[threadIdx.x] : 0.f;
    if (w == 0) t = warp_sum(t);
    if (threadIdx.x == 0) red[0] = t;
    __syncthreads();
    return red[0];
}

extern "C" __global__ void addnorm16_b(float* h, const __half* add, const __half* w, int C, float eps, __half* out) {
    __shared__ float red[32];
    const size_t o = (size_t)blockIdx.x * C;
    h += o; out += o;
    if (add) add += o;
    float ss = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = h[i];
        if (add) { v += __half2float(add[i]); h[i] = v; }
        ss += v * v;
    }
    const float r = rsqrtf(block_sum(ss, red) / C + eps);
    for (int i = threadIdx.x; i < C; i += blockDim.x) out[i] = __float2half(h[i] * r * __half2float(w[i]));
}

extern "C" __global__ void rope_cache_b(const __half* q, const __half* k, const __half* v, const __half* bq,
                                        const __half* bk, const __half* bv, const long long* pos_p, float theta,
                                        int maxlen, __half* qo, __half* kc, __half* vc) {
    const int b = blockIdx.x, i = threadIdx.x, half = HD / 2;
    const size_t bb = blockIdx.y;
    q += bb * NH * HD; k += bb * NKV * HD; v += bb * NKV * HD; qo += bb * NH * HD;
    kc += bb * NKV * maxlen * HD; vc += bb * NKV * maxlen * HD;
    const int p = (int)pos_p[bb];
    const float inv = 1.f / powf(theta, (2.f * i) / HD);
    const float ang = (float)p * inv;
    const float c = __half2float(__float2half(cosf(ang))), s = __half2float(__float2half(sinf(ang)));
    const __half z = __float2half(0.f);
    if (b < NH) {
        const int o = b * HD;
        const float x1 = __half2float(__hadd(q[o + i], bq ? bq[o + i] : z));
        const float x2 = __half2float(__hadd(q[o + i + half], bq ? bq[o + i + half] : z));
        qo[o + i] = __float2half(x1 * c - x2 * s);
        qo[o + i + half] = __float2half(x2 * c + x1 * s);
    } else {
        const int kv = b - NH, o = kv * HD;
        const size_t co = ((size_t)kv * maxlen + p) * HD;
        const float x1 = __half2float(__hadd(k[o + i], bk ? bk[o + i] : z));
        const float x2 = __half2float(__hadd(k[o + i + half], bk ? bk[o + i + half] : z));
        kc[co + i] = __float2half(x1 * c - x2 * s);
        kc[co + i + half] = __float2half(x2 * c + x1 * s);
        vc[co + i] = __hadd(v[o + i], bv ? bv[o + i] : z);
        vc[co + i + half] = __hadd(v[o + i + half], bv ? bv[o + i + half] : z);
    }
}

extern "C" __global__ void silu_mul16(const __half* g, const __half* u, int n, __half* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float a = __half2float(g[i]);
        out[i] = __hmul(__float2half(a / (1.f + __expf(-a))), u[i]);
    }
}
"""

_mods: dict = {}


def module(cfg):
    """Kernels for a decode_e2e.Cfg."""
    import cupy as cp
    from .qwen35_batch_kernels import GQA_SRC
    key = (cfg.n_heads, cfg.n_kv, cfg.head_dim)
    if key not in _mods:
        defs = (f"#define NH {cfg.n_heads}\n#define NKV {cfg.n_kv}\n#define HD {cfg.head_dim}\n"
                "#define QT __half\n#define GATED 0\n")
        _mods[key] = cp.RawModule(code=defs + _SRC + GQA_SRC, options=("-std=c++17",))
    return _mods[key]
