"""Fused CUDA kernels for the Qwen3.5 decode step (shapequant.qwen35_runtime, fused=True).

  addnorm      h += add (fp16, optional); out = fp16(rmsnorm(h) * w1p)            1 block
  conv_step    depthwise causal conv over [state, x] + silu -> u (fp32); state shift
  delta_step   one block per value head: a, b projections of x, l2norm(q), l2norm(k),
               g / beta gates, recurrent state S [dk, dv] update (thread = one column, held in
               registers), o = S^T q, gated RMSNorm (w, silu(z)) -> fp16
  attn_prep    q_norm / k_norm (1 + w), partial rope, K/V cache write at pos
  attn         one block per query head, flash-style over the cache (warp per position
               stripe, online softmax), times sigmoid(gate) -> fp16
  silu_mul     fp16(silu(g) * u)

All numerics in fp32; the position comes from a device scalar (graph capturable).
"""

from __future__ import annotations

import torch

_SRC = r"""
#include <cuda_fp16.h>

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// block-wide sum (blockDim.x multiple of 32, <= 1024); red: >= 32 floats of shared memory
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
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float softplus(float x) { return x > 20.f ? x : log1pf(__expf(x)); }

extern "C" __global__ void addnorm(float* h, const __half* add, const float* w1p, int C, float eps, __half* out) {
    __shared__ float red[32];
    float ss = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = h[i];
        if (add) { v += __half2float(add[i]); h[i] = v; }
        ss += v * v;
    }
    const float r = rsqrtf(block_sum(ss, red) / C + eps);
    for (int i = threadIdx.x; i < C; i += blockDim.x) out[i] = __float2half(h[i] * r * w1p[i]);
}

// u[c] = silu(sum_k w[c,k] * win[c,k]), win = [state[c,1:], x[c]]; state <- win
extern "C" __global__ void conv_step(const __half* x, float* state, const float* w, int D, float* u) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float* s = state + (size_t)c * KC;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < KC - 1; ++k) { const float v = s[k + 1]; s[k] = v; acc += w[c * KC + k] * v; }
    const float xv = __half2float(x[c]);
    s[KC - 1] = xv;
    acc += w[c * KC + KC - 1] * xv;
    u[c] = silu(acc);
}

// grid = HV blocks, blockDim = DV threads (thread t owns column t of the head's state)
extern "C" __global__ void __launch_bounds__(DV) delta_step(
        const float* u, const __half* z, const __half* x, const float* wa, const float* wb,
        const float* A, const float* dt_bias, const float* gnorm, float* S, int H, float eps, __half* out) {
    __shared__ float red[32];
    __shared__ float qs[DK], ks[DK];
    const int h = blockIdx.x, t = threadIdx.x, kh = h / (HV / HK);
    // a, b projections of x for this head
    float pa = 0.f, pb = 0.f;
    for (int i = t; i < H; i += DV) {
        const float xv = __half2float(x[i]);
        pa += wa[(size_t)h * H + i] * xv;
        pb += wb[(size_t)h * H + i] * xv;
    }
    const float a = block_sum(pa, red);
    const float b = block_sum(pb, red);
    const float decay = __expf(-A[h] * softplus(a + dt_bias[h]));
    const float beta = sigm(b);
    // q, k (l2-normalised; q scaled by DK^-1/2): threads t < DK
    const float qv = t < DK ? u[kh * DK + t] : 0.f;
    const float kv = t < DK ? u[HK * DK + kh * DK + t] : 0.f;
    const float qn = rsqrtf(block_sum(qv * qv, red) + 1e-6f) * rsqrtf((float)DK);
    const float kn = rsqrtf(block_sum(kv * kv, red) + 1e-6f);
    if (t < DK) { qs[t] = qv * qn; ks[t] = kv * kn; }
    __syncthreads();
    const float vv = u[2 * HK * DK + h * DV + t];
    float* Sh = S + (size_t)h * DK * DV;
    float s[DK];
    float mem = 0.f;
    #pragma unroll
    for (int i = 0; i < DK; ++i) { s[i] = Sh[i * DV + t] * decay; mem += s[i] * ks[i]; }
    const float delta = (vv - mem) * beta;
    float o = 0.f;
    #pragma unroll
    for (int i = 0; i < DK; ++i) { s[i] += ks[i] * delta; o += s[i] * qs[i]; Sh[i * DV + t] = s[i]; }
    // gated RMSNorm over the head's DV outputs
    const float r = rsqrtf(block_sum(o * o, red) / DV + eps);
    out[h * DV + t] = __float2half(o * r * gnorm[t] * silu(__half2float(z[h * DV + t])));
}

// grid = NH + NKV blocks, blockDim = HD. Block < NH: query head (normed + roped in place into
// qr fp32); else KV head: k normed + roped, k / v written to the caches at pos.
extern "C" __global__ void attn_prep(const __half* qg, const __half* k, const __half* v, const float* qn,
        const float* kn, const float* inv_freq, const long long* pos_p, float eps, float* qr,
        __half* kc, __half* vc, int maxlen) {
    __shared__ float red[32];
    __shared__ float buf[HD];
    const int b = blockIdx.x, t = threadIdx.x;
    const long long pos = *pos_p;
    const bool isq = b < NH;
    const int hh = isq ? b : b - NH;
    const float x = isq ? __half2float(qg[hh * 2 * HD + t]) : __half2float(k[hh * HD + t]);
    const float r = rsqrtf(block_sum(x * x, red) / HD + eps);
    const float y = x * r * (isq ? qn[t] : kn[t]);
    buf[t] = y;
    __syncthreads();
    float o = y;
    if (t < ROT) {
        const int i = t % (ROT / 2);
        const float f = (float)pos * inv_freq[i];
        float sn, cs;
        sincosf(f, &sn, &cs);
        const float rot = t < ROT / 2 ? -buf[t + ROT / 2] : buf[t - ROT / 2];
        o = y * cs + rot * sn;
    }
    if (isq) qr[hh * HD + t] = o;
    else {
        kc[((size_t)hh * maxlen + pos) * HD + t] = __float2half(o);
        vc[((size_t)hh * maxlen + pos) * HD + t] = v[hh * HD + t];
    }
}

// grid = NH blocks, 256 threads (8 warps). Warp w takes positions w, w+8, ...; each lane holds
// HD/32 dims. Online softmax per warp, merged across warps in shared memory.
extern "C" __global__ void __launch_bounds__(256) attn(const float* qr, const __half* qg, const __half* kc,
        const __half* vc, const long long* pos_p, int maxlen, __half* out) {
    constexpr int PL = HD / 32;
    __shared__ float wm[8], wl[8];
    __shared__ float wo[8][HD];
    const int h = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5, kvh = h / (NH / NKV);
    const int n = (int)(*pos_p) + 1;
    float q[PL];
    #pragma unroll
    for (int j = 0; j < PL; ++j) q[j] = qr[h * HD + lane * PL + j] * rsqrtf((float)HD);
    float m = -1e30f, l = 0.f, acc[PL];
    #pragma unroll
    for (int j = 0; j < PL; ++j) acc[j] = 0.f;
    const __half* K = kc + (size_t)kvh * maxlen * HD;
    const __half* V = vc + (size_t)kvh * maxlen * HD;
    for (int p = w; p < n; p += 8) {
        float d = 0.f;
        #pragma unroll
        for (int j = 0; j < PL; ++j) d += q[j] * __half2float(K[(size_t)p * HD + lane * PL + j]);
        d = warp_sum(d);
        const float mn = fmaxf(m, d), c = __expf(m - mn), e = __expf(d - mn);
        l = l * c + e;
        #pragma unroll
        for (int j = 0; j < PL; ++j) acc[j] = acc[j] * c + e * __half2float(V[(size_t)p * HD + lane * PL + j]);
        m = mn;
    }
    if (lane == 0) { wm[w] = m; wl[w] = l; }
    #pragma unroll
    for (int j = 0; j < PL; ++j) wo[w][lane * PL + j] = acc[j];
    __syncthreads();
    if (threadIdx.x < HD) {
        float M = -1e30f;
        for (int i = 0; i < 8; ++i) M = fmaxf(M, wm[i]);
        float L = 0.f, O = 0.f;
        for (int i = 0; i < 8; ++i) { const float c = __expf(wm[i] - M); L += wl[i] * c; O += wo[i][threadIdx.x] * c; }
        const int d = threadIdx.x;
        out[h * HD + d] = __float2half(O / L * sigm(__half2float(qg[h * 2 * HD + HD + d])));
    }
}

extern "C" __global__ void silu_mul(const __half* g, const __half* u, int n, __half* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(silu(__half2float(g[i])) * __half2float(u[i]));
}
"""

_mods: dict = {}


def module(c):
    """Compiled kernels for engine config c (qwen35_runtime._cfg)."""
    import cupy as cp
    key = (c.hk, c.hv, c.dk, c.dv, c.conv_k, c.n_heads, c.n_kv, c.head_dim, c.rot)
    if key not in _mods:
        defs = (f"#define HK {c.hk}\n#define HV {c.hv}\n#define DK {c.dk}\n#define DV {c.dv}\n#define KC {c.conv_k}\n"
                f"#define NH {c.n_heads}\n#define NKV {c.n_kv}\n#define HD {c.head_dim}\n#define ROT {c.rot}\n")
        _mods[key] = cp.RawModule(code=defs + _SRC, options=("-std=c++17",))
    return _mods[key]


def supported(c) -> bool:
    return (c.dv % 32 == 0 and c.dv <= 1024 and c.dk <= c.dv and c.head_dim % 32 == 0 and c.head_dim <= 256
            and c.hv % c.hk == 0 and c.n_heads % c.n_kv == 0 and c.rot % 2 == 0 and c.rot <= c.head_dim)
