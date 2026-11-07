"""CUDA kernels for batched prefill (shapequant.prefill, shapequant.qwen35_runtime.prefill).

The torch reference prefill spends most of its non-GEMM time in small elementwise ops (fp32
<-> fp16 casts, rmsnorm as 6 kernels, rope as ~15) and, for Qwen3.5, in the CPU-bound Python
loops of HF's torch_chunk_gated_delta_rule. These kernels fuse them, one launch per op class:

  addnorm      per row: h += add (fp16, optional); out = fp16(rmsnorm(h) * w)  (w fp16 or fp32)
  silu_mul     per row of a [T, 2I] gate|up GEMM output: fp16(silu(g) * u), Qwen2 or Qwen3.5 rounding
  rope_qkv     Qwen2: bias add, rope (fp16-rounded cos/sin tables), q -> [NH, T, HD], k / v -> cache
  attn_prep    Qwen3.5 full attention: q/k norm (1 + w), partial rope, q -> [NH, T, HD] fp16, k / v -> cache
  gate_out     Qwen3.5: fp16(o * sigmoid(gate)) back to [T, NH*HD]
  conv         Qwen3.5 DeltaNet: causal depthwise conv over [state, x] + silu (fp32), then state update
  gdn_prep     l2norm(q) * dk^-1/2, l2norm(k), v, g = -A softplus(a + dt_bias), beta = sigmoid(b),
               head-major and padded to whole chunks
  gdn_scan     the sequential part of the chunked gated delta rule: per chunk, v_new = u - w S,
               o_inter = (q e^g) S, S = e^{g_last} S + (k e^{g_last - g})^T v_new (fp32, S in smem)
  gated_norm   fp16(rmsnorm(o) * w * silu(z))

All math in fp32 unless noted; layouts are documented per kernel.
"""

from __future__ import annotations

import math

import torch

_SRC = r"""
#include <cuda_fp16.h>
typedef unsigned int u32;

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// block-wide sum (blockDim.x multiple of 32); red: >= 32 floats of shared memory
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
__device__ __forceinline__ float silu(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }

// ---------------------------------------------------------------- addnorm
// h [T, C] fp32 (in/out), add fp16 rows of stride lda (nullable), w fp16 (WF32 0) or fp32 (WF32 1)
extern "C" __global__ void addnorm(float* __restrict__ h, const __half* __restrict__ add, int lda,
                                   const void* __restrict__ wv, int wf32, int C, float eps,
                                   __half* __restrict__ out, int ldo) {
    __shared__ float red[32];
    const size_t t = blockIdx.x;
    float* hr = h + t * C;
    const __half* ar = add ? add + t * lda : nullptr;
    float ss = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = hr[i];
        if (ar) { v += __half2float(ar[i]); hr[i] = v; }
        ss += v * v;
    }
    const float r = rsqrtf(block_sum(ss, red) / C + eps);
    __half* o = out + t * ldo;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        const float w = wf32 ? ((const float*)wv)[i] : __half2float(((const __half*)wv)[i]);
        o[i] = __float2half(hr[i] * r * w);
    }
}

// addnorm with 4-wide accesses (C, lda, ldo multiples of 4; 8-byte aligned fp16 rows)
__device__ __forceinline__ float4 h4f(uint2 a) {
    const float2 x = __half22float2(*(const __half2*)&a.x), y = __half22float2(*(const __half2*)&a.y);
    return make_float4(x.x, x.y, y.x, y.y);
}
extern "C" __global__ void addnorm4(float* __restrict__ h, const __half* __restrict__ add, int lda,
                                    const void* __restrict__ wv, int wf32, int C, float eps,
                                    __half* __restrict__ out, int ldo) {
    __shared__ float red[32];
    const size_t t = blockIdx.x;
    float4* hr = (float4*)(h + t * C);
    const uint2* ar = add ? (const uint2*)(add + t * lda) : nullptr;
    const int C4 = C >> 2;
    float ss = 0.f;
    for (int i = threadIdx.x; i < C4; i += blockDim.x) {
        float4 v = hr[i];
        if (ar) {
            const float4 a = h4f(ar[i]);
            v.x += a.x; v.y += a.y; v.z += a.z; v.w += a.w;
            hr[i] = v;
        }
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    const float r = rsqrtf(block_sum(ss, red) / C + eps);
    uint2* o = (uint2*)(out + t * ldo);
    for (int i = threadIdx.x; i < C4; i += blockDim.x) {
        const float4 v = hr[i];
        const float4 w = wf32 ? ((const float4*)wv)[i] : h4f(((const uint2*)wv)[i]);
        __half2 a = __floats2half2_rn(v.x * r * w.x, v.y * r * w.y), b = __floats2half2_rn(v.z * r * w.z, v.w * r * w.w);
        o[i] = make_uint2(*(u32*)&a, *(u32*)&b);
    }
}

// silu_mul over 8 elements per thread (I % 8 == 0)
extern "C" __global__ void silu_mul8(const __half* __restrict__ gu, int I, long long n8, int mode, __half* __restrict__ out) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n8) return;
    const int I8 = I >> 3;
    const long long t = idx / I8, i = idx - t * I8;
    const uint4 gv = ((const uint4*)(gu + t * 2 * I))[i], uv = ((const uint4*)(gu + t * 2 * I + I))[i];
    const __half* g = (const __half*)&gv;
    const __half* u = (const __half*)&uv;
    uint4 ov;
    __half* o = (__half*)&ov;
    #pragma unroll
    for (int q = 0; q < 8; ++q) {
        const float gf = __half2float(g[q]);
        o[q] = mode == 0 ? __hmul(__float2half(silu(gf)), u[q]) : __float2half(silu(gf) * __half2float(u[q]));
    }
    ((uint4*)out)[idx] = ov;
}

// ---------------------------------------------------------------- silu_mul
// gu [T, 2I] fp16 (gate | up), out [T, I] fp16. mode 0: half(silu(g)) * u in fp16 (Qwen2 reference);
// mode 1: half(silu(g) * u) in fp32 (Qwen3.5 reference)
extern "C" __global__ void silu_mul(const __half* __restrict__ gu, int I, long long n, int mode, __half* __restrict__ out) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const long long t = idx / I, i = idx - t * I;
    const float g = __half2float(gu[t * 2 * I + i]);
    const __half u = gu[t * 2 * I + I + i];
    out[idx] = mode == 0 ? __hmul(__float2half(silu(g)), u) : __float2half(silu(g) * __half2float(u));
}

// ---------------------------------------------------------------- rope_qkv (Qwen2)
// qkv [T, (NH + 2 NKV) * HD] fp16; biases fp16 (nullable); cs, sn [T, HD/2] fp16 tables.
// grid (T, NH + 2 NKV), block HD/2. q -> qo [NH, T, HD]; k, v -> kc / vc [NKV, maxlen, HD] at p0 + t.
extern "C" __global__ void rope_qkv(const __half* __restrict__ qkv, const __half* __restrict__ bq,
                                    const __half* __restrict__ bk, const __half* __restrict__ bv,
                                    const __half* __restrict__ cs, const __half* __restrict__ sn,
                                    int T, int NH, int NKV, int HD, int p0, int maxlen,
                                    __half* __restrict__ qo, __half* __restrict__ kc, __half* __restrict__ vc) {
    const int t = blockIdx.x, hh = blockIdx.y, half = HD / 2;
    const __half* row = qkv + (size_t)t * (NH + 2 * NKV) * HD + (size_t)hh * HD;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        if (hh >= NH + NKV) {                       // v: bias only
            const int kv = hh - NH - NKV;
            __half* dst = vc + ((size_t)kv * maxlen + p0 + t) * HD;
            const __half* b = bv ? bv + kv * HD : nullptr;
            dst[i] = b ? __hadd(row[i], b[i]) : row[i];
            dst[i + half] = b ? __hadd(row[i + half], b[i + half]) : row[i + half];
            continue;
        }
        const bool isq = hh < NH;
        const __half* b = isq ? (bq ? bq + hh * HD : nullptr) : (bk ? bk + (hh - NH) * HD : nullptr);
        const __half a1 = b ? __hadd(row[i], b[i]) : row[i];
        const __half a2 = b ? __hadd(row[i + half], b[i + half]) : row[i + half];
        const float x1 = __half2float(a1), x2 = __half2float(a2);
        const float c = __half2float(cs[(size_t)t * half + i]), s = __half2float(sn[(size_t)t * half + i]);
        const __half o1 = __float2half(__fsub_rn(__fmul_rn(x1, c), __fmul_rn(x2, s)));
        const __half o2 = __float2half(__fadd_rn(__fmul_rn(x2, c), __fmul_rn(x1, s)));
        __half* dst = isq ? qo + ((size_t)hh * T + t) * HD
                          : kc + ((size_t)(hh - NH) * maxlen + p0 + t) * HD;
        dst[i] = o1;
        dst[i + half] = o2;
    }
}

// ---------------------------------------------------------------- attn_prep (Qwen3.5)
// qkv rows [T, ld] fp16 holding q|gate per head (NH * 2HD), then k (NKV * HD), then v (NKV * HD);
// qn / kn: 1 + w (fp32 [HD]); cs, sn [T, ROT/2] fp32. grid (T, NH + 2 NKV), block 128.
extern "C" __global__ void attn_prep(const __half* __restrict__ qkv, int ld, const float* __restrict__ qn,
                                     const float* __restrict__ kn, const float* __restrict__ cs,
                                     const float* __restrict__ sn, int T, int NH, int NKV, int HD, int ROT,
                                     float eps, int p0, int maxlen, __half* __restrict__ qo,
                                     __half* __restrict__ kc, __half* __restrict__ vc) {
    extern __shared__ float xs[];                  // HD floats
    __shared__ float red[32];
    const int t = blockIdx.x, hh = blockIdx.y;
    const __half* row = qkv + (size_t)t * ld;
    if (hh >= NH + NKV) {
        const int kv = hh - NH - NKV;
        const __half* src = row + NH * 2 * HD + NKV * HD + kv * HD;
        __half* dst = vc + ((size_t)kv * maxlen + p0 + t) * HD;
        for (int i = threadIdx.x; i < HD; i += blockDim.x) dst[i] = src[i];
        return;
    }
    const bool isq = hh < NH;
    const __half* src = isq ? row + hh * 2 * HD : row + NH * 2 * HD + (hh - NH) * HD;
    const float* w = isq ? qn : kn;
    float ss = 0.f;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        const float v = __half2float(src[i]);
        xs[i] = v;
        ss += v * v;
    }
    const float r = rsqrtf(block_sum(ss, red) / HD + eps);    // block_sum syncs: xs is complete
    __half* dst = isq ? qo + ((size_t)hh * T + t) * HD : kc + ((size_t)(hh - NH) * maxlen + p0 + t) * HD;
    const int h2 = ROT / 2;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        float v = xs[i] * r * w[i];
        if (i < ROT) {
            const int j = i < h2 ? i : i - h2;
            const float c = cs[(size_t)t * h2 + j], s = sn[(size_t)t * h2 + j];
            const float o = i < h2 ? -(xs[i + h2] * r * w[i + h2]) : xs[i - h2] * r * w[i - h2];
            v = v * c + o * s;
        }
        dst[i] = __float2half(v);
    }
}

// o [NH, T, HD] fp16 (attention output), gate from qkv rows (q|gate per head); out [T, NH*HD] fp16
extern "C" __global__ void gate_out(const __half* __restrict__ o, const __half* __restrict__ qkv, int ld,
                                    int T, int NH, int HD, __half* __restrict__ out) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long n = (long long)T * NH * HD;
    if (idx >= n) return;
    const int t = (int)(idx / (NH * HD)), r = (int)(idx - (long long)t * NH * HD), h = r / HD, i = r - h * HD;
    const float a = __half2float(o[((size_t)h * T + t) * HD + i]);
    const float g = __half2float(qkv[(size_t)t * ld + h * 2 * HD + HD + i]);
    out[idx] = __float2half(a * sigm(g));
}

// ---------------------------------------------------------------- conv (Qwen3.5 DeltaNet)
// x rows [T, ld] fp16 (first D columns), state [D, K] fp32 (oldest first), w [D, K] fp32.
// u [T, D] fp32 = silu(sum_m w[m] * seq[t + 1 + m]), seq = [state (K), x (T)]. grid (D/256, T/TS).
extern "C" __global__ void conv(const __half* __restrict__ x, int ld, const float* __restrict__ state,
                                const float* __restrict__ w, int D, int K, int T, int TS, float* __restrict__ u) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    const int t0 = blockIdx.y * TS, t1 = min(T, t0 + TS);
    // window held right-aligned in win[8 - K .. 7] (compile-time indices: registers, not local memory)
    float wr[8], win[8];
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        const int mm = m - (8 - K);                    // tap index, < 0: unused slot
        wr[m] = mm >= 0 ? w[c * K + mm] : 0.f;
        win[m] = 0.f;
        // window = seq[t0 + 1 .. t0 + K - 1] (the K-1 inputs before x[t0]) in slots 8-K .. 6
        if (mm >= 0 && m < 7) {
            const int s = t0 + 1 + mm;
            win[m] = s < K ? state[c * K + s] : __half2float(x[(size_t)(s - K) * ld + c]);
        }
    }
    for (int t = t0; t < t1; ++t) {
        win[7] = __half2float(x[(size_t)t * ld + c]);
        float acc = 0.f;
        #pragma unroll
        for (int m = 0; m < 8; ++m) acc += wr[m] * win[m];
        u[(size_t)t * D + c] = silu(acc);
        #pragma unroll
        for (int m = 0; m < 7; ++m) win[m] = win[m + 1];
    }
}

// new state = last K of seq
extern "C" __global__ void conv_state(const __half* __restrict__ x, int ld, float* __restrict__ state, int D, int K, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float nv[8];
    for (int j = 0; j < K; ++j) {
        const int s = T + j;
        nv[j] = s < K ? state[c * K + s] : __half2float(x[(size_t)(s - K) * ld + c]);
    }
    for (int j = 0; j < K; ++j) state[c * K + j] = nv[j];
}

// ---------------------------------------------------------------- gdn_prep
// u [T, D] fp32 (conv output: q (HK*DK) | k (HK*DK) | v (HV*DV)); ab [T, 2HV] fp32 (a | b).
// Outputs head-major, Tp rows (Tp >= T, zero padded): q, k [HV, Tp, DK]; v [HV, Tp, DV]; g, beta [HV, Tp].
// grid (Tp, HV), block 128.
extern "C" __global__ void gdn_prep(const float* __restrict__ u, const float* __restrict__ ab,
                                    const float* __restrict__ A, const float* __restrict__ dtb,
                                    int T, int Tp, int HK, int HV, int DK, int DV, float qscale,
                                    float* __restrict__ qo, float* __restrict__ ko, float* __restrict__ vo,
                                    float* __restrict__ go, float* __restrict__ bo) {
    __shared__ float red[32];
    const int t = blockIdx.x, h = blockIdx.y, hk = h / (HV / HK);
    const size_t oq = ((size_t)h * Tp + t) * DK, ov = ((size_t)h * Tp + t) * DV;
    if (t >= T) {
        for (int i = threadIdx.x; i < DK; i += blockDim.x) { qo[oq + i] = 0.f; ko[oq + i] = 0.f; }
        for (int i = threadIdx.x; i < DV; i += blockDim.x) vo[ov + i] = 0.f;
        if (threadIdx.x == 0) { go[(size_t)h * Tp + t] = 0.f; bo[(size_t)h * Tp + t] = 0.f; }
        return;
    }
    const int D = 2 * HK * DK + HV * DV;
    const float* ur = u + (size_t)t * D;
    const float* qr = ur + hk * DK;
    const float* kr = ur + HK * DK + hk * DK;
    float sq = 0.f, sk = 0.f;
    for (int i = threadIdx.x; i < DK; i += blockDim.x) { sq += qr[i] * qr[i]; sk += kr[i] * kr[i]; }
    const float rq = rsqrtf(block_sum(sq, red) + 1e-6f) * qscale;
    const float rk = rsqrtf(block_sum(sk, red) + 1e-6f);
    for (int i = threadIdx.x; i < DK; i += blockDim.x) { qo[oq + i] = qr[i] * rq; ko[oq + i] = kr[i] * rk; }
    const float* vr = ur + 2 * HK * DK + h * DV;
    for (int i = threadIdx.x; i < DV; i += blockDim.x) vo[ov + i] = vr[i];
    if (threadIdx.x == 0) {
        const float a = ab[(size_t)t * 2 * HV + h], b = ab[(size_t)t * 2 * HV + HV + h];
        go[(size_t)h * Tp + t] = -A[h] * softplus(a + dtb[h]);
        bo[(size_t)h * Tp + t] = sigm(b);
    }
}

// ---------------------------------------------------------------- gdn_scan
// Per head h and value slice [v0, v0 + BV): S [DK, BV] in shared memory, chunks of C = 64 rows.
//   W, QG, KD: [H, N, C, DK] (w = k_cumdecay, q e^g, k e^{g_last - g}); U: [H, N, C, DV]; DEC: [H, N] e^{g_last}
//   S: [H, DK, DV] fp32 (in: initial, out: final); VN, OI: [H, N, C, DV] (v_new, q e^g S)
// block 256 threads, grid H * DV / BV.
#define GC 64
#define KT 32
extern "C" __global__ void __launch_bounds__(256) gdn_scan(
        const float* __restrict__ W, const float* __restrict__ QG, const float* __restrict__ KD,
        const float* __restrict__ U, const float* __restrict__ DEC, float* __restrict__ S,
        float* __restrict__ VN, float* __restrict__ OI, int N) {
    constexpr int CG = SBV / 4;                 // column groups of 4
    constexpr int RG = 256 / CG;               // row groups
    constexpr int RP = 2 * GC / RG;            // phase 1 rows per thread (of 2C)
    constexpr int RK = SDK / RG;                // phase 2 rows per thread (of SDK)
    __shared__ __align__(16) float Ss[SDK * SBV];
    __shared__ __align__(16) float st[2 * GC * (KT + 1) > GC / 2 * SDK ? 2 * GC * (KT + 1) : GC / 2 * SDK];
    __shared__ __align__(16) float vns[GC * SBV];
    const int nvs = SDV / SBV;
    const int h = blockIdx.x / nvs, v0 = (blockIdx.x - h * nvs) * SBV;
    const int tid = threadIdx.x, cg = tid % CG, rg = tid / CG;
    float* Sg = S + (size_t)h * SDK * SDV;
    for (int i = tid; i < SDK * SBV; i += 256) Ss[i] = Sg[(i / SBV) * SDV + v0 + i % SBV];
    __syncthreads();
    for (int n = 0; n < N; ++n) {
        const size_t cb = ((size_t)h * N + n) * GC;          // first row of this chunk
        // phase 1: P = [W; QG] (2C x SDK) @ S (SDK x SBV)
        float acc[RP][4];
        #pragma unroll
        for (int r = 0; r < RP; ++r) { acc[r][0] = acc[r][1] = acc[r][2] = acc[r][3] = 0.f; }
        for (int kt = 0; kt < SDK; kt += KT) {
            for (int i = tid; i < 2 * GC * KT / 4; i += 256) {
                const int r = i / (KT / 4), c4 = i - r * (KT / 4);
                const float* src = (r < GC ? W : QG) + (cb + (r % GC)) * SDK + kt + c4 * 4;
                const float4 v = *(const float4*)src;
                float* d = st + r * (KT + 1) + c4 * 4;
                d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
            }
            __syncthreads();
            #pragma unroll 8
            for (int kk = 0; kk < KT; ++kk) {
                const float4 s4 = *(const float4*)(Ss + (kt + kk) * SBV + cg * 4);
                #pragma unroll
                for (int r = 0; r < RP; ++r) {
                    const float a = st[(rg * RP + r) * (KT + 1) + kk];
                    acc[r][0] += a * s4.x; acc[r][1] += a * s4.y; acc[r][2] += a * s4.z; acc[r][3] += a * s4.w;
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int r = 0; r < RP; ++r) {
            const int row = rg * RP + r;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = cg * 4 + j;
                if (row < GC) {
                    const size_t gi = (cb + row) * SDV + v0 + col;
                    const float vn = U[gi] - acc[r][j];
                    vns[row * SBV + col] = vn;
                    VN[gi] = vn;
                } else {
                    OI[(cb + row - GC) * SDV + v0 + col] = acc[r][j];
                }
            }
        }
        // phase 2: S = dec * S + KD^T (SDK x C) @ v_new (C x SBV)
        const float dec = DEC[(size_t)h * N + n];
        float a2[RK][4];
        #pragma unroll
        for (int r = 0; r < RK; ++r) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) a2[r][j] = Ss[(rg * RK + r) * SBV + cg * 4 + j] * dec;
        }
        for (int t0 = 0; t0 < GC; t0 += GC / 2) {
            __syncthreads();                   // vns complete / previous half consumed
            for (int i = tid; i < GC / 2 * SDK / 4; i += 256) {
                const int r = i / (SDK / 4), c4 = i - r * (SDK / 4);
                *(float4*)(st + r * SDK + c4 * 4) = *(const float4*)(KD + (cb + t0 + r) * SDK + c4 * 4);
            }
            __syncthreads();
            #pragma unroll 4
            for (int t = 0; t < GC / 2; ++t) {
                const float4 v4 = *(const float4*)(vns + (t0 + t) * SBV + cg * 4);
                #pragma unroll
                for (int r = 0; r < RK; ++r) {
                    const float a = st[t * SDK + rg * RK + r];
                    a2[r][0] += a * v4.x; a2[r][1] += a * v4.y; a2[r][2] += a * v4.z; a2[r][3] += a * v4.w;
                }
            }
        }
        __syncthreads();                       // every thread is done reading Ss (phase 1 / init)
        #pragma unroll
        for (int r = 0; r < RK; ++r) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) Ss[(rg * RK + r) * SBV + cg * 4 + j] = a2[r][j];
        }
        __syncthreads();
    }
    for (int i = tid; i < SDK * SBV; i += 256) Sg[(i / SBV) * SDV + v0 + i % SBV] = Ss[i];
}

// ---------------------------------------------------------------- gdn_chunk_prep / gdn_tri
// Per chunk (h, n) of C = 64 rows; q, k [H, Tp, DK], v [H, Tp, DV], g, beta [H, Tp] (Tp = N * 64).
// prep: gc = cumsum(g) within the chunk; vb = v * beta [H, N, C, DV], kbe = k * beta * e^gc;
//       qg = q e^gc, kd = k e^{gc_last - gc} [H, N, C, DK]; dec = e^{gc_last} [H, N]. grid H*N, block 256.
extern "C" __global__ void gdn_chunk_prep(const float* __restrict__ q, const float* __restrict__ k,
                                          const float* __restrict__ v, const float* __restrict__ g,
                                          const float* __restrict__ beta, int DK, int DV,
                                          float* __restrict__ gc, float* __restrict__ vb, float* __restrict__ kbe,
                                          float* __restrict__ qg,
                                          float* __restrict__ kd, float* __restrict__ dec) {
    __shared__ float sg[64], sb[64];
    const size_t c0 = (size_t)blockIdx.x * 64;                  // first row (h * Tp + n * 64)
    if (threadIdx.x < 32) {                                     // warp scan of the 64 gates
        const int l = threadIdx.x;
        float a = g[c0 + 2 * l], b = g[c0 + 2 * l + 1];
        b += a;                                                 // pair sums, then an inclusive scan
        float s = b;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float t = __shfl_up_sync(0xffffffffu, s, o);
            if (l >= o) s += t;
        }
        const float ex = s - b;                                 // exclusive prefix of the pairs
        sg[2 * l] = ex + a;
        sg[2 * l + 1] = s;
        sb[2 * l] = beta[c0 + 2 * l];
        sb[2 * l + 1] = beta[c0 + 2 * l + 1];
    }
    __syncthreads();
    const float gl = sg[63];
    if (threadIdx.x < 64) gc[c0 + threadIdx.x] = sg[threadIdx.x];
    if (threadIdx.x == 0) dec[blockIdx.x] = expf(gl);
    for (int idx = threadIdx.x; idx < 64 * DV; idx += blockDim.x) {
        const int i = idx / DV;
        const size_t o = (c0 + i) * DV + (idx - i * DV);
        vb[o] = v[o] * sb[i];
    }
    for (int idx = threadIdx.x; idx < 64 * DK; idx += blockDim.x) {
        const int i = idx / DK;
        const size_t o = (c0 + i) * DK + (idx - i * DK);
        const float e = expf(sg[i]), kk = k[o];
        kbe[o] = kk * sb[i] * e;
        qg[o] = q[o] * e;
        kd[o] = kk * expf(gl - sg[i]);
    }
}

// tri: from KK = k k^T and QK = q k^T [B, 64, 64] (B = H*N chunks), gc, beta:
//   M = beta_i KK_ij e^{gc_i - gc_j} (j < i);  Tinv = (I + M)^-1 (written over KK);
//   P = QK_ij e^{gc_i - gc_j} (j <= i, else 0) (written over QK). grid B, block 64.
extern "C" __global__ void __launch_bounds__(64) gdn_tri(float* __restrict__ KK, float* __restrict__ QK,
                                                         const float* __restrict__ gc, const float* __restrict__ beta) {
    __shared__ float M[64][65], sg[64], sb[64];       // 17 KB: several blocks per SM
    const int j = threadIdx.x;
    const size_t b = blockIdx.x;
    float* kk = KK + b * 4096;
    float* qk = QK + b * 4096;
    sg[j] = gc[b * 64 + j];
    sb[j] = beta[b * 64 + j];
    __syncthreads();
    for (int i = 0; i < 64; ++i) {                    // thread j: column j (coalesced row reads)
        const float e = j <= i ? expf(sg[i] - sg[j]) : 0.f;
        M[i][j] = j < i ? kk[i * 64 + j] * e * sb[i] : 0.f;
        qk[i * 64 + j] = qk[i * 64 + j] * e;
    }
    __syncthreads();
    // forward substitution in place, row by row: X[i][jj] = -(M[i][jj] + sum_{k=jj+1}^{i-1} M[i][k] X[k][jj])
    // for jj < i (X[k][k] = 1). Row i is read (M) before it is overwritten (X): two barriers per row.
    for (int i = 1; i < 64; ++i) {
        float s = 0.f;
        if (j < i) {
            s = M[i][j];
            for (int k2 = j + 1; k2 < i; ++k2) s += M[i][k2] * M[k2][j];
        }
        __syncthreads();
        if (j < i) M[i][j] = -s;
        __syncthreads();
    }
    for (int i = 0; i < 64; ++i) kk[i * 64 + j] = j < i ? M[i][j] : (j == i ? 1.f : 0.f);
}

// ---------------------------------------------------------------- q8_rows
// rows [s, e) of an 8-bit block-32 matrix (q int8 [R, C], sc fp16 [R, C/32]) -> fp16 [e - s, C]
extern "C" __global__ void q8_rows(const signed char* __restrict__ q, const __half* __restrict__ sc, int C,
                                   long long s, long long n, __half* __restrict__ out) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const long long g = s * C + idx;
    out[idx] = __float2half((float)q[g] * __half2float(sc[g >> 5]));
}

// ---------------------------------------------------------------- gated_norm
// o [HV, Tp, DV] fp32, z rows [T, ldz] fp16 (HV*DV), w [DV] fp32 -> out [T, HV*DV] fp16. grid (T, HV), block 128.
extern "C" __global__ void gated_norm(const float* __restrict__ o, int Tp, const __half* __restrict__ z, int ldz,
                                      const float* __restrict__ w, int HV, int DV, float eps, __half* __restrict__ out) {
    __shared__ float red[32];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* orow = o + ((size_t)h * Tp + t) * DV;
    float ss = 0.f;
    for (int i = threadIdx.x; i < DV; i += blockDim.x) ss += orow[i] * orow[i];
    const float r = rsqrtf(block_sum(ss, red) / DV + eps);
    for (int i = threadIdx.x; i < DV; i += blockDim.x) {
        const float zz = __half2float(z[(size_t)t * ldz + h * DV + i]);
        out[(size_t)t * HV * DV + h * DV + i] = __float2half(orow[i] * r * w[i] * silu(zz));
    }
}
"""

_mods: dict = {}


def _mod(dk: int = 128, dv: int = 128, bv: int = 32):
    import cupy as cp
    key = (dk, dv, bv)
    if key not in _mods:
        src = f"#define SDK {dk}\n#define SDV {dv}\n#define SBV {bv}\n" + _SRC
        _mods[key] = cp.RawModule(code=src, options=("-std=c++17",))
    return _mods[key]


_fns: dict = {}
_streams: dict = {}


def stream():
    """CuPy handle of torch's current stream (cached: building one per launch costs CPU time)."""
    import cupy as cp
    ptr = torch.cuda.current_stream().cuda_stream
    s = _streams.get(ptr)
    if s is None:
        s = _streams[ptr] = cp.cuda.ExternalStream(ptr)
    return s


def _launch(name, grid, block, args, smem=0, mod=None):
    key = (id(mod), name)
    fn = _fns.get(key)
    if fn is None:
        fn = _fns[key] = (mod or _mod()).get_function(name)
    fn(grid if isinstance(grid, tuple) else (grid,), (block,), args, shared_mem=smem, stream=stream())


def _p(t):
    import cupy as cp
    return cp.uint64(0 if t is None else t.data_ptr())


def _i(v):
    import cupy as cp
    return cp.int32(v)


def _f(v):
    import cupy as cp
    return cp.float32(v)


def addnorm(h: torch.Tensor, add: torch.Tensor | None, w: torch.Tensor, eps: float,
            out: torch.Tensor | None = None) -> torch.Tensor:
    """h fp32 [T, C] (updated in place: h += add), add fp16 [T, >=C] rows (or None), w fp16/fp32 [C]
    -> fp16 [T, C] rmsnorm(h) * w."""
    T, C = h.shape
    assert h.dtype == torch.float32 and h.is_contiguous()
    if out is None:
        out = torch.empty(T, C, dtype=torch.float16, device=h.device)
    if add is not None:
        assert add.dtype == torch.float16 and add.stride(1) == 1
    lda = 0 if add is None else add.stride(0)
    vec = (C % 4 == 0 and lda % 4 == 0 and out.stride(0) % 4 == 0 and out.stride(1) == 1
           and all(t is None or t.data_ptr() % 16 == 0 for t in (h, w)) and (add is None or add.data_ptr() % 8 == 0)
           and out.data_ptr() % 8 == 0)
    n = C // 4 if vec else C
    threads = min(1024, max(32, 32 * math.ceil(min(n, 1024) / 32)))
    _launch("addnorm4" if vec else "addnorm", T, threads,
            (_p(h), _p(add), _i(lda), _p(w), _i(int(w.dtype == torch.float32)), _i(C), _f(eps), _p(out),
             _i(out.stride(0))))
    return out


def silu_mul(gu: torch.Tensor, mode: int) -> torch.Tensor:
    """gu fp16 [T, 2I] -> fp16 [T, I]; mode 0 = Qwen2 rounding, 1 = Qwen3.5 rounding."""
    T, I2 = gu.shape
    I = I2 // 2
    assert gu.is_contiguous()
    out = torch.empty(T, I, dtype=torch.float16, device=gu.device)
    n = T * I
    if I % 8 == 0 and gu.data_ptr() % 16 == 0:
        _launch("silu_mul8", (n // 8 + 255) // 256, 256, (_p(gu), _i(I), _ll(n // 8), _i(mode), _p(out)))
    else:
        _launch("silu_mul", (n + 255) // 256, 256, (_p(gu), _i(I), _ll(n), _i(mode), _p(out)))
    return out


def _ll(v):
    import cupy as cp
    return cp.int64(v)


def rope_qkv(qkv, bq, bk, bv, cs, sn, NH, NKV, HD, p0, kc, vc):
    """Qwen2: qkv fp16 [T, (NH+2NKV)*HD] -> q fp16 [NH, T, HD]; k / v written to kc / vc
    ([NKV, maxlen, HD] views of one layer) at positions p0 .. p0+T-1."""
    T = qkv.shape[0]
    assert qkv.is_contiguous() and kc.is_contiguous() and vc.is_contiguous()
    q = torch.empty(NH, T, HD, dtype=torch.float16, device=qkv.device)
    _launch("rope_qkv", (T, NH + 2 * NKV), max(32, min(HD // 2, 256)),
            (_p(qkv), _p(bq), _p(bk), _p(bv), _p(cs), _p(sn), _i(T), _i(NH), _i(NKV), _i(HD), _i(p0),
             _i(kc.shape[1]), _p(q), _p(kc), _p(vc)))
    return q


def attn_prep(qkv, qn, kn, cs, sn, NH, NKV, HD, ROT, eps, p0, kc, vc):
    """Qwen3.5: qkv fp16 rows [T, ld] (q|gate per head, k, v) -> q fp16 [NH, T, HD] (normed, roped);
    k (normed, roped) / v written to kc / vc at p0 .. p0+T-1."""
    T = qkv.shape[0]
    assert qkv.stride(1) == 1 and kc.is_contiguous() and vc.is_contiguous()
    q = torch.empty(NH, T, HD, dtype=torch.float16, device=qkv.device)
    _launch("attn_prep", (T, NH + 2 * NKV), 128,
            (_p(qkv), _i(qkv.stride(0)), _p(qn), _p(kn), _p(cs), _p(sn), _i(T), _i(NH), _i(NKV), _i(HD), _i(ROT),
             _f(eps), _i(p0), _i(kc.shape[1]), _p(q), _p(kc), _p(vc)), smem=HD * 4)
    return q


def gate_out(o, qkv, NH, HD):
    """o fp16 [NH, T, HD] contiguous, gates from qkv rows -> fp16 [T, NH*HD]."""
    T = o.shape[1]
    out = torch.empty(T, NH * HD, dtype=torch.float16, device=o.device)
    n = T * NH * HD
    _launch("gate_out", (n + 255) // 256, 256, (_p(o), _p(qkv), _i(qkv.stride(0)), _i(T), _i(NH), _i(HD), _p(out)))
    return out


def conv(x, state, w, D):
    """x fp16 rows [T, ld] (first D columns), state fp32 [D, K] (updated), w fp32 [D, K] -> u fp32 [T, D]."""
    T = x.shape[0]
    K = state.shape[1]
    assert K <= 8 and state.is_contiguous() and w.is_contiguous() and x.stride(1) == 1
    u = torch.empty(T, D, dtype=torch.float32, device=x.device)
    TS = 32
    _launch("conv", ((D + 255) // 256, (T + TS - 1) // TS), 256,
            (_p(x), _i(x.stride(0)), _p(state), _p(w), _i(D), _i(K), _i(T), _i(TS), _p(u)))
    _launch("conv_state", (D + 255) // 256, 256, (_p(x), _i(x.stride(0)), _p(state), _i(D), _i(K), _i(T)))
    return u


def gdn_prep(u, ab, A, dt_bias, HK, HV, DK, DV, chunk: int = 64):
    """-> q, k [HV, Tp, DK], v [HV, Tp, DV], g, beta [HV, Tp] fp32, Tp = T rounded up to the chunk."""
    T = u.shape[0]
    Tp = (T + chunk - 1) // chunk * chunk
    f = dict(dtype=torch.float32, device=u.device)
    q, k = torch.empty(HV, Tp, DK, **f), torch.empty(HV, Tp, DK, **f)
    v = torch.empty(HV, Tp, DV, **f)
    g, beta = torch.empty(HV, Tp, **f), torch.empty(HV, Tp, **f)
    _launch("gdn_prep", (Tp, HV), 128, (_p(u), _p(ab), _p(A), _p(dt_bias), _i(T), _i(Tp), _i(HK), _i(HV), _i(DK),
                                        _i(DV), _f(DK ** -0.5), _p(q), _p(k), _p(v), _p(g), _p(beta)))
    return q, k, v, g, beta


def scan_bv(H: int, dv: int, device=None) -> int:
    """Value-slice width of gdn_scan: 32 when that still gives >= 2 blocks per SM, else 16."""
    sms = torch.cuda.get_device_properties(device or torch.cuda.current_device()).multi_processor_count
    return 32 if H * dv // 32 >= 2 * sms else 16


def gdn_scan(w, qg, kd, u, dec, S, bv: int | None = None):
    """Sequential chunk scan (see the module doc). w, qg, kd [H, N, 64, DK]; u [H, N, 64, DV];
    dec [H, N]; S [H, DK, DV] (updated in place). Returns v_new, o_inter [H, N, 64, DV]."""
    H, N, C, DK = w.shape
    DV = u.shape[-1]
    assert C == 64 and all(t.is_contiguous() and t.dtype == torch.float32 for t in (w, qg, kd, u, dec, S))
    bv = bv or scan_bv(H, DV, w.device)
    assert DV % bv == 0 and DK % (256 // (bv // 4)) == 0
    vn, oi = torch.empty_like(u), torch.empty_like(u)
    _launch("gdn_scan", H * DV // bv, 256, (_p(w), _p(qg), _p(kd), _p(u), _p(dec), _p(S), _p(vn), _p(oi), _i(N)),
            mod=_mod(DK, DV, bv))
    return vn, oi


def chunk_gated_delta_rule(q, k, v, g, beta, S, chunk: int = 64, bv: int | None = None):
    """Chunked gated delta rule, the math of HF's torch_chunk_gated_delta_rule (q already l2-normed
    and scaled, k l2-normed): per chunk, gdn_chunk_prep (gate cumsum, scaled operands), k k^T and
    q k^T as batched matmuls, gdn_tri (decay masks and the intra-chunk inverse (I + M)^-1 that HF
    builds row by row), u / w = inverse @ (v beta | k beta e^g), then the inter-chunk recurrence in
    gdn_scan and o = o_inter + P v_new. q, k [H, Tp, DK], v [H, Tp, DV], g, beta [H, Tp] (fp32,
    zero padded to whole chunks), S [H, DK, DV] fp32 initial state (updated in place to the final
    state). Returns o [H, Tp, DV] fp32."""
    assert chunk == 64
    H, Tp, DK = q.shape
    DV = v.shape[-1]
    N = Tp // chunk
    B = H * N
    f = dict(dtype=torch.float32, device=q.device)
    gc, dec = torch.empty(H, Tp, **f), torch.empty(H, N, **f)
    vb, kbe = torch.empty(H, Tp, DV, **f), torch.empty(H, Tp, DK, **f)
    qg, kd = torch.empty(H, Tp, DK, **f), torch.empty(H, Tp, DK, **f)
    _launch("gdn_chunk_prep", B, 256, (_p(q), _p(k), _p(v), _p(g), _p(beta), _i(DK), _i(DV), _p(gc), _p(vb),
                                       _p(kbe), _p(qg), _p(kd), _p(dec)))
    qc, kc = q.view(B, chunk, DK), k.view(B, chunk, DK)
    KK = torch.bmm(kc, kc.transpose(1, 2))
    QK = torch.bmm(qc, kc.transpose(1, 2))
    _launch("gdn_tri", B, 64, (_p(KK), _p(QK), _p(gc), _p(beta)))
    uu = torch.bmm(KK, vb.view(B, chunk, DV)).view(H, N, chunk, DV)
    w = torch.bmm(KK, kbe.view(B, chunk, DK)).view(H, N, chunk, DK)
    vn, oi = gdn_scan(w, qg.view(H, N, chunk, DK), kd.view(H, N, chunk, DK), uu, dec, S, bv)
    o = torch.baddbmm(oi.view(B, chunk, DV), QK, vn.view(B, chunk, DV))
    return o.view(H, Tp, DV)


def q8_rows(q, sc, s: int, e: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """fp16 rows s:e of an 8-bit block-32 matrix (== packed.q8_rows(...).half())."""
    C = q.shape[1]
    n = (e - s) * C
    if out is None:
        out = torch.empty(e - s, C, dtype=torch.float16, device=q.device)
    out = out.view(-1)[:n].view(e - s, C)
    _launch("q8_rows", (n + 255) // 256, 256, (_p(q), _p(sc), _i(C), _ll(s), _ll(n), _p(out)))
    return out


def gated_norm(o, z, w, eps, T):
    """o fp32 [HV, Tp, DV], z fp16 rows [T, ld] -> fp16 [T, HV*DV]."""
    HV, Tp, DV = o.shape
    out = torch.empty(T, HV * DV, dtype=torch.float16, device=o.device)
    _launch("gated_norm", (T, HV), 128, (_p(o), _i(Tp), _p(z), _i(z.stride(0)), _p(w), _i(HV), _i(DV), _f(eps), _p(out)))
    return out
