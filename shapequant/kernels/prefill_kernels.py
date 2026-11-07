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
    float wr[8], win[8];
    for (int m = 0; m < K; ++m) wr[m] = w[c * K + m];
    // window = seq[t0 + 1 .. t0 + K - 1] (the K-1 inputs before x[t0])
    for (int m = 0; m < K - 1; ++m) {
        const int s = t0 + 1 + m;
        win[m] = s < K ? state[c * K + s] : __half2float(x[(size_t)(s - K) * ld + c]);
    }
    for (int t = t0; t < t1; ++t) {
        win[K - 1] = __half2float(x[(size_t)t * ld + c]);
        float acc = 0.f;
        for (int m = 0; m < K; ++m) acc += wr[m] * win[m];
        u[(size_t)t * D + c] = silu(acc);
        for (int m = 0; m < K - 1; ++m) win[m] = win[m + 1];
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
    threads = min(1024, max(32, 32 * math.ceil(min(C, 1024) / 32)))
    _launch("addnorm", T, threads, (_p(h), _p(add), _i(0 if add is None else add.stride(0)), _p(w),
                                     _i(int(w.dtype == torch.float32)), _i(C), _f(eps), _p(out), _i(out.stride(0))))
    return out


def silu_mul(gu: torch.Tensor, mode: int) -> torch.Tensor:
    """gu fp16 [T, 2I] -> fp16 [T, I]; mode 0 = Qwen2 rounding, 1 = Qwen3.5 rounding."""
    T, I2 = gu.shape
    I = I2 // 2
    out = torch.empty(T, I, dtype=torch.float16, device=gu.device)
    n = T * I
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
    and scaled, k l2-normed) with the intra-chunk inverse as one triangular solve and the
    inter-chunk recurrence in gdn_scan. q, k [H, Tp, DK], v [H, Tp, DV], g, beta [H, Tp] (fp32,
    zero padded to whole chunks), S [H, DK, DV] fp32 initial state (updated in place to the final
    state). Returns o [H, Tp, DV] fp32."""
    H, Tp, DK = q.shape
    DV = v.shape[-1]
    N = Tp // chunk
    q, k, v = q.view(H, N, chunk, DK), k.view(H, N, chunk, DK), v.view(H, N, chunk, DV)
    g, beta = g.view(H, N, chunk), beta.view(H, N, chunk)
    gc = g.cumsum(-1)
    ar = torch.arange(chunk, device=q.device)
    lower = ar[:, None] >= ar[None, :]
    D = torch.exp((gc[..., :, None] - gc[..., None, :]).masked_fill(~lower, float("-inf")))   # [H,N,C,C]
    kb = k * beta[..., None]
    M = (kb @ k.transpose(-1, -2)) * D
    A = torch.tril(M, -1) + torch.eye(chunk, device=q.device, dtype=q.dtype)
    eg = gc.exp()
    rhs = torch.cat([v * beta[..., None], kb * eg[..., None]], -1)
    sol = torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True)
    uu, w = sol[..., :DV].contiguous(), sol[..., DV:].contiguous()
    gl = gc[..., -1:]
    qg = (q * eg[..., None]).contiguous()
    kd = (k * (gl - gc).exp()[..., None]).contiguous()
    dec = gl[..., 0].exp().contiguous()
    vn, oi = gdn_scan(w, qg, kd, uu, dec, S, bv)
    o = oi + ((q @ k.transpose(-1, -2)) * D) @ vn
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
