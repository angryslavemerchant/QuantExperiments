"""Batched versions of the fused Qwen3.5 decode-step kernels (kernels.qwen35_kernels) for
shapequant.qwen35_batch: one sequence per blockIdx.y (addnorm_b: per blockIdx.x), every buffer
gaining a leading batch dimension, each sequence with its own position (ragged positions work).
The bodies are the batch-1 kernels' with per-sequence pointer offsets; delta_step_b takes the
a / b projections precomputed for the whole batch (one small GEMM instead of every (head,
sequence) block re-reading in_proj_a / in_proj_b).

  addnorm_b     grid B                 h [B, C] (+= add [B, C]); out [B, C] fp16
  conv_step_b   grid (ceil(D/256), B)  x [B, D] fp16, state [B, D, KC], u [B, D]
  delta_step_b  grid (HV, B), DV thr   u [B, D], z [B, HV*DV], ab [B, 2*HV] (a | b), S [B, HV, DK, DV]
  attn_prep_b   grid (NH + NKV, B)     qg [B, NH*2*HD], k / v [B, NKV*HD], pos [B], qr [B, NH*HD],
                                       kc / vc [B, NKV, maxlen, HD]
  attn_b        grid (NH, B)           out [B, NH*HD]
silu_mul is elementwise: the batch-1 kernel over B * inter elements.
"""

from __future__ import annotations

from . import qwen35_kernels as qk

_BSRC = r"""
extern "C" __global__ void addnorm_b(float* h, const __half* add, const float* w1p, int C, float eps, __half* out) {
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
    for (int i = threadIdx.x; i < C; i += blockDim.x) out[i] = __float2half(h[i] * r * w1p[i]);
}

extern "C" __global__ void conv_step_b(const __half* x, float* state, const float* w, int D, float* u) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    const size_t bb = blockIdx.y;
    x += bb * D; u += bb * D;
    float* s = state + (bb * D + c) * KC;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < KC - 1; ++k) { const float v = s[k + 1]; s[k] = v; acc += w[c * KC + k] * v; }
    const float xv = __half2float(x[c]);
    s[KC - 1] = xv;
    acc += w[c * KC + KC - 1] * xv;
    u[c] = silu(acc);
}

extern "C" __global__ void __launch_bounds__(DV) delta_step_b(
        const float* u, const __half* z, const float* ab, int D, const float* A, const float* dt_bias,
        const float* gnorm, float* S, float eps, __half* out) {
    __shared__ float red[32];
    __shared__ float qs[DK], ks[DK];
    const int h = blockIdx.x, t = threadIdx.x, kh = h / (HV / HK);
    const size_t bb = blockIdx.y;
    u += bb * D; z += bb * HV * DV; ab += bb * 2 * HV; S += bb * HV * DK * DV; out += bb * HV * DV;
    const float a = ab[h], b = ab[HV + h];
    const float decay = __expf(-A[h] * softplus(a + dt_bias[h]));
    const float beta = sigm(b);
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
    const float r = rsqrtf(block_sum(o * o, red) / DV + eps);
    out[h * DV + t] = __float2half(o * r * gnorm[t] * silu(__half2float(z[h * DV + t])));
}

extern "C" __global__ void attn_prep_b(const __half* qg, const __half* k, const __half* v, const float* qn,
        const float* kn, const float* inv_freq, const long long* pos_p, float eps, float* qr,
        __half* kc, __half* vc, int maxlen) {
    __shared__ float red[32];
    __shared__ float buf[HD];
    const int b = blockIdx.x, t = threadIdx.x;
    const size_t bb = blockIdx.y;
    qg += bb * NH * 2 * HD; k += bb * NKV * HD; v += bb * NKV * HD; qr += bb * NH * HD;
    kc += bb * NKV * maxlen * HD; vc += bb * NKV * maxlen * HD;
    const long long pos = pos_p[bb];
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

extern "C" __global__ void __launch_bounds__(256) attn_b(const float* qr, const __half* qg, const __half* kc,
        const __half* vc, const long long* pos_p, int maxlen, __half* out) {
    constexpr int PL = HD / 32;
    __shared__ float wm[8], wl[8];
    __shared__ float wo[8][HD];
    const int h = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5, kvh = h / (NH / NKV);
    const size_t bb = blockIdx.y;
    qr += bb * NH * HD; qg += bb * NH * 2 * HD; out += bb * NH * HD;
    kc += bb * NKV * maxlen * HD; vc += bb * NKV * maxlen * HD;
    const int n = (int)pos_p[bb] + 1;
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
"""

_mods: dict = {}


def module(c):
    """Batch-1 + batched kernels for engine config c (qwen35_runtime._cfg)."""
    import cupy as cp
    key = (c.hk, c.hv, c.dk, c.dv, c.conv_k, c.n_heads, c.n_kv, c.head_dim, c.rot)
    if key not in _mods:
        defs = (f"#define HK {c.hk}\n#define HV {c.hv}\n#define DK {c.dk}\n#define DV {c.dv}\n#define KC {c.conv_k}\n"
                f"#define NH {c.n_heads}\n#define NKV {c.n_kv}\n#define HD {c.head_dim}\n#define ROT {c.rot}\n")
        _mods[key] = cp.RawModule(code=defs + qk._SRC + _BSRC, options=("-std=c++17",))
    return _mods[key]
