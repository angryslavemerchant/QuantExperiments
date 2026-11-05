"""Batch-1 decode engine for Qwen2-architecture models, for end-to-end tokens/s.

Every linear is pluggable (fp16 cuBLAS, our Q4_0-like reference kernel, the
lane-interleaved run kernel), and everything else is a handful of fused CUDA
kernels so the step is ~13 launches per layer, close to what llama.cpp runs:

  embed -> per layer: addnorm_perm -> q,k,v -> rope_cache -> attn -> perm -> o
                      -> addnorm_perm -> gate,up -> silu_mul_perm -> down
        -> addnorm_perm -> lm_head -> argmax (greedy) -> pos += 1

addnorm_perm / perm / silu_mul_perm write the M column-order copies of a
linear's input directly ([M, C] fp16), so the run kernel needs no separate
gather. Matrices that read the same input (q/k/v, gate/up) must share their
column orders. The whole step is captured in one CUDA graph (token and
position live on the device), as llama-bench tg does.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .pdl import PDL_DEFS

_SRC = r"""
#include <cuda_fp16.h>

__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < nw ? red[lane] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    return red[0];
}

__device__ __forceinline__ float block_max(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < nw ? red[lane] : -3.0e38f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    return red[0];
}

extern "C" __global__ void embed(const __half* E, const long long* tok, int rows, int C, float* h) {
    PDL_TRIGGER();
    PDL_WAIT();     // reads the previous kernel's output right away
    const long long t = tok[0] % rows;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < C; i += gridDim.x * blockDim.x)
        h[i] = __half2float(E[t * C + i]);
}

// h_out = h_in + add (add may be null); out[m, i] = rmsnorm(h_out)[perm[m, i]] (perm null: M = 1,
// identity). Every block recomputes the norm (C is small) and writes its share of the M copies;
// block 0 writes h_out (ping-pong buffers, so no block reads a half-updated residual).
extern "C" __global__ void addnorm_perm(const float* h_in, float* h_out, const __half* add, const __half* w,
                                        const unsigned short* perms, int M, int C, float eps, __half* out) {
    PDL_TRIGGER();
    PDL_WAIT();     // reads the previous kernel's output right away
    extern __shared__ __half xs[];
    __shared__ float red[32];
    float ss = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = h_in[i];
        if (add) v += __half2float(add[i]);
        if (blockIdx.x == 0) h_out[i] = v;
        ss += v * v;
    }
    const float rstd = rsqrtf(block_sum(ss, red) / C + eps);
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = h_in[i];
        if (add) v += __half2float(add[i]);
        xs[i] = __float2half(v * rstd * __half2float(w[i]));
    }
    __syncthreads();
    const int MC = perms ? M * C : C;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < MC; k += gridDim.x * blockDim.x)
        out[k] = xs[perms ? perms[k] : k];
}

// out[m, i] = src[perm[m, i]] with src = x (perm_copy) or silu(g) * u (silu_mul_perm). Each block
// stages the whole source vector in shared memory (random gathers from global memory cost one
// sector per element) and writes its share of the M copies.
__device__ __forceinline__ void gather_out(const __half* xs, const unsigned short* perms, int C, int MC, __half* out) {
    __syncthreads();
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < MC; k += gridDim.x * blockDim.x)
        out[k] = xs[perms ? perms[k] : k];
}

extern "C" __global__ void perm_copy(const __half* x, const unsigned short* perms, int C, int MC, __half* out) {
    PDL_TRIGGER();
    PDL_WAIT();     // reads the previous kernel's output right away
    extern __shared__ __half xs[];
    for (int i = threadIdx.x; i < C; i += blockDim.x) xs[i] = x[i];
    gather_out(xs, perms, C, MC, out);
}

extern "C" __global__ void silu_mul_perm(const __half* g, const __half* u, const unsigned short* perms, int C, int MC,
                                         __half* out) {
    PDL_TRIGGER();
    PDL_WAIT();     // reads the previous kernel's output right away
    extern __shared__ __half xs[];
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        const float a = __half2float(g[i]);
        // HF computes silu(gate) and the product in the model dtype
        xs[i] = __hmul(__float2half(a / (1.f + __expf(-a))), u[i]);
    }
    gather_out(xs, perms, C, MC, out);
}

// q, k, v (+ bias) -> rotated q, and rotated k / v written into the cache at *pos.
// Blocks 0..NH-1: query heads; NH..NH+NKV-1: kv heads. HD/2 threads.
extern "C" __global__ void rope_cache(const __half* q, const __half* k, const __half* v,
                                      const __half* bq, const __half* bk, const __half* bv,
                                      const int* pos, float theta, __half* qo, __half* kc, __half* vc) {
    PDL_TRIGGER();
    PDL_WAIT();     // reads the previous kernel's output right away
    const int b = blockIdx.x, i = threadIdx.x, half = HD / 2, p = *pos;
    const float inv = 1.f / powf(theta, (2.f * i) / HD);
    const float ang = (float)p * inv;
    const float c = __half2float(__float2half(cosf(ang))), s = __half2float(__float2half(sinf(ang)));
    if (b < NH) {
        const int o = b * HD;
        const float x1 = __half2float(__hadd(q[o + i], bq ? bq[o + i] : __float2half(0.f)));
        const float x2 = __half2float(__hadd(q[o + i + half], bq ? bq[o + i + half] : __float2half(0.f)));
        qo[o + i] = __float2half(x1 * c - x2 * s);
        qo[o + i + half] = __float2half(x2 * c + x1 * s);
    } else {
        const int kv = b - NH, o = kv * HD;
        const size_t co = ((size_t)kv * MAXLEN + p) * HD;
        const float x1 = __half2float(__hadd(k[o + i], bk ? bk[o + i] : __float2half(0.f)));
        const float x2 = __half2float(__hadd(k[o + i + half], bk ? bk[o + i + half] : __float2half(0.f)));
        kc[co + i] = __float2half(x1 * c - x2 * s);
        kc[co + i + half] = __float2half(x2 * c + x1 * s);
        vc[co + i] = __hadd(v[o + i], bv ? bv[o + i] : __float2half(0.f));
        vc[co + i + half] = __hadd(v[o + i + half], bv ? bv[o + i + half] : __float2half(0.f));
    }
}

// one block of HD (=128) threads per query head; attends over cache positions 0..*pos
extern "C" __global__ void attn(const __half* q, const __half* kc, const __half* vc, const int* pos,
                                float scale, __half* out) {
    PDL_TRIGGER();
    PDL_WAIT();     // reads the previous kernel's output right away
    __shared__ float sc[MAXLEN];
    __shared__ float red[32];
    const int h = blockIdx.x, kv = h / (NH / NKV), L = *pos + 1;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    float qv[HD / 32];
    #pragma unroll
    for (int j = 0; j < HD / 32; ++j) qv[j] = __half2float(q[h * HD + lane * (HD / 32) + j]);
    for (int t = warp; t < L; t += nw) {
        const __half* kp = kc + ((size_t)kv * MAXLEN + t) * HD + lane * (HD / 32);
        float d = 0.f;
        #pragma unroll
        for (int j = 0; j < HD / 32; ++j) d += qv[j] * __half2float(kp[j]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        if (lane == 0) sc[t] = d * scale;
    }
    __syncthreads();
    float m = -3.0e38f;
    for (int t = threadIdx.x; t < L; t += blockDim.x) m = fmaxf(m, sc[t]);
    m = block_max(m, red);
    float s = 0.f;
    for (int t = threadIdx.x; t < L; t += blockDim.x) { const float e = __expf(sc[t] - m); sc[t] = e; s += e; }
    s = block_sum(s, red);
    const int d = threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < L; ++t) acc += sc[t] * __half2float(vc[((size_t)kv * MAXLEN + t) * HD + d]);
    out[h * HD + d] = __float2half(acc / s);
}

// rope_cache + attn + the o input's column-order copies in one kernel. One block of HD threads per
// query head: it rotates its q and its group's current k itself (bias added and rounded as in
// rope_cache), the group's first block writes k/v at *pos into the cache, position *pos is read from
// shared memory (no block reads it from the cache in this launch), and the head's outputs are
// scattered into the M copies through the inverse orders: out[m*C + inv[m*C + j]] = o[j].
extern "C" __global__ void attn_rope(const __half* q, const __half* k, const __half* v,
                                     const __half* bq, const __half* bk, const __half* bv,
                                     const int* pos, float theta, __half* kc, __half* vc, float scale,
                                     const unsigned short* inv, int M, int C, __half* out) {
    PDL_TRIGGER();
    PDL_WAIT();     // reads the previous kernel's output right away
    __shared__ float sc[MAXLEN];
    __shared__ float red[32];
    __shared__ __half qs[HD], ks[HD], vs[HD];
    const int h = blockIdx.x, kv = h / (NH / NKV), p = *pos, L = p + 1, i = threadIdx.x, half = HD / 2;
    const __half z = __float2half(0.f);
    if (i < half) {
        const float inv_f = 1.f / powf(theta, (2.f * i) / HD);
        const float ang = (float)p * inv_f;
        const float c = __half2float(__float2half(cosf(ang))), s = __half2float(__float2half(sinf(ang)));
        const int o = h * HD;
        float x1 = __half2float(__hadd(q[o + i], bq ? bq[o + i] : z));
        float x2 = __half2float(__hadd(q[o + i + half], bq ? bq[o + i + half] : z));
        qs[i] = __float2half(x1 * c - x2 * s);
        qs[i + half] = __float2half(x2 * c + x1 * s);
        const int ok = kv * HD;
        x1 = __half2float(__hadd(k[ok + i], bk ? bk[ok + i] : z));
        x2 = __half2float(__hadd(k[ok + i + half], bk ? bk[ok + i + half] : z));
        ks[i] = __float2half(x1 * c - x2 * s);
        ks[i + half] = __float2half(x2 * c + x1 * s);
    }
    vs[i] = __hadd(v[kv * HD + i], bv ? bv[kv * HD + i] : z);
    __syncthreads();
    if (h % (NH / NKV) == 0) {
        const size_t co = ((size_t)kv * MAXLEN + p) * HD;
        kc[co + i] = ks[i];
        vc[co + i] = vs[i];
    }
    const int lane = i & 31, warp = i >> 5, nw = blockDim.x >> 5;
    float qv[HD / 32];
    #pragma unroll
    for (int j = 0; j < HD / 32; ++j) qv[j] = __half2float(qs[lane * (HD / 32) + j]);
    for (int t = warp; t < L; t += nw) {
        const __half* kp = (t == p ? ks : kc + ((size_t)kv * MAXLEN + t) * HD) + lane * (HD / 32);
        float d = 0.f;
        #pragma unroll
        for (int j = 0; j < HD / 32; ++j) d += qv[j] * __half2float(kp[j]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        if (lane == 0) sc[t] = d * scale;
    }
    __syncthreads();
    float m = -3.0e38f;
    for (int t = i; t < L; t += blockDim.x) m = fmaxf(m, sc[t]);
    m = block_max(m, red);
    float s = 0.f;
    for (int t = i; t < L; t += blockDim.x) { const float e = __expf(sc[t] - m); sc[t] = e; s += e; }
    s = block_sum(s, red);
    float acc = 0.f;
    for (int t = 0; t < p; ++t) acc += sc[t] * __half2float(vc[((size_t)kv * MAXLEN + t) * HD + i]);
    acc += sc[p] * __half2float(vs[i]);
    const __half val = __float2half(acc / s);
    const int j = h * HD + i;
    if (inv) {
        for (int mm = 0; mm < M; ++mm) out[mm * C + inv[mm * C + j]] = val;
    } else {
        out[j] = val;
    }
}
"""


@dataclass
class Cfg:
    hidden: int = 3584
    n_heads: int = 28
    n_kv: int = 4
    head_dim: int = 128
    inter: int = 18944
    vocab: int = 152064
    layers: int = 28
    rope_theta: float = 1e6
    eps: float = 1e-6
    max_len: int = 512


# ---------------------------------------------------------------- linears

class DenseLin:
    """fp16 weight [R, C], cuBLAS GEMV."""

    def __init__(self, w: torch.Tensor):
        self.w = w.half().contiguous()
        self.R, self.C = self.w.shape
        self.perms = None

    def run(self, xp, y):
        torch.mv(self.w, xp[:self.C], out=y)

    def nbytes(self):
        return self.w.numel() * 2


class Q4Lin:
    """Our Q4_0-like reference kernel (4-bit + fp16 scale per 32 weights = 4.5 bpw)."""

    def __init__(self, q, sc, R, C):
        self.q, self.sc, self.R, self.C, self.perms = q, sc, R, C, None

    @classmethod
    def from_weight(cls, w: torch.Tensor, dev="cuda"):
        from . import proto_formats as pf
        q, sc, _ = pf.q4_pack(w.float().cpu())
        return cls(q.to(dev), sc.to(dev), *w.shape)

    @classmethod
    def random(cls, R, C, g: torch.Generator, dev="cuda"):
        q = torch.randint(-2 ** 31, 2 ** 31 - 1, (R, C // 8), dtype=torch.int32, device=dev, generator=g)
        sc = (torch.rand(R, C // 32, device=dev, generator=g) * 0.004 + 0.001).half()
        return cls(q, sc, R, C)

    def run(self, xp, y):
        from . import proto_formats as pf
        pf.gemv_q4(self.q, self.sc, xp, y, self.R, self.C)

    def nbytes(self):
        return self.q.numel() * 4 + self.sc.numel() * 2


class RunsLin:
    """Lane-interleaved run kernel (proto_runs). threads = 0: one block per SM (small matrices)."""

    def __init__(self, rl, threads=512, blocks_per_sm=1, pipe=False):
        self.rl, self.R, self.C = rl, rl.R, rl.nch * 4
        assert self.C < 32768
        self.perms = rl.perms_l.to(torch.int16).contiguous() if rl.has_mask else None   # read as uint16
        self.threads, self.bps, self.pipe = threads, blocks_per_sm, pipe
        self.pdl = False        # set by DecodeEngine

    @classmethod
    def random(cls, R, C, bits, perms, g: torch.Generator, ru=1, dev="cuda", G=1, **launch):
        """Random tags/codebook (speed only). perms: int64 [M, C] column orders (shared within a group).
        G: mixture books (table G*K rows, each row picks one)."""
        from . import proto_runs as pr
        M = perms.shape[0]
        K = 1 << bits
        rl = object.__new__(pr.RunsLinear)
        rl.ru, rl.R, rl.nch, rl.bits = ru, R, C // 4, bits
        rl.words = torch.randint(-2 ** 31, 2 ** 31 - 1, (R, pr.words_per_row(C // 4, bits)), dtype=torch.int32,
                                 device=dev, generator=g)
        rl.table, rl.yscale = pr.scaled_table((torch.randn(G * K, 4, device=dev, generator=g) * 0.5).half(),
                                              pr.TABLE_MAX)
        rl.row_scale = (torch.rand(R, device=dev, generator=g) * 0.02 + 0.01).half()
        rl.book_off = (torch.randint(0, G, (R,), device=dev, generator=g) * K).to(torch.int32)
        rl.has_mask = M > 1
        rl.row_mask = (torch.randint(0, M, (R,), device=dev, generator=g) if M > 1
                       else torch.zeros(R, dtype=torch.int64, device=dev)).to(torch.int32)
        rl.perms_l = perms.to(dev)
        rl.tiles = pr.make_tiles(rl.row_mask, ru)
        return cls(rl, **launch)

    def run(self, xp, y):
        from . import proto_runs as pr
        pr.gemv(self.rl, xp, y, xp=xp, threads=self.threads, blocks_per_sm=self.bps, pipe=self.pipe, pdl=self.pdl)

    def nbytes(self):
        return self.rl.stored_bytes() + self.rl.R * 2 + self.rl.table.numel() * 2


def mergeable(a, b) -> bool:
    """Two linears can run as one launch: run kernel, same tag bits / width / RU / column orders."""
    return (isinstance(a, RunsLin) and isinstance(b, RunsLin) and a.rl.bits == b.rl.bits and a.C == b.C
            and a.rl.ru == b.rl.ru and a.rl.has_mask == b.rl.has_mask
            and (a.perms is b.perms or (a.perms is not None and b.perms is not None and torch.equal(a.perms, b.perms))))


def merge_runs(lins: list, threads: int, blocks_per_sm: int) -> RunsLin:
    """One RunsLin over the stacked rows of `lins` (all mergeable): output rows are concatenated.
    Each matrix's books become books of the merged table (book_off shifted). The tables keep their
    own power-of-two scales; the differences go into the fp16 row scales (exact powers of two), so
    the merged launch computes bit-identical outputs."""
    import math
    from . import proto_runs as pr
    r0 = lins[0].rl
    ks = [round(math.log2(float(l.rl.yscale))) for l in lins]
    k = max(ks)
    tabs, offs, rss, base = [], [], [], 0
    for l, ki in zip(lins, ks):
        tabs.append(l.rl.table)
        offs.append(l.rl.book_off + base)
        rs = (l.rl.row_scale.float() * 2.0 ** (ki - k)).half()
        assert torch.equal(rs.float() * 2.0 ** (k - ki), l.rl.row_scale.float()), "row scale underflow"
        rss.append(rs)
        base += l.rl.table.shape[0]
    rl = object.__new__(pr.RunsLinear)
    rl.ru, rl.R, rl.nch, rl.bits = r0.ru, sum(l.R for l in lins), r0.nch, r0.bits
    rl.words = torch.cat([l.rl.words for l in lins]).contiguous()
    rl.table = torch.cat(tabs).contiguous()
    rl.yscale = torch.tensor([2.0 ** k], dtype=torch.float32, device=r0.table.device)
    rl.row_scale = torch.cat(rss).contiguous()
    rl.book_off = torch.cat(offs).to(torch.int32).contiguous()
    rl.row_mask = torch.cat([l.rl.row_mask for l in lins]).contiguous()
    rl.perms_l, rl.has_mask = r0.perms_l, r0.has_mask
    rl.tiles = pr.make_tiles(rl.row_mask, rl.ru)
    off = 0
    for l in lins:      # the originals keep working, as views (no second copy of the weights)
        l.rl.words = rl.words[off:off + l.R]
        off += l.R
    return RunsLin(rl, threads=threads, blocks_per_sm=blocks_per_sm, pipe=lins[0].pipe)


# ---------------------------------------------------------------- engine

_mods: dict = {}


def _fn(name, cfg: Cfg):
    import cupy as cp
    key = (cfg.n_heads, cfg.n_kv, cfg.head_dim, cfg.max_len)
    if key not in _mods:
        defs = f"#define NH {cfg.n_heads}\n#define NKV {cfg.n_kv}\n#define HD {cfg.head_dim}\n#define MAXLEN {cfg.max_len}\n"
        _mods[key] = cp.RawModule(code=defs + PDL_DEFS + _SRC)
    return _mods[key].get_function(name)


def _ptr(t):
    import cupy as cp
    return cp.uint64(0 if t is None else t.data_ptr())


class DecodeEngine:
    """layers: list of dicts with ln1, ln2 (fp16 [hidden]), q, k, v, o, gate, up, down (linears),
    bq, bk, bv (fp16 bias or None). Linears in a group (q/k/v; gate/up) must share `perms`."""

    def __init__(self, cfg: Cfg, embed: torch.Tensor, layers: list, final_norm: torch.Tensor, lm_head, dev="cuda",
                 fuse: bool = True, large=(512, 2), small=(0, 1), pdl: bool | None = None):
        """fuse: rope + attention + the o input's copies in one kernel, and linears reading the same
        input with the same tag bits (q/k/v, gate/up) merged into one launch (launch config
        `large` for merged matrices over 4096 rows, else `small`)."""
        assert cfg.head_dim == 128 and cfg.n_heads % cfg.n_kv == 0 and cfg.max_len <= 8192
        self.cfg, self.embed, self.layers, self.final_norm, self.lm_head = cfg, embed, layers, final_norm, lm_head
        for L in layers:
            for grp in (("q", "k", "v"), ("gate", "up")):
                p0 = L[grp[0]].perms
                for n in grp[1:]:
                    p = L[n].perms
                    assert (p is None) == (p0 is None) and (p is None or p is p0 or torch.equal(p, p0)), \
                        f"{grp} must share column orders"
        H, NH, NKV, HD = cfg.hidden, cfg.n_heads, cfg.n_kv, cfg.head_dim
        f16 = dict(dtype=torch.float16, device=dev)
        mx = lambda lins: max((l.perms.shape[0] if l.perms is not None else 1) * l.C for l in lins)  # noqa: E731
        all_lins = [l for L in layers for l in (L["q"], L["o"], L["gate"], L["down"])] + [lm_head]
        self.xp = torch.empty(mx(all_lins), **f16)
        self.hb = [torch.empty(H, dtype=torch.float32, device=dev) for _ in range(2)]   # residual, ping-pong
        self.cur = 0
        # q|k|v and gate|up outputs are contiguous so a merged launch writes them in one go
        self.yqkv = torch.empty(NH * HD + 2 * NKV * HD, **f16)
        self.yq, self.yk, self.yv = self.yqkv[:NH * HD], self.yqkv[NH * HD:(NH + NKV) * HD], self.yqkv[(NH + NKV) * HD:]
        self.qr, self.ao, self.yo = torch.empty(NH * HD, **f16), torch.empty(NH * HD, **f16), torch.empty(H, **f16)
        self.ygu = torch.empty(2 * cfg.inter, **f16)
        self.yg, self.yu, self.yd = self.ygu[:cfg.inter], self.ygu[cfg.inter:], torch.empty(H, **f16)
        self.fuse = fuse
        from . import pdl as pdl_mod
        self.pdl = pdl_mod.supported(dev) if pdl is None else pdl
        outs = dict(q=self.yq, k=self.yk, v=self.yv, gate=self.yg, up=self.yu)
        self.groups = []    # per layer: [(linear, output view)] for the q/k/v and gate/up stages
        for L in layers:
            gl = {}
            for stage in (("q", "k", "v"), ("gate", "up")):
                runs = [[stage[0]]]
                for n in stage[1:]:
                    if fuse and mergeable(L[runs[-1][-1]], L[n]):
                        runs[-1].append(n)
                    else:
                        runs.append([n])
                plan = []
                for r in runs:
                    if len(r) == 1:
                        plan.append((L[r[0]], outs[r[0]]))
                    else:
                        R = sum(L[n].R for n in r)
                        th, bps = small if R <= 4096 else large
                        start = outs[r[0]].data_ptr()
                        base = (self.yqkv if stage[0] == "q" else self.ygu)
                        off = (start - base.data_ptr()) // 2
                        plan.append((merge_runs([L[n] for n in r], th, bps), base[off:off + R]))
                gl[stage[0]] = plan
            if fuse and L["o"].perms is not None:
                p = L["o"].perms.long()
                inv = torch.empty_like(p)
                inv.scatter_(1, p, torch.arange(p.shape[1], device=p.device).expand_as(p).contiguous())
                gl["oinv"] = inv.to(torch.int16).contiguous()
            self.groups.append(gl)
        lins = [lm_head] + [l for L in layers for l in L.values() if isinstance(l, RunsLin)]
        lins += [lin for G in self.groups for st in ("q", "gate") for lin, _ in G[st]]
        for lin in lins:
            if isinstance(lin, RunsLin):
                lin.pdl = self.pdl
        self.logits = torch.empty(lm_head.R, **f16)
        self.kc = torch.zeros(len(layers), NKV, cfg.max_len, HD, **f16)
        self.vc = torch.zeros(len(layers), NKV, cfg.max_len, HD, **f16)
        self.tok = torch.zeros(1, dtype=torch.int64, device=dev)
        self.pos = torch.zeros(1, dtype=torch.int32, device=dev)
        self.graph = None
        self.only_linears = False
        self.skip: set = set()   # op classes left out of step() (timing only)

    # -- fused helpers
    def _launch(self, name, grid, block, args, smem=0):
        import cupy as cp
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            if self.pdl:
                from . import pdl as pdl_mod
                pdl_mod.launch(_fn(name, self.cfg), grid, block, args, smem)
            else:
                _fn(name, self.cfg)((grid,), (block,), args, shared_mem=smem)

    def _addnorm(self, add, w, lin):
        import cupy as cp
        M = lin.perms.shape[0] if lin.perms is not None else 1
        sms = torch.cuda.get_device_properties(self.xp.device).multi_processor_count
        grid = max(1, min(sms, M * self.cfg.hidden // 4096))
        h_in, h_out = self.hb[self.cur], self.hb[1 - self.cur]
        self.cur = 1 - self.cur
        self._launch("addnorm_perm", grid, 1024,
                     (_ptr(h_in), _ptr(h_out), _ptr(add), _ptr(w), _ptr(lin.perms), cp.int32(M),
                      cp.int32(self.cfg.hidden), cp.float32(self.cfg.eps), _ptr(self.xp)), smem=self.cfg.hidden * 2)

    def _elementwise(self, name, srcs, lin):
        import cupy as cp
        MC = (lin.perms.shape[0] if lin.perms is not None else 1) * lin.C
        sms = torch.cuda.get_device_properties(self.xp.device).multi_processor_count
        grid = max(1, min(sms, MC // 8192))
        self._launch(name, grid, 1024, (*(_ptr(s) for s in srcs), _ptr(lin.perms), cp.int32(lin.C), cp.int32(MC),
                                        _ptr(self.xp)), smem=lin.C * 2)

    def step(self):
        """One greedy decode step (graph-capturable): reads tok/pos, writes logits, tok, pos += 1."""
        import cupy as cp
        cfg = self.cfg
        if self.only_linears:   # GEMV-only timing: the same linears in the same order, nothing else
            for L, G in zip(self.layers, self.groups):
                for lin, y in G["q"]:
                    lin.run(self.xp, y)
                L["o"].run(self.xp, self.yo)
                for lin, y in G["gate"]:
                    lin.run(self.xp, y)
                L["down"].run(self.xp, self.yd)
            self.lm_head.run(self.xp, self.logits)
            return
        self.cur = 0
        on = lambda k: k not in self.skip   # noqa: E731  (leave-one-out timing; outputs then meaningless)
        if on("embed"):
            self._launch("embed", 16, 256, (_ptr(self.embed), _ptr(self.tok), cp.int32(self.embed.shape[0]),
                                            cp.int32(cfg.hidden), _ptr(self.hb[0])))
        add = None
        for li, (L, G) in enumerate(zip(self.layers, self.groups)):
            if on("addnorm"):
                self._addnorm(add, L["ln1"], L["q"])
            if on("qkv"):
                for lin, y in G["q"]:
                    lin.run(self.xp, y)
            kc, vc = self.kc[li], self.vc[li]
            if not on("attn"):
                pass
            elif self.fuse:
                inv = G.get("oinv")
                self._launch("attn_rope", cfg.n_heads, cfg.head_dim,
                             (_ptr(self.yq), _ptr(self.yk), _ptr(self.yv), _ptr(L.get("bq")), _ptr(L.get("bk")),
                              _ptr(L.get("bv")), _ptr(self.pos), cp.float32(cfg.rope_theta), _ptr(kc), _ptr(vc),
                              cp.float32(cfg.head_dim ** -0.5), _ptr(inv),
                              cp.int32(1 if inv is None else inv.shape[0]), cp.int32(L["o"].C), _ptr(self.xp)))
            else:
                self._launch("rope_cache", cfg.n_heads + cfg.n_kv, cfg.head_dim // 2,
                             (_ptr(self.yq), _ptr(self.yk), _ptr(self.yv), _ptr(L.get("bq")), _ptr(L.get("bk")),
                              _ptr(L.get("bv")), _ptr(self.pos), cp.float32(cfg.rope_theta), _ptr(self.qr),
                              _ptr(kc), _ptr(vc)))
                self._launch("attn", cfg.n_heads, cfg.head_dim,
                             (_ptr(self.qr), _ptr(kc), _ptr(vc), _ptr(self.pos), cp.float32(cfg.head_dim ** -0.5),
                              _ptr(self.ao)))
                self._elementwise("perm_copy", (self.ao,), L["o"])
            if on("o"):
                L["o"].run(self.xp, self.yo)
            if on("addnorm"):
                self._addnorm(self.yo, L["ln2"], L["gate"])
            if on("gateup"):
                for lin, y in G["gate"]:
                    lin.run(self.xp, y)
            if on("silu"):
                self._elementwise("silu_mul_perm", (self.yg, self.yu), L["down"])
            if on("down"):
                L["down"].run(self.xp, self.yd)
            add = self.yd
        if on("addnorm"):
            self._addnorm(add, self.final_norm, self.lm_head)
        if on("lm_head"):
            self.lm_head.run(self.xp, self.logits)
        if on("argmax"):
            torch.argmax(self.logits, out=self.tok.view(()))
        self.pos.add_(1)

    def reset(self, tok: int = 0):
        self.tok.fill_(tok)
        self.pos.zero_()

    def capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):          # compile / load every kernel before capture
                self.step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.step()
        return self.graph

    def generate(self, n: int, tok: int = 0) -> list[int]:
        """Greedy-decode n tokens starting from `tok` at position 0 (eager; for checks)."""
        self.reset(tok)
        out = []
        for _ in range(n):
            self.step()
            out.append(int(self.tok))
        return out

    def time_tokens(self, n: int = 128, reps: int = 3) -> float:
        """Seconds per token, averaged over n graph replays starting from an empty context (median of reps)."""
        assert n <= self.cfg.max_len
        if self.graph is None:
            self.capture()
        ts = []
        for _ in range(reps):
            self.reset()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(n):
                self.graph.replay()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) / 1e3 / n)
        return sorted(ts)[len(ts) // 2]

    def weight_bytes(self) -> int:
        lins = [L[n] for L in self.layers for n in ("q", "k", "v", "o", "gate", "up", "down")] + [self.lm_head]
        return sum(l.nbytes() for l in lins)


# ---------------------------------------------------------------- builders

def random_engine(cfg: Cfg, kind: str, *, bits: int = 10, M: int = 16, ru: int = 1, large=(512, 1),
                  small=(0, 1), pipe=False, embed_rows: int = 32768, seed: int = 0, dev="cuda",
                  G: int = 1, alloc: dict | None = None, head_bits: int | None = None, fuse: bool = True,
                  pdl: bool | None = None) -> DecodeEngine:
    """Random-weight engine (speed only). kind: fp16 | q4 | runs. Column orders are shared within
    q/k/v and gate/up; each layer/group draws its own. embed_rows: the token embedding is only read
    one row per step, so a truncated table (tok % rows) keeps memory down without changing speed.
    runs only: G mixture books per matrix; alloc {"model.layers.i.self_attn.q_proj": K, ...} gives
    per-matrix K (missing -> `bits`); head_bits: lm_head tag bits (default `bits`, always G=1)."""
    from .. import masks as mk
    g = torch.Generator(device=dev).manual_seed(seed)
    gc = torch.Generator().manual_seed(seed + 1)
    H, I, NH, NKV, HD = cfg.hidden, cfg.inter, cfg.n_heads, cfg.n_kv, cfg.head_dim

    def lin(R, C, perms, name=None):
        if kind == "fp16":
            return DenseLin(torch.randn(R, C, device=dev, generator=g, dtype=torch.float16) * 0.02)
        if kind == "q4":
            return Q4Lin.random(R, C, g, dev)
        th, bps = small if R <= 4096 else large
        if name is None:
            b, gg = head_bits or bits, 1
        else:
            b, gg = (int(alloc[name]).bit_length() - 1 if alloc and name in alloc else bits), G
        return RunsLin.random(R, C, b, perms, g, ru=ru, dev=dev, threads=th, blocks_per_sm=bps, pipe=pipe, G=gg)

    def perms(C):
        return mk.seeded_perms(C, M, gc) if kind == "runs" else None

    def vec(n, s=1.0, base=0.0):
        return (torch.randn(n, device=dev, generator=g) * s + base).half()

    layers = []
    for i in range(cfg.layers):
        pa, po, pg, pd = perms(H), perms(H), perms(H), perms(I)
        a, m = f"model.layers.{i}.self_attn.", f"model.layers.{i}.mlp."
        layers.append(dict(ln1=vec(H, 0.1, 1.0), ln2=vec(H, 0.1, 1.0),
                           q=lin(NH * HD, H, pa, a + "q_proj"), k=lin(NKV * HD, H, pa, a + "k_proj"),
                           v=lin(NKV * HD, H, pa, a + "v_proj"), o=lin(H, NH * HD, po, a + "o_proj"),
                           gate=lin(I, H, pg, m + "gate_proj"), up=lin(I, H, pg, m + "up_proj"),
                           down=lin(H, I, pd, m + "down_proj"),
                           bq=vec(NH * HD, 0.1), bk=vec(NKV * HD, 0.1), bv=vec(NKV * HD, 0.1)))
        torch.cuda.empty_cache()
    head = lin(cfg.vocab, H, perms(H))
    embed = (torch.randn(min(embed_rows, cfg.vocab), H, device=dev, generator=g) * 0.02).half()
    return DecodeEngine(cfg, embed, layers, vec(H, 0.1, 1.0), head, dev, fuse=fuse, large=large, small=small, pdl=pdl)
