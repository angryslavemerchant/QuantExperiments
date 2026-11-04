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
    const long long t = tok[0] % rows;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < C; i += gridDim.x * blockDim.x)
        h[i] = __half2float(E[t * C + i]);
}

// h_out = h_in + add (add may be null); out[m, i] = rmsnorm(h_out)[perm[m, i]] (perm null: M = 1,
// identity). Every block recomputes the norm (C is small) and writes its share of the M copies;
// block 0 writes h_out (ping-pong buffers, so no block reads a half-updated residual).
extern "C" __global__ void addnorm_perm(const float* h_in, float* h_out, const __half* add, const __half* w,
                                        const unsigned short* perms, int M, int C, float eps, __half* out) {
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
    extern __shared__ __half xs[];
    for (int i = threadIdx.x; i < C; i += blockDim.x) xs[i] = x[i];
    gather_out(xs, perms, C, MC, out);
}

extern "C" __global__ void silu_mul_perm(const __half* g, const __half* u, const unsigned short* perms, int C, int MC,
                                         __half* out) {
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

    def __init__(self, rl, threads=512, blocks_per_sm=1, pipe=True):
        self.rl, self.R, self.C = rl, rl.R, rl.nch * 4
        assert self.C < 32768
        self.perms = rl.perms_l.to(torch.int16).contiguous() if rl.has_mask else None   # read as uint16
        self.threads, self.bps, self.pipe = threads, blocks_per_sm, pipe

    @classmethod
    def random(cls, R, C, bits, perms, g: torch.Generator, ru=1, dev="cuda", **launch):
        """Random tags/codebook (speed only). perms: int64 [M, C] column orders (shared within a group)."""
        from . import proto_runs as pr
        M = perms.shape[0]
        K = 1 << bits
        rl = object.__new__(pr.RunsLinear)
        rl.ru, rl.R, rl.nch, rl.bits = ru, R, C // 4, bits
        rl.words = torch.randint(-2 ** 31, 2 ** 31 - 1, (R, pr.words_per_row(C // 4, bits)), dtype=torch.int32,
                                 device=dev, generator=g)
        rl.table, rl.yscale = pr.scaled_table((torch.randn(K, 4, device=dev, generator=g) * 0.5).half(),
                                              pr.TABLE_MAX)
        rl.row_scale = (torch.rand(R, device=dev, generator=g) * 0.02 + 0.01).half()
        rl.book_off = torch.zeros(R, dtype=torch.int32, device=dev)
        rl.has_mask = M > 1
        rl.row_mask = (torch.randint(0, M, (R,), device=dev, generator=g) if M > 1
                       else torch.zeros(R, dtype=torch.int64, device=dev)).to(torch.int32)
        rl.perms_l = perms.to(dev)
        rl.tiles = pr.make_tiles(rl.row_mask, ru)
        return cls(rl, **launch)

    def run(self, xp, y):
        from . import proto_runs as pr
        pr.gemv(self.rl, xp, y, xp=xp, threads=self.threads, blocks_per_sm=self.bps, pipe=self.pipe)

    def nbytes(self):
        return self.rl.stored_bytes() + self.rl.R * 2


# ---------------------------------------------------------------- engine

_mods: dict = {}


def _fn(name, cfg: Cfg):
    import cupy as cp
    key = (cfg.n_heads, cfg.n_kv, cfg.head_dim, cfg.max_len)
    if key not in _mods:
        defs = f"#define NH {cfg.n_heads}\n#define NKV {cfg.n_kv}\n#define HD {cfg.head_dim}\n#define MAXLEN {cfg.max_len}\n"
        _mods[key] = cp.RawModule(code=defs + _SRC)
    return _mods[key].get_function(name)


def _ptr(t):
    import cupy as cp
    return cp.uint64(0 if t is None else t.data_ptr())


class DecodeEngine:
    """layers: list of dicts with ln1, ln2 (fp16 [hidden]), q, k, v, o, gate, up, down (linears),
    bq, bk, bv (fp16 bias or None). Linears in a group (q/k/v; gate/up) must share `perms`."""

    def __init__(self, cfg: Cfg, embed: torch.Tensor, layers: list, final_norm: torch.Tensor, lm_head, dev="cuda"):
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
        self.yq, self.yk, self.yv = (torch.empty(n, **f16) for n in (NH * HD, NKV * HD, NKV * HD))
        self.qr, self.ao, self.yo = torch.empty(NH * HD, **f16), torch.empty(NH * HD, **f16), torch.empty(H, **f16)
        self.yg, self.yu, self.yd = torch.empty(cfg.inter, **f16), torch.empty(cfg.inter, **f16), torch.empty(H, **f16)
        self.logits = torch.empty(lm_head.R, **f16)
        self.kc = torch.zeros(len(layers), NKV, cfg.max_len, HD, **f16)
        self.vc = torch.zeros(len(layers), NKV, cfg.max_len, HD, **f16)
        self.tok = torch.zeros(1, dtype=torch.int64, device=dev)
        self.pos = torch.zeros(1, dtype=torch.int32, device=dev)
        self.graph = None
        self.only_linears = False

    # -- fused helpers
    def _launch(self, name, grid, block, args, smem=0):
        import cupy as cp
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
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
            for L in self.layers:
                for n, y in (("q", self.yq), ("k", self.yk), ("v", self.yv), ("o", self.yo), ("gate", self.yg),
                             ("up", self.yu), ("down", self.yd)):
                    L[n].run(self.xp, y)
            self.lm_head.run(self.xp, self.logits)
            return
        self.cur = 0
        self._launch("embed", 16, 256, (_ptr(self.embed), _ptr(self.tok), cp.int32(self.embed.shape[0]),
                                        cp.int32(cfg.hidden), _ptr(self.hb[0])))
        add = None
        for li, L in enumerate(self.layers):
            self._addnorm(add, L["ln1"], L["q"])
            L["q"].run(self.xp, self.yq)
            L["k"].run(self.xp, self.yk)
            L["v"].run(self.xp, self.yv)
            kc, vc = self.kc[li], self.vc[li]
            self._launch("rope_cache", cfg.n_heads + cfg.n_kv, cfg.head_dim // 2,
                         (_ptr(self.yq), _ptr(self.yk), _ptr(self.yv), _ptr(L.get("bq")), _ptr(L.get("bk")),
                          _ptr(L.get("bv")), _ptr(self.pos), cp.float32(cfg.rope_theta), _ptr(self.qr),
                          _ptr(kc), _ptr(vc)))
            self._launch("attn", cfg.n_heads, cfg.head_dim,
                         (_ptr(self.qr), _ptr(kc), _ptr(vc), _ptr(self.pos), cp.float32(cfg.head_dim ** -0.5),
                          _ptr(self.ao)))
            self._elementwise("perm_copy", (self.ao,), L["o"])
            L["o"].run(self.xp, self.yo)
            self._addnorm(self.yo, L["ln2"], L["gate"])
            L["gate"].run(self.xp, self.yg)
            L["up"].run(self.xp, self.yu)
            self._elementwise("silu_mul_perm", (self.yg, self.yu), L["down"])
            L["down"].run(self.xp, self.yd)
            add = self.yd
        self._addnorm(add, self.final_norm, self.lm_head)
        self.lm_head.run(self.xp, self.logits)
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
                  small=(0, 1), pipe=True, embed_rows: int = 32768, seed: int = 0, dev="cuda") -> DecodeEngine:
    """Random-weight engine (speed only). kind: fp16 | q4 | runs. Column orders are shared within
    q/k/v and gate/up; each layer/group draws its own. embed_rows: the token embedding is only read
    one row per step, so a truncated table (tok % rows) keeps memory down without changing speed."""
    from .. import masks as mk
    g = torch.Generator(device=dev).manual_seed(seed)
    gc = torch.Generator().manual_seed(seed + 1)
    H, I, NH, NKV, HD = cfg.hidden, cfg.inter, cfg.n_heads, cfg.n_kv, cfg.head_dim

    def lin(R, C, perms):
        if kind == "fp16":
            return DenseLin(torch.randn(R, C, device=dev, generator=g, dtype=torch.float16) * 0.02)
        if kind == "q4":
            return Q4Lin.random(R, C, g, dev)
        th, bps = small if R <= 4096 else large
        return RunsLin.random(R, C, bits, perms, g, ru=ru, dev=dev, threads=th, blocks_per_sm=bps, pipe=pipe)

    def perms(C):
        return mk.seeded_perms(C, M, gc) if kind == "runs" else None

    def vec(n, s=1.0, base=0.0):
        return (torch.randn(n, device=dev, generator=g) * s + base).half()

    layers = []
    for _ in range(cfg.layers):
        pa, po, pg, pd = perms(H), perms(H), perms(H), perms(I)
        layers.append(dict(ln1=vec(H, 0.1, 1.0), ln2=vec(H, 0.1, 1.0),
                           q=lin(NH * HD, H, pa), k=lin(NKV * HD, H, pa), v=lin(NKV * HD, H, pa),
                           o=lin(H, NH * HD, po), gate=lin(I, H, pg), up=lin(I, H, pg), down=lin(H, I, pd),
                           bq=vec(NH * HD, 0.1), bk=vec(NKV * HD, 0.1), bv=vec(NKV * HD, 0.1)))
        torch.cuda.empty_cache()
    head = lin(cfg.vocab, H, perms(H))
    embed = (torch.randn(min(embed_rows, cfg.vocab), H, device=dev, generator=g) * 0.02).half()
    return DecodeEngine(cfg, embed, layers, vec(H, 0.1, 1.0), head, dev)
