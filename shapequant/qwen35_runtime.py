"""Batch-1 decode engine for Qwen3.5 / Qwen3.8 (model_type qwen3_5: Gated DeltaNet + gated full
attention), built from a packed file.

Every compressed block linear (in_proj_qkv, in_proj_z, out_proj, q/k/v/o, gate/up/down) runs on
the run kernel (kernels.decode_e2e.RunsLin) straight from the file's codes. The rest follows HF's
Qwen3_5 modules (transformers.models.qwen3_5.modeling_qwen3_5) in plain torch ops on fixed
buffers, so one step is CUDA-graph capturable:

  linear-attention layer (3 of 4): qkv = in_proj_qkv(x); causal depthwise conv over the last 4
      qkv vectors (conv state) + silu; q, k l2-normalised (q scaled by dk^-1/2); beta = sigmoid(b),
      g = -exp(A_log) * softplus(a + dt_bias) (a, b: tiny dense projections); recurrent state
      S [Hv, dk, dv]: S *= exp(g); S += k (beta (v - S^T k))^T; o = S^T q; gated RMSNorm
      (weight w, times silu(z)); out_proj.
  full-attention layer (every 4th): q_proj gives per head [query, gate]; q_norm / k_norm
      (weight 1 + w); rope on the first rotary_dim dims (text-only MRoPE == plain rope);
      softmax attention over the KV cache; times sigmoid(gate); o_proj.
  RMSNorms on the residual use weight 1 + w (Qwen3_5RMSNorm), computed in fp32.

State: conv [n_lin, conv_dim, K] and recurrent [n_lin, Hv, dk, dv] in fp32, KV cache fp16. The
prompt is fed one token at a time (no batched prefill yet). Embedding / head: the file's 4-bit
embedding and 8-bit head (self-contained file), else fp16 embedding and an 8-bit head from `model`.
"""

from __future__ import annotations

import math
import types

import torch
import torch.nn.functional as F

from .kernels import decode_e2e as de
from .packed import PackedFile

LIN = {"qkv": "linear_attn.in_proj_qkv", "z": "linear_attn.in_proj_z", "out": "linear_attn.out_proj"}
ATT = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj"}
MLP = {"gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}


def _cfg(c, max_len: int) -> types.SimpleNamespace:
    g = (lambda k, d=None: c.get(k, d)) if isinstance(c, dict) else (lambda k, d=None: getattr(c, k, d))
    rp = g("rope_parameters") or {}
    hd = g("head_dim") or g("hidden_size") // g("num_attention_heads")
    prf = rp.get("partial_rotary_factor", g("partial_rotary_factor", 1.0))
    return types.SimpleNamespace(
        hidden=g("hidden_size"), layers=g("num_hidden_layers"), layer_types=list(g("layer_types")),
        n_heads=g("num_attention_heads"), n_kv=g("num_key_value_heads"), head_dim=hd,
        rot=int(hd * prf), theta=float(rp.get("rope_theta", g("rope_theta", 1e7))),
        hk=g("linear_num_key_heads"), hv=g("linear_num_value_heads"), dk=g("linear_key_head_dim"),
        dv=g("linear_value_head_dim"), conv_k=g("linear_conv_kernel_dim"), eps=float(g("rms_norm_eps")),
        vocab=g("vocab_size"), inter=g("intermediate_size"), max_len=max_len)


def _rms1p(x: torch.Tensor, w1p: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3_5RMSNorm in fp32; w1p = 1 + weight (fp32)."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w1p


class Qwen35Engine:
    """See the module doc. step() advances one token: reads self.tok at self.pos, writes
    self.logits (fp16 [vocab]) and self.tok (greedy next token)."""

    def __init__(self, cfg, layers: list[dict], embed, final_norm: torch.Tensor, lm_head, dev="cuda",
                 fused: bool | None = None):
        from .kernels import qwen35_kernels as qk
        self.cfg, self.layers, self.embed, self.lm_head = cfg, layers, embed, lm_head
        self.fused = qk.supported(cfg) if fused is None else fused
        self.final_norm = final_norm.float()
        f32 = dict(dtype=torch.float32, device=dev)
        c = cfg
        self.lin_idx = [i for i, t in enumerate(c.layer_types) if t == "linear_attention"]
        self.att_idx = [i for i, t in enumerate(c.layer_types) if t == "full_attention"]
        conv_dim = 2 * c.hk * c.dk + c.hv * c.dv
        self.conv = torch.zeros(len(self.lin_idx), conv_dim, c.conv_k, **f32)
        self.S = torch.zeros(len(self.lin_idx), c.hv, c.dk, c.dv, **f32)
        self.kc = torch.zeros(len(self.att_idx), c.n_kv, c.max_len, c.head_dim, dtype=torch.float16, device=dev)
        self.vc = torch.zeros_like(self.kc)
        self.tok = torch.zeros(1, dtype=torch.int64, device=dev)
        self.pos = torch.zeros(1, dtype=torch.int64, device=dev)
        self.logits = torch.zeros(c.vocab, dtype=torch.float16, device=dev)
        self.ar = torch.arange(c.max_len, device=dev)
        self.inv_freq = 1.0 / (c.theta ** (torch.arange(0, c.rot, 2, dtype=torch.float32, device=dev) / c.rot))
        mx = max(max(l.R for L in layers for l in L.values() if hasattr(l, "R")), c.hidden, c.inter)
        self._y = {k: torch.empty(mx, dtype=torch.float16, device=dev) for k in ("a", "b", "c", "d")}
        self.h = torch.zeros(c.hidden, **f32)
        self.x16 = torch.zeros(c.hidden, dtype=torch.float16, device=dev)
        self.u = torch.zeros(conv_dim, **f32)
        self.o16 = torch.zeros(max(c.hv * c.dv, c.n_heads * c.head_dim, c.inter), dtype=torch.float16, device=dev)
        self.qr = torch.zeros(c.n_heads * c.head_dim, **f32)
        self.conv_dim = conv_dim
        self.graph = None

    # -- helpers
    def _lin(self, lin, x16: torch.Tensor, key: str) -> torch.Tensor:
        y = self._y[key][:lin.R]
        lin.run(x16.contiguous(), y)
        return y

    def _embed(self) -> torch.Tensor:
        if isinstance(self.embed, de.Q4Embed):
            return self.embed.rows(self.tok % self.embed.shape[0])[0]
        return self.embed[self.tok % self.embed.shape[0]][0].float()

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        """x [heads, hd] fp32: rotate_half rope on the first rot dims at self.pos."""
        r = self.cfg.rot
        fr = self.pos.float() * self.inv_freq
        cos, sin = torch.cat([fr, fr]).cos(), torch.cat([fr, fr]).sin()
        xr = x[:, :r]
        rot = torch.cat([-xr[:, r // 2:], xr[:, :r // 2]], -1)
        return torch.cat([xr * cos + rot * sin, x[:, r:]], -1)

    def _linear_attn(self, j: int, L: dict, x16: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        qkv = self._lin(L["qkv"], x16, "a").float()
        z = self._lin(L["z"], x16, "b").float()
        b = L["wb"] @ x16.float()
        a = L["wa"] @ x16.float()
        win = torch.cat([self.conv[j, :, 1:], qkv[:, None]], 1)              # [conv_dim, K]
        self.conv[j].copy_(win)
        u = F.silu((win * L["convw"]).sum(-1))
        kd = c.hk * c.dk
        q, k, v = u[:kd].view(c.hk, c.dk), u[kd:2 * kd].view(c.hk, c.dk), u[2 * kd:].view(c.hv, c.dv)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * c.dk ** -0.5
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        if c.hv != c.hk:
            q = q.repeat_interleave(c.hv // c.hk, 0)
            k = k.repeat_interleave(c.hv // c.hk, 0)
        beta = torch.sigmoid(b)
        g = -L["A"] * F.softplus(a + L["dt_bias"])
        S = self.S[j]
        S.mul_(g.exp()[:, None, None])
        delta = (v - (S * k[:, :, None]).sum(1)) * beta[:, None]               # [hv, dv]
        S.add_(k[:, :, None] * delta[:, None, :])
        o = (S * q[:, :, None]).sum(1)                                         # [hv, dv]
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + c.eps) * L["gnorm"] * F.silu(z.view(c.hv, c.dv))
        return self._lin(L["out"], o.reshape(-1).half(), "c").float()

    def _full_attn(self, j: int, L: dict, x16: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        qg = self._lin(L["q"], x16, "a").float().view(c.n_heads, 2 * c.head_dim)
        q, gate = qg[:, :c.head_dim], qg[:, c.head_dim:]
        k = self._lin(L["k"], x16, "b").float().view(c.n_kv, c.head_dim)
        v = self._lin(L["v"], x16, "c").view(c.n_kv, c.head_dim)
        q = self._rope(_rms1p(q, L["qn"], c.eps))
        k = self._rope(_rms1p(k, L["kn"], c.eps))
        self.kc[j].index_copy_(1, self.pos, k.half()[:, None])
        self.vc[j].index_copy_(1, self.pos, v[:, None])
        rep = c.n_heads // c.n_kv
        s = torch.einsum("hgd,htd->hgt", q.view(c.n_kv, rep, c.head_dim), self.kc[j].float()) * c.head_dim ** -0.5
        s = s.masked_fill(self.ar > self.pos, float("-inf"))
        o = torch.einsum("hgt,htd->hgd", torch.softmax(s, -1), self.vc[j].float()).reshape(c.n_heads, c.head_dim)
        o = o * torch.sigmoid(gate)
        return self._lin(L["o"], o.reshape(-1).half(), "a").float()

    # -- fused path (kernels.qwen35_kernels)
    def _k(self, name, grid, block, args):
        import cupy as cp
        from .kernels import qwen35_kernels as qk
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            qk.module(self.cfg).get_function(name)((grid,), (block,), args)

    def _addnorm(self, add, w1p):
        import cupy as cp
        P = lambda t: cp.uint64(0 if t is None else t.data_ptr())   # noqa: E731
        self._k("addnorm", 1, 1024, (P(self.h), P(add), P(w1p), cp.int32(self.cfg.hidden),
                                     cp.float32(self.cfg.eps), P(self.x16)))

    def _step_fused(self):
        import cupy as cp
        c = self.cfg
        P = lambda t: cp.uint64(t.data_ptr())   # noqa: E731
        self.h.copy_(self._embed())
        add = None
        jl = ja = 0
        y = self._y
        for i, L in enumerate(self.layers):
            self._addnorm(add, L["ln1"])
            if c.layer_types[i] == "linear_attention":
                qkv = self._lin(L["qkv"], self.x16, "a")
                z = self._lin(L["z"], self.x16, "b")
                self._k("conv_step", (self.conv_dim + 255) // 256, 256,
                        (P(qkv), P(self.conv[jl]), P(L["convw"]), cp.int32(self.conv_dim), P(self.u)))
                self._k("delta_step", c.hv, c.dv,
                        (P(self.u), P(z), P(self.x16), P(L["wa"]), P(L["wb"]), P(L["A"]), P(L["dt_bias"]),
                         P(L["gnorm"]), P(self.S[jl]), cp.int32(c.hidden), cp.float32(c.eps), P(self.o16)))
                add = self._lin(L["out"], self.o16[:c.hv * c.dv], "c")
                jl += 1
            else:
                qg = self._lin(L["q"], self.x16, "a")
                k = self._lin(L["k"], self.x16, "b")
                v = self._lin(L["v"], self.x16, "c")
                self._k("attn_prep", c.n_heads + c.n_kv, c.head_dim,
                        (P(qg), P(k), P(v), P(L["qn"]), P(L["kn"]), P(self.inv_freq), P(self.pos),
                         cp.float32(c.eps), P(self.qr), P(self.kc[ja]), P(self.vc[ja]), cp.int32(c.max_len)))
                self._k("attn", c.n_heads, 256, (P(self.qr), P(qg), P(self.kc[ja]), P(self.vc[ja]), P(self.pos),
                                                 cp.int32(c.max_len), P(self.o16)))
                add = self._lin(L["o"], self.o16[:c.n_heads * c.head_dim], "c")
                ja += 1
            self._addnorm(add, L["ln2"])
            g = self._lin(L["gate"], self.x16, "a")
            u = self._lin(L["up"], self.x16, "b")
            self._k("silu_mul", (c.inter + 255) // 256, 256, (P(g), P(u), cp.int32(c.inter), P(self.o16)))
            add = self._lin(L["down"], self.o16[:c.inter], "d")
        self._addnorm(add, self.final_norm)
        self.lm_head.run(self.x16, self.logits)
        self.tok.copy_(self.logits.float().argmax().view(1))
        self.pos.add_(1)

    @torch.no_grad()
    def step(self):
        if self.fused:
            return self._step_fused()
        c = self.cfg
        h = self._embed()
        jl = ja = 0
        for i, L in enumerate(self.layers):
            x16 = _rms1p(h, L["ln1"], c.eps).half()
            if c.layer_types[i] == "linear_attention":
                h = h + self._linear_attn(jl, L, x16)
                jl += 1
            else:
                h = h + self._full_attn(ja, L, x16)
                ja += 1
            x16 = _rms1p(h, L["ln2"], c.eps).half()
            g = self._lin(L["gate"], x16, "a")
            u = self._lin(L["up"], x16, "b")
            act = (F.silu(g.float()) * u.float()).half()
            h = h + self._lin(L["down"], act, "c").float()
        x16 = _rms1p(h, self.final_norm, c.eps).half()
        self.lm_head.run(x16.contiguous(), self.logits)
        self.tok.copy_(self.logits.float().argmax().view(1))
        self.pos.add_(1)

    def reset(self, tok: int = 0):
        self.conv.zero_()
        self.S.zero_()
        self.kc.zero_()
        self.vc.zero_()
        self.pos.zero_()
        self.tok.fill_(tok)

    def capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.reset()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.step()
        self.reset()
        return self.graph

    def weight_bytes(self) -> int:
        n = self.lm_head.nbytes()
        for L in self.layers:
            for v in L.values():
                n += v.nbytes() if hasattr(v, "nbytes") and callable(v.nbytes) else v.numel() * v.element_size()
        return n

    @torch.no_grad()
    def time_tokens(self, n: int = 128, reps: int = 3) -> float:
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


def default_large(dev="cuda") -> tuple[int, int]:
    """(threads, blocks_per_sm) for matrices with R > 4096: Ampere (sm_80 / sm_86) prefers 1024-thread
    blocks (3090 27B tg 48.8 vs 43.3 tok/s at 512, s38 sweep), newer GPUs 512 (PRO 6000 / 5090)."""
    return (1024, 1) if (8, 0) <= torch.cuda.get_device_capability(dev) < (8, 9) else (512, 1)


def engine_from_packed(packed_path, *, model=None, max_len: int = 2048, dev="cuda", small=(0, 1),
                       large=None, fused: bool | None = None) -> tuple[Qwen35Engine, dict]:
    """Qwen3.5 engine from a packed file: self-contained (aux section) unless `model` is given,
    in which case the non-block tensors come from the HF model (fp16 embedding, 8-bit head)."""
    from .runtime import _AuxSource, _ModelSource, runs_linear
    pf = PackedFile(packed_path)
    use_aux = pf.aux is not None and model is None
    if use_aux:
        src = _AuxSource(pf)
    else:
        if model is None:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(pf.summary["base_model"], dtype=torch.bfloat16)
        src = _ModelSource(model)
    cfg = _cfg(src.config, max_len)
    f32 = lambda t: t.detach().to(dev, torch.float32).contiguous()          # noqa: E731
    w1p = lambda t: (1.0 + t.detach().float()).to(dev).contiguous()         # noqa: E731
    layers = []
    for i, typ in enumerate(cfg.layer_types):
        p = f"model.layers.{i}."
        L = {"ln1": w1p(src.get(p + "input_layernorm.weight")), "ln2": w1p(src.get(p + "post_attention_layernorm.weight"))}
        names = dict(MLP, **(LIN if typ == "linear_attention" else ATT))
        for k, n in names.items():
            L[k] = runs_linear(pf, p + n, dev, 1, small, large or default_large(dev))
        if typ == "linear_attention":
            la = p + "linear_attn."
            L.update(wa=f32(src.get(la + "in_proj_a.weight")), wb=f32(src.get(la + "in_proj_b.weight")),
                     convw=f32(src.get(la + "conv1d.weight")).view(-1, cfg.conv_k),
                     dt_bias=f32(src.get(la + "dt_bias")), A=f32(src.get(la + "A_log")).exp(),
                     gnorm=f32(src.get(la + "norm.weight")))
        else:
            L.update(qn=w1p(src.get(p + "self_attn.q_norm.weight")), kn=w1p(src.get(p + "self_attn.k_norm.weight")))
        layers.append(L)
    if use_aux:
        R, C = pf.aux["vocab"], pf.aux["hidden"]
        lm_head = de.Q8Lin(pf.aux_tensor("lm_head.q").to(dev), pf.aux_tensor("lm_head.scale").to(dev), R, C)
        emb = de.Q4Embed(pf.aux_tensor("embed.q").to(dev), pf.aux_tensor("embed.scale").to(dev))
    else:
        lm_head = de.Q8Lin.from_weight(model.lm_head.weight.detach(), dev)
        emb = model.model.embed_tokens.weight.detach().to(dev, torch.float16).contiguous()
    eng = Qwen35Engine(cfg, layers, emb, src.get("model.norm.weight").to(dev).float() + 1.0, lm_head, dev, fused=fused)
    return eng, {"summary": pf.summary, "self_contained": use_aux, "weight_bytes": eng.weight_bytes()}


@torch.no_grad()
def feed(eng: Qwen35Engine, tokens: list[int], keep_logits: bool = False) -> list:
    """Run tokens from position 0, one step each; afterwards eng.tok = greedy next token."""
    eng.reset(tokens[0])
    out = []
    for i in range(len(tokens)):
        eng.step()
        if keep_logits:
            out.append(eng.logits.float().clone())
        if i + 1 < len(tokens):
            eng.tok.fill_(tokens[i + 1])
    return out


@torch.no_grad()
def generate(eng: Qwen35Engine, prompt: list[int], n: int, graph: bool = True) -> list[int]:
    """Greedy continuation of prompt by n tokens (prompt fed step by step)."""
    if graph and eng.graph is None:
        eng.capture()
    feed(eng, prompt)
    out = [int(eng.tok)]
    while len(out) < n:
        if graph:
            eng.graph.replay()
        else:
            eng.step()
        out.append(int(eng.tok))
    return out


# ---------------------------------------------------------------- batched prefill

def attach_prefill(eng: Qwen35Engine) -> None:
    """Give the engine batched prefill: every block linear decodes to a shared fp16 scratch from
    its run words (kernels.cuda_dequant) for one GEMM over the prompt."""
    from .kernels import cuda_dequant
    from .prefill import RunsPrefill
    rls = [v.rl for L in eng.layers for v in L.values() if hasattr(v, "rl")]
    bad = [rl for rl in rls if not cuda_dequant.supported(rl)]
    if bad:
        raise ValueError(f"{len(bad)} matrices not supported by the GPU dequant (column orders / odd bits)")
    n = max(rl.R * rl.nch * 4 for rl in rls)
    scratch = torch.empty(n, dtype=torch.float16, device=eng.h.device)
    eng.prefill_lins = [{k: RunsPrefill(v.rl, scratch) for k, v in L.items() if hasattr(v, "rl")} for L in eng.layers]
    eng.prefill_scratch_bytes = n * 2
    eng.dense_head = None


@torch.no_grad()
def prefill(eng: Qwen35Engine, tokens, logits: str = "last", impl: str | None = None,
            gemm: str | None = None) -> torch.Tensor:
    """Run tokens through the engine at its current position in one pass (conv / recurrent /
    KV states continue from eng's). logits: "last" -> fp32 [vocab], "all" -> fp32 [T, vocab].
    Leaves eng.pos += T, eng.tok = argmax of the last row, eng.logits = last row.
    impl: "fast" (default, prefill.IMPL) = shapequant.qwen35_prefill (fused kernels, grouped GEMMs,
    chunked DeltaNet scan kernel; gemm "dequant" | "fused" as in prefill.prefill); "ref" = the
    torch path below: HF's chunked gated delta rule (torch_chunk_gated_delta_rule), fp32 SDPA."""
    from . import prefill as pre
    impl = impl or pre.IMPL
    if impl == "fast":
        from . import qwen35_prefill
        return qwen35_prefill.prefill(eng, tokens, logits, gemm or pre.GEMM)
    if impl != "ref":
        raise ValueError(f"impl {impl!r}: 'fast' | 'ref'")
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    from .prefill import _head
    c = eng.cfg
    dev = eng.h.device
    ids = torch.as_tensor(tokens, dtype=torch.int64, device=dev).view(-1)
    T = ids.numel()
    p0 = int(eng.pos)
    if p0 + T > c.max_len:
        raise ValueError(f"context {p0 + T} > max_len {c.max_len}")
    rid = ids % eng.embed.shape[0]
    h = eng.embed.rows(rid) if isinstance(eng.embed, de.Q4Embed) else eng.embed[rid].float()
    pos = torch.arange(p0, p0 + T, device=dev)
    fr = pos.float()[:, None] * eng.inv_freq[None]
    cos, sin = torch.cat([fr, fr], -1).cos(), torch.cat([fr, fr], -1).sin()          # [T, rot]
    r = c.rot

    def rope(x):                                   # x fp32 [T, n, hd]
        xr = x[..., :r]
        rot = torch.cat([-xr[..., r // 2:], xr[..., :r // 2]], -1)
        return torch.cat([xr * cos[:, None] + rot * sin[:, None], x[..., r:]], -1)

    jl = ja = 0
    kd = c.hk * c.dk
    for i, (L, P) in enumerate(zip(eng.layers, eng.prefill_lins)):
        x = _rms1p(h, L["ln1"], c.eps).half()
        if c.layer_types[i] == "linear_attention":
            qkv = F.linear(x, P["qkv"].dense())                                       # [T, conv_dim] fp16
            z = F.linear(x, P["z"].dense()).float()
            a = x.float() @ L["wa"].T
            b = x.float() @ L["wb"].T
            seq = torch.cat([eng.conv[jl], qkv.float().T], 1)                         # [conv_dim, K + T]
            u = F.silu(F.conv1d(seq[None], L["convw"][:, None], groups=eng.conv_dim)[0, :, 1:]).T   # [T, conv_dim]
            eng.conv[jl].copy_(seq[:, -c.conv_k:])
            q = u[:, :kd].view(1, T, c.hk, c.dk)
            k = u[:, kd:2 * kd].view(1, T, c.hk, c.dk)
            v = u[:, 2 * kd:].view(1, T, c.hv, c.dv)
            if c.hv != c.hk:
                q = q.repeat_interleave(c.hv // c.hk, 2)
                k = k.repeat_interleave(c.hv // c.hk, 2)
            g = (-L["A"] * F.softplus(a + L["dt_bias"]))[None]
            beta = torch.sigmoid(b)[None]
            o, S = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=eng.S[jl][None],
                                                output_final_state=True, use_qk_l2norm_in_kernel=True)
            eng.S[jl].copy_(S[0])
            o = o[0].float()                                                           # [T, hv, dv]
            o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + c.eps) * L["gnorm"] * F.silu(z.view(T, c.hv, c.dv))
            h = h + F.linear(o.reshape(T, -1).half(), P["out"].dense()).float()
            jl += 1
        else:
            qg = F.linear(x, P["q"].dense()).float().view(T, c.n_heads, 2 * c.head_dim)
            q, gate = qg[..., :c.head_dim], qg[..., c.head_dim:]
            k = F.linear(x, P["k"].dense()).float().view(T, c.n_kv, c.head_dim)
            v = F.linear(x, P["v"].dense()).view(T, c.n_kv, c.head_dim)
            q = rope(_rms1p(q, L["qn"], c.eps))
            k = rope(_rms1p(k, L["kn"], c.eps))
            eng.kc[ja, :, p0:p0 + T] = k.half().transpose(0, 1)
            eng.vc[ja, :, p0:p0 + T] = v.transpose(0, 1)
            rep = c.n_heads // c.n_kv
            kk = eng.kc[ja, :, :p0 + T].float().repeat_interleave(rep, 0)
            vv = eng.vc[ja, :, :p0 + T].float().repeat_interleave(rep, 0)
            mask = (torch.arange(p0 + T, device=dev)[None, :] <= pos[:, None])
            o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], kk[None], vv[None], attn_mask=mask[None, None],
                                               scale=c.head_dim ** -0.5)[0].transpose(0, 1)       # [T, nh, hd]
            o = o * torch.sigmoid(gate)
            h = h + F.linear(o.reshape(T, -1).half(), P["o"].dense()).float()
            ja += 1
        x = _rms1p(h, L["ln2"], c.eps).half()
        act = (F.silu(F.linear(x, P["gate"].dense()).float()) * F.linear(x, P["up"].dense()).float()).half()
        h = h + F.linear(act, P["down"].dense()).float()
    x = _rms1p(h if logits == "all" else h[-1:], eng.final_norm, c.eps).half()
    lg = _head(eng, x)
    eng.logits.copy_(lg[-1].half())
    eng.tok.copy_(lg[-1].half().float().argmax().view(1))
    eng.pos.fill_(p0 + T)
    return lg if logits == "all" else lg[-1]
