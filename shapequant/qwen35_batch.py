"""Batched decode for Qwen3.5 / Qwen3.8: B sequences per step (B = 1..32), sharing the weights
of a batch-1 shapequant.qwen35_runtime.Qwen35Engine.

    eng, info = qwen35_runtime.engine_from_packed("model.safetensors", max_len=2048)
    beng = Qwen35BatchEngine(eng, B=8)
    outs = generate_batch(beng, prompts, n=64)       # prompts: B token lists (lengths may differ)

Per sequence the engine holds its own DeltaNet recurrent state S [Hv, dk, dv] (fp32), conv state,
KV cache and position (pos [B]: sequences may sit at different positions). One step = the fused
batch-1 step with a batch dimension everywhere:
  - block linears: kernels.batched_runs.RunsBatch (each weight code decoded once for all B,
    tensor-core mma over the batch); the lm_head: batched q8 kernel on a tile-major copy of the
    head (kernels.batched_runs.Q8Batch, made once per head: costs a second copy of the head);
  - DeltaNet / attention / norms: kernels.qwen35_batch_kernels (grid.y = sequence); the tiny
    in_proj_a / in_proj_b projections as one fp32 GEMM for the batch;
  - greedy argmax per sequence; the whole step CUDA-graph capturable (capture() per engine, i.e.
    per B).
B = 1 runs the batch-1 kernels throughout (bit-identical to Qwen35Engine; batch1_kernels=False
uses the batched ones). For B > 1 the
linears accumulate in fp32 (the batch-1 run kernel partially in fp16), so logits match the
batch-1 engine to fp16 rounding, not bitwise.

Prompts are loaded per sequence through the batch-1 engine (its batched prefill if attached,
else step by step) and its states copied into the batch slots (load_slot), or, for equal-length
prompts, fed through the batched step itself (feed_batch).
"""

from __future__ import annotations

import torch

from .kernels import decode_e2e as de
from .qwen35_runtime import Qwen35Engine


class BatchEngineBase:
    """Engine-agnostic parts of the batch engines: subclasses set B, cfg, eng (the batch-1
    engine), tok / pos [B], logits, graph, implement step(), reset(), _states() (per-sequence state
    tensors, batch dim 1) and _load_one(b, tokens, use_prefill) / has_prefill()."""

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

    def state_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._states())

    @torch.no_grad()
    def time_steps(self, n: int = 64, ctx: int = 0, reps: int = 3, stat: str = "median") -> float:
        """Seconds per batched step (graph replay), averaged over n steps starting at position
        ctx for every sequence (the states are not meaningful: speed only; attention cost grows
        with ctx). stat: "median" or "min" of reps."""
        if ctx + n > self.cfg.max_len:
            raise ValueError("ctx + n exceeds max_len")
        if self.graph is None:
            self.capture()
        ts = []
        for _ in range(reps):
            self.reset()
            self.pos.fill_(ctx)
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(n):
                self.graph.replay()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) / 1e3 / n)
        return min(ts) if stat == "min" else sorted(ts)[len(ts) // 2]


class Qwen35BatchEngine(BatchEngineBase):
    """step(): reads tok [B] at pos [B], writes logits [B, vocab] (fp16) and tok (greedy next).
    layout: "tile" (default) gives the batched GEMMs a tile-major copy of every block matrix's
    words (made once per matrix and shared by all batch engines: a second copy of the block
    weights, the batch-1 engine keeps its own), "lane" reads the batch-1 words in place (no extra
    memory, slower). head_loop_max: batch sizes up to this run the batch-1 q8 head kernel once per
    sequence (else a tile-major copy of the head). warps / bps / max_split / acc16: RunsBatch.
    attn: "gqa" / "heads" / "auto" (gqa for B > 1); nsplit: gqa position slices (None: ~2 blocks
    per SM). tune: time the launch configs per matrix shape at this B first (kernels.batched_runs.
    tune_runs; seconds) and use the fastest."""

    def __init__(self, eng: Qwen35Engine, B: int, *, layout: str = "tile", head_loop_max: int = 1, warps: int = 8,
                 bps: int = 2, max_split: int = 8, acc16: bool = False, attn: str = "auto",
                 nsplit: int | None = None, batch1_kernels: bool = True, tune: bool = False, tune_log=None):
        from .kernels import batched_runs as br
        from .kernels import qwen35_kernels as qk
        c = eng.cfg
        if not qk.supported(c):
            raise ValueError("config not supported by the fused kernels")
        if not 1 <= B <= br.BMAX:
            raise ValueError(f"B = {B} outside 1..{br.BMAX}")
        self.eng, self.cfg, self.B, self.layers = eng, c, B, eng.layers
        self.embed, self.lm_head, self.final_norm = eng.embed, eng.lm_head, eng.final_norm
        self.batched = B > 1 or not batch1_kernels      # batched kernels (else B = 1 on the batch-1 ones)
        if not self.batched:
            head_loop_max = max(head_loop_max, 1)
        elif B == 1:
            head_loop_max = 0
        self.head_loop_max = head_loop_max
        self.qhead = None       # tile-major copy of the q8 head (kernels.batched_runs.Q8Batch), shared per head
        if B > head_loop_max and isinstance(eng.lm_head, de.Q8Lin) and eng.lm_head.C % 64 == 0:
            if getattr(eng.lm_head, "_q8batch", None) is None:
                eng.lm_head._q8batch = br.Q8Batch.from_lin(eng.lm_head)
            self.qhead = eng.lm_head._q8batch
        dev = eng.h.device
        f32 = dict(dtype=torch.float32, device=dev)
        f16 = dict(dtype=torch.float16, device=dev)
        self.conv_dim = eng.conv_dim
        nl, na = len(eng.lin_idx), len(eng.att_idx)
        self.conv = torch.zeros(nl, B, self.conv_dim, c.conv_k, **f32)
        self.S = torch.zeros(nl, B, c.hv, c.dk, c.dv, **f32)
        self.kc = torch.zeros(na, B, c.n_kv, c.max_len, c.head_dim, **f16)
        self.vc = torch.zeros_like(self.kc)
        self.tok = torch.zeros(B, dtype=torch.int64, device=dev)
        self.pos = torch.zeros(B, dtype=torch.int64, device=dev)
        self.logits = torch.zeros(B, c.vocab, **f16)
        self.inv_freq = eng.inv_freq
        mx = max(max(l.R for L in eng.layers for l in L.values() if hasattr(l, "R")), c.hidden, c.inter)
        self._y = {k: torch.empty(B * mx, **f16) for k in ("a", "b", "c", "d")}
        self.h = torch.zeros(B, c.hidden, **f32)
        self.x16 = torch.zeros(B, c.hidden, **f16)
        self.xf = torch.zeros(B, c.hidden, **f32)
        self.ab = torch.zeros(B, 2 * c.hv, **f32)
        self.u = torch.zeros(B, self.conv_dim, **f32)
        self.o16 = torch.zeros(B * max(c.hv * c.dv, c.n_heads * c.head_dim, c.inter), **f16)
        self.qr = torch.zeros(B, c.n_heads * c.head_dim, **f32)
        # per linear-attention layer: [in_proj_a; in_proj_b]^T (fp32 [hidden, 2 hv]) for one GEMM
        self.wabT = {i: torch.cat([L["wa"], L["wb"]]).T.contiguous() for i, L in enumerate(eng.layers)
                     if c.layer_types[i] == "linear_attention"}
        # attention: "gqa" (one block per kv head and position slice for all its query heads, then a
        # combine pass; default for B > 1) or "heads" (block per query head: the batch-1 kernel's scheme)
        rep = c.n_heads // c.n_kv
        gqa_ok = 4 * rep * c.head_dim * 4 <= 40 * 1024
        self.attn = ("gqa" if self.batched and gqa_ok else "heads") if attn == "auto" else attn
        if self.attn == "gqa":
            if not gqa_ok:
                raise ValueError("attn='gqa': too many query heads per kv head for its shared memory")
            sms = torch.cuda.get_device_properties(dev).multi_processor_count
            self.nsplit = nsplit or max(1, min(-(-2 * sms // (c.n_kv * B)), c.max_len // 64))
            npart = B * c.n_kv * self.nsplit * rep
            self.pm, self.pl = torch.zeros(npart, **f32), torch.zeros(npart, **f32)
            self.po = torch.zeros(npart * c.head_dim, **f32)
        self.blin = None
        if self.batched:
            cfg = {}
            if tune:        # per matrix shape: the fastest (warps, bps, max_split) at this B
                rls = [v.rl for L in eng.layers for v in L.values() if isinstance(v, de.RunsLin)]
                cfg = br.tune_runs(rls, B, layout=layout, acc16=acc16, log=tune_log)
            self.launch = cfg

            def mk(rl):
                w, b, m = cfg.get(br.runs_key(rl), (warps, bps, max_split))
                return br.RunsBatch(rl, bmax=B, warps=w, bps=b, max_split=m, layout=layout, acc16=acc16)
            self.blin = [{k: mk(v.rl) for k, v in L.items() if isinstance(v, de.RunsLin)} for L in eng.layers]
        self.graph = None

    # -- helpers
    def _lin(self, i: int, k: str, x16: torch.Tensor, key: str) -> torch.Tensor:
        """x16 [B, C] fp16 -> [B, R] fp16 (a view of buffer `key`)."""
        lin = self.layers[i][k]
        y = self._y[key][:self.B * lin.R].view(self.B, lin.R)
        if self.blin is None:
            lin.run(x16[0], y[0])
        else:
            self.blin[i][k].run(x16, y, self.B)
        return y

    def _k(self, name, grid, block, args):
        import cupy as cp
        from .kernels import qwen35_batch_kernels as qbk
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            qbk.module(self.cfg).get_function(name)(grid, (block,), args)

    def _addnorm(self, add, w1p):
        import cupy as cp
        P = lambda t: cp.uint64(0 if t is None else t.data_ptr())   # noqa: E731
        self._k("addnorm_b", (self.B,), 1024, (P(self.h), P(add), P(w1p), cp.int32(self.cfg.hidden),
                                               cp.float32(self.cfg.eps), P(self.x16)))

    def _embed(self):
        ids = self.tok % self.embed.shape[0]
        if isinstance(self.embed, de.Q4Embed):
            self.h.copy_(self.embed.rows(ids))
        else:
            self.h.copy_(self.embed[ids].float())

    def _head(self):
        from .kernels import batched_runs as br
        lm = self.lm_head
        if self.qhead is not None:
            self.qhead.run(self.x16, self.logits, self.B)
        elif self.B <= self.head_loop_max or not isinstance(lm, de.Q8Lin):
            for b in range(self.B):
                lm.run(self.x16[b], self.logits[b])
        else:
            br.gemv_q8_batched(lm.q, lm.sc, self.x16, self.logits, lm.R, lm.C, self.B)

    @torch.no_grad()
    def step(self):
        import cupy as cp
        c, B = self.cfg, self.B
        P = lambda t: cp.uint64(t.data_ptr())   # noqa: E731
        self._embed()
        add = None
        jl = ja = 0
        for i, L in enumerate(self.layers):
            self._addnorm(add, L["ln1"])
            if c.layer_types[i] == "linear_attention":
                qkv = self._lin(i, "qkv", self.x16, "a")
                z = self._lin(i, "z", self.x16, "b")
                self._k("conv_step_b", ((self.conv_dim + 255) // 256, B), 256,
                        (P(qkv), P(self.conv[jl]), P(L["convw"]), cp.int32(self.conv_dim), P(self.u)))
                o = self.o16[:B * c.hv * c.dv]
                if not self.batched:    # the batch-1 kernel (a / b projections in-kernel): bit-identical
                    self._k("delta_step", (c.hv,), c.dv,
                            (P(self.u), P(z), P(self.x16), P(L["wa"]), P(L["wb"]), P(L["A"]), P(L["dt_bias"]),
                             P(L["gnorm"]), P(self.S[jl]), cp.int32(c.hidden), cp.float32(c.eps), P(o)))
                else:
                    self.xf.copy_(self.x16)
                    torch.mm(self.xf, self.wabT[i], out=self.ab)
                    self._k("delta_step_b", (c.hv, B), c.dv,
                            (P(self.u), P(z), P(self.ab), cp.int32(self.conv_dim), P(L["A"]), P(L["dt_bias"]),
                             P(L["gnorm"]), P(self.S[jl]), cp.float32(c.eps), P(o)))
                add = self._lin(i, "out", o.view(B, -1), "c")
                jl += 1
            else:
                qg = self._lin(i, "q", self.x16, "a")
                k = self._lin(i, "k", self.x16, "b")
                v = self._lin(i, "v", self.x16, "c")
                o = self.o16[:B * c.n_heads * c.head_dim]
                self._k("attn_prep_b", (c.n_heads + c.n_kv, B), c.head_dim,
                        (P(qg), P(k), P(v), P(L["qn"]), P(L["kn"]), P(self.inv_freq), P(self.pos),
                         cp.float32(c.eps), P(self.qr), P(self.kc[ja]), P(self.vc[ja]), cp.int32(c.max_len)))
                if self.attn == "gqa":
                    ns = self.nsplit
                    self._k("attn_gqa_b", (c.n_kv * ns, B), 128,
                            (P(self.qr), P(qg), P(self.kc[ja]), P(self.vc[ja]), P(self.pos), cp.int32(c.max_len),
                             cp.int32(ns), P(self.pm), P(self.pl), P(self.po), P(o)))
                    if ns > 1:
                        self._k("attn_comb_b", (c.n_heads, B), c.head_dim,
                                (P(qg), cp.int32(ns), P(self.pm), P(self.pl), P(self.po), P(o)))
                else:
                    self._k("attn_b", (c.n_heads, B), 256, (P(self.qr), P(qg), P(self.kc[ja]), P(self.vc[ja]),
                                                           P(self.pos), cp.int32(c.max_len), P(o)))
                add = self._lin(i, "o", o.view(B, -1), "c")
                ja += 1
            self._addnorm(add, L["ln2"])
            g = self._lin(i, "gate", self.x16, "a")
            u = self._lin(i, "up", self.x16, "b")
            act = self.o16[:B * c.inter]
            self._k("silu_mul", ((B * c.inter + 255) // 256,), 256, (P(g), P(u), cp.int32(B * c.inter), P(act)))
            add = self._lin(i, "down", act.view(B, c.inter), "d")
        self._addnorm(add, self.final_norm)
        self._head()
        torch.argmax(self.logits.float(), dim=-1, out=self.tok)
        self.pos.add_(1)

    # -- state
    def reset(self, tok=0):
        self.conv.zero_()
        self.S.zero_()
        self.kc.zero_()
        self.vc.zero_()
        self.pos.zero_()
        if isinstance(tok, int):
            self.tok.fill_(tok)
        else:
            self.tok.copy_(torch.as_tensor(tok, dtype=torch.int64))

    def load_slot(self, b: int, eng: Qwen35Engine | None = None):
        """Copy a batch-1 engine's sequence state (conv, S, KV cache, position, next token) into
        slot b. eng defaults to the engine this batch engine was built from."""
        eng = self.eng if eng is None else eng
        self.conv[:, b].copy_(eng.conv)
        self.S[:, b].copy_(eng.S)
        self.kc[:, b].copy_(eng.kc)
        self.vc[:, b].copy_(eng.vc)
        self.pos[b] = eng.pos[0]
        self.tok[b] = eng.tok[0]
        self.logits[b].copy_(eng.logits)

    def _states(self):
        return (self.conv, self.S, self.kc, self.vc)

    def has_prefill(self) -> bool:
        return getattr(self.eng, "prefill_lins", None) is not None

    def _load_one(self, b: int, tokens, use_prefill: bool):
        from . import qwen35_runtime as q35
        if use_prefill:
            self.eng.reset()
            q35.prefill(self.eng, tokens)
        else:
            q35.feed(self.eng, tokens)
        self.load_slot(b)


@torch.no_grad()
def feed_batch(beng: BatchEngineBase, prompts: list[list[int]], keep_logits: bool = False, graph: bool = False):
    """Equal-length prompts fed from position 0 through the batched step (one step per prompt
    position). Afterwards beng.tok = each sequence's greedy next token. keep_logits: fp32
    [B, vocab] after each step."""
    T = len(prompts[0])
    if len(prompts) != beng.B or any(len(p) != T for p in prompts):
        raise ValueError(f"feed_batch needs {beng.B} prompts of equal length")
    if T > beng.cfg.max_len:
        raise ValueError("prompt longer than max_len")
    if graph and beng.graph is None:
        beng.capture()
    ids = torch.tensor(prompts, dtype=torch.int64, device=beng.tok.device)        # [B, T]
    beng.reset()
    out = []
    for i in range(T):
        beng.tok.copy_(ids[:, i])
        if graph:
            beng.graph.replay()
        else:
            beng.step()
        if keep_logits:
            out.append(beng.logits.float().clone())
    return out


@torch.no_grad()
def load_prompts(beng: BatchEngineBase, prompts: list[list[int]], use_prefill: bool | None = None):
    """Run each prompt alone through the batch-1 engine (batched prefill when attached, else step
    by step) and copy its state into its slot. Prompt lengths may differ (ragged positions)."""
    if len(prompts) != beng.B:
        raise ValueError(f"need {beng.B} prompts")
    if use_prefill is None:
        use_prefill = beng.has_prefill()
    beng.reset()
    for b, p in enumerate(prompts):
        if len(p) > beng.cfg.max_len:
            raise ValueError("prompt longer than max_len")
        beng._load_one(b, p, use_prefill)


@torch.no_grad()
def generate_batch(beng: BatchEngineBase, prompts: list[list[int]], n: int, graph: bool = True,
                   use_prefill: bool | None = None, feed: str = "auto") -> list[list[int]]:
    """Greedy continuation of each prompt by n tokens. feed: "slots" (load_prompts), "batch"
    (feed_batch: equal lengths) or "auto" (batch when the lengths are equal and the batch-1
    engine has no batched prefill, else slots)."""
    if graph and beng.graph is None:
        beng.capture()
    if max(len(p) for p in prompts) + n > beng.cfg.max_len:
        raise ValueError("prompt + n exceeds max_len")
    if feed == "auto":
        equal = len({len(p) for p in prompts}) == 1
        feed = "batch" if equal and not beng.has_prefill() else "slots"
    if feed == "batch":
        feed_batch(beng, prompts, graph=graph)
    else:
        load_prompts(beng, prompts, use_prefill)
    out = [[int(t)] for t in beng.tok.tolist()]
    for _ in range(n - 1):
        if graph:
            beng.graph.replay()
        else:
            beng.step()
        for b, t in enumerate(beng.tok.tolist()):
            out[b].append(int(t))
    return out
