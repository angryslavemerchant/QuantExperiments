"""Batched decode for Qwen2-architecture models: B sequences per step (B = 1..32), sharing the
weights of a batch-1 kernels.decode_e2e.DecodeEngine (shapequant.runtime.engine_from_packed or
decode_e2e.random_engine).

    eng, info = runtime.engine_from_packed("model.safetensors", max_len=2048)
    beng = Qwen2BatchEngine(eng, B=8)
    outs = qwen35_batch.generate_batch(beng, prompts, n=64)

Per sequence: its own KV cache and position. One step: embedding rows, per layer add + RMSNorm
(fp16 weight), q / k / v on the batched run kernel (kernels.batched_runs.RunsBatch), rope + KV
write (bias added, cos / sin rounded to fp16 as decode_e2e), GQA-sharing flash-decoding attention,
o, add + norm, gate / up, fp16 silu * up, down; final norm, head (q8: tile-major batched kernel;
fp16: cuBLAS GEMM; q4 / other: the batch-1 kernel per sequence), greedy argmax per sequence.
CUDA-graph capturable. Block linears with several column orders (M > 1) are not supported.
Logits match the batch-1 engine to fp16 rounding (fp32 accumulation in the batched GEMMs).
"""

from __future__ import annotations

import torch

from .kernels import decode_e2e as de
from .qwen35_batch import BatchEngineBase

NAMES = ("q", "k", "v", "o", "gate", "up", "down")


class Qwen2BatchEngine(BatchEngineBase):
    """step(): reads tok [B] at pos [B], writes logits [B, vocab] (fp16) and tok (greedy next).
    layout / warps / bps / max_split / acc16 / tune: as qwen35_batch.Qwen35BatchEngine;
    nsplit: attention position slices (None: ~2 blocks per SM)."""

    def __init__(self, eng: de.DecodeEngine, B: int, *, layout: str = "tile", warps: int = 8, bps: int = 2,
                 max_split: int = 8, acc16: bool = False, nsplit: int | None = None, tune: bool = False,
                 tune_log=None):
        from .kernels import batched_runs as br
        c = eng.cfg
        if not 1 <= B <= br.BMAX:
            raise ValueError(f"B = {B} outside 1..{br.BMAX}")
        for L in eng.layers:
            for k in NAMES:
                if not isinstance(L[k], de.RunsLin) or L[k].perms is not None:
                    raise ValueError("Qwen2BatchEngine needs run-kernel block linears with one column order (M = 1)")
        rep = c.n_heads // c.n_kv
        if c.head_dim % 64 or 4 * rep * c.head_dim * 4 > 40 * 1024:
            raise ValueError("attention shape not supported by the batched kernels")
        self.eng, self.cfg, self.B, self.layers = eng, c, B, eng.layers
        self.embed, self.lm_head, self.final_norm = eng.embed, eng.lm_head, eng.final_norm
        dev = eng.kc.device
        f32 = dict(dtype=torch.float32, device=dev)
        f16 = dict(dtype=torch.float16, device=dev)
        H, I, NH, NKV, HD = c.hidden, c.inter, c.n_heads, c.n_kv, c.head_dim
        self.kc = torch.zeros(len(eng.layers), B, NKV, c.max_len, HD, **f16)
        self.vc = torch.zeros_like(self.kc)
        self.tok = torch.zeros(B, dtype=torch.int64, device=dev)
        self.pos = torch.zeros(B, dtype=torch.int64, device=dev)
        self.logits = torch.zeros(B, eng.lm_head.R, **f16)
        self.h = torch.zeros(B, H, **f32)
        self.x16 = torch.zeros(B, H, **f16)
        self.y = {k: torch.zeros(B, eng.layers[0][k].R, **f16) for k in NAMES}
        self.qo = torch.zeros(B, NH * HD, **f16)
        self.ao = torch.zeros(B, NH * HD, **f16)
        self.act = torch.zeros(B, I, **f16)
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.nsplit = nsplit or max(1, min(-(-2 * sms // (NKV * B)), c.max_len // 64))
        npart = B * NKV * self.nsplit * rep
        self.pm, self.pl = torch.zeros(npart, **f32), torch.zeros(npart, **f32)
        self.po = torch.zeros(npart * HD, **f32)
        cfg = {}
        if tune:
            rls = [L[k].rl for L in eng.layers for k in NAMES]
            cfg = br.tune_runs(rls, B, layout=layout, acc16=acc16, log=tune_log)
        self.launch = cfg

        def mk(rl):
            w, b, m = cfg.get(br.runs_key(rl), (warps, bps, max_split))
            return br.RunsBatch(rl, bmax=B, warps=w, bps=b, max_split=m, layout=layout, acc16=acc16)
        self.blin = [{k: mk(L[k].rl) for k in NAMES} for L in eng.layers]
        lm = eng.lm_head
        self.qhead = self.bhead = None
        if isinstance(lm, de.Q8Lin) and lm.C % 64 == 0:
            if getattr(lm, "_q8batch", None) is None:
                lm._q8batch = br.Q8Batch.from_lin(lm)
            self.qhead = lm._q8batch
        elif isinstance(lm, de.RunsLin) and lm.perms is None:
            self.bhead = mk(lm.rl)
        self.graph = None

    def _k(self, name, grid, block, args):
        import cupy as cp
        from .kernels import qwen2_batch_kernels as q2k
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            q2k.module(self.cfg).get_function(name)(grid, (block,), args)

    def _addnorm(self, add, w):
        import cupy as cp
        P = lambda t: cp.uint64(0 if t is None else t.data_ptr())   # noqa: E731
        self._k("addnorm16_b", (self.B,), 1024, (P(self.h), P(add), P(w), cp.int32(self.cfg.hidden),
                                                 cp.float32(self.cfg.eps), P(self.x16)))

    def _head(self):
        lm = self.lm_head
        if self.qhead is not None:
            self.qhead.run(self.x16, self.logits, self.B)
        elif self.bhead is not None:
            self.bhead.run(self.x16, self.logits, self.B)
        elif isinstance(lm, de.DenseLin):
            torch.mm(self.x16, lm.w.t(), out=self.logits)
        else:
            for b in range(self.B):
                lm.run(self.x16[b], self.logits[b])

    @torch.no_grad()
    def step(self):
        import cupy as cp
        c, B = self.cfg, self.B
        P = lambda t: cp.uint64(0 if t is None else t.data_ptr())   # noqa: E731
        ids = self.tok % self.embed.shape[0]
        self.h.copy_(self.embed.rows(ids) if isinstance(self.embed, de.Q4Embed) else self.embed[ids].float())
        y = self.y
        add = None
        for i, L in enumerate(self.layers):
            bl = self.blin[i]
            self._addnorm(add, L["ln1"])
            for k in ("q", "k", "v"):
                bl[k].run(self.x16, y[k], B)
            kc, vc = self.kc[i], self.vc[i]
            self._k("rope_cache_b", (c.n_heads + c.n_kv, B), c.head_dim // 2,
                    (P(y["q"]), P(y["k"]), P(y["v"]), P(L.get("bq")), P(L.get("bk")), P(L.get("bv")), P(self.pos),
                     cp.float32(c.rope_theta), cp.int32(c.max_len), P(self.qo), P(kc), P(vc)))
            ns = self.nsplit
            self._k("attn_gqa_b", (c.n_kv * ns, B), 128,
                    (P(self.qo), P(None), P(kc), P(vc), P(self.pos), cp.int32(c.max_len), cp.int32(ns),
                     P(self.pm), P(self.pl), P(self.po), P(self.ao)))
            if ns > 1:
                self._k("attn_comb_b", (c.n_heads, B), c.head_dim,
                        (P(None), cp.int32(ns), P(self.pm), P(self.pl), P(self.po), P(self.ao)))
            bl["o"].run(self.ao, y["o"], B)
            self._addnorm(y["o"], L["ln2"])
            bl["gate"].run(self.x16, y["gate"], B)
            bl["up"].run(self.x16, y["up"], B)
            n = B * c.inter
            self._k("silu_mul16", ((n + 255) // 256,), 256, (P(y["gate"]), P(y["up"]), cp.int32(n), P(self.act)))
            bl["down"].run(self.act, y["down"], B)
            add = y["down"]
        self._addnorm(add, self.final_norm)
        self._head()
        torch.argmax(self.logits.float(), dim=-1, out=self.tok)
        self.pos.add_(1)

    # -- state
    def reset(self, tok=0):
        self.kc.zero_()
        self.vc.zero_()
        self.pos.zero_()
        if isinstance(tok, int):
            self.tok.fill_(tok)
        else:
            self.tok.copy_(torch.as_tensor(tok, dtype=torch.int64))

    def _states(self):
        return (self.kc, self.vc)

    def load_slot(self, b: int, eng: de.DecodeEngine | None = None):
        """Copy a batch-1 engine's sequence state (KV cache, position, next token) into slot b."""
        eng = self.eng if eng is None else eng
        self.kc[:, b].copy_(eng.kc)
        self.vc[:, b].copy_(eng.vc)
        self.pos[b] = eng.pos[0].long()
        self.tok[b] = eng.tok[0]
        self.logits[b].copy_(eng.logits)

    def has_prefill(self) -> bool:
        from .runtime import has_prefill
        return has_prefill(self.eng)

    def _load_one(self, b: int, tokens, use_prefill: bool):
        from .runtime import start
        start(self.eng, list(tokens), use_prefill=use_prefill)
        self.load_slot(b)
