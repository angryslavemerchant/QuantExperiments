"""Batched prefill for the decode engine (kernels/decode_e2e.py): a whole prompt in one pass.

Per layer, every block linear's weight is dequantized to fp16 from its codes (local tags kept
on the GPU as int16, book offsets, fp16 table and row scales, column orders) into a temporary,
then the T tokens go through cuBLAS matmuls, torch SDPA (causal, grouped KV heads) and the
norms / SiLU in torch. The numerics follow the decode kernels: fp32 residual, rmsnorm output
rounded to fp16, q/k/v bias added in fp16, RoPE with fp32 inverse frequencies and fp16-rounded
cos/sin, rotated k and biased v written to the engine's cache (kc/vc [layer, kv, pos, HD]),
SiLU and the gate*up product in fp16. Afterwards the engine continues decoding exactly where
`runtime.feed` would have left it (pos advanced, tok = greedy next token, logits = last row).

Weights are decoded per matrix into one shared fp16 scratch buffer (the largest matrix) right
before its matmul: on the GPU from the run kernel's own packed words (RunsPrefill) when the matrix
has no column orders, else from int16 tags kept for it (PrefillLinear, 4 extra bits per weight).
The lm_head is kept as a dense fp16 copy (the Q4 head's dequantized values for head="q4").

Prefill appends at the engine's current position: from a reset engine (pos 0) it is a fresh
prompt; otherwise the tokens extend the cached context (multi-turn, or eval continuations).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .packed import PackedFile


@dataclass
class PrefillLinear:
    tags: torch.Tensor        # int16 [R, nch] local codebook indices
    table: torch.Tensor       # fp16 [G*K, d]
    row_scale: torch.Tensor   # fp32 [R]
    book_off: torch.Tensor    # int64 [R, 1]
    groups: list | None       # M > 1: [(rows int64, perm int64 [C])]; None: identity order
    R: int
    C: int

    @classmethod
    def from_packed(cls, pf: PackedFile, name: str, dev="cuda", local_tags: torch.Tensor | None = None):
        m = pf.meta[name]
        t = pf.tensors(name)
        tags = local_tags if local_tags is not None else pf.local_tags(name, dev)
        R, C, K = m["R"], m["C"], m["K"]
        rb = t.get("row_book")
        book_off = (rb.to(dev, torch.int64) * K if rb is not None
                    else torch.zeros(R, dtype=torch.int64, device=dev))[:, None]
        groups = None
        if m["M"] > 1:
            perms = pf.perms(name).to(dev)
            rm = t["row_mask"].to(dev, torch.int64)
            groups = [((rm == j).nonzero().squeeze(1), perms[j]) for j in range(m["M"]) if bool((rm == j).any())]
        return cls(tags.to(dev, torch.int16).contiguous(), t["table"].to(dev, torch.float16).contiguous(),
                   t["row_scale"].to(dev, torch.float32), book_off, groups, R, C)

    def dense(self) -> torch.Tensor:
        """fp16 [R, C] weight in the original row and column order."""
        idx = self.tags.long() + self.book_off
        u = (self.table.float()[idx].view(self.R, -1)[:, :self.C] * self.row_scale[:, None]).half()
        if self.groups is None:
            return u
        w = torch.empty_like(u)
        for rows, perm in self.groups:
            blk = torch.empty(rows.numel(), self.C, dtype=u.dtype, device=u.device)
            blk[:, perm] = u[rows]       # chunk column j of row r holds original column perm[j]
            w[rows] = blk
        return w

    def nbytes(self) -> int:
        return self.tags.numel() * 2


class RunsPrefill:
    """Dense fp16 weight decoded on the GPU from the decode kernel's own packed words
    (kernels.cuda_dequant): no extra copy of the codes. dense() writes into a scratch buffer
    shared by every matrix of the engine (each result is consumed by the next matmul on the
    same stream before the buffer is reused)."""

    def __init__(self, rl, scratch: torch.Tensor):
        self.rl, self.scratch, self.R, self.C = rl, scratch, rl.R, rl.nch * 4

    def dense(self) -> torch.Tensor:
        from .kernels import cuda_dequant
        return cuda_dequant.dequant(self.rl, self.scratch)

    def nbytes(self) -> int:
        return 0


def _rmsnorm(h: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """h fp32 [T, C] -> fp16 rmsnorm(h) * w, as addnorm_perm."""
    rstd = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return (h * rstd * w.float()).half()


def _rope(x: torch.Tensor, bias, pos: torch.Tensor, theta: float, hd: int) -> torch.Tensor:
    """x fp16 [T, n, hd] (+ fp16 bias [n*hd]) rotated at positions pos [T], as rope_cache."""
    if bias is not None:
        x = x + bias.view(1, -1, hd)                       # fp16 add, as __hadd
    half = hd // 2
    inv = 1.0 / torch.pow(torch.tensor(theta, dtype=torch.float32, device=x.device),
                          torch.arange(half, device=x.device, dtype=torch.float32) * 2.0 / hd)
    ang = pos.float()[:, None] * inv[None, :]
    c = torch.cos(ang).half().float()[:, None, :]
    s = torch.sin(ang).half().float()[:, None, :]
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).half()


@torch.no_grad()
def prefill(eng, tokens: list[int] | torch.Tensor, logits: str = "all") -> torch.Tensor:
    """Run `tokens` through the engine at its current position in one pass. Needs
    eng.prefill_lins (runtime.engine_from_packed(..., prefill=True)) and eng.dense_head.
    logits: "all" -> fp32 [T, vocab] (row i predicts token i+1), "last" -> fp32 [vocab].
    Leaves eng.pos += T, eng.tok = argmax of the last row, eng.logits = last row (fp16)."""
    cfg = eng.cfg
    dev = eng.kc.device
    ids = torch.as_tensor(tokens, dtype=torch.int64, device=dev).view(-1)
    T = ids.numel()
    p0 = int(eng.pos)
    if p0 + T > cfg.max_len:
        raise ValueError(f"context {p0 + T} > max_len {cfg.max_len}")
    NH, NKV, HD = cfg.n_heads, cfg.n_kv, cfg.head_dim
    pos = torch.arange(p0, p0 + T, device=dev)
    h = eng.embed[ids % eng.embed.shape[0]].float()                     # [T, H] fp32 residual
    mask = None
    if p0 > 0:
        mask = (torch.arange(p0 + T, device=dev)[None, :] <= pos[:, None])   # [T, p0+T]
    for li, (L, P) in enumerate(zip(eng.layers, eng.prefill_lins)):
        x = _rmsnorm(h, L["ln1"], cfg.eps)
        q = F.linear(x, P["q"].dense()).view(T, NH, HD)
        k = F.linear(x, P["k"].dense()).view(T, NKV, HD)
        v = F.linear(x, P["v"].dense()).view(T, NKV, HD)
        q = _rope(q, L.get("bq"), pos, cfg.rope_theta, HD)
        k = _rope(k, L.get("bk"), pos, cfg.rope_theta, HD)
        if L.get("bv") is not None:
            v = v + L["bv"].view(1, NKV, HD)
        eng.kc[li, :, p0:p0 + T] = k.transpose(0, 1)
        eng.vc[li, :, p0:p0 + T] = v.transpose(0, 1)
        kk = eng.kc[li, :, :p0 + T].repeat_interleave(NH // NKV, dim=0)   # [NH, p0+T, HD]
        vv = eng.vc[li, :, :p0 + T].repeat_interleave(NH // NKV, dim=0)
        a = F.scaled_dot_product_attention(q.transpose(0, 1)[None], kk[None], vv[None],
                                           attn_mask=None if mask is None else mask[None, None],
                                           is_causal=mask is None, scale=HD ** -0.5)
        a = a[0].transpose(0, 1).reshape(T, NH * HD)
        h = h + F.linear(a, P["o"].dense()).float()
        x = _rmsnorm(h, L["ln2"], cfg.eps)
        g = F.linear(x, P["gate"].dense())
        u = F.linear(x, P["up"].dense())
        act = F.silu(g.float()).half() * u                                  # fp16, as silu_mul_perm
        h = h + F.linear(act, P["down"].dense()).float()
    x = _rmsnorm(h if logits == "all" else h[-1:], eng.final_norm, cfg.eps)
    lg = F.linear(x, eng.dense_head).float()
    eng.logits.copy_(lg[-1].half())
    eng.tok.copy_(lg[-1].half().float().argmax().view(1))
    eng.pos.fill_(p0 + T)
    return lg if logits == "all" else lg[-1]


def attach(eng, pf: PackedFile, names: list[dict], dense_head: torch.Tensor, local_tags: dict | None = None,
           dev="cuda") -> None:
    """Give an engine what prefill needs: per layer {q, k, ..., down: RunsPrefill | PrefillLinear}
    (names: per layer {short: full matrix name}) and the dense fp16 lm_head. Matrices the GPU
    dequant handles (no column orders) decode from the engine's run-kernel words; the rest keep
    int16 tags (local_tags, else unpacked from the file)."""
    from .kernels import cuda_dequant
    rls = [{k: getattr(eng.layers[i][k], "rl", None) for k in layer} for i, layer in enumerate(names)]
    fast = [{k: rl is not None and cuda_dequant.supported(rl) for k, rl in L.items()} for L in rls]
    n = max([rls[i][k].R * rls[i][k].nch * 4 for i, F_ in enumerate(fast) for k, ok in F_.items() if ok] or [0])
    scratch = torch.empty(n, dtype=torch.float16, device=dev) if n else None
    eng.prefill_lins = [{k: (RunsPrefill(rls[i][k], scratch) if fast[i][k]
                             else PrefillLinear.from_packed(pf, nm, dev, (local_tags or {}).get(nm)))
                         for k, nm in layer.items()} for i, layer in enumerate(names)]
    eng.prefill_scratch_bytes = n * 2
    eng.dense_head = dense_head.to(dev, torch.float16).contiguous()


def prefill_bytes(eng) -> int:
    return (sum(p.nbytes() for L in eng.prefill_lins for p in L.values()) + eng.dense_head.numel() * 2
            + getattr(eng, "prefill_scratch_bytes", 0))
