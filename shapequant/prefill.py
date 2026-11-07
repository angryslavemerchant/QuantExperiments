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
The lm_head: an 8-bit head is dequantized in row chunks at the end (no dense copy); fp16 and q4 heads keep a dense fp16 copy.

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


IMPL = "fast"          # default prefill implementation: "fast" (fused kernels, grouped GEMMs) | "ref"
GEMM = "dequant"       # default fast-path GEMM: "dequant" (decode to fp16 scratch + cuBLAS) | "fused"


@torch.no_grad()
def prefill(eng, tokens: list[int] | torch.Tensor, logits: str = "all", impl: str | None = None,
            gemm: str | None = None) -> torch.Tensor:
    """Run `tokens` through the engine at its current position in one pass. Needs
    eng.prefill_lins (runtime.engine_from_packed(..., prefill=True)); the head is eng.dense_head when
    kept (fp16 / q4 heads), else the engine's 8-bit head dequantized in chunks.
    logits: "all" -> fp32 [T, vocab] (row i predicts token i+1), "last" -> fp32 [vocab].
    impl: "fast" (default, IMPL) = fused prefill kernels (kernels.prefill_kernels) and one GEMM per
    group of matrices sharing an input (q|k|v, gate|up); "ref" = the original torch path.
    gemm (fast path): "dequant" (default, GEMM) | "fused" (kernels.runs_gemm, no fp16 scratch).
    Leaves eng.pos += T, eng.tok = argmax of the last row, eng.logits = last row (fp16)."""
    if (impl or IMPL) == "fast":
        return _prefill_fast(eng, tokens, logits, gemm or GEMM)
    return _prefill_ref(eng, tokens, logits)


@torch.no_grad()
def _prefill_ref(eng, tokens, logits: str = "all") -> torch.Tensor:
    cfg = eng.cfg
    dev = eng.kc.device
    ids = torch.as_tensor(tokens, dtype=torch.int64, device=dev).view(-1)
    T = ids.numel()
    p0 = int(eng.pos)
    if p0 + T > cfg.max_len:
        raise ValueError(f"context {p0 + T} > max_len {cfg.max_len}")
    NH, NKV, HD = cfg.n_heads, cfg.n_kv, cfg.head_dim
    pos = torch.arange(p0, p0 + T, device=dev)
    rid = ids % eng.embed.shape[0]
    h = eng.embed.rows(rid) if hasattr(eng.embed, "rows") else eng.embed[rid].float()   # [T, H] fp32 residual
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
    lg = _head(eng, x)
    eng.logits.copy_(lg[-1].half())
    eng.tok.copy_(lg[-1].half().float().argmax().view(1))
    eng.pos.fill_(p0 + T)
    return lg if logits == "all" else lg[-1]


class Group:
    """Prefill matrices sharing one input (q|k|v, gate|up, ...): their weights decode into adjacent
    rows of one fp16 scratch, so one GEMM computes all outputs ([T, sum R], members in order)."""

    def __init__(self, members: list):
        self.members = members
        self.C = members[0].C
        assert all(m.C == self.C for m in members)
        self.R = sum(m.R for m in members)
        self.n = self.R * self.C
        self.runs = [getattr(m, "rl", None) for m in members]

    def fused_ok(self) -> bool:
        from .kernels import runs_gemm
        return all(rl is not None and runs_gemm.supported(rl) for rl in self.runs)

    def dense(self, scratch: torch.Tensor) -> torch.Tensor:
        from .kernels import cuda_dequant
        off = 0
        for m in self.members:
            n = m.R * m.C
            if isinstance(m, RunsPrefill):
                cuda_dequant.dequant(m.rl, scratch[off:off + n])
            else:
                scratch[off:off + n].view(m.R, m.C).copy_(m.dense())
            off += n
        return scratch[:self.n].view(self.R, self.C)


def group_scratch(eng, groups, nbuf: int = 1) -> torch.Tensor:
    """The engine's fp16 scratch for grouped dequant: nbuf buffers of its largest group (allocated
    once, grown on demand). Returns [nbuf, n]."""
    n = max(g.n for L in groups for g in L.values() if isinstance(g, Group))
    n = (n + 63) // 64 * 64
    s = getattr(eng, "prefill_group_scratch", None)
    if s is None or s.numel() < n * nbuf:
        eng.prefill_group_scratch = None
        s = eng.prefill_group_scratch = torch.empty(n * nbuf, dtype=torch.float16, device=eng.logits.device)
    return s[:n * nbuf].view(nbuf, n)


def needs_scratch(groups, gemm: str) -> bool:
    return gemm != "fused" or not all(g.fused_ok() for L in groups for g in L.values() if isinstance(g, Group))


class Gemms:
    """The fast prefill's GEMMs, called with the groups in a fixed order (seq) per prefill.
    gemm "dequant": each group decodes into the shared scratch right before its cuBLAS GEMM;
    "overlap": the next group decodes on a side stream into a second buffer while the current
    GEMM (and whatever runs between GEMMs) proceeds - same values, the dequant mostly hidden;
    "fused": kernels.runs_gemm straight from the codes (groups it cannot run fall back to dequant)."""

    def __init__(self, eng, groups, seq: list, gemm: str = "dequant"):
        self.seq, self.gemm, self.i = seq, gemm, 0
        self.overlap = gemm == "overlap"
        self.bufs = (group_scratch(eng, groups, 2 if self.overlap else 1) if needs_scratch(groups, gemm) else None)
        if self.overlap:
            self.side = getattr(eng, "_prefill_side_stream", None)
            if self.side is None:
                self.side = eng._prefill_side_stream = torch.cuda.Stream(device=eng.logits.device)
            self.pending, self.free = {}, [None, None]

    def _issue(self, j: int):
        main = torch.cuda.current_stream()
        b = j % 2
        if self.free[b] is not None:
            self.side.wait_event(self.free[b])        # the GEMM that last read this buffer is done
        else:
            self.side.wait_stream(main)               # earlier work (a previous prefill) may still read it
        with torch.cuda.stream(self.side):
            w = self.seq[j].dense(self.bufs[b])
            ev = torch.cuda.Event()
            ev.record(self.side)
        self.pending[j] = (w, ev)

    def __call__(self, g: Group, x: torch.Tensor) -> torch.Tensor:
        j = self.i
        assert self.seq[j] is g, "groups must be multiplied in the declared order"
        self.i += 1
        if self.gemm == "fused" and g.fused_ok():
            from .kernels import runs_gemm
            return runs_gemm.gemm(x, g.runs)
        if not self.overlap:
            return F.linear(x, g.dense(self.bufs[0]))
        if j == 0:
            self._issue(0)
        if j + 1 < len(self.seq):
            self._issue(j + 1)
        w, ev = self.pending.pop(j)
        main = torch.cuda.current_stream()
        main.wait_event(ev)
        y = F.linear(x, w)
        fe = torch.cuda.Event()
        fe.record(main)
        self.free[j % 2] = fe
        return y


def _groups(eng):
    """Per layer {qkv, o, gu, down: Group} from eng.prefill_lins (cached on the engine)."""
    gs = getattr(eng, "_prefill_groups", None)
    if gs is None or gs[0] is not eng.prefill_lins:
        G = [{"qkv": Group([P["q"], P["k"], P["v"]]), "o": Group([P["o"]]),
              "gu": Group([P["gate"], P["up"]]), "down": Group([P["down"]])} for P in eng.prefill_lins]
        gs = eng._prefill_groups = (eng.prefill_lins, G)
    return gs[1]


def _rope_tables(pos: torch.Tensor, theta: float, hd: int):
    """fp16 cos / sin [T, hd/2] exactly as _rope computes them."""
    half = hd // 2
    inv = 1.0 / torch.pow(torch.tensor(theta, dtype=torch.float32, device=pos.device),
                          torch.arange(half, device=pos.device, dtype=torch.float32) * 2.0 / hd)
    ang = pos.float()[:, None] * inv[None, :]
    return torch.cos(ang).half().contiguous(), torch.sin(ang).half().contiguous()


@torch.no_grad()
def _prefill_fast(eng, tokens, logits: str = "all", gemm: str = "dequant") -> torch.Tensor:
    """prefill() with the fused kernels: same math as _prefill_ref (fp32 residual, fp16 GEMM
    inputs / outputs, rope with fp16-rounded tables, fp16 SiLU * up), fewer and larger launches."""
    from .kernels import prefill_kernels as pk
    cfg = eng.cfg
    dev = eng.kc.device
    ids = torch.as_tensor(tokens, dtype=torch.int64, device=dev).view(-1)
    T = ids.numel()
    p0 = int(eng.pos)
    if p0 + T > cfg.max_len:
        raise ValueError(f"context {p0 + T} > max_len {cfg.max_len}")
    NH, NKV, HD = cfg.n_heads, cfg.n_kv, cfg.head_dim
    groups = _groups(eng)
    mm = Gemms(eng, groups, [G[k] for G in groups for k in ("qkv", "o", "gu", "down")], gemm)
    pos = torch.arange(p0, p0 + T, device=dev)
    cs, sn = _rope_tables(pos, cfg.rope_theta, HD)
    rid = ids % eng.embed.shape[0]
    h = (eng.embed.rows(rid) if hasattr(eng.embed, "rows") else eng.embed[rid]).float().contiguous()
    mask = None
    if p0 > 0:
        mask = (torch.arange(p0 + T, device=dev)[None, :] <= pos[:, None])
    add = None
    for li, (L, G) in enumerate(zip(eng.layers, groups)):
        x = pk.addnorm(h, add, L["ln1"], cfg.eps)
        qkv = mm(G["qkv"], x)
        q = pk.rope_qkv(qkv, L.get("bq"), L.get("bk"), L.get("bv"), cs, sn, NH, NKV, HD, p0, eng.kc[li], eng.vc[li])
        kk = eng.kc[li, :, :p0 + T].repeat_interleave(NH // NKV, dim=0)
        vv = eng.vc[li, :, :p0 + T].repeat_interleave(NH // NKV, dim=0)
        a = F.scaled_dot_product_attention(q[None], kk[None], vv[None],
                                           attn_mask=None if mask is None else mask[None, None],
                                           is_causal=mask is None, scale=HD ** -0.5)
        a = a[0].transpose(0, 1).reshape(T, NH * HD)
        add = mm(G["o"], a)
        x = pk.addnorm(h, add, L["ln2"], cfg.eps)
        act = pk.silu_mul(mm(G["gu"], x), 0)
        add = mm(G["down"], act)
    if logits == "all":
        x = pk.addnorm(h, add, eng.final_norm, cfg.eps)
    else:
        x = pk.addnorm(h[-1:], add[-1:], eng.final_norm, cfg.eps)
    lg = head_fast(eng, x)
    eng.logits.copy_(lg[-1].half())
    eng.tok.copy_(lg[-1].half().float().argmax().view(1))
    eng.pos.fill_(p0 + T)
    return lg if logits == "all" else lg[-1]


def head_fast(eng, x: torch.Tensor) -> torch.Tensor:
    """_head for the fast paths: one row through the engine's own head kernel (the decode GEMV),
    several rows through 8-bit chunks dequantized by one kernel each."""
    if eng.dense_head is not None:
        return F.linear(x, eng.dense_head).float()
    hd = eng.lm_head
    if x.shape[0] == 1 and hasattr(hd, "run"):
        y = torch.empty(hd.R, dtype=torch.float16, device=x.device)
        hd.run(x[0].contiguous(), y)
        return y.float()[None]
    from .kernels import prefill_kernels as pk
    out = torch.empty(x.shape[0], hd.R, dtype=torch.float32, device=x.device)
    buf = torch.empty(min(HEAD_CHUNK, hd.R) * hd.C, dtype=torch.float16, device=x.device)
    for s in range(0, hd.R, HEAD_CHUNK):
        e = min(s + HEAD_CHUNK, hd.R)
        out[:, s:e] = F.linear(x, pk.q8_rows(hd.q, hd.sc, s, e, buf))
    return out


HEAD_CHUNK = 16384      # rows of the 8-bit head dequantized at a time (no dense head copy)


def _head(eng, x: torch.Tensor) -> torch.Tensor:
    """fp32 logits of the normed rows x [T, H]: the dense fp16 head if the engine keeps one,
    else the 8-bit head dequantized chunk by chunk."""
    if eng.dense_head is not None:
        return F.linear(x, eng.dense_head).float()
    from .packed import q8_rows
    hd = eng.lm_head
    out = torch.empty(x.shape[0], hd.R, dtype=torch.float32, device=x.device)
    for s in range(0, hd.R, HEAD_CHUNK):
        e = min(s + HEAD_CHUNK, hd.R)
        out[:, s:e] = F.linear(x, q8_rows(hd.q, hd.sc, s, e).to(x.dtype)).float()
    return out


def attach(eng, pf: PackedFile, names: list[dict], dense_head: torch.Tensor | None, local_tags: dict | None = None,
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
    eng.dense_head = None if dense_head is None else dense_head.to(dev, torch.float16).contiguous()


def prefill_bytes(eng) -> int:
    dh = 0 if eng.dense_head is None else eng.dense_head.numel() * 2
    return (sum(p.nbytes() for L in eng.prefill_lins for p in L.values()) + dh
            + getattr(eng, "prefill_scratch_bytes", 0))
