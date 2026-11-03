"""Stage 3/4 research code: column orders (shuffle, masks) and local scales.

One general compressor, `compress_masked`, covers every pre-screen variant:

- Column orders. M permutations of the input columns per matrix; each row uses
  one of them (`row_mask`). M = 1 with the identity is the Stage 2 codebook,
  M = 1 with a sort is the Stage 3 shuffle, M > 1 is the Stage 4 mask codebook.
- Local scales (gain-shape VQ). After row normalization and reordering, each
  block of B consecutive weights gets its own scale, quantized to `scale_bits`
  bits on a per-matrix log grid; chunks are divided by it before snapping, so
  the codebook learns shapes, not magnitudes.
- Importance weighting. Element weight s_r^2 * h_c (times the block scale^2),
  the diagonal approximation of layer-output error, as in Stage 2c.

`fit_masks` runs the alternating loop: rows pick the order with least weighted
snap error; each mask is recomputed from the rows that chose it (columns sorted
by those rows' mean |normalized weight|); the codebook is refit (warm start).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import codebook as cb


@dataclass
class Fit:
    decoded: torch.Tensor          # [R, C] float32, original column order
    perms: torch.Tensor            # [M, C] int64
    row_mask: torch.Tensor         # [R] int64
    codebook: torch.Tensor         # [K, d] fp16
    bits: dict                     # term -> bits
    stats: dict
    tags: list | None = None       # per stage [R * n_chunks] int16 (for fine-tuning)
    row_scale: torch.Tensor | None = None  # [R] fp32 (fp16-rounded)
    pad: int = 0


def _row_scale(w: torch.Tensor) -> torch.Tensor:
    rms = w.pow(2).mean(dim=1).sqrt()
    return torch.where(rms == 0, torch.ones_like(rms), rms).to(torch.float16).to(torch.float32)


def _pad_cols(x: torch.Tensor, mult: int) -> tuple[torch.Tensor, int]:
    pad = (-x.shape[1]) % mult
    return (torch.nn.functional.pad(x, (0, pad)) if pad else x), pad


def _block_scales(u: torch.Tensor, B: int, scale_bits: int) -> tuple[torch.Tensor, float, float]:
    """Per-block RMS of u [R, Cp] (Cp % B == 0), quantized on a log2 grid with
    2^scale_bits levels spanning the matrix's [0.1%, 100%] quantiles."""
    R, Cp = u.shape
    blocks = u.reshape(R, Cp // B, B)
    rms = blocks.pow(2).mean(-1).sqrt().clamp_min(1e-8)
    lg = rms.log2()
    lo = float(torch.quantile(lg.flatten()[:: max(1, lg.numel() // 1_000_000)], 0.001))
    hi = float(lg.max())
    lo, hi = float(torch.tensor(lo).half()), float(torch.tensor(hi).half())  # stored fp16
    levels = 2 ** scale_bits - 1
    step = (hi - lo) / levels if hi > lo else 1.0
    q = ((lg - lo) / step).round().clamp(0, levels)
    return torch.exp2(lo + q * step), lo, hi  # [R, nb]


def _prepare(w, h, rs, perms, row_mask, d, B, scale_bits):
    """Reorder, normalize, (block-scale), chunk. Returns chunks, element weights
    and what decode needs."""
    R, C = w.shape
    idx = perms[row_mask]                                   # [R, C]
    u = torch.gather(w / rs[:, None], 1, idx)
    ew = torch.gather(h[None, :].expand(R, C), 1, idx) * rs.pow(2)[:, None]
    mult = math.lcm(d, B) if B else d
    u, pad = _pad_cols(u, mult)
    ew, _ = _pad_cols(ew, mult)                              # padding weight 0
    s = None
    if B:
        s, lo, hi = _block_scales(u, B, scale_bits)
        sx = s.repeat_interleave(B, dim=1)
        u = u / sx
        ew = ew * sx.pow(2)
    # Constant normalizer (independent of the order), so row errors under
    # different masks stay comparable.
    ew = ew / (h.mean() * rs.pow(2).mean())
    chunks = u.reshape(-1, d)
    return chunks, ew.reshape(-1, d), idx, pad, s


def _decode(codebook, tags, R, C, rs, idx, s, B):
    flat = codebook.to(torch.float32)[tags].reshape(R, -1)
    if s is not None:
        flat = flat * s.repeat_interleave(B, dim=1)
    u = flat[:, :C]
    out = torch.empty_like(u)
    out.scatter_(1, idx, u)                                  # undo each row's order
    return out * rs[:, None]


def _row_errors(chunks, ew, codebook, R, batch):
    _, err = cb.assign(chunks, codebook, batch, ew)
    return err.reshape(R, -1).sum(1)


def sort_perm(score: torch.Tensor) -> torch.Tensor:
    return torch.argsort(score, descending=True, stable=True)



def rvq_encode(x, wt, books, batch, refine: int = 2):
    """Residual (multi-stage) snap: greedy stage by stage, then `refine` rounds
    of coordinate re-selection of each stage's tag given the others.
    Returns (list of tags, weighted squared error per chunk)."""
    books32 = [b.to(torch.float32) for b in books]
    tags, r = [], x
    for b in books32:
        i, _ = cb.assign(r, b, batch, wt)
        tags.append(i)
        r = r - b[i]
    if len(books32) > 1:
        for _ in range(refine):
            for st, b in enumerate(books32):
                target = x - sum(bb[t] for j, (bb, t) in enumerate(zip(books32, tags)) if j != st)
                tags[st], _ = cb.assign(target, b, batch, wt)
    recon = sum(b[t] for b, t in zip(books32, tags))
    return tags, ((x - recon).pow(2) * wt).sum(1)


def rvq_fit(x, wt, Ks, iters, gen, batch, warm=None, rounds: int = 3, refit_iters: int = 2):
    """Fit one codebook per stage. Cold start: greedy k-means on successive
    residuals. Then `rounds` of alternating refinement: re-encode, and update
    each stage's codebook (warm-started Lloyd) on x minus the other stages."""
    if warm is None:
        books, r = [], x
        for K in Ks:
            c = cb.kmeans(r, K, iters, gen, batch, wt).to(torch.float16)
            books.append(c)
            i, _ = cb.assign(r, c, batch, wt)
            r = r - c.to(torch.float32)[i]
    else:
        books = [b.clone() for b in warm]
    for _ in range(rounds):
        tags, _ = rvq_encode(x, wt, books, batch, refine=1)
        for st in range(len(books)):
            others = sum(books[j].to(torch.float32)[tags[j]] for j in range(len(books)) if j != st)
            target = x - others
            books[st] = cb.kmeans(target, Ks[st], refit_iters, gen, batch, wt,
                                  init=books[st]).to(torch.float16)
            tags[st], _ = cb.assign(target, books[st], batch, wt)
    return books


def _decode_multi(books, tags, R, C, rs, idx, s, B):
    flat = sum(b.to(torch.float32)[t] for b, t in zip(books, tags)).reshape(R, -1)
    if s is not None:
        flat = flat * s.repeat_interleave(B, dim=1)
    u = flat[:, :C]
    out = torch.empty_like(u)
    out.scatter_(1, idx, u)
    return out * rs[:, None]


def compress_masked(
    w: torch.Tensor, h: torch.Tensor, d: int, K: int | tuple[int, ...], *,
    M: int = 1, init: str = "identity", update: bool = False, rounds: int = 5,
    B: int | None = None, scale_bits: int = 4,
    iters: int = 25, refit_iters: int = 10, max_fit_chunks: int = 1_000_000,
    seed: int = 0, batch: int = 65536, select_frac: float = 1.0,
    keep_tags: bool = False,
) -> Fit:
    """init: 'identity' | 'sort' (M=1 shuffle) | 'random+sort' (spec masks) |
    'cluster' (rows k-means'd on |u| profiles, one sorted order per cluster) |
    'identity+random' / 'random' (seeded orders, no storage).
    K: one codebook size, or a tuple for a residual (multi-stage) codebook."""
    Ks = (K,) if isinstance(K, int) else tuple(K)
    R, C = w.shape
    dev = w.device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rs = _row_scale(w)
    absu = (w / rs[:, None]).abs()

    if init == "identity":
        perms = torch.arange(C, device=dev)[None]
        row_mask = torch.zeros(R, dtype=torch.long, device=dev)
    elif init == "sort":
        perms = sort_perm(absu.mean(0))[None]
        row_mask = torch.zeros(R, dtype=torch.long, device=dev)
    elif init == "random+sort":
        rand = [torch.randperm(C, generator=gen).to(dev) for _ in range(M - 1)]
        perms = torch.stack([sort_perm(absu.mean(0))] + rand)
        row_mask = torch.zeros(R, dtype=torch.long, device=dev)  # chosen below
    elif init in ("identity+random", "random"):
        # Seeded random orders cost no storage (regenerated from the seed);
        # 'identity+random' keeps the original order as one of the M choices.
        rand = [torch.randperm(C, generator=gen).to(dev) for _ in range(M)]
        if init == "identity+random":
            rand[0] = torch.arange(C, device=dev)
        perms = torch.stack(rand)
        row_mask = torch.zeros(R, dtype=torch.long, device=dev)
    elif init == "cluster":
        prof = absu / absu.norm(dim=1, keepdim=True).clamp_min(1e-8)
        cent = cb.kmeans(prof, M, 15, gen, batch)
        row_mask, _ = cb.assign(prof, cent, batch)
        perms = torch.stack([sort_perm(absu[row_mask == m].mean(0)) if (row_mask == m).any()
                             else sort_perm(absu.mean(0)) for m in range(M)])
    else:
        raise ValueError(init)

    def subsample(chunks, ew):
        n = chunks.shape[0]
        if n > max_fit_chunks:
            sub = torch.randperm(n, generator=gen)[:max_fit_chunks].to(dev)
            return chunks[sub], ew[sub]
        return chunks, ew

    def fit_books(chunks, ew, warm):
        fx, fw = subsample(chunks, ew)
        if len(Ks) == 1:  # single codebook: unchanged Stage 2/2c behaviour
            if warm is None:
                return [cb.kmeans(fx, Ks[0], iters, gen, batch, fw).to(torch.float16)]
            return [cb.kmeans(fx, Ks[0], refit_iters, gen, batch, fw, init=warm[0]).to(torch.float16)]
        return rvq_fit(fx, fw, Ks, iters, gen, batch, warm=warm, rounds=3 if warm is None else 1)

    slots = None  # fixed random subset of chunk positions for mask selection

    def row_errors(chunks, ew, books):
        nonlocal slots
        nch = chunks.shape[0] // R
        if select_frac < 1.0:
            if slots is None:
                k = max(1, math.ceil(select_frac * nch))
                slots = torch.randperm(nch, generator=gen)[:k].sort().values.to(dev)
            chunks = chunks.view(R, nch, d)[:, slots].reshape(-1, d)
            ew = ew.view(R, nch, d)[:, slots].reshape(-1, d)
        _, err = rvq_encode(chunks, ew, books, batch)
        return err.reshape(R, -1).sum(1)

    chunks, ew, idx, pad, s = _prepare(w, h, rs, perms, row_mask, d, B, scale_bits)
    books = fit_books(chunks, ew, None)
    history = []
    n_rounds = rounds if perms.shape[0] > 1 else 0
    for _ in range(n_rounds):
        # (a) each row picks the order with least weighted snap error
        errs = torch.stack([
            row_errors(*_prepare(w, h, rs, perms, torch.full((R,), m, device=dev), d, B, scale_bits)[:2], books)
            for m in range(perms.shape[0])])
        row_mask = errs.argmin(0)
        history.append(float(errs.min(0).values.sum()))
        # (b) masks learn from their rows
        if update:
            for m in range(perms.shape[0]):
                sel = row_mask == m
                if sel.any():
                    perms[m] = sort_perm(absu[sel].mean(0))
        # (c) refit the codebook(s) on the reordered chunks
        chunks, ew, idx, pad, s = _prepare(w, h, rs, perms, row_mask, d, B, scale_bits)
        books = fit_books(chunks, ew, books)

    tags, _ = rvq_encode(chunks, ew, books, batch)
    decoded = _decode_multi(books, tags, R, C, rs, idx, s, B)
    Cp = C + pad
    Mn = perms.shape[0]
    reordered = not (Mn == 1 and init == "identity")
    bits = {
        "tags": R * (Cp // d) * sum(math.ceil(math.log2(k)) for k in Ks),
        "codebook": sum(Ks) * d * 16,
        "row_scales": R * 16,
        "block_scales": (R * (Cp // B) * scale_bits + 2 * 16) if B else 0,
        # seeded random orders are regenerated from a 32-bit seed; a sorted or
        # learned order is stored explicitly (16 bits per column index)
        "orders": (32 if init in ("identity+random", "random") else Mn * C * 16) if reordered else 0,
        "row_mask": R * math.ceil(math.log2(Mn)) if Mn > 1 else 0,
    }
    usage = torch.bincount(row_mask, minlength=Mn).tolist()
    return Fit(decoded, perms, row_mask, books[0], bits,
               {"mask_usage": usage, "round_errors": history, "books": books},
               tags=[t.to(torch.int16) for t in tags] if keep_tags else None,
               row_scale=rs, pad=pad)


def weighted_rel_error(w: torch.Tensor, w_hat: torch.Tensor, h: torch.Tensor) -> float:
    """sum_c h_c (w - w^)^2 / sum_c h_c w^2: relative layer-output error proxy."""
    return float(((w - w_hat).pow(2) * h[None, :]).sum() / (w.pow(2) * h[None, :]).sum())
