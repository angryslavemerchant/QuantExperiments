"""Learned weight-shape codebook: chunk, fit (k-means), snap (assign), repaint (decode).

Vocabulary follows the spec: a *chunk* is d consecutive weights in one output
row along the input dimension; the *codebook* holds K whole shapes (K counts
shapes, never per-weight values); each chunk is replaced by an integer tag
`idx` pointing at its nearest codeword.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import torch


@dataclass
class CodebookMatrix:
    idx: torch.Tensor  # [R, n_chunks] int64 tags
    codebook: torch.Tensor  # [K, d] fp16 shapes
    row_scale: torch.Tensor  # [R] fp16
    pad: int  # zero-padding added to the row length
    cols: int  # original row length C
    perm: torch.Tensor | None = None  # optional column permutation (Stage 3+)

    @property
    def d(self) -> int:
        return self.codebook.shape[1]

    @property
    def K(self) -> int:
        return self.codebook.shape[0]


def layer_seed(seed: int, name: str) -> int:
    """Stable per-matrix seed (independent of Python's hash randomization)."""
    return (seed * 1_000_003 + zlib.crc32(name.encode())) % (2**31 - 1)


def _assign_batch(x: torch.Tensor, c: torch.Tensor, c_sq: torch.Tensor,
                  wt: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest codeword and its (weighted) squared error for one batch.

    argmin_k ||x - c_k||^2 = argmin_k (c_sq_k - 2 x.c_k): one fused addmm writes
    the only [batch, K] tensor; the per-row ||x||^2 term is added after the min.
    Weighted: sum_i w_i (x_i - c_ki)^2 = sum w x^2 + [-2 w*x, w] . [c_k, c_k^2],
    again a single matmul (inner dim 2d)."""
    if wt is None:
        d2 = torch.addmm(c_sq[None, :], x, c.T, alpha=-2.0)
        m, i = d2.min(dim=1)
        return i, (m + (x * x).sum(1)).clamp_min(0)
    a = torch.cat([-2.0 * wt * x, wt], dim=1)
    b = torch.cat([c, c * c], dim=1)
    m, i = (a @ b.T).min(dim=1)
    return i, (m + (wt * x * x).sum(1)).clamp_min(0)


def assign(x: torch.Tensor, codebook: torch.Tensor, batch: int = 65536,
           wt: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Snap: index of the nearest codeword and its error, per chunk. Squared
    Euclidean, or per-element weighted squared error when `wt` (same shape as x)
    is given."""
    c = codebook.to(torch.float32)
    c_sq = (c * c).sum(1)
    idx = torch.empty(x.shape[0], dtype=torch.int64, device=x.device)
    err = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    for s in range(0, x.shape[0], batch):
        i, e = _assign_batch(x[s : s + batch], c, c_sq, None if wt is None else wt[s : s + batch])
        idx[s : s + batch] = i
        err[s : s + batch] = e
    return idx, err


# How kmeans() fits. "lloyd" (default): k-means++ then `iters` full Lloyd passes over the given chunks.
# "ema": e-means-style online fit (arXiv 2611.09501, as used by LookAtMyQuant 2 for its trellis tables):
# fresh random mini-batches drawn from ALL given chunks, each centroid a decaying weighted mean whose
# new-batch weight falls along a cosine (0.5 -> 0.01 cold, EMA_WARM_A0 -> 0.01 warm-started). The step
# count matches Lloyd's compute: iters * min(n, EMA_REF_CHUNKS) chunk assignments, times EMA_COMPUTE.
# Set by pipeline.apply_run from kcfg["fit"] / kcfg["ema_compute"].
FIT = "lloyd"
EMA_BATCH = 131072
EMA_REF_CHUNKS = 1_000_000
EMA_COMPUTE = 1.0
EMA_WARM_A0 = 0.1


def _kmeans_ema(x, centers, iters, generator, batch, wt, warm):
    import math
    n, d = x.shape
    K = centers.shape[0]
    dev = x.device
    B = min(EMA_BATCH, n)
    steps = max(8, int(round(EMA_COMPUTE * iters * min(n, EMA_REF_CHUNKS) / B)))
    a0 = EMA_WARM_A0 if warm else 0.5
    sums = torch.zeros(K, d, dtype=torch.float64, device=dev)
    wsum = torch.zeros(K, d, dtype=torch.float64, device=dev)
    for j in range(steps):
        a = 0.01 + (a0 - 0.01) * 0.5 * (1 + math.cos(math.pi * j / max(steps - 1, 1)))
        pick = torch.randint(n, (B,), generator=generator).to(dev)
        xb = x[pick]
        wb = None if wt is None else wt[pick]
        idx, _ = assign(xb, centers, batch, wb)
        bs = torch.zeros(K, d, dtype=torch.float64, device=dev)
        bw = torch.zeros(K, d, dtype=torch.float64, device=dev)
        if wb is None:
            bs.index_add_(0, idx, xb.to(torch.float64))
            bw.index_add_(0, idx, torch.ones_like(xb, dtype=torch.float64))
        else:
            bs.index_add_(0, idx, (wb * xb).to(torch.float64))
            bw.index_add_(0, idx, wb.to(torch.float64))
        if j == 0 and not warm:
            sums, wsum = bs, bw
        else:
            sums = (1 - a) * sums + a * bs
            wsum = (1 - a) * wsum + a * bw
        upd = wsum > 0
        new = torch.where(upd, sums / wsum.clamp_min(1e-300), centers.to(torch.float64))
        centers.copy_(new.to(torch.float32))
    return centers


def kmeans(
    x: torch.Tensor,
    K: int,
    iters: int,
    generator: torch.Generator,
    batch: int = 65536,
    wt: torch.Tensor | None = None,
    init: torch.Tensor | None = None,
) -> torch.Tensor:
    """k-means++ init then Lloyd iterations. Empty clusters are re-seeded from the
    chunks with the largest current error. Centroid sums accumulate in float64 so
    results are reproducible despite atomic-add ordering on GPU.

    With per-element weights `wt`, distances are weighted squared errors and each
    centroid coordinate is the weight-averaged member coordinate (the exact
    minimizer of the weighted objective)."""
    n, d = x.shape
    if n < K:
        raise ValueError(f"k-means needs at least K={K} chunks, got {n}")
    dev = x.device
    if FIT == "ema":
        if init is not None:
            return _kmeans_ema(x, init.to(dev, torch.float32).clone(), iters, generator, batch, wt, warm=True)
        m = min(n, EMA_REF_CHUNKS)   # k-means++ seeds from a subsample, then the online fit sees everything
        s = torch.randperm(n, generator=generator)[:m].to(dev) if m < n else None
        xs, ws = (x, wt) if s is None else (x[s], None if wt is None else wt[s])
        seeds = _kmeanspp(xs, K, generator, ws)
        return _kmeans_ema(x, seeds, iters, generator, batch, wt, warm=False)
    if init is not None:  # warm start (used by the mask fitting loop): skip k-means++
        centers = init.to(dev, torch.float32).clone()
        return _lloyd(x, centers, iters, batch, wt)
    return _lloyd(x, _kmeanspp(x, K, generator, wt), iters, batch, wt)


def _kmeanspp(x: torch.Tensor, K: int, generator: torch.Generator, wt: torch.Tensor | None) -> torch.Tensor:
    n, d = x.shape
    dev = x.device
    centers = torch.empty(K, d, dtype=torch.float32, device=dev)
    first = torch.randint(n, (1,), generator=generator, device="cpu").item()
    centers[0] = x[first]

    def _d2(c: torch.Tensor) -> torch.Tensor:
        e = (x - c) ** 2
        return (e if wt is None else e * wt).sum(1)

    dmin = _d2(centers[0])
    # D^2 sampling by inverse CDF on-device; uniforms come from the CPU generator
    # so the draw sequence is reproducible from the seed.
    uniforms = torch.rand(K, generator=generator, dtype=torch.float64).to(dev)
    for k in range(1, K):
        cdf = torch.cumsum(dmin.to(torch.float64), 0)
        total = cdf[-1]
        # No host sync: both candidates are computed on device and selected with
        # torch.where (identical picks to the former `if total <= 0` branch).
        uniform_pick = torch.clamp((uniforms[k] * n).long(), max=n - 1).reshape(1)
        cdf_pick = torch.clamp(torch.searchsorted(cdf, (uniforms[k] * total).reshape(1)), max=n - 1)
        pick = torch.where(total > 0, cdf_pick, uniform_pick)
        centers[k] = x[pick].reshape(d)
        dmin = torch.minimum(dmin, _d2(centers[k]))
    return centers


def _lloyd(x: torch.Tensor, centers: torch.Tensor, iters: int, batch: int,
           wt: torch.Tensor | None) -> torch.Tensor:
    K, d = centers.shape
    dev = x.device
    for _ in range(iters):
        idx, err = assign(x, centers, batch, wt)
        sums = torch.zeros(K, d, dtype=torch.float64, device=dev)
        counts = torch.bincount(idx, minlength=K)
        nonempty = counts > 0
        if wt is None:
            sums.index_add_(0, idx, x.to(torch.float64))
            centers[nonempty] = (sums[nonempty] / counts[nonempty, None]).to(torch.float32)
        else:
            sums.index_add_(0, idx, (wt * x).to(torch.float64))
            wsum = torch.zeros(K, d, dtype=torch.float64, device=dev)
            wsum.index_add_(0, idx, wt.to(torch.float64))
            # A coordinate whose members all have zero weight (e.g. only padding)
            # keeps its old value: it never affects the weighted error.
            upd = nonempty[:, None] & (wsum > 0)
            new = torch.where(upd, sums / wsum.clamp_min(1e-300), centers.to(torch.float64))
            centers.copy_(new.to(torch.float32))
        n_empty = int((~nonempty).sum())
        if n_empty:
            worst = torch.topk(err, n_empty).indices
            centers[~nonempty] = x[worst]
    return centers


def chunk_rows(w: torch.Tensor, d: int) -> tuple[torch.Tensor, int]:
    """[R, C] -> [R * ceil(C/d), d], zero-padding each row to a multiple of d."""
    rows, cols = w.shape
    pad = (-cols) % d
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    return w.reshape(rows * ((cols + pad) // d), d), pad


def compress(
    w: torch.Tensor,
    d: int,
    K: int,
    *,
    iters: int,
    max_fit_chunks: int,
    seed: int,
    batch: int = 65536,
    perm: torch.Tensor | None = None,
    col_weight: torch.Tensor | None = None,
    row_weighted: bool = False,
) -> tuple[CodebookMatrix, dict]:
    """Row-normalize, chunk, fit the codebook on a subsample, snap every chunk.

    Weighting (Stage 2c): the layer-output error of row r is approximately
    sum_c h_c (w_rc - w^_rc)^2 with h_c = E[x_c^2] over calibration inputs. In
    normalized space w = s_r * u, so each element's weight is s_r^2 * h_c.
    `col_weight` supplies h (length C, original column order); `row_weighted`
    adds the s_r^2 factor. Weights are normalized to mean 1 and padding gets 0."""
    rows, cols = w.shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if perm is not None:
        w = w[:, perm]
    rms = w.pow(2).mean(dim=1).sqrt()
    row_scale = torch.where(rms == 0, torch.ones_like(rms), rms).to(torch.float16)
    wn = w / row_scale.to(torch.float32)[:, None]
    chunks, pad = chunk_rows(wn, d)
    wt = None
    if col_weight is not None or row_weighted:
        h = (torch.ones(cols, device=w.device) if col_weight is None
             else col_weight.to(w.device, torch.float32))
        if perm is not None and col_weight is not None:
            h = h[perm]
        full = h[None, :].expand(rows, cols)
        if row_weighted:
            full = full * row_scale.to(torch.float32).pow(2)[:, None]
        full = full / full.mean()
        wt, _ = chunk_rows(full.contiguous(), d)  # padding columns get weight 0
    n = chunks.shape[0]
    if n > max_fit_chunks:
        sub = torch.randperm(n, generator=gen)[:max_fit_chunks].to(chunks.device)
        fit = chunks[sub]
        fit_wt = None if wt is None else wt[sub]
    else:
        fit, fit_wt = chunks, wt
    centers = kmeans(fit, K, iters, gen, batch, fit_wt)
    codebook = centers.to(torch.float16)  # stored at 16-bit; snap against what is stored
    idx, err = assign(chunks, codebook, batch, wt)
    used = torch.unique(idx).numel()
    cm = CodebookMatrix(idx=idx.reshape(rows, -1), codebook=codebook, row_scale=row_scale,
                        pad=pad, cols=cols, perm=perm)
    stats = {"codebook_usage": used / K, "normalized_chunk_mse": float(err.mean())}
    return cm, stats


def decode(cm: CodebookMatrix) -> torch.Tensor:
    """Repaint: tags -> shapes -> rows, strip padding, rescale rows, undo the shuffle."""
    rows = cm.idx.shape[0]
    flat = cm.codebook.to(torch.float32)[cm.idx.reshape(-1)]
    w = flat.reshape(rows, -1)
    if cm.pad:
        w = w[:, : cm.cols]
    w = w * cm.row_scale.to(torch.float32)[:, None]
    if cm.perm is not None:
        inv = torch.empty_like(cm.perm)
        inv[cm.perm] = torch.arange(cm.perm.numel(), device=cm.perm.device)
        w = w[:, inv]
    return w
