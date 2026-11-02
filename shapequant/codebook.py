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


def _sq_dists(x: torch.Tensor, c: torch.Tensor, c_sq: torch.Tensor,
              wt: torch.Tensor | None = None) -> torch.Tensor:
    if wt is None:
        return (x * x).sum(1, keepdim=True) - 2.0 * (x @ c.T) + c_sq[None, :]
    # Weighted: sum_i w_i (x_i - c_i)^2 = sum w x^2 - 2 (w*x).c + w.(c*c)
    return (wt * x * x).sum(1, keepdim=True) - 2.0 * ((wt * x) @ c.T) + wt @ (c * c).T


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
        d2 = _sq_dists(x[s : s + batch], c, c_sq, None if wt is None else wt[s : s + batch])
        m, i = d2.min(dim=1)
        idx[s : s + batch] = i
        err[s : s + batch] = m.clamp_min(0)
    return idx, err


def kmeans(
    x: torch.Tensor,
    K: int,
    iters: int,
    generator: torch.Generator,
    batch: int = 65536,
    wt: torch.Tensor | None = None,
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
        if total <= 0:
            pick = torch.clamp((uniforms[k] * n).long(), max=n - 1)
        else:
            pick = torch.clamp(torch.searchsorted(cdf, (uniforms[k] * total).reshape(1)), max=n - 1)
        centers[k] = x[pick].reshape(d)
        dmin = torch.minimum(dmin, _d2(centers[k]))

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
