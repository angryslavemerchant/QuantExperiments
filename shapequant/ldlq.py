"""LDLQ / GPTQ-style error feedback for the vector codebook (the rounding used by QuIP#
and QTIP, adapted to d-dimensional learned codebooks with per-row books and orders).

Plain assignment snaps every chunk to its nearest codeword under the diagonal weight
h_c. Error feedback instead walks the columns of a row in order, snaps chunk j, and
pushes its error onto the columns not yet quantized, using the full input second
moment H = E[x x^T], so later chunks compensate for earlier ones. With the upper
Cholesky factor U of H^-1 (H^-1 = U^T U), the cost of the chunk at columns S, given
the remaining columns are re-optimized afterwards, is (x - c)^T A (x - c) with
A = (U_SS^T U_SS)^-1; the error pushed onward is E U_{S,rest} with E U_SS = x - c.
Rows sharing a column order share one U (rows in a group are processed together).
"""

from __future__ import annotations

import torch


F64_MAX_C = 4096   # above this, factor in float32 (as GPTQ does): fp64 is slow on consumer GPUs


def upper_chol_inv(H: torch.Tensor, damp: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    """Upper Cholesky factor of (H + damp * mean(diag H) * I)^-1 (float32 out) and the mask
    of dead columns (zero activation), which get unit diagonal. float64 math up to
    F64_MAX_C columns; float32 above, falling back to float64 if a factorization fails."""
    dead = torch.diag(H) == 0

    def factor(dtype):
        Hd = H.to(dtype).clone()
        Hd[dead, dead] = 1.0
        Hd.diagonal().add_(damp * float(torch.diag(Hd).mean()))
        L, info = torch.linalg.cholesky_ex(Hd)
        if int(info):
            return None
        Uf, info = torch.linalg.cholesky_ex(torch.cholesky_inverse(L), upper=True)
        if int(info) or not torch.isfinite(Uf).all():
            return None
        return Uf.to(torch.float32)

    U = factor(torch.float32) if H.shape[0] > F64_MAX_C else None
    if U is None:
        U = factor(torch.float64)
    if U is None:
        raise RuntimeError("LDLQ: Cholesky failed even in float64; raise ldlq_damp")
    return U, dead


@torch.no_grad()
def ldlq_assign(u: torch.Tensor, H: torch.Tensor, books: list[torch.Tensor], row_book: torch.Tensor,
                d: int, damp: float = 0.01, block: int = 128, targets: torch.Tensor | None = None,
                U: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """u [n, C]: rows (row-normalized) in the group's column order; H [C, C] in the same
    order. books: G codebooks [K, d]; row_book [n]: each row's book.
    Returns tags [n, C // d] (index within the row's book) and the decoded rows [n, C].
    targets [n, C] (optional, filled in): each chunk's feedback-adjusted value at the
    moment it was snapped (what a codebook refit should fit). U: a precomputed
    upper_chol_inv(H, damp)[0] (reused across refit rounds)."""
    n, C = u.shape
    if C % d:
        raise ValueError("LDLQ needs C divisible by d")
    dev = u.device
    block = max(d, block - block % d)
    nch = C // d
    dead = (torch.diag(H) == 0).to(dev)
    if U is None:
        U = upper_chol_inv(H, damp)[0]
    U = U.to(dev)
    # Every step's d x d quantities up front, batched (one launch, one error check, instead of
    # an inverse + triangular solve + device sync per step): the diagonal blocks Uss, the
    # snapping metric A = (Uss^T Uss)^-1 and Uss^-1 (error pushed onward: E = (X - q) Uss^-1).
    ar = torch.arange(nch, device=dev)
    Ublk = U.view(nch, d, nch, d)[ar, :, ar, :].to(torch.float64)              # [nch, d, d]
    A_all = torch.linalg.inv(Ublk.transpose(1, 2) @ Ublk)
    A_all = (0.5 * (A_all + A_all.transpose(1, 2))).to(torch.float32)
    Uinv_all = torch.linalg.inv(Ublk).to(torch.float32)
    # Rows sorted by book so each book scores only its own rows, as contiguous slices
    # (the dominant step was every row against all G*K codewords).
    order = torch.argsort(row_book, stable=True)
    counts = torch.bincount(row_book, minlength=len(books)).tolist()
    spans, s0 = [], 0
    for g, c in enumerate(counts):
        if c:
            spans.append((g, s0, s0 + c))
        s0 += c
    W = u.to(torch.float32)[order].clone()
    W[:, dead] = 0.0
    bk = [b.to(device=dev, dtype=torch.float32) for b in books]                 # [K, d] each
    bkT = [b.T.contiguous() for b in bk]
    # c^T A c for every step and codeword, in slabs of steps (bounded memory on wide inputs)
    cAc_all = [torch.empty(nch, b.shape[0], device=dev) for b in bk]
    for c0 in range(0, nch, 256):
        Ac = A_all[c0:c0 + 256]
        for g, b in enumerate(bk):
            cAc_all[g][c0:c0 + 256] = torch.einsum("kd,sde,ke->sk", b, Ac, b)
    tags_s = torch.empty(n, nch, dtype=torch.long, device=dev)
    Q_s = torch.empty(n, C, dtype=torch.float32, device=dev)
    tg_s = torch.empty(n, C, dtype=torch.float32, device=dev) if targets is not None else None
    for b0 in range(0, C, block):
        b1 = min(b0 + block, C)
        err = torch.empty(n, b1 - b0, device=dev)
        for i in range(b0, b1, d):
            j = slice(i, i + d)
            c = i // d
            X = W[:, j]
            if tg_s is not None:
                tg_s[:, j] = X
            A = A_all[c]
            XA = X @ A
            for g, r0, r1 in spans:
                t = torch.addmm(cAc_all[g][c][None, :], XA[r0:r1], bkT[g], alpha=-2.0).argmin(1)
                tags_s[r0:r1, c] = t
                Q_s[r0:r1, j] = bk[g][t]
            E = (X - Q_s[:, j]) @ Uinv_all[c]                                   # E @ Uss = X - q
            if i + d < b1:
                W[:, i + d:b1] -= E @ U[j, i + d:b1]
            err[:, i - b0:i - b0 + d] = E
        if b1 < C:
            W[:, b1:] -= err @ U[b0:b1, b1:]
    tags = torch.empty_like(tags_s)
    tags[order] = tags_s
    Q = torch.empty_like(Q_s)
    Q[order] = Q_s
    if targets is not None:
        targets[order] = tg_s
    return tags, Q


def output_error(w: torch.Tensor, w_hat: torch.Tensor, H: torch.Tensor) -> float:
    """tr((W - W^) H (W - W^)^T) / tr(W H W^T): relative layer-output error under the full H."""
    e = (w - w_hat).to(torch.float64)
    Hd = H.to(torch.float64).to(w.device)
    num = float(((e @ Hd) * e).sum())
    w64 = w.to(torch.float64)
    return num / float(((w64 @ Hd) * w64).sum())
