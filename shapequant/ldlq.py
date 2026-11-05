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
    block = max(d, block - block % d)
    dead = (torch.diag(H) == 0).to(u.device)
    if U is None:
        U = upper_chol_inv(H, damp)[0]
    U = U.to(u.device)
    W = u.to(torch.float32).clone()
    W[:, dead] = 0.0
    allb = torch.cat([b.to(torch.float32) for b in books]).to(u.device)       # [G*K, d]
    K = books[0].shape[0]
    pen = torch.full((n, allb.shape[0]), float("inf"), device=u.device)
    for g in range(len(books)):
        pen[row_book == g, g * K:(g + 1) * K] = 0.0
    tags = torch.empty(n, C // d, dtype=torch.long, device=u.device)
    Q = torch.empty(n, C, dtype=torch.float32, device=u.device)
    for b0 in range(0, C, block):
        b1 = min(b0 + block, C)
        err = torch.empty(n, b1 - b0, device=u.device)
        for i in range(b0, b1, d):
            j = slice(i, i + d)
            X = W[:, j]
            if targets is not None:
                targets[:, j] = X
            Uss = U[j, j]
            A = torch.linalg.inv(Uss.T @ Uss)
            A = 0.5 * (A + A.T)
            cAc = ((allb @ A) * allb).sum(1)
            t = (cAc[None, :] - 2.0 * (X @ A) @ allb.T + pen).argmin(1)
            q = allb[t]
            tags[:, i // d] = t - row_book * K
            Q[:, j] = q
            E = torch.linalg.solve_triangular(Uss, X - q, upper=True, left=False)   # E @ Uss = X - q
            if i + d < b1:
                W[:, i + d:b1] -= E @ U[j, i + d:b1]
            err[:, i - b0:i - b0 + d] = E
        if b1 < C:
            W[:, b1:] -= err @ U[b0:b1, b1:]
    return tags, Q


def output_error(w: torch.Tensor, w_hat: torch.Tensor, H: torch.Tensor) -> float:
    """tr((W - W^) H (W - W^)^T) / tr(W H W^T): relative layer-output error under the full H."""
    e = (w - w_hat).to(torch.float64)
    Hd = H.to(torch.float64).to(w.device)
    num = float(((e @ Hd) * e).sum())
    w64 = w.to(torch.float64)
    return num / float(((w64 @ Hd) * w64).sum())
