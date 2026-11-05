"""LDLQ error feedback (shapequant/ldlq.py), its use inside compress_mixture, and rotated
fits through the pipeline and the fine-tune wrapper."""

import torch

from shapequant import codebook as cb
from shapequant import masks as mk
from shapequant.finetune import CodebookLinear
from shapequant.ldlq import ldlq_assign, output_error
from shapequant.pipeline import compress_matrix, random_orthogonal

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _correlated_H(C, seed=0, rho=0.9):
    g = torch.Generator().manual_seed(seed)
    idx = torch.arange(C, dtype=torch.float64)
    H = rho ** (idx[:, None] - idx[None, :]).abs()               # AR(1): strong neighbour correlation
    s = torch.rand(C, generator=g, dtype=torch.float64) * 3 + 0.2
    return (H * s[:, None] * s[None, :]).float()


def test_identity_H_equals_nearest_snap():
    g = torch.Generator().manual_seed(0)
    u = torch.randn(64, 128, generator=g).to(DEV)
    books = [torch.randn(32, 4, generator=g).to(DEV) for _ in range(2)]
    rb = torch.randint(0, 2, (64,), generator=g).to(DEV)
    tags, q = ldlq_assign(u, torch.eye(128), books, rb, 4)
    for r in range(64):
        x = u[r].view(-1, 4)
        ref, _ = cb.assign(x, books[int(rb[r])])
        assert torch.equal(tags[r], ref)
        assert torch.allclose(q[r], books[int(rb[r])][ref].reshape(-1))


def test_feedback_lowers_output_error_under_correlated_H():
    g = torch.Generator().manual_seed(1)
    C = 256
    H = _correlated_H(C)
    w = torch.randn(96, C, generator=g).to(DEV)
    book = torch.randn(256, 4, generator=g).to(DEV) * 0.8
    rb = torch.zeros(96, dtype=torch.long, device=DEV)
    _, q_ldlq = ldlq_assign(w, H, [book], rb, 4, damp=1e-4)
    t, _ = cb.assign(w.reshape(-1, 4), book)
    q_near = book[t].reshape(96, C)
    e_ldlq, e_near = output_error(w, q_ldlq, H), output_error(w, q_near, H)
    assert e_ldlq < 0.8 * e_near, (e_ldlq, e_near)


def test_mixture_with_isotropic_H_matches_plain_assignment():
    """H = c I gives the same tags as the diagonal path (uniform h), through masks + books:
    checks the per-order bookkeeping of the LDLQ branch."""
    g = torch.Generator().manual_seed(2)
    R, C = 128, 64
    w = torch.randn(R, C, generator=g).to(DEV)
    h = torch.ones(C, device=DEV)
    kw = dict(G=2, M=4, rounds=2, iters=5, refit_iters=2, seed=3, keep_tags=True)
    a = mk.compress_mixture(w, h, 4, 16, **kw)
    b = mk.compress_mixture(w, h, 4, 16, H=torch.eye(C, device=DEV) * 2.0, ldlq_damp=0.0, **kw)
    assert torch.equal(a.row_mask, b.row_mask)
    assert torch.equal(a.tags[0], b.tags[0])
    assert torch.allclose(a.decoded, b.decoded, atol=1e-6)


def test_random_orthogonal_is_orthogonal_and_seeded():
    q = random_orthogonal(96, 5, DEV)
    assert torch.allclose(q @ q.T, torch.eye(96, device=DEV), atol=1e-5)
    assert torch.equal(q, random_orthogonal(96, 5, DEV))


def test_rotated_ldlq_fit_roundtrips_through_finetune_wrapper():
    g = torch.Generator().manual_seed(4)
    R, C = 96, 128
    w = torch.randn(R, C, generator=g).to(DEV) * 0.05
    name = "model.layers.0.mlp.up_proj"
    H = _correlated_H(C, seed=4).to(DEV)
    run = dict(method="stack", d=4, K=16, G=2, M=4, rounds=1, rotate="both", ldlq=True, finetune={"steps": 1})
    kcfg = dict(iters=4, refit_iters=2, max_fit_chunks=100000, assign_batch=4096)
    out, bits, _, stats = compress_matrix(w, run, kcfg, 0, name, importance={name: torch.diag(H).cpu()},
                                          hessians={name: H.cpu()})
    fit = stats["fit"]
    assert out.shape == w.shape and torch.isfinite(out).all()
    assert float((out - w).pow(2).sum() / w.pow(2).sum()) < 1.0
    lin = torch.nn.Linear(C, R, bias=False).to(DEV)
    lin.weight.data.copy_(out)
    wrap = CodebookLinear(lin, fit, 4).to(DEV)
    x = torch.randn(5, C, generator=g).to(DEV)
    with torch.no_grad():
        y = wrap(x)
        assert torch.allclose(y, x @ out.T, atol=1e-4), float((y - x @ out.T).abs().max())
        baked = wrap.bake()
    assert torch.allclose(baked, out, atol=2e-3)                 # fp16 rounding of books/scales only
