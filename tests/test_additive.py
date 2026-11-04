import torch
from torch import nn

from shapequant import alloc as al
from shapequant.masks import compress_masked, rvq_encode


def _w(seed, R=40, C=64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(R, C, generator=g), torch.rand(C, generator=g) + 0.1


def test_two_small_books_encode_exactly():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(500, 4, generator=g)
    wt = torch.ones_like(x)
    books = [torch.randn(8, 4, generator=g).half(), torch.randn(8, 4, generator=g).half()]
    tags, err = rvq_encode(x, wt, books, batch=4096)
    combo = (books[0].float()[:, None] + books[1].float()[None]).reshape(-1, 4)
    best = ((x[:, None] - combo[None]) ** 2).sum(-1).min(1).values
    assert torch.allclose(err, best, atol=1e-5)
    recon = books[0].float()[tags[0]] + books[1].float()[tags[1]]
    assert torch.allclose(((x - recon) ** 2).sum(1), best, atol=1e-5)


def test_additive_fit_feeds_the_finetune_wrapper_exactly():
    from shapequant.finetune import CodebookLinear
    w, h = _w(1)
    for M in (1, 4):
        fit = compress_masked(w, h, 4, (8, 8), M=M, init="identity+random" if M > 1 else "identity",
                              rounds=1, iters=3, refit_iters=2, seed=5, keep_tags=True)
        assert len(fit.tags) == 2 and len(fit.stats["books"]) == 2
        lin = nn.Linear(64, 40)
        wrap = CodebookLinear(lin, fit, 4)
        x = torch.randn(3, 64)
        assert torch.allclose(wrap(x), torch.nn.functional.linear(x, fit.decoded, lin.bias), atol=1e-5)
        assert torch.allclose(wrap.bake(), fit.decoded, atol=1e-6)


def test_budget_spec_accepts_additive_options():
    Ks = ((16, 16), (32, 32), (64, 64))
    bits = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
    assert al.budget_bits(bits, Ks, "uniform:32x32") == 22.0
