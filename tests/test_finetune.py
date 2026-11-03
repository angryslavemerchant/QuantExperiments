import torch
from torch import nn

from shapequant.finetune import CodebookLinear, kl_loss
from shapequant.masks import compress_masked


def _fit(M):
    g = torch.Generator().manual_seed(0)
    w = torch.randn(24, 40, generator=g) * 0.05
    h = torch.rand(40, generator=g) + 0.5
    fit = compress_masked(w, h, 4, 8, M=M, init="identity+random" if M > 1 else "identity",
                          rounds=2, iters=5, seed=1, keep_tags=True)
    return w, fit


def test_wrapper_matches_decoded_weight_and_bakes_exactly():
    for M in (1, 4):
        w, fit = _fit(M)
        lin = nn.Linear(40, 24)
        decoded = fit.decoded.clone()
        wrap = CodebookLinear(lin, fit, 4)
        x = torch.randn(3, 5, 40)
        y = wrap(x)
        ref = torch.nn.functional.linear(x, decoded, lin.bias)
        assert torch.allclose(y, ref, atol=1e-5), M
        baked = wrap.bake()  # fp16 rounding of books / scales is already in `decoded`
        assert torch.allclose(baked, decoded, atol=1e-6), M


def test_gradients_reach_codebooks_and_gains():
    w, fit = _fit(4)
    wrap = CodebookLinear(nn.Linear(40, 24), fit, 4)
    wrap(torch.randn(2, 40)).pow(2).sum().backward()
    assert wrap.books[0].grad is not None and wrap.books[0].grad.abs().sum() > 0
    assert wrap.log_gain.grad.abs().sum() > 0


def test_kl_loss_zero_for_identical_logits():
    a = torch.randn(4, 7, 11)
    assert float(kl_loss(a, a)) < 1e-6
    assert float(kl_loss(a, torch.zeros_like(a))) > 0
