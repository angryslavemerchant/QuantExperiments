"""Discrete code re-pick during fine-tune (CodebookLinear.repick)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from shapequant.finetune import CodebookLinear


def _wrap(R=64, C=32, d=4, K=64, G=2, seed=0, bad=True):
    g = torch.Generator().manual_seed(seed)
    book = torch.randn(G * K, d, generator=g)
    row_book = torch.randint(0, G, (R,), generator=g)
    nch = C // d
    # target weight exactly representable: each chunk a codeword of its row's book
    true = torch.randint(0, K, (R, nch), generator=g) + row_book[:, None] * K
    tags = torch.randint(0, K, (R, nch), generator=g) + row_book[:, None] * K if bad else true.clone()
    fit = SimpleNamespace(stats={"books": [book.half()], "row_book": row_book},
                          row_mask=torch.zeros(R, dtype=torch.long), perms=torch.arange(C)[None],
                          row_scale=torch.ones(R), tags=[tags.reshape(-1).to(torch.int16)])
    lin = nn.Linear(C, R, bias=False)
    w = CodebookLinear(lin, fit, d)
    W_true = book.half().float()[true.reshape(-1)].view(R, C)
    return w, W_true


def _loss(w, W_true, x, ckpt=False):
    y = checkpoint(w, x, use_reentrant=False) if ckpt else w(x)
    return 0.5 * (y - x @ W_true.T).pow(2).sum(-1).mean()


def _captured_step(w, W_true, x, ckpt=False):
    """Mirror of the fine-tune loop's capture step for a Gaussian model y ~ N(f(x), I):
    the sampled-target log-likelihood backward ('fisher') then the real loss ('grad')."""
    y = checkpoint(w, x, use_reentrant=False) if ckpt else w(x)
    w._cap["mode"] = "fisher"
    yhat = (y + torch.randn_like(y)).detach()
    (-0.5 * (yhat - y).pow(2).sum()).backward(retain_graph=True)
    w._cap["mode"] = "grad"
    loss = 0.5 * (y - x @ W_true.T).pow(2).sum(-1).mean()
    loss.backward()
    return loss


@pytest.mark.parametrize("ckpt", [False, True])
def test_capture_counts_once_per_step(ckpt):
    w, W_true = _wrap()
    w.start_capture(torch.float32)
    x = torch.randn(3, 128, 32)
    for _ in range(3):
        _captured_step(w, W_true, x, ckpt)
    # one gradient and one curvature sample per step, however often the forward re-ran
    assert w._cap["n"] == 3 and w._cap["nA"] == 3
    # Gaussian model: E[dy^2] = 1 per output, so A / nA ~ 1
    assert torch.allclose(w._cap["A"] / 3, torch.ones(w.R), atol=0.25)


def test_repick_lowers_loss_and_respects_quota():
    torch.manual_seed(0)
    w, W_true = _wrap()
    x = torch.randn(4, 256, 32)
    before = float(_loss(w, W_true, x).detach())
    for _ in range(4):
        w.start_capture(torch.float32)
        _captured_step(w, W_true, x)
        r = w.repick(frac=0.25, damp=0.1)
        assert 0 < r["changed"] <= int(0.25 * r["chunks"])
        assert r["pred_drop"] > 0
    after = float(_loss(w, W_true, x).detach())
    assert after < 0.5 * before


def test_repick_stays_put_at_optimum():
    w, W_true = _wrap(bad=False)
    x = torch.randn(4, 256, 32)
    tags = w.tags0.clone()
    w.start_capture(torch.float32)
    _captured_step(w, W_true, x)
    r = w.repick(frac=0.5)
    assert r["changed"] == 0
    assert torch.equal(w.tags0, tags)


def test_repick_only_within_row_book():
    w, W_true = _wrap(G=2, K=64)
    x = torch.randn(4, 256, 32)
    w.start_capture(torch.float32)
    _captured_step(w, W_true, x)
    w.repick(frac=1.0)
    t = w.tags0.long().view(w.R, -1)
    assert torch.equal(t // 64, w.row_book.long()[:, None].expand_as(t))
