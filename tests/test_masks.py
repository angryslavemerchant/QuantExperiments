import torch

from shapequant.masks import compress_masked, weighted_rel_error


def _patterned(R=64, C=32, d=4, n_pat=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    pats = torch.randn(n_pat, d, generator=g)
    pick = torch.randint(n_pat, (R, C // d), generator=g)
    return pats[pick].reshape(R, C)


def test_identity_roundtrip_is_exact_when_codebook_covers_patterns():
    w = _patterned()
    h = torch.ones(w.shape[1])
    fit = compress_masked(w, h, 4, 4, iters=20, seed=1)
    # row normalization makes patterns row-dependent; allow small error
    assert weighted_rel_error(w, fit.decoded, h) < 0.05


def test_orders_are_undone_exactly():
    w = _patterned(R=48, C=40)
    h = torch.rand(40) + 0.5
    for init, M in [("sort", 1), ("random+sort", 4), ("cluster", 3)]:
        fit = compress_masked(w, h, 4, 64, M=M, init=init, update=(init == "cluster"),
                              rounds=2, iters=10, seed=2)
        assert fit.decoded.shape == w.shape
        # decoding with K >= distinct chunks must land near the original in the
        # original column order (a scrambled inverse would give error ~2)
        assert weighted_rel_error(w, fit.decoded, h) < 0.5, init
        assert sum(fit.stats["mask_usage"]) == 48


def test_block_scales_absorb_magnitude_drift():
    g = torch.Generator().manual_seed(3)
    base = _patterned(R=64, C=64, n_pat=2)
    drift = torch.exp(torch.randn(64, 2, generator=g) * 1.5).repeat_interleave(32, dim=1)
    w = base * drift
    h = torch.ones(64)
    plain = compress_masked(w, h, 4, 2, iters=20, seed=4)
    scaled = compress_masked(w, h, 4, 2, B=32, scale_bits=6, iters=20, seed=4)
    assert weighted_rel_error(w, scaled.decoded, h) < 0.5 * weighted_rel_error(w, plain.decoded, h)


def test_bits_by_hand():
    w = torch.randn(16, 64)
    h = torch.ones(64)
    fit = compress_masked(w, h, 4, 16, M=4, init="random+sort", rounds=1, B=32, iters=3, seed=0)
    b = fit.bits
    assert b["tags"] == 16 * 16 * 4
    assert b["codebook"] == 16 * 4 * 16
    assert b["row_scales"] == 16 * 16
    assert b["block_scales"] == 16 * 2 * 4 + 32
    assert b["orders"] == 4 * 64 * 16
    assert b["row_mask"] == 16 * 2


def test_residual_bits_and_quality():
    g = torch.Generator().manual_seed(7)
    w = torch.randn(256, 64, generator=g)
    h = torch.ones(64)
    single = compress_masked(w, h, 4, 16, iters=15, seed=0)          # 4 bits / 4 weights
    two = compress_masked(w, h, 8, (16, 16), iters=15, seed=0)       # 8 bits / 8 weights
    assert two.bits["tags"] == single.bits["tags"]
    assert two.bits["codebook"] == 2 * 16 * 8 * 16
    one_stage = compress_masked(w, h, 8, 16, iters=15, seed=0)       # same d, first stage only
    assert weighted_rel_error(w, two.decoded, h) < weighted_rel_error(w, one_stage.decoded, h)


def test_residual_with_masks_roundtrip():
    w = _patterned(R=32, C=48, d=8)
    h = torch.rand(48) + 0.5
    fit = compress_masked(w, h, 8, (8, 8), M=4, init="identity+random", rounds=2, iters=10, seed=1)
    assert fit.decoded.shape == w.shape
    assert weighted_rel_error(w, fit.decoded, h) < 0.5
    assert fit.bits["tags"] == 32 * 6 * 6
