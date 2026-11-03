import torch

from shapequant.masks import compress_masked, compress_mixture, weighted_rel_error


def _w(seed=0, R=64, C=96):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(R, C, generator=g)
    w[: R // 2] = torch.randn(R // 2, C, generator=g).pow(3)  # two row "shapes"
    return w, torch.rand(C, generator=g) + 0.1


def test_mixture_g1_matches_shape_and_bits():
    w, h = _w()
    fit = compress_mixture(w, h, 4, 16, G=1, seed=0)
    assert fit.decoded.shape == w.shape
    assert fit.bits["row_book"] == 0 and fit.bits["codebook"] == 16 * 4 * 16


def test_mixture_helps_and_counts_bits():
    w, h = _w()
    base = compress_masked(w, h, 4, 16, seed=0)
    mix = compress_mixture(w, h, 4, 16, G=4, seed=0)
    assert mix.bits["row_book"] == 64 * 2 and mix.bits["codebook"] == 4 * 16 * 4 * 16
    assert sum(mix.stats["book_usage"]) == 64
    assert weighted_rel_error(w, mix.decoded, h) < weighted_rel_error(w, base.decoded, h)


def test_mixture_with_masks_decodes_in_original_order():
    w, h = _w(1)
    fit = compress_mixture(w, h, 4, 16, G=2, M=4, seed=0)
    assert fit.bits["row_mask"] == 64 * 2
    assert weighted_rel_error(w, fit.decoded, h) < 0.5


def test_mixture_alternating_selection_with_many_pairs():
    w, h = _w(2)
    fit = compress_mixture(w, h, 4, 16, G=4, M=8, rounds=2, seed=0)  # 32 pairs -> alternating
    assert sum(fit.stats["book_usage"]) == 64 and sum(fit.stats["mask_usage"]) == 64
    base = compress_masked(w, h, 4, 16, seed=0)
    assert weighted_rel_error(w, fit.decoded, h) < weighted_rel_error(w, base.decoded, h)


def test_mixture_tags_feed_the_finetune_wrapper_exactly():
    from torch import nn
    from shapequant.finetune import CodebookLinear
    w, h = _w(3, R=24, C=40)
    for G, M in ((1, 1), (3, 1), (2, 4), (4, 8)):
        fit = compress_mixture(w, h, 4, 8, G=G, M=M, rounds=2, iters=5, seed=1, keep_tags=True)
        assert fit.stats["books"][0].shape == (G * 8, 4)
        lin = nn.Linear(40, 24)
        decoded = fit.decoded.clone()
        wrap = CodebookLinear(lin, fit, 4)
        x = torch.randn(3, 40)
        assert torch.allclose(wrap(x), torch.nn.functional.linear(x, decoded, lin.bias), atol=1e-5), (G, M)
        assert torch.allclose(wrap.bake(), decoded, atol=1e-6), (G, M)
