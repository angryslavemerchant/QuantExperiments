import math
from decimal import Decimal

import pytest
import torch

from shapequant import codebook as cb
from shapequant import memory
from shapequant.evaluate import extract_answer, gold_answer
from shapequant.rtn import rtn_quantize, ternary_quantize


def _w(r=64, c=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(r, c, generator=g) * 0.02


# ---------------------------------------------------------------- RTN

@pytest.mark.parametrize("bits", [2, 3, 4])
def test_rtn_level_count_and_range(bits):
    w = _w()
    out = rtn_quantize(w, bits, 32)
    groups = out.reshape(64, -1, 32)
    for g in groups.reshape(-1, 32)[:50]:
        assert torch.unique(g).numel() <= 2 ** bits
    # the max-|w| element in each group is represented exactly (up to fp16 scale rounding)
    src = w.reshape(64, -1, 32)
    i = src.abs().argmax(-1, keepdim=True)
    assert torch.allclose(torch.gather(groups, -1, i), torch.gather(src, -1, i), rtol=2e-3)


def test_rtn_error_shrinks_with_bits():
    w = _w()
    errs = [((rtn_quantize(w, b) - w) ** 2).mean() for b in (2, 3, 4)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] / (w ** 2).mean() < 0.02  # Q4 is close


def test_rtn_handles_padding_and_zero_groups():
    w = _w(4, 40)
    w[0, :32] = 0
    out = rtn_quantize(w, 4, 32)
    assert out.shape == w.shape and torch.all(out[0, :32] == 0)


def test_ternary_values():
    w = _w()
    out = ternary_quantize(w, 32).reshape(64, -1, 32)
    for g in out.reshape(-1, 32)[:50]:
        vals = torch.unique(g)
        assert vals.numel() <= 3
        nz = vals[vals != 0]
        assert torch.allclose(nz.abs(), nz.abs()[0].expand_as(nz))


# ---------------------------------------------------------------- memory

def test_codebook_memory_by_hand_d8_k4096():
    # 1536 x 1536 matrix, d = 8, K = 4,096
    # tags: 1536 rows * 192 chunks * 12 bits = 3,538,944
    # codebook: 4096 * 8 * 16 = 524,288 ; row scales: 1536 * 16 = 24,576
    rounded, ideal = memory.codebook_bits(1536, 1536, 8, 4096)
    assert rounded == 3_538_944 + 524_288 + 24_576
    assert ideal == rounded  # log2(4096) is a whole number


def test_codebook_memory_ideal_vs_rounded_and_padding():
    # d=1, K=3: rounded tag width 2 bits, ideal log2(3)
    r, i = memory.codebook_bits(10, 10, 1, 3)
    assert r == 100 * 2 + 3 * 16 + 10 * 16
    assert i == pytest.approx(100 * math.log2(3) + 3 * 16 + 10 * 16)
    # C not divisible by d pads up: 10 cols, d=8 -> 2 chunks per row
    r, _ = memory.codebook_bits(1, 10, 8, 4096)
    assert r == 2 * 12 + 4096 * 8 * 16 + 16


def test_codebook_memory_shuffle_and_masks():
    base, _ = memory.codebook_bits(100, 64, 8, 256)
    sh, _ = memory.codebook_bits(100, 64, 8, 256, shuffle=True)
    mk, _ = memory.codebook_bits(100, 64, 8, 256, masks=16)
    assert sh - base == 64 * 16
    assert mk - base == 16 * 64 * 16 + 100 * 4


def test_rtn_and_ternary_memory():
    assert memory.rtn_bits(32, 64, 4) == (32 * 64 * 4 + 64 * 16,) * 2
    r, i = memory.ternary_bits(32, 64)
    assert r == 32 * 64 * 2 + 64 * 16
    assert i == pytest.approx(32 * 64 * math.log2(3) + 64 * 16)


# ---------------------------------------------------------------- codebook

def test_kmeans_recovers_separated_clusters():
    g = torch.Generator().manual_seed(1)
    centers = torch.randn(8, 4, generator=g) * 10
    x = centers.repeat_interleave(200, 0) + 0.01 * torch.randn(1600, 4, generator=g)
    found = cb.kmeans(x, 8, 10, torch.Generator().manual_seed(0))
    d = torch.cdist(centers, found)
    assert torch.all(d.min(1).values < 0.1)


def test_compress_decode_exact_and_reproducible():
    w = _w(64, 100)  # 100 not divisible by 8 -> padding path
    kw = dict(iters=5, max_fit_chunks=10_000, seed=3)
    cm, stats = cb.compress(w, 8, 64, **kw)
    a, b = cb.decode(cm), cb.decode(cm)
    assert a.shape == w.shape and torch.equal(a, b)
    cm2, _ = cb.compress(w, 8, 64, **kw)
    assert torch.equal(cm.idx, cm2.idx) and torch.equal(cm.codebook, cm2.codebook)
    assert 0 < stats["codebook_usage"] <= 1
    assert ((a - w) ** 2).sum() / (w ** 2).sum() < 1.0


def test_decode_with_permutation_roundtrip():
    w = _w(16, 64)
    perm = torch.randperm(64, generator=torch.Generator().manual_seed(0))
    # with K = number of distinct chunks the codebook is lossless up to fp16 rounding
    cm, _ = cb.compress(w, 1, 4, iters=1, max_fit_chunks=10_000, seed=0, perm=perm)
    cm.idx = torch.zeros_like(cm.idx)
    cm.codebook = torch.ones_like(cm.codebook)
    out = cb.decode(cm)
    assert torch.allclose(out, cm.row_scale.float()[:, None].expand(16, 64))
    # explicit check that the permutation is undone
    cm.codebook = torch.arange(4, dtype=torch.float16)[:, None]
    cm.idx = (torch.arange(64) % 4).repeat(16, 1)
    cm.row_scale = torch.ones(16, dtype=torch.float16)
    out = cb.decode(cm)
    assert torch.equal(out[:, perm], (torch.arange(64) % 4).float().repeat(16, 1))


def test_layer_seed_stable():
    assert cb.layer_seed(0, "model.layers.0.mlp.down_proj") == cb.layer_seed(0, "model.layers.0.mlp.down_proj")
    assert cb.layer_seed(0, "a") != cb.layer_seed(1, "a")


# ---------------------------------------------------------------- GSM8K parsing

def test_extract_answer():
    assert extract_answer("so 3+4=7\n#### 7") == Decimal("7")
    assert extract_answer("#### 1,234.50") == Decimal("1234.5")
    assert extract_answer("costs $18 in total.") == Decimal("18")
    assert extract_answer("#### -5") == Decimal("-5")
    assert extract_answer("no numbers here") is None
    assert extract_answer("#### 10\nthen later 99") == Decimal("10")
    assert gold_answer("blah <<2*3=6>>6\n#### 6") == Decimal("6")
    assert extract_answer("The answer is 72.") == gold_answer("#### 72")
