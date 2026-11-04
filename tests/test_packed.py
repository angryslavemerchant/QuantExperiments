import math

import pytest
import torch
from torch import nn

from shapequant import packed
from shapequant.masks import compress_mixture


def _w(seed, R=48, C=64):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(R, C, generator=g) * (torch.rand(R, 1, generator=g) + 0.2)
    h = torch.rand(C, generator=g) + 0.1
    return w, h


@pytest.mark.parametrize("bits", [1, 8, 10, 13, 14, 16])
def test_pack_bits_round_trip(bits):
    g = torch.Generator().manual_seed(bits)
    v = torch.randint(0, 1 << bits, (70, 37), generator=g)
    words = packed.pack_bits(v, bits)
    assert words.dtype == torch.int32 and words.shape == (70, math.ceil(37 * bits / 32))
    assert torch.equal(packed.unpack_bits(words, bits, 37), v)


def test_pack_bits_layout_is_little_endian_stream():
    # value 1 at index 0 -> bit 0 of word 0; value 1 at index 1 (bits=10) -> bit 10
    v = torch.tensor([[1, 1, 0, 0]])
    w = packed.pack_bits(v, 10)
    assert int(w[0, 0]) == (1 | (1 << 10))
    # a value straddling the word boundary: index 3 occupies bits 30..39
    v = torch.tensor([[0, 0, 0, 0b1111111111]])
    w = packed.pack_bits(v, 10).to(torch.int64) & 0xFFFFFFFF
    assert int(w[0, 0]) == 0b11 << 30 and int(w[0, 1]) == 0b11111111


@pytest.mark.parametrize("G,M,K", [(1, 1, 64), (4, 1, 32), (1, 8, 64), (4, 8, 16)])
def test_packed_file_decodes_the_fit_exactly(tmp_path, G, M, K):
    w, h = _w(G * 10 + M)
    seed = 1234 + G + M
    fit = compress_mixture(w, h, 4, K, G=G, M=M, rounds=2, iters=4, seed=seed, keep_tags=True)
    t, meta = packed.pack_matrix(fit, d=4, K=K, G=G, M=M, seed=seed)
    summary = packed.save(tmp_path / "q.safetensors", {"model.layers.0.mlp.up_proj": (t, meta)},
                          base_model="test/base")
    pf = packed.PackedFile(tmp_path / "q.safetensors")
    dec = pf.decode("model.layers.0.mlp.up_proj")
    assert torch.equal(dec, fit.decoded)
    # real storage: tags + table + scales (+ row_book/row_mask bytes) + seed
    R, C = w.shape
    expect = (R * math.ceil(C // 4 * int(math.log2(K)) / 32) * 32 + G * K * 4 * 16 + R * 16
              + (R * 8 if G > 1 else 0) + (R * 8 if M > 1 else 0) + 32)
    assert meta["bits"] == expect and summary["bits"] == expect


def test_packed_rejects_changed_orders():
    w, h = _w(5)
    fit = compress_mixture(w, h, 4, 16, G=1, M=4, rounds=1, iters=2, seed=7, keep_tags=True)
    with pytest.raises(ValueError, match="regenerate"):
        packed.pack_matrix(fit, d=4, K=16, G=1, M=4, seed=8)


def test_finetuned_values_reach_the_packed_file(tmp_path):
    from shapequant.finetune import CodebookLinear, write_back
    w, h = _w(9)
    fit = compress_mixture(w, h, 4, 16, G=2, M=4, rounds=1, iters=3, seed=3, keep_tags=True)
    lin = nn.Linear(64, 48)
    wrap = CodebookLinear(lin, fit, 4)
    with torch.no_grad():  # stand-in for training: perturb books and gains
        for b in wrap.books:
            b.add_(0.05 * torch.randn_like(b))
        wrap.log_gain.add_(0.1 * torch.randn_like(wrap.log_gain))
    baked = wrap.bake()
    write_back(fit, wrap)
    t, meta = packed.pack_matrix(fit, d=4, K=16, G=2, M=4, seed=3)
    packed.save(tmp_path / "ft.safetensors", {"m": (t, meta)}, base_model="test/base")
    assert torch.equal(packed.PackedFile(tmp_path / "ft.safetensors").decode("m"), baked)
