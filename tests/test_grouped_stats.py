"""Layer-grouped calibration statistics (for models whose accumulators don't fit the GPU at once)
equal the all-at-once ones exactly: input Hessians and Fisher diagonals."""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from shapequant import alloc as al  # noqa: E402
from shapequant.calibrate import input_hessians, layer_groups  # noqa: E402
from shapequant.pipeline import compressible_linears  # noqa: E402


def _qwen2():
    c = transformers.Qwen2Config(hidden_size=128, intermediate_size=256, num_hidden_layers=4, num_attention_heads=4,
                                 num_key_value_heads=2, vocab_size=256, max_position_embeddings=64)
    c._attn_implementation = "eager"
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(c).float().eval()


def _qwen35():
    from transformers.models.qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    c = Qwen3_5TextConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=4, num_attention_heads=2,
                          num_key_value_heads=1, head_dim=64, vocab_size=256, max_position_embeddings=64,
                          layer_types=["linear_attention"] * 3 + ["full_attention"], linear_num_key_heads=2,
                          linear_num_value_heads=2, linear_key_head_dim=32, linear_value_head_dim=32,
                          linear_conv_kernel_dim=4)
    c._attn_implementation = "eager"
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(c).float().eval()


def test_layer_groups():
    assert layer_groups([(0, 5), (1, 5), (2, 5)], 10) == [[0, 1], [2]]
    assert layer_groups([(0, 50), (1, 5)], 10) == [[0], [1]]          # one layer over budget: alone
    assert layer_groups([(0, 1), (1, 1)], 1 << 40) == [[0, 1]]


@pytest.mark.parametrize("make", [_qwen2, _qwen35])
def test_hessians_grouped_identical(make):
    m = make()
    seqs = torch.randint(0, 256, (6, 32), generator=torch.Generator().manual_seed(1))
    full = input_hessians(m, seqs, batch_size=2, max_gpu_bytes=1 << 40)
    grouped = input_hessians(m, seqs, batch_size=2, max_gpu_bytes=1)      # one layer per pass
    assert set(full) == set(grouped) == {n for n, _ in compressible_linears(m)}
    for n in full:
        assert torch.equal(full[n], grouped[n]), n


@pytest.mark.parametrize("make", [_qwen2, _qwen35])
def test_fisher_grouped_identical(make):
    m = make()
    seqs = torch.randint(0, 256, (3, 32), generator=torch.Generator().manual_seed(2))
    full = al.fisher_diag(m, seqs, log=lambda *_: None)
    names = [n for n, _ in compressible_linears(m)]
    part = {}
    for g in (names[: len(names) // 2], names[len(names) // 2:]):
        part.update(al.fisher_diag(m, seqs, log=lambda *_: None, names=g))
    for n in names:
        assert torch.equal(full[n], part[n]), n
