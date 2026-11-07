"""shapequant.vllm_ext.core: stacked packed matrices against their dense decode (GPU + cupy)."""

from pathlib import Path

import pytest
import torch

from shapequant.vllm_ext import core

ROOT = Path(__file__).resolve().parents[1]
PACKED = ROOT / "artifacts/remote/s29_bench_qwen35_2b_20261106a/extracted/artifacts/qwen35_2b/stage13b_bench/packed/s13_25_ft.safetensors"


def test_packed_pieces():
    names = {"model.layers.3.self_attn.q_proj", "model.layers.3.self_attn.k_proj", "model.layers.3.self_attn.v_proj",
             "model.layers.0.linear_attn.in_proj_qkv", "model.layers.0.linear_attn.in_proj_z",
             "model.layers.0.mlp.down_proj"}
    assert core.packed_pieces("model.layers.3.self_attn.qkv_proj", names) == [
        "model.layers.3.self_attn.q_proj", "model.layers.3.self_attn.k_proj", "model.layers.3.self_attn.v_proj"]
    assert core.packed_pieces("language_model.model.layers.0.linear_attn.in_proj_qkvz", names) == [
        "model.layers.0.linear_attn.in_proj_qkv", "model.layers.0.linear_attn.in_proj_z"]
    assert core.packed_pieces("model.layers.0.mlp.down_proj", names) == ["model.layers.0.mlp.down_proj"]
    assert core.packed_pieces("model.layers.0.linear_attn.in_proj_ba", names) is None
    assert core.packed_pieces("model.layers.0.linear_attn.conv1d", names) is None
    assert core.packed_pieces("lm_head", names) is None
    with pytest.raises(ValueError):
        core.packed_pieces("model.layers.0.mlp.gate_up_proj", names | {"model.layers.0.mlp.gate_proj"})


needs_gpu = pytest.mark.skipif(not torch.cuda.is_available() or not PACKED.exists(), reason="needs CUDA + the 2B packed file")


@needs_gpu
@pytest.mark.parametrize("pieces", [("mlp.down_proj",), ("mlp.gate_proj", "mlp.up_proj"),
                                    ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z")])
@pytest.mark.parametrize("M", [1, 3, 32, 33, 100])
def test_forward_matches_dense(pieces, M):
    pytest.importorskip("cupy")
    from shapequant.packed import PackedFile
    pf = PackedFile(PACKED)
    ps = core.PackedShards.from_packed(pf, [f"model.layers.0.{p}" for p in pieces])
    W = ps.dense().float()
    assert W.shape == (ps.R, ps.C)
    torch.manual_seed(M)
    x = torch.randn(M, ps.C, device="cuda", dtype=torch.bfloat16)
    y = ps.forward(x)
    assert y.dtype == torch.bfloat16 and y.shape == (M, ps.R)
    ref = x.half().float() @ W.t()
    err = (y.float() - ref).norm() / ref.norm()
    assert err < 1e-2, err
    h = core.register(ps)
    y2 = core.linear_op(x.view(M, 1, ps.C), h)
    assert y2.shape == (M, 1, ps.R)
    assert torch.equal(y2.view(M, ps.R), y)
