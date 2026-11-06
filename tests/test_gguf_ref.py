import numpy as np
import pytest
import torch

gguf = pytest.importorskip("gguf")
from gguf.quants import quantize  # noqa: E402

from shapequant import gguf_ref  # noqa: E402
from shapequant.pipeline import compress_matrix  # noqa: E402


def test_tensor_name_mapping():
    assert gguf_ref.gguf_tensor_name("model.layers.3.mlp.down_proj") == "blk.3.ffn_down.weight"
    assert gguf_ref.gguf_tensor_name("model.layers.0.self_attn.o_proj") == "blk.0.attn_output.weight"
    assert gguf_ref.gguf_tensor_name("model.layers.2.linear_attn.in_proj_qkv") == "blk.2.attn_qkv.weight"
    assert gguf_ref.gguf_tensor_name("model.layers.2.linear_attn.in_proj_z") == "blk.2.attn_gate.weight"
    assert gguf_ref.gguf_tensor_name("model.layers.2.linear_attn.out_proj") == "blk.2.ssm_out.weight"
    with pytest.raises(ValueError):
        gguf_ref.gguf_tensor_name("lm_head")


def _write(path, name, arr, qtype):
    w = gguf.GGUFWriter(str(path), "qwen2")
    w.add_tensor(name, quantize(arr, qtype), raw_dtype=qtype)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def test_q8_roundtrip_bits_and_guard(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((64, 96)).astype(np.float32) * 0.02
    path = tmp_path / "t.gguf"
    _write(path, "blk.0.mlp_placeholder", a, gguf.GGMLQuantizationType.Q8_0)  # unrelated name ignored
    _write(path, "blk.0.ffn_up.weight", a, gguf.GGMLQuantizationType.Q8_0)
    src = gguf_ref.GGUFSource(str(path))
    w = torch.from_numpy(a)
    out, rb, ib, stats = compress_matrix(w, {"method": "gguf"}, {}, 0, "model.layers.0.mlp.up_proj", src)
    assert rb == ib == 64 * 96 * 8.5          # Q8_0: 34 bytes per 32 weights
    assert stats["ggml_type"] == "Q8_0"
    assert float(((out - w) ** 2).sum() / (w ** 2).sum()) < 1e-3
    # A transposed (wrong) mapping must be caught, not silently evaluated.
    with pytest.raises(RuntimeError, match="mapping is wrong"):
        compress_matrix(w.T.contiguous().reshape(64, 96), {"method": "gguf"}, {}, 0,
                        "model.layers.0.mlp.up_proj", src)


@pytest.mark.parametrize("with_output", [True, False])
def test_head_and_embed(tmp_path, with_output):
    """The file's own token_embd / output, dequantized in HF [vocab, hidden] layout; tied files
    (no output tensor) reuse token_embd."""
    rng = np.random.default_rng(1)
    E = rng.standard_normal((48, 64)).astype(np.float32) * 0.02
    H = rng.standard_normal((48, 64)).astype(np.float32) * 0.02
    path = tmp_path / "t.gguf"
    w = gguf.GGUFWriter(str(path), "qwen2")
    w.add_tensor("token_embd.weight", quantize(E, gguf.GGMLQuantizationType.Q8_0),
                 raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    if with_output:
        w.add_tensor("output.weight", quantize(H, gguf.GGMLQuantizationType.Q8_0),
                     raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    he = gguf_ref.GGUFSource(str(path)).head_and_embed(48, 64)
    assert he["embed"].shape == (48, 64) and he["embed_type"] == "Q8_0" and he["head_bits"] == 48 * 64 * 8.5
    ref_h = H if with_output else E
    assert float(((he["head"] - torch.from_numpy(ref_h)) ** 2).sum() / (ref_h ** 2).sum()) < 1e-3
    assert float(((he["embed"] - torch.from_numpy(E)) ** 2).sum() / (E ** 2).sum()) < 1e-3


def test_qwen35_value_head_reorder(tmp_path):
    """GGUF files of Qwen3.5 / 3.8 with more value than key heads store the DeltaNet value heads
    tiled over the key heads (llama.cpp); GGUFSource.matrix returns them in HF order."""
    HK, HV, DK, DV, H = 2, 6, 4, 4, 32
    rep = HV // HK
    g = torch.Generator().manual_seed(0)
    z = torch.randn(HV * DV, H, generator=g)
    qkv = torch.randn(2 * HK * DK + HV * DV, H, generator=g)
    out = torch.randn(H, HV * DV, generator=g)
    perm = [rep * (j % HK) + j // HK for j in range(HV)]
    rows = torch.tensor([p * DV + i for p in perm for i in range(DV)])
    path = tmp_path / "t.gguf"
    w = gguf.GGUFWriter(str(path), "qwen35")
    w.add_uint32("qwen35.ssm.group_count", HK)
    w.add_uint32("qwen35.ssm.time_step_rank", HV)
    w.add_uint32("qwen35.ssm.inner_size", HV * DV)
    off = 2 * HK * DK
    for name, a in [("blk.0.attn_gate.weight", z[rows]),
                    ("blk.0.attn_qkv.weight", torch.cat([qkv[:off], qkv[off:][rows]])),
                    ("blk.0.ssm_out.weight", out[:, rows])]:
        w.add_tensor(name, a.numpy().astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    src = gguf_ref.GGUFSource(str(path))
    for hf, ref in [("model.layers.0.linear_attn.in_proj_z", z), ("model.layers.0.linear_attn.in_proj_qkv", qkv),
                    ("model.layers.0.linear_attn.out_proj", out)]:
        got, _, _ = src.matrix(hf, tuple(ref.shape))
        assert torch.equal(got, ref), hf
