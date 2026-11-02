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
