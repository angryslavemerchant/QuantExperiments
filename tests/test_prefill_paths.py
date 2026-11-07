"""Every prefill path (ref, fast with gemm = dequant | overlap | fused) == step-by-step decoding,
for the Qwen2 engine (shapequant.prefill) and the Qwen3.5 engine (qwen35_runtime.prefill),
including a prompt that is not a multiple of the GEMM / DeltaNet tiles and a continuation."""

import pytest
import torch

cp = pytest.importorskip("cupy")
pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from shapequant import qwen35_runtime as q35, runtime  # noqa: E402
from shapequant.prefill import prefill  # noqa: E402

PATHS = [("ref", None), ("fast", "dequant"), ("fast", "overlap"), ("fast", "fused")]


def _rel(a, b):
    return float((a - b).norm() / b.norm())


@pytest.mark.parametrize("M,G", [(1, 4), (4, 1)])
def test_qwen2_paths(tmp_path, M, G):
    from test_runtime import _hf, _write_packed
    path = tmp_path / "m.safetensors"
    _write_packed(path, M, G)
    eng, _ = runtime.engine_from_packed(path, model=_hf(True), max_len=64, prefill=True)
    seq = [int(t) for t in torch.randint(0, 1024, (45,), generator=torch.Generator().manual_seed(2))]
    ref = torch.stack(runtime.feed(eng, seq, keep_logits=True))
    outs = {}
    for impl, gemm in PATHS:
        eng.reset(0)
        a = prefill(eng, seq[:30], impl=impl, gemm=gemm)
        b = prefill(eng, seq[30:], impl=impl, gemm=gemm)      # continues from the cache
        got = torch.cat([a, b])
        outs[(impl, gemm)] = got
        assert _rel(got, ref) < 1e-2, (impl, gemm, _rel(got, ref))
        assert int(eng.pos) == len(seq)
    # the fast paths agree with the reference prefill more tightly than with token-by-token decode
    for k, v in outs.items():
        assert _rel(v, outs[("ref", None)]) < 5e-3, k


@pytest.mark.parametrize("hv", [4, 2])
def test_qwen35_paths(tmp_path, hv):
    from test_qwen35_runtime import _hf, _write_packed
    path = tmp_path / "m.safetensors"
    _write_packed(path, _hf(hv=hv), aux=True)
    eng, _ = q35.engine_from_packed(path, max_len=256)
    q35.attach_prefill(eng)
    seq = [int(t) for t in torch.randint(0, 1024, (150,), generator=torch.Generator().manual_seed(3))]
    ref = torch.stack(q35.feed(eng, seq, keep_logits=True))
    S_ref, conv_ref = eng.S.clone(), eng.conv.clone()
    for impl, gemm in PATHS:
        eng.reset()
        a = q35.prefill(eng, seq[:70], logits="all", impl=impl, gemm=gemm)
        b = q35.prefill(eng, seq[70:], logits="all", impl=impl, gemm=gemm)
        got = torch.cat([a, b])
        assert _rel(got, ref) < 5e-3, (impl, gemm, _rel(got, ref))
        assert _rel(eng.S, S_ref) < 5e-3 and _rel(eng.conv, conv_ref) < 1e-3
    # "last" logits == the last row of "all"
    eng.reset()
    lg_last = q35.prefill(eng, seq, logits="last")
    assert _rel(lg_last, ref[-1]) < 5e-3
    # EXPERIMENTAL fp8 rounding (emulated on any GPU): runs end to end, stays near the fp16 result
    eng.reset()
    lg8 = q35.prefill(eng, seq, logits="all", gemm="fp8_emulated")
    assert torch.isfinite(lg8).all() and _rel(lg8, ref) < 0.15, _rel(lg8, ref)
