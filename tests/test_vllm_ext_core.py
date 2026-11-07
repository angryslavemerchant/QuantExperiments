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
@pytest.mark.parametrize("large", ["dequant", "chunked", "fused"])
def test_forward_matches_dense(pieces, M, large):
    pytest.importorskip("cupy")
    from shapequant.packed import PackedFile
    pf = PackedFile(PACKED)
    ps = core.PackedShards.from_packed(pf, [f"model.layers.0.{p}" for p in pieces], large=large,
                                       chunk_bytes=1 << 20)     # several chunks per matrix
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


@needs_gpu
def test_builds_under_vllm_defaults():
    """vLLM builds models under a CUDA default device and a bf16 default dtype."""
    pytest.importorskip("cupy")
    from shapequant.packed import PackedFile
    pf = PackedFile(PACKED)
    names = ["model.layers.0.linear_attn.in_proj_qkv", "model.layers.0.linear_attn.in_proj_z"]
    dt = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            ps = core.PackedShards.from_packed(pf, names, dev=torch.device("cuda", 0), large="fused")
    finally:
        torch.set_default_dtype(dt)
    assert torch.get_default_dtype() == dt
    ref = core.PackedShards.from_packed(pf, names)
    x = torch.randn(40, ps.C, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(ps.forward(x[:5]), ref.forward(x[:5]))
    W = ref.dense().float()
    err = (ps.forward(x).float() - x.half().float() @ W.t()).norm() / (x.half().float() @ W.t()).norm()
    assert err < 1e-2


@needs_gpu
@pytest.mark.parametrize("large", ["dequant", "fused"])
def test_custom_op_compiles_and_graph_captures(large):
    """What vLLM does with the op: dynamo traces it (fullgraph, fake impl gives the shape) and the
    compiled steps are replayed from CUDA graphs, for small- and large-M batches."""
    pytest.importorskip("cupy")
    from shapequant.packed import PackedFile
    pf = PackedFile(PACKED)
    ps = core.PackedShards.from_packed(pf, ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"], large=large)
    h = core.register(ps)

    def f(x):
        return torch.ops.shapequant.linear(x, h).relu()

    fc = torch.compile(f, backend="aot_eager", fullgraph=True, dynamic=True)
    for M in (1, 8, 40):
        x = torch.randn(M, ps.C, device="cuda", dtype=torch.bfloat16)
        want = f(x)
        assert torch.equal(fc(x), want)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fc(x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = fc(x)
        x.copy_(torch.randn_like(x))
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(y, f(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_tuned_dispatch_synthetic():
    """tune() picks a crossover among the measured Ms; forward agrees with the dense product on both
    sides of it (xs batched kernel up to 64 rows, fused GEMM above)."""
    pytest.importorskip("cupy")
    from test_runs_gemm import _rl, _ref
    a, b = _rl(300, 1280, 10, 4, 31), _rl(200, 1280, 12, 4, 32)
    ps = core.PackedShards([a, b], xs=True, bmax=64, large="fused")
    cross = ps.tune(Ms=(8, 32, 64))
    assert cross in (0, 8, 32, 64) and ps.cross == cross
    assert core.PackedShards([a, b], xs=True, bmax=64, large="fused").tune(Ms=(8, 32, 64)) == cross   # cached
    for M in (1, 8, 31, 33, 64, 65, 130):
        x = torch.randn(M, 1280 * 4, device="cuda", dtype=torch.bfloat16)
        ref = _ref(x.half(), [a, b])
        err = (ps.forward(x).float() - ref).norm() / ref.norm()
        assert err < 1e-2, (M, err)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fused_hands_long_batches_to_dequant():
    pytest.importorskip("cupy")
    from test_runs_gemm import _rl, _ref
    a = _rl(200, 1280, 10, 4, 41)
    ps = core.PackedShards([a], large="fused", dequant_from=100)
    for M in (40, 99, 100, 300):
        x = torch.randn(M, 1280 * 4, device="cuda", dtype=torch.bfloat16)
        ref = _ref(x.half(), [a])
        assert (ps.forward(x).float() - ref).norm() / ref.norm() < 1e-2
