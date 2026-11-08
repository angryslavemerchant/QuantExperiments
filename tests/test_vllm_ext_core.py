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
    assert core.packed_pieces("mtp.layers.0.mlp.down_proj", names) is None
    assert core.packed_pieces("model.mtp.layers.0.mlp.down_proj", names) is None
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


def _q8_parts(V=300, C=128, seed=0):
    from shapequant.kernels import proto_formats as pfm
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(V, C, generator=g) * 0.05
    q, sc, _ = pfm.q8_pack(w)
    return q, sc


def test_q4_embed_matches_reference_rows():
    from shapequant.packed import q4_embed_pack, q4_embed_rows
    g = torch.Generator().manual_seed(1)
    q, sc = q4_embed_pack(torch.randn(500, 256, generator=g) * 0.03)
    ids = torch.tensor([[0, 7, 499], [3, 3, 250]])
    out = core.q4_embed(q, sc, ids, torch.bfloat16)
    assert out.shape == (2, 3, 256) and out.dtype == torch.bfloat16
    ref = q4_embed_rows(q, sc, ids.reshape(-1)).to(torch.bfloat16).view(2, 3, 256)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("tiled", [True, False])
def test_q8_head_chunked_path_cpu(tiled):
    from shapequant.packed import q8_rows
    q, sc = _q8_parts()
    head = core.Q8Head(q, sc, R=320, small_max=0, chunk_rows=64, tiled=tiled)   # padded vocab, every M on the chunked path
    x = torch.randn(5, 128) * 0.1
    y = head.forward(x)
    assert y.shape == (5, 320) and y.dtype == x.dtype
    w = q8_rows(q, sc)
    assert torch.allclose(y[:, :300], x @ w.t(), atol=2e-3, rtol=1e-2)
    assert torch.count_nonzero(y[:, 300:]) == 0
    assert torch.allclose(head.dense().float(), w, atol=1e-3)
    with pytest.raises(ValueError):
        core.Q8Head(q, sc, R=299)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("tiled", [True, False])
@pytest.mark.parametrize("M", [1, 7, 32, 33, 80])
def test_q8_head_gpu_matches_dense(M, tiled):
    pytest.importorskip("cupy")
    from shapequant.packed import q8_rows
    q, sc = _q8_parts(V=1000, C=256)
    head = core.Q8Head(q.cuda(), sc.cuda(), chunk_rows=384, tiled=tiled)
    x = (torch.randn(M, 256) * 0.1).to(torch.bfloat16).cuda()
    y = torch.ops.shapequant.linear(x, core.register(head))
    ref = x.float() @ q8_rows(q, sc).cuda().t()
    assert y.shape == (M, 1000) and y.dtype == torch.bfloat16
    assert torch.allclose(y.float(), ref, atol=3e-2, rtol=3e-2)


def test_export_config_embed_head_flag():
    import importlib.util
    spec = importlib.util.spec_from_file_location("export_vllm_checkpoint", ROOT / "scripts" / "export_vllm_checkpoint.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    cfg = ex.vllm_config({"vocab_size": 10, "dtype": "bfloat16"})
    assert cfg["quantization_config"]["embed_head"] == "packed" and "dtype" not in cfg
    assert ex.vllm_config({}, "bf16")["quantization_config"]["embed_head"] == "bf16"
    names = ex.to_multimodal_names({"model.layers.3.mlp.down_proj.weight": 0, "model.norm.weight": 0,
                                    "model.embed_tokens.weight": 0, "lm_head.weight": 0})
    assert sorted(names) == ["lm_head.weight", "model.language_model.embed_tokens.weight",
                             "model.language_model.layers.3.mlp.down_proj.weight", "model.language_model.norm.weight"]
    # the multimodal prefix still maps every text layer onto its packed matrices; the vision tower does not
    assert core.packed_pieces("model.language_model.layers.3.mlp.gate_up_proj",
                              {"model.layers.3.mlp.gate_proj", "model.layers.3.mlp.up_proj"}) == \
        ["model.layers.3.mlp.gate_proj", "model.layers.3.mlp.up_proj"]
    assert core.packed_pieces("model.visual.blocks.3.mlp.linear_fc1", {"model.layers.3.mlp.gate_proj"}) is None


def test_embed_op_matches_q4_embed():
    from shapequant.packed import q4_embed_pack
    g = torch.Generator().manual_seed(2)
    q, sc = q4_embed_pack(torch.randn(100, 64, generator=g) * 0.03)
    h = core.register(core.Q4Embedding(q, sc, torch.bfloat16))
    ids = torch.tensor([3, 99, 0])
    out = torch.ops.shapequant.embed(ids, h)
    assert out.shape == (3, 64) and out.dtype == torch.bfloat16
    assert torch.equal(out, core.q4_embed(q, sc, ids, torch.bfloat16)) and out.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_embed_op_compiles_on_gpu():
    from shapequant.packed import q4_embed_pack
    q, sc = q4_embed_pack(torch.randn(100, 64) * 0.03)
    h = core.register(core.Q4Embedding(q.cuda(), sc.cuda(), torch.bfloat16))
    ids = torch.tensor([3, 99, 0], device="cuda")
    f = torch.compile(lambda i: torch.ops.shapequant.embed(i, h) * 2, fullgraph=True)
    assert torch.equal(f(ids), torch.ops.shapequant.embed(ids, h) * 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("M", [1, 16, 64, 65, 300])
def test_compact_forward_matches_dense(M):
    """compact: lane words freed, batched kernel to cross, tile dequant + GEMM above."""
    pytest.importorskip("cupy")
    from shapequant.kernels import proto_runs
    from test_proto_runs import _lin
    rls = []
    for i, R in enumerate((768, 256)):
        lin, tags = _lin(R, 2048, 10, G=4, M=1, seed=11 + i)
        rls.append(proto_runs.RunsLinear(lin, tags, 1))
    ref_ps = core.PackedShards(rls, bmax=64, layout="tile", xs=True)
    W = ref_ps.dense().float()
    ps = core.PackedShards(rls, bmax=64, layout="tile", xs=True, compact=True)
    assert all(rl.words.numel() == 0 for rl in rls)
    x = (torch.randn(M, 2048, device="cuda") * 0.5).to(torch.bfloat16)
    y = ps.forward(x).float()
    ref = x.float() @ W.t()
    assert y.shape == (M, 1024)
    assert float((y - ref).abs().max() / ref.abs().max()) < 1e-2
    assert torch.equal(ps.dense().float(), W)
    with pytest.raises(ValueError):
        core.PackedShards(rls, layout="lane", compact=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_tune_launch_buckets_match_default():
    pytest.importorskip("cupy")
    from shapequant.kernels import proto_runs
    from test_proto_runs import _lin
    rls = []
    for i, R in enumerate((512, 256)):
        lin, tags = _lin(R, 2048, 10, G=4, M=1, seed=21 + i)
        rls.append(proto_runs.RunsLinear(lin, tags, 1))
    ps = core.PackedShards(rls, bmax=32, layout="tile", xs=True)
    x = (torch.randn(32, 2048, device="cuda") * 0.5).to(torch.bfloat16)
    ref = {M: ps.forward(x[:M]).float() for M in (1, 3, 17, 32)}
    cfgs = ps.tune_launch(reps=1)
    assert ps.buckets == [1, 2, 4, 8, 16, 32] and len(cfgs) == 12
    for M, r in ref.items():
        y = ps.forward(x[:M]).float()
        assert float((y - r).abs().max() / r.abs().max()) < 1e-2, M


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_pool_tile_words_keeps_outputs_and_repoints_launchers():
    """pool_tile_words: every registered matrix's tile words move into one buffer; outputs (small-M
    tuned buckets and the compact large-M dequant path) are bit-identical; a second call is a no-op."""
    pytest.importorskip("cupy")
    from shapequant.kernels import proto_runs
    from test_proto_runs import _lin
    saved = list(core._REGISTRY)
    core._REGISTRY.clear()
    try:
        shards = []
        for j, Rs in enumerate(((512, 256), (384,))):
            rls = []
            for i, R in enumerate(Rs):
                lin, tags = _lin(R, 2048, 10, G=4, M=1, seed=31 + 10 * j + i)
                rls.append(proto_runs.RunsLinear(lin, tags, 1))
            ps = core.PackedShards(rls, bmax=32, layout="tile", xs=True, compact=True)
            ps.tune_launch(reps=1)
            core.register(ps)
            shards.append(ps)
        x = (torch.randn(80, 2048, device="cuda") * 0.5).to(torch.bfloat16)
        before = [{M: ps.forward(x[:M]) for M in (1, 5, 32, 80)} for ps in shards]
        old = [rl._tile_words[0].data_ptr() for ps in shards for rl in ps.rls]
        n = core.pool_tile_words()
        words = [rl._tile_words[0] for ps in shards for rl in ps.rls]
        assert n >= sum(w.numel() * 4 for w in words)
        base = words[0].untyped_storage().data_ptr()
        assert all(w.untyped_storage().data_ptr() == base for w in words)      # one shared buffer
        assert all(w.data_ptr() % 256 == 0 for w in words)
        assert not set(old) & {w.data_ptr() for w in words}
        for ps in shards:
            for lb in list(ps.launchers) + [lb for per in ps.blaunch for lb in per]:
                assert lb.wt.untyped_storage().data_ptr() == base
        for ps, ref in zip(shards, before):
            for M, y in ref.items():
                assert torch.equal(ps.forward(x[:M]), y), M
        assert core.pool_tile_words() == 0
    finally:
        core._REGISTRY[:] = saved
