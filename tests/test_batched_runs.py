"""Batched decode GEMMs (shapequant/kernels/batched_runs.py) == B separate batch-1 kernel calls
(to fp16 rounding: the batched kernels accumulate in fp32). Needs CUDA + CuPy."""

import pytest
import torch

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from shapequant.kernels import batched_runs as br  # noqa: E402
from shapequant.kernels import proto_formats as pf  # noqa: E402
from shapequant.kernels import proto_runs  # noqa: E402
from test_proto_runs import _lin  # noqa: E402


def _ref(rl, X):
    return torch.stack([proto_runs.gemv(rl, X[b]).float() for b in range(X.shape[0])])


def _close(Y, ref):
    return float((Y.float() - ref).abs().max() / ref.abs().max())


# shapes: no full batch (C=256: tail only), full batches only (C=2048), full + tail (C=3584, 5120),
# rows not a multiple of 16 (300); K 256 / 1024 / 4096 with 1 or 4 books (smem, book-major, global)
@pytest.mark.parametrize("R,C", [(512, 3584), (300, 2048), (128, 256), (1040, 5120)])
@pytest.mark.parametrize("bits,G", [(8, 4), (10, 1), (10, 4), (12, 4), (14, 1)])
def test_runs_matches_batch1(R, C, bits, G):
    lin, tags = _lin(R, C, bits, G=G, M=1)
    rl = proto_runs.RunsLinear(lin, tags, 1)
    X = (torch.randn(32, C, device="cuda") * 2).half()
    ref = _ref(rl, X)
    for layout in ("lane", "tile"):
        lb = br.RunsBatch(rl, layout=layout)
        for B in (1, 2, 5, 8, 13, 16, 24, 32):
            Y = torch.full((B, R), float("nan"), dtype=torch.float16, device="cuda")
            lb.run(X[:B].contiguous(), Y)
            assert _close(Y, ref[:B]) < 2e-3, (layout, B)


@pytest.mark.parametrize("split", [1, 3, 8])
@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("layout,acc16", [("lane", False), ("tile", False), ("tile", True)])
def test_runs_split_and_replay(split, warps, layout, acc16):
    """Split-K partial sums (deterministic: identical on repeated launches, also inside a graph)."""
    lin, tags = _lin(1024, 3584, 10, G=4, M=1, seed=3)
    rl = proto_runs.RunsLinear(lin, tags, 1)
    X = torch.randn(12, 3584, device="cuda").half()
    ref = _ref(rl, X)
    lb = br.RunsBatch(rl, bmax=12, warps=warps, max_split=split, bps=8, layout=layout, acc16=acc16)
    assert lb.S == min(split, lb.S) and (split == 1) == (lb.S == 1)
    Y1 = torch.empty(12, 1024, dtype=torch.float16, device="cuda")
    lb.run(X, Y1)
    assert _close(Y1, ref) < 2e-3
    Y2 = torch.empty_like(Y1)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        lb.run(X, Y2)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        lb.run(X, Y2)
    for _ in range(3):
        Y2.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(Y1, Y2)


@pytest.mark.parametrize("R,C", [(1000, 2048), (4096, 5120), (333, 96)])
def test_q8_matches_batch1(R, C):
    w = torch.randn(R, C) * 0.02
    q, sc, _ = pf.q8_pack(w)
    q, sc = q.cuda(), sc.cuda()
    X = (torch.randn(32, C, device="cuda") * 2).half()
    ref = torch.stack([pf.gemv_q8(q, sc, X[b], torch.empty(R, dtype=torch.float16, device="cuda"), R, C).float()
                       for b in range(32)])
    tile = br.Q8Batch(q, sc, R, C) if C % 64 == 0 else None
    for B in (1, 3, 8, 9, 17, 32):
        for wide in ((False, True) if C % 64 == 0 else (False,)):
            Y = torch.empty(B, R, dtype=torch.float16, device="cuda")
            br.gemv_q8_batched(q, sc, X[:B].contiguous(), Y, R, C, wide=wide)
            assert _close(Y, ref[:B]) < 2e-3, (B, wide)
        if tile is not None:
            Y = torch.full((B, R), float("nan"), dtype=torch.float16, device="cuda")
            tile.run(X[:B].contiguous(), Y)
            assert _close(Y, ref[:B]) < 2e-3, (B, "tile")


def test_tune_runs():
    """The tuner returns one grid config per distinct shape, and those configs compute correctly."""
    rls = [proto_runs.RunsLinear(*_lin(R, C, 10, G=4, M=1, seed=R), 1) for R, C in ((256, 2048), (512, 1024), (256, 2048))]
    grid = [(4, 1, 1), (8, 2, 4)]
    best = br.tune_runs(rls, 5, grid=grid, n=2, reps=1)
    assert set(best) == {br.runs_key(rls[0]), br.runs_key(rls[1])}
    assert all(v in grid for v in best.values())
    X = torch.randn(5, 2048, device="cuda").half()
    w, b, m = best[br.runs_key(rls[0])]
    lb = br.RunsBatch(rls[0], warps=w, bps=b, max_split=m, layout="tile")
    Y = lb.run(X, torch.empty(5, 256, dtype=torch.float16, device="cuda"))
    assert _close(Y, _ref(rls[0], X)) < 2e-3


# xs: X staged in shared memory (swizzled), batches up to 128
@pytest.mark.parametrize("R,C", [(300, 2048), (128, 256), (1040, 5120)])
@pytest.mark.parametrize("bits,G", [(8, 4), (10, 1), (12, 4)])
@pytest.mark.parametrize("layout", ["lane", "tile"])
def test_runs_xs_matches_batch1(R, C, bits, G, layout):
    lin, tags = _lin(R, C, bits, G=G, M=1)
    rl = proto_runs.RunsLinear(lin, tags, 1)
    X = (torch.randn(128, C, device="cuda") * 2).half()
    ref = _ref(rl, X)
    for split in (1, 4):
        lb = br.RunsBatch(rl, bmax=128, layout=layout, xs=True, max_split=split)
        for B in (1, 7, 32, 33, 64, 100, 128):
            Y = torch.full((B, R), float("nan"), dtype=torch.float16, device="cuda")
            lb.run(X[:B].contiguous(), Y)
            assert _close(Y, ref[:B]) < 2e-3, (layout, split, B)


@pytest.mark.parametrize("R,C", [(512, 3584), (300, 2048), (128, 256), (1040, 5120)])
@pytest.mark.parametrize("bits,G", [(8, 4), (10, 1), (12, 4)])
def test_dequant_tile_matches_lane_dequant(R, C, bits, G):
    """The tile-layout dequant (compact mode) is bit-identical to the lane-major one."""
    from shapequant.kernels import cuda_dequant
    lin, tags = _lin(R, C, bits, G=G, M=1, seed=5)
    rl = proto_runs.RunsLinear(lin, tags, 1)
    ref = cuda_dequant.dequant(rl).clone().view(R, C)
    lb = br.RunsBatch(rl, layout="tile", xs=True, bmax=64)
    out = torch.full((R * C + 77,), float("nan"), dtype=torch.float16, device="cuda")
    W = br.dequant_tile(lb, out)
    assert W.shape == (R, C) and torch.equal(W, ref)
