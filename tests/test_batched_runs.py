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
    lb = br.RunsBatch(rl)
    for B in (1, 2, 5, 8, 13, 16, 24, 32):
        Y = torch.full((B, R), float("nan"), dtype=torch.float16, device="cuda")
        lb.run(X[:B].contiguous(), Y)
        assert _close(Y, ref[:B]) < 2e-3, B


@pytest.mark.parametrize("split", [1, 3, 8])
@pytest.mark.parametrize("warps", [4, 8])
def test_runs_split_and_replay(split, warps):
    """Split-K partial sums (deterministic: identical on repeated launches, also inside a graph)."""
    lin, tags = _lin(1024, 3584, 10, G=4, M=1, seed=3)
    rl = proto_runs.RunsLinear(lin, tags, 1)
    X = torch.randn(12, 3584, device="cuda").half()
    ref = _ref(rl, X)
    lb = br.RunsBatch(rl, bmax=12, warps=warps, max_split=split, bps=8)
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
