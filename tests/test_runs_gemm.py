"""kernels.runs_gemm: fused dequant + GEMM == cuda_dequant.dequant followed by a matmul."""

import pytest
import torch

pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from shapequant.kernels import cuda_dequant, proto_runs, runs_gemm  # noqa: E402
from shapequant.kernels.gemv import PackedLinear  # noqa: E402
from shapequant.packed import pack_bits  # noqa: E402


def _rl(R, nch, bits, G, seed):
    g = torch.Generator().manual_seed(seed)
    K = 2 ** bits
    tags = torch.randint(0, K, (R, nch), generator=g)
    table = (torch.randn(G * K, 4, generator=g) * 0.7).half()
    rs = (torch.rand(R, generator=g) * 0.03 + 1e-3).half()
    rb = torch.randint(0, G, (R,), generator=g).to(torch.uint8) if G > 1 else None
    lin = PackedLinear(pack_bits(tags, bits).cuda(), table.cuda(), rs.cuda(), None if rb is None else rb.cuda(),
                       None, torch.arange(nch * 4)[None].cuda(), K, 4)
    return proto_runs.RunsLinear(lin, tags.cuda())


def _ref(x, rls):
    w = torch.cat([cuda_dequant.dequant(rl) for rl in rls])
    return (x.float() @ w.float().T)


def _check(got, ref):
    rel = float((got.float() - ref).norm() / ref.norm())
    assert rel < 2e-3, rel
    assert torch.isfinite(got).all()


# nch = C/4: 64 (tail only, JT=1), 128 (JT=2), 320 (JT=5), 384 (JT=6), 448 (JT=7), 512 (full only),
# 1280 (2 full batches + JT=4)
@pytest.mark.parametrize("bits", [8, 10, 12])
@pytest.mark.parametrize("nch", [64, 128, 320, 384, 448, 512, 1280])
def test_matches_dequant_matmul(bits, nch):
    rl = _rl(200, nch, bits, 4, bits * 7 + nch)
    for T in (1, 37, 128, 300):
        x = (torch.randn(T, nch * 4, device="cuda") * 2).half()
        _check(runs_gemm.gemm(x, [rl]), _ref(x, [rl]))


@pytest.mark.parametrize("table", ["smem", "global"])
def test_variants_and_groups(table):
    a, b = _rl(96, 256, 10, 1, 1), _rl(333, 256, 10, 4, 2)
    x = torch.randn(130, 1024, device="cuda").half()
    if table == "smem" and not all(runs_gemm.plan(r)[0] for r in (a, b)):
        pytest.skip("codebook does not fit in shared memory here")
    _check(runs_gemm.gemm(x, [a, b], table=table), _ref(x, [a, b]))
    # strided input rows (a column slice of a wider activation), output into a buffer
    xw = torch.randn(130, 1024 + 64, device="cuda").half()
    out = torch.empty(130, 96 + 333, dtype=torch.float16, device="cuda")
    runs_gemm.gemm(xw[:, :1024], [a, b], out=out, table=table)
    _check(out, _ref(xw[:, :1024].contiguous(), [a, b]))


def test_big_codebook_global_table():
    """K = 4096 x 4 books (128 KB) is read through L1 instead of shared memory."""
    rl = _rl(130, 512, 12, 4, 5)
    assert runs_gemm.plan(rl)[0] is False or runs_gemm._smem_optin() >= 128 * 1024
    x = torch.randn(70, 2048, device="cuda").half()
    _check(runs_gemm.gemm(x, [rl]), _ref(x, [rl]))


@pytest.mark.parametrize("split", [1, 2, 3, 7, 40])
@pytest.mark.parametrize("nch", [128, 1280])
def test_split_k(split, nch):
    """Split-K slices (including more slices than super-tiles: empty slices) sum to the same result."""
    a, b = _rl(150, nch, 10, 4, 11), _rl(70, nch, 10, 1, 12)
    for T in (1, 64, 200):
        x = torch.randn(T, nch * 4, device="cuda").half()
        _check(runs_gemm.gemm(x, [a, b], split=split), _ref(x, [a, b]))


def test_auto_split_fills_gpu():
    rl = _rl(200, 1280, 10, 1, 3)
    assert runs_gemm.auto_split(rl, 64, sms=188) > 1
    assert runs_gemm.auto_split(rl, 64, sms=188) <= 16


@pytest.mark.parametrize("bm", [16, 32, 64, 128])
@pytest.mark.parametrize("split", [1, 5])
def test_token_tiles(bm, split):
    """Every token-tile height gives the same product, for T below, at and above the tile."""
    a, b = _rl(150, 1280, 10, 4, 21), _rl(70, 1280, 8, 1, 22)
    for T in (1, bm - 1, bm, bm + 3, 3 * bm + 5):
        x = torch.randn(T, 1280 * 4, device="cuda").half()
        _check(runs_gemm.gemm(x, [a, b], split=split, bm=bm), _ref(x, [a, b]))


def test_auto_bm():
    assert [runs_gemm.auto_bm(t) for t in (1, 16, 17, 33, 64, 65, 500)] == [16, 16, 32, 64, 64, 128, 128]
