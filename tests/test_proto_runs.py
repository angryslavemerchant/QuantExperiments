"""Lane-interleaved GEMV prototype (shapequant/kernels/proto_runs.py) and the
fp16-overflow guard shared with cuda_gemv. Needs CUDA + CuPy."""

import pytest
import torch

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from shapequant import masks as mk  # noqa: E402
from shapequant.kernels import cuda_gemv, proto_runs  # noqa: E402
from shapequant.kernels.gemv import PackedLinear  # noqa: E402
from shapequant.packed import pack_bits  # noqa: E402


def _lin(R, C, bits, G=1, M=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    K = 1 << bits
    tags = torch.randint(0, K, (R, C // 4), generator=g)
    table = (torch.randn(G * K, 4, generator=g) * 0.5).half()
    rs = (torch.rand(R, generator=g) * 0.02 + 0.01).half()
    rb = torch.randint(0, G, (R,), generator=g) if G > 1 else None
    rm = torch.randint(0, M, (R,), generator=g)
    perms = mk.seeded_perms(C, M, torch.Generator().manual_seed(seed + 1))
    lin = PackedLinear(pack_bits(tags, bits).cuda(), table.cuda(), rs.cuda(), None if rb is None else rb.cuda(),
                       rm.cuda(), perms.cuda(), K, 4)
    return lin, tags


@pytest.mark.parametrize("R,C", [(512, 3584), (300, 18944), (256, 4864), (128, 256)])
@pytest.mark.parametrize("bits", [10, 12])
@pytest.mark.parametrize("ru", [1, 3, 4])
def test_matches_dense(R, C, bits, ru):
    lin, tags = _lin(R, C, bits)
    rl = proto_runs.RunsLinear(lin, tags, ru)
    assert rl.words.shape[1] * 32 == (C // 4) * bits          # no padding
    x = torch.randn(C, device="cuda").half()
    ref = lin.dense() @ x.float()
    y = proto_runs.gemv(rl, x).float()
    assert float((y - ref).abs().max() / ref.abs().max()) < 3e-3


def test_mixture_books():
    lin, tags = _lin(256, 3584, 10, G=4)
    x = torch.randn(3584, device="cuda").half()
    ref = lin.dense() @ x.float()
    y = proto_runs.gemv(proto_runs.RunsLinear(lin, tags, 2), x).float()
    assert float((y - ref).abs().max() / ref.abs().max()) < 3e-3


@pytest.mark.parametrize("bits", [8, 10, 12, 14])
@pytest.mark.parametrize("R,C", [(512, 3584), (256, 18944)])
def test_stack_bits_with_four_books(R, C, bits):
    """The real stacks' per-matrix K (256..16384) with G = 4: tables of 8 KB..512 KB, the
    larger ones read through L1 (global path)."""
    lin, tags = _lin(R, C, bits, G=4)
    rl = proto_runs.RunsLinear(lin, tags, 1)
    x = torch.randn(C, device="cuda").half()
    ref = lin.dense() @ x.float()
    for cfg in ((512, 1), (0, 1)):
        y = proto_runs.gemv(rl, x, threads=cfg[0], blocks_per_sm=cfg[1]).float()
        assert float((y - ref).abs().max() / ref.abs().max()) < 3e-3, cfg


@pytest.mark.parametrize("kernel", ["runs", "cur"])
def test_outlier_activation_does_not_overflow(kernel):
    lin, tags = _lin(256, 3584, 10)
    lin.table[:8] = 60.0                                        # huge normalized codewords
    x = torch.randn(3584, device="cuda").half()
    x[123], x[999] = 5536.0, -4000.0
    ref = lin.dense() @ x.float()
    y = (proto_runs.gemv(proto_runs.RunsLinear(lin, tags, 2), x) if kernel == "runs"
         else cuda_gemv.gemv(lin, x)).float()
    assert torch.isfinite(y).all()
    assert float((y - ref).norm() / ref.norm()) < 3e-3


@pytest.mark.parametrize("cmax", [0.01, 1.0, 60.0])
def test_scaled_table_bound_is_exact_power_of_two(cmax):
    t = torch.randn(1024, 4, device="cuda").half()
    t = (t / t.abs().max() * cmax).half()
    s, ys = proto_runs.scaled_table(t, proto_runs.TABLE_MAX)
    assert float(s.float().abs().max()) <= proto_runs.TABLE_MAX
    assert float(ys).is_integer() or (1 / float(ys)).is_integer()
    # exact except where scaling down lands in fp16 subnormals (spacing 2^-24)
    assert float(((s.float() * ys) - t.float()).abs().max()) <= 2.0 ** -24 * float(ys)


def test_saturated_activations_stay_finite():
    lin, tags = _lin(64, 3584, 10)
    lin.table[:] = lin.table.sign() * 60.0
    x = torch.full((3584,), 30000.0, device="cuda").half()
    x[::2] = -30000.0                                          # partial sums cancel; products are huge
    for y in (proto_runs.gemv(proto_runs.RunsLinear(lin, tags, 1), x), cuda_gemv.gemv(lin, x)):
        assert torch.isfinite(y.float()).all() or not torch.isfinite((lin.dense() @ x.float()).half()).all()


@pytest.mark.parametrize("R,C", [(512, 3584), (300, 18944), (256, 4864), (128, 256)])
@pytest.mark.parametrize("ru", [1, 2])
@pytest.mark.parametrize("cfg", [(64, 1, True), (64, 1, False), (0, 1, True), (1024, 1, True)])
def test_launch_configs_and_pipeline(R, C, ru, cfg):
    # 64 threads x 1 block/SM: every warp runs many tiles (exercises the cross-tile pipeline)
    threads, bps, pipe = cfg
    lin, tags = _lin(R, C, 12, G=2)
    rl = proto_runs.RunsLinear(lin, tags, ru)
    x = torch.randn(C, device="cuda").half()
    ref = lin.dense() @ x.float()
    y = proto_runs.gemv(rl, x, threads=threads, blocks_per_sm=bps, pipe=pipe).float()
    assert float((y - ref).abs().max() / ref.abs().max()) < 3e-3
