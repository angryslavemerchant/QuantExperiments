import importlib.util

import pytest
import torch

from shapequant import masks as mk
from shapequant.kernels.gemv import PackedLinear
from shapequant.packed import pack_bits

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
cupy = pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="needs CuPy")


def _lin(R, C, bits, G, M, seed=0):
    g = torch.Generator().manual_seed(seed)
    K = 1 << bits
    tags = torch.randint(0, K, (R, C // 4), generator=g)
    table = (torch.randn(G * K, 4, generator=g) * 0.5).half()
    rs = (torch.rand(R, generator=g) * 0.02 + 0.01).half()
    rb = torch.randint(0, G, (R,), generator=g) if G > 1 else None
    rm = torch.randint(0, M, (R,), generator=g) if M > 1 else None
    perms = mk.seeded_perms(C, M, torch.Generator().manual_seed(seed + 1))
    to = (lambda t: None if t is None else t.cuda())
    return PackedLinear(pack_bits(tags, bits).cuda(), table.cuda(), rs.cuda(), to(rb), to(rm), perms.cuda(), K, 4)


CASES = [(96, 256, 10, 4, 16), (64, 384, 12, 1, 1), (40, 128, 14, 2, 4), (33, 640, 8, 1, 8)]


def _check(fn, case):
    lin = _lin(*case)
    x = torch.randn(case[1], device="cuda", dtype=torch.float16)
    ref = lin.dense() @ x.float()
    got = fn(lin, x).float()
    assert float((got - ref).abs().max() / ref.abs().max()) < 3e-3


@cuda
@pytest.mark.parametrize("case", CASES)
def test_triton_gemv(case):
    _check(lambda lin, x: lin.gemv(x), case)


@cuda
@cupy
@pytest.mark.parametrize("case", CASES)
def test_cuda_gemv(case):
    from shapequant.kernels import cuda_gemv
    _check(lambda lin, x: cuda_gemv.gemv(lin, x), case)


@cuda
@cupy
@pytest.mark.parametrize("case", CASES)
def test_cuda_l1_gemv(case):
    from shapequant.kernels import cuda_l1
    _check(lambda lin, x: cuda_l1.gemv(lin, x), case)


@cuda
@cupy
@pytest.mark.parametrize("case", [c for c in CASES if (c[1] // 4 // 32 * c[2]) % 4 == 0])
def test_cuda_slab_gemv(case):
    from shapequant.kernels import cuda_slab
    _check(lambda lin, x: cuda_slab.gemv(lin, x), case)


@cuda
@cupy
@pytest.mark.parametrize("case", CASES)
def test_cuda_tiles_gemv(case):
    from shapequant.kernels import cuda_tiles
    _check(lambda lin, x: cuda_tiles.gemv(cuda_tiles.TiledLayout(lin, 16), x), case)


@cuda
def test_packed_linear_forward_matches_dense_for_batches():
    lin = _lin(48, 256, 10, 4, 16)
    x = torch.randn(3, 256, device="cuda", dtype=torch.float16)
    ref = x.float() @ lin.dense().T
    assert torch.allclose(lin(x).float(), ref, atol=2e-2, rtol=2e-2)
