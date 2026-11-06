"""kernels.cuda_dequant: GPU dequant from lane-major run words == the reference decode, bit for bit."""

import pytest
import torch

pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from shapequant.kernels import cuda_dequant, proto_runs  # noqa: E402
from shapequant.kernels.gemv import PackedLinear  # noqa: E402
from shapequant.packed import pack_bits  # noqa: E402


@pytest.mark.parametrize("bits", [8, 10, 12])
@pytest.mark.parametrize("nch", [64, 128, 576, 1024])     # tail only, full batches only, both
@pytest.mark.parametrize("G", [1, 4])
def test_dequant_matches_reference(bits, nch, G):
    g = torch.Generator().manual_seed(bits * 1000 + nch + G)
    R, K = 40, 2 ** bits
    tags = torch.randint(0, K, (R, nch), generator=g)
    table = (torch.randn(G * K, 4, generator=g) * 0.7).half()
    rs = (torch.rand(R, generator=g) * 0.03 + 1e-3).half()
    rb = torch.randint(0, G, (R,), generator=g).to(torch.uint8) if G > 1 else None
    perms = torch.arange(nch * 4)[None]
    lin = PackedLinear(pack_bits(tags, bits).cuda(), table.cuda(), rs.cuda(), None if rb is None else rb.cuda(),
                       None, perms.cuda(), K, 4)
    rl = proto_runs.RunsLinear(lin, tags.cuda())
    assert cuda_dequant.supported(rl)
    got = cuda_dequant.dequant(rl)
    off = (rb.long() * K if rb is not None else torch.zeros(R, dtype=torch.long))[:, None]
    ref = (table.float()[tags + off].view(R, -1) * rs.float()[:, None]).half()
    assert torch.equal(got.cpu(), ref)
    # into a larger shared buffer
    buf = torch.full((R * nch * 4 + 100,), float("nan"), dtype=torch.float16, device="cuda")
    assert torch.equal(cuda_dequant.dequant(rl, buf).cpu(), ref)
