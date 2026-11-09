"""kernels.had_quant: fused block-128 Hadamard + per-row int8 vs an fp32 reference (GPU)."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("triton")
from shapequant.kernels.had_quant import had_quant, hadamard128  # noqa: E402


def _ref(t):
    y = (t.float().view(-1, 128) @ hadamard128(t.device).float()).view(t.shape)
    s = y.abs().amax(1, keepdim=True).clamp_min(1e-10) / 127
    return torch.round(y / s).clamp(-127, 127).to(torch.int8), s


@pytest.mark.parametrize("N,C", [(1, 128), (37, 5120), (300, 17408), (64, 6144)])
def test_matches_reference_and_given_scale(N, C):
    torch.manual_seed(0)
    t = (torch.randn(N, C, device="cuda") * 0.05).to(torch.bfloat16)
    t[:, 3] *= 40
    q, s = had_quant(t)
    qr, sr = _ref(t)
    assert (q.int() - qr.int()).abs().max() <= 1                 # rint ties / fp32 summation order
    assert (q != qr).float().mean() < 1e-4
    assert torch.allclose(s, sr, rtol=1e-5)
    big = torch.zeros(N + 8, C, dtype=torch.int8, device="cuda")
    sb = s.clone()
    had_quant(t, big[4:4 + N], sb, given=True)                 # rows of a wider matrix, known scales
    assert torch.equal(big[4:4 + N], q) and big[:4].abs().sum() == 0 and big[4 + N:].abs().sum() == 0
