"""kernels.fp8_gemm (EXPERIMENTAL): the e4m3 weight decode and the scaled math, checked on any GPU;
the real FP8 GEMM (torch._scaled_mm) only runs on compute capability >= 8.9."""

import pytest
import torch

pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from shapequant.kernels import cuda_dequant, fp8_gemm  # noqa: E402
from test_runs_gemm import _rl  # noqa: E402


@pytest.mark.parametrize("bits,nch", [(8, 320), (10, 512), (12, 1280)])
def test_dequant_fp8_matches_torch_cast(bits, nch):
    rl = _rl(150, nch, bits, 4, bits + nch)
    w16 = cuda_dequant.dequant(rl).float()
    sw = (fp8_gemm.weight_absmax(rl) / fp8_gemm.E4M3_MAX).reshape(1)
    # an upper bound from the codebooks (every codeword of a row's book, used or not)
    assert float(fp8_gemm.weight_absmax(rl)) >= float(w16.abs().max()) * (1 - 1e-3)
    buf = torch.empty(rl.R * nch * 4, dtype=torch.uint8, device="cuda")
    got = fp8_gemm.dequant_fp8(rl, sw, buf).float() * sw
    # torch's e4m3 cast of the same fp32 values (round to nearest even; in range by construction)
    ref8 = (w16 / sw).to(torch.float8_e4m3fn).float() * sw
    assert float((got - ref8).abs().max()) <= 0.13 * float(w16.abs().max())   # w16 is fp16-rounded: at most 1 e4m3 ulp apart
    assert float((got - ref8).norm() / ref8.norm()) < 5e-3
    rel = float((got - w16).norm() / w16.norm())
    assert rel < 0.04, rel                                          # e4m3: ~2-3% RMS rounding


def test_emulated_gemm_close_to_fp16():
    a, b = _rl(256, 512, 10, 4, 1), _rl(128, 512, 10, 1, 2)
    x = torch.randn(100, 2048, device="cuda").half()
    scratch = torch.empty(384 * 2048, dtype=torch.float16, device="cuda")
    ref = x.float() @ torch.cat([cuda_dequant.dequant(a), cuda_dequant.dequant(b)]).float().T
    got = fp8_gemm.gemm_emulated(x, [a, b], scratch).float()
    rel = float((got - ref).norm() / ref.norm())
    assert rel < 0.06, rel


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
                    reason="FP8 tensor cores need cc >= 8.9")
def test_fp8_gemm_matches_emulation():
    a = _rl(256, 512, 10, 4, 3)
    x = torch.randn(100, 2048, device="cuda").half()
    scratch = torch.empty(256 * 2048, dtype=torch.float16, device="cuda")
    emu = fp8_gemm.gemm_emulated(x, [a], scratch).float()
    got = fp8_gemm.gemm(x, [a], scratch).float()
    assert float((got - emu).norm() / emu.norm()) < 5e-3


def test_fp8_gemm_refuses_old_gpus():
    if fp8_gemm.device_ok():
        pytest.skip("this GPU supports FP8")
    a = _rl(128, 64, 8, 1, 4)
    with pytest.raises(RuntimeError):
        fp8_gemm.gemm(torch.randn(4, 256, device="cuda").half(), [a],
                      torch.empty(128 * 256, dtype=torch.float16, device="cuda"))
