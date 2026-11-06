"""evaluate.block_quant matches the runtime head formats' dequantized weights exactly."""

import pytest
import torch

from shapequant.evaluate import block_quant
from shapequant.kernels import proto_formats as pf


@pytest.mark.parametrize("bits,pack", [(4, pf.q4_pack), (8, pf.q8_pack)])
def test_block_quant_matches_runtime_head(bits, pack):
    w = torch.randn(300, 256, generator=torch.Generator().manual_seed(bits)) * 0.05
    w[3, :32] = 0.0                                              # all-zero block
    assert torch.equal(block_quant(w, bits), pack(w)[2].float())
