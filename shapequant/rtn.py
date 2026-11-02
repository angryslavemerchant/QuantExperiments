"""Round-to-nearest scalar baselines (group-wise, one 16-bit scale per group).

All functions take a float32 weight matrix [R, C] and return the dequantized
float32 matrix. Scales are rounded to fp16 before dequantization so the
simulated model matches what 16-bit stored scales would give.
"""

from __future__ import annotations

import torch


def _grouped(w: torch.Tensor, group: int) -> tuple[torch.Tensor, int]:
    rows, cols = w.shape
    pad = (-cols) % group
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    return w.reshape(rows, -1, group), pad


def _ungrouped(g: torch.Tensor, rows: int, pad: int) -> torch.Tensor:
    w = g.reshape(rows, -1)
    return w[:, : w.shape[1] - pad] if pad else w


def rtn_quantize(w: torch.Tensor, bits: int, group: int = 32) -> torch.Tensor:
    """Symmetric RTN with integer levels -2^(b-1) .. 2^(b-1)-1 (Q4: -8..7).

    The scale is chosen like llama.cpp Q4_0: the signed weight with the
    largest magnitude maps exactly onto the -2^(b-1) level, so all 2^b levels
    are reachable with a single scale per group.
    """
    if bits < 2:
        raise ValueError("rtn_quantize needs bits >= 2; use ternary_quantize")
    rows = w.shape[0]
    g, pad = _grouped(w, group)
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    idx = g.abs().argmax(dim=-1, keepdim=True)
    extreme = torch.gather(g, -1, idx)  # signed value with max |w|
    scale = (extreme / lo).to(torch.float16).to(torch.float32)
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(g / safe), lo, hi)
    return _ungrouped(q * scale, rows, pad)


def ternary_quantize(w: torch.Tensor, group: int = 32) -> torch.Tensor:
    """Per group: scale = mean |w|, round w/scale to {-1, 0, +1}."""
    rows = w.shape[0]
    g, pad = _grouped(w, group)
    scale = g.abs().mean(dim=-1, keepdim=True).to(torch.float16).to(torch.float32)
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(g / safe), -1, 1)
    return _ungrouped(q * scale, rows, pad)
