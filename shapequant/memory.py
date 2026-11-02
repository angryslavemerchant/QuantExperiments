"""Memory accounting by formula (never measured from files).

Each function returns (rounded_bits, ideal_bits) for one compressed matrix with
R output rows and C input columns. "Rounded" uses whole-bit tag widths;
"ideal" uses log2 of the number of states.
"""

from __future__ import annotations

import math


def rtn_bits(R: int, C: int, bits: int, group: int = 32) -> tuple[float, float]:
    n_groups = R * math.ceil(C / group)
    total = R * C * bits + n_groups * 16
    return total, total


def ternary_bits(R: int, C: int, group: int = 32) -> tuple[float, float]:
    n_groups = R * math.ceil(C / group)
    rounded = R * C * 2 + n_groups * 16
    ideal = R * C * math.log2(3) + n_groups * 16
    return rounded, ideal


def codebook_bits(
    R: int,
    C: int,
    d: int,
    K: int,
    *,
    shuffle: bool = False,
    masks: int = 0,
) -> tuple[float, float]:
    """tags + codebook (K shapes x d values x 16 bits) + row scales + shuffle + masks."""
    n_tags = R * math.ceil(C / d)
    fixed = K * d * 16 + R * 16
    if shuffle:
        fixed += C * 16
    mask_round = mask_ideal = 0.0
    if masks:
        fixed += masks * C * 16
        mask_round = R * math.ceil(math.log2(masks))
        mask_ideal = R * math.log2(masks)
    rounded = n_tags * math.ceil(math.log2(K)) + fixed + mask_round
    ideal = n_tags * math.log2(K) + fixed + mask_ideal
    return rounded, ideal
