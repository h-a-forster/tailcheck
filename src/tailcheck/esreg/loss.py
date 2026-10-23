"""Strictly consistent joint (VaR, ES) loss (Fissler and Ziegel 2016; DB 2019 eq. 2.2)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tailcheck._validation import as_float_1d, check_alpha, check_same_length
from tailcheck.esreg.gfuns import G1, G2, NEGATIVE_DOMAIN, curly_g2, g1, g2


def esr_loss_raw(
    y: NDArray[np.float64],
    q: NDArray[np.float64],
    e: NDArray[np.float64],
    alpha: float,
    g1_kind: G1,
    g2_kind: G2,
) -> NDArray[np.float64]:
    """Unvalidated per-observation loss (used inside the optimiser)."""
    hit = (y <= q).astype(np.float64)
    return (
        (hit - alpha) * g1(q, g1_kind)
        - hit * g1(y, g1_kind)
        + g2(e, g2_kind) * (e - q + (q - y) * hit / alpha)
        - curly_g2(e, g2_kind)
    )


def esr_loss(
    r: ArrayLike,
    q: ArrayLike,
    e: ArrayLike,
    alpha: float,
    g1: G1 = G1.ZERO,
    g2: G2 = G2.NEG_INV,
) -> NDArray[np.float64]:
    """Per-observation Fissler-Ziegel loss; with the defaults it equals FZ0 exactly."""
    alpha = check_alpha(alpha)
    ra, qa, ea = as_float_1d(r, "r"), as_float_1d(q, "q"), as_float_1d(e, "e")
    check_same_length([ra, qa, ea], ["r", "q", "e"])
    if G2(g2) in NEGATIVE_DOMAIN and np.any(ea >= 0):
        raise ValueError(f"G2 type {int(g2)} requires negative ES values")
    return esr_loss_raw(ra, qa, ea, alpha, G1(g1), G2(g2))
