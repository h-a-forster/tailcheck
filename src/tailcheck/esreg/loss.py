"""Strictly consistent joint (VaR, ES) loss."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tailcheck.esreg.gfuns import G1, G2


def esr_loss(
    r: ArrayLike,
    q: ArrayLike,
    e: ArrayLike,
    alpha: float,
    g1: G1 = G1.ZERO,
    g2: G2 = G2.NEG_INV,
) -> NDArray[np.float64]:
    """Per-observation Fissler-Ziegel loss; with the defaults it equals FZ0 exactly."""
    raise NotImplementedError("M4")
