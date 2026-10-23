"""Input validation shared by the public functions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def as_float_1d(x: ArrayLike, name: str) -> FloatArray:
    """Return ``x`` as a finite 1-D float64 array or raise ``ValueError``."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return a


def check_alpha(alpha: float) -> float:
    """Validate a tail probability level in (0, 1)."""
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return a


def check_same_length(arrays: Sequence[FloatArray], names: Sequence[str]) -> int:
    """Raise unless all arrays have the same length; return that length."""
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        detail = ", ".join(f"{n}={len(a)}" for n, a in zip(names, arrays, strict=True))
        raise ValueError(f"length mismatch: {detail}")
    return lengths.pop()
