"""Forecast comparison: FZ0 loss, Diebold-Mariano test, model confidence set (via ``arch``).

Formulas: ``docs/DESIGN.md`` section 4.4. Milestone M5.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from tailcheck.results import BacktestResult


def fz0_loss(r: ArrayLike, q: ArrayLike, e: ArrayLike, alpha: float) -> NDArray[np.float64]:
    """Patton-Ziegel-Chen (2019) FZ0 loss per observation. Requires e <= q < 0."""
    raise NotImplementedError("M5")


def diebold_mariano(
    loss_a: ArrayLike,
    loss_b: ArrayLike,
    lags: int | None = None,
    small_sample: bool = True,
) -> BacktestResult:
    """DM test of equal expected loss with a Newey-West long-run variance.

    Negative statistic: model ``a`` has lower loss. ``small_sample`` applies the
    Harvey-Leybourne-Newbold (1997) correction and Student-t(T-1) reference.
    """
    raise NotImplementedError("M5")


@dataclass(frozen=True)
class MCSResult:
    """Model confidence set at level ``1 - size``."""

    included: tuple[str, ...]
    excluded: tuple[str, ...]
    pvalues: Mapping[str, float]
    size: float
    method: str
    reps: int
    block_size: int
    seed: int | None


def model_confidence_set(
    losses: pd.DataFrame,
    size: float = 0.10,
    reps: int = 10_000,
    block_size: int | None = None,
    method: Literal["R", "max"] = "R",
    seed: int | None = None,
) -> MCSResult:
    """Hansen-Lunde-Nason (2011) MCS on a T x M loss frame, wrapping ``arch.bootstrap.MCS``."""
    raise NotImplementedError("M5")
