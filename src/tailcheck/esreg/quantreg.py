"""Exact linear quantile regression by linear programming (scipy HiGHS)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def quantile_regression(y: ArrayLike, x: ArrayLike, tau: float) -> NDArray[np.float64]:
    """Koenker-Bassett (1978) estimate of ``beta`` minimising the check loss.

    ``x`` must already contain an intercept column if one is wanted.
    """
    raise NotImplementedError("M4")


def quantile_regression_se_iid(
    y: ArrayLike, x: ArrayLike, tau: float, beta: ArrayLike
) -> NDArray[np.float64]:
    """IID-error standard errors (Koenker 2005, sec. 3.4), used only to scale perturbations."""
    raise NotImplementedError("M4")
