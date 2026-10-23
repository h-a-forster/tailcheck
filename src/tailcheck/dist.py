"""Closed-form VaR/ES of standardized distributions and samplers (DESIGN 4.5).

All laws have mean 0 and variance 1; values are on the return scale (left tail, negative).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from tailcheck._validation import check_alpha


def normal_var_es(alpha: float) -> tuple[float, float]:
    """(VaR, ES) of the standard normal at tail level ``alpha``."""
    alpha = check_alpha(alpha)
    z = float(stats.norm.ppf(alpha))
    return z, -float(stats.norm.pdf(z)) / alpha


def std_t_var_es(nu: float, alpha: float) -> tuple[float, float]:
    """(VaR, ES) of the unit-variance Student-t with ``nu > 2`` degrees of freedom."""
    alpha = check_alpha(alpha)
    if not nu > 2:
        raise ValueError("nu must exceed 2 for a unit-variance t")
    ta = float(stats.t.ppf(alpha, nu))
    s = math.sqrt((nu - 2) / nu)
    es = -s * (nu + ta**2) / (nu - 1) * float(stats.t.pdf(ta, nu)) / alpha
    return s * ta, es


def draw_std_t(rng: np.random.Generator, nu: float, size: int | tuple[int, ...]) -> NDArray:
    """Unit-variance Student-t draws."""
    return rng.standard_t(nu, size=size) * math.sqrt((nu - 2) / nu)
