"""VaR backtests: Kupiec POF, Christoffersen independence and conditional coverage,
Engle-Manganelli dynamic quantile, Basel traffic light.

Formulas, edge cases and reference values: ``docs/DESIGN.md`` section 4.1. Milestone M2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tailcheck.results import BacktestResult

Zone = Literal["green", "yellow", "red"]


def hits(r: ArrayLike, q: ArrayLike) -> NDArray[np.bool_]:
    """VaR exceedance indicator ``r <= q`` (validated, same length)."""
    raise NotImplementedError("M2")


def kupiec_pof(r: ArrayLike, q: ArrayLike, alpha: float) -> BacktestResult:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test, chi2(1)."""
    raise NotImplementedError("M2")


def christoffersen_independence(r: ArrayLike, q: ArrayLike, alpha: float) -> BacktestResult:
    """Christoffersen (1998) first-order Markov independence LR test, chi2(1)."""
    raise NotImplementedError("M2")


def christoffersen_cc(r: ArrayLike, q: ArrayLike, alpha: float) -> BacktestResult:
    """Christoffersen (1998) conditional coverage test: LR_pof + LR_ind, chi2(2)."""
    raise NotImplementedError("M2")


def dynamic_quantile(
    r: ArrayLike, q: ArrayLike, alpha: float, lags: int = 4, include_var: bool = True
) -> BacktestResult:
    """Engle and Manganelli (2004) dynamic quantile Wald test, chi2(lags + 1 + include_var)."""
    raise NotImplementedError("M2")


@dataclass(frozen=True)
class TrafficLight:
    """Basel (1996) traffic-light classification of a VaR exception count."""

    zone: Zone
    exceptions: int
    nobs: int
    cumulative_probability: float
    plus_factor: float


def traffic_light(
    r: ArrayLike,
    q: ArrayLike,
    alpha: float = 0.01,
    green_below: float = 0.95,
    red_from: float = 0.9999,
) -> TrafficLight:
    """Classify the exception count by its binomial cumulative probability."""
    raise NotImplementedError("M2")
