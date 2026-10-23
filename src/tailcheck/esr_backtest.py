"""Bayer-Dimitriadis (2022) ES regression backtests.

version 1 "strict ESR":       r ~ e | e,        H0: ES coefficients (0, 1), Wald chi2(2)
version 2 "auxiliary ESR":    r ~ q | e,        H0: ES coefficients (0, 1), Wald chi2(2)
version 3 "strict intercept": (r - e) ~ e | 1,  H0: ES intercept 0, t-test (one- and two-sided)

Specification: ``docs/DESIGN.md`` section 4.3.4. Milestone M4.
"""

from __future__ import annotations

from typing import Literal

from numpy.typing import ArrayLike

from tailcheck.esreg.covariance import SigmaEstimator, Sparsity
from tailcheck.results import BacktestResult


def esr_backtest(
    r: ArrayLike,
    e: ArrayLike,
    alpha: float,
    version: Literal[1, 2, 3] = 1,
    q: ArrayLike | None = None,
    n_boot: int = 0,
    sparsity: Sparsity = "nid",
    sigma_est: SigmaEstimator = "scl_sp",
    misspec: bool = True,
    seed: int | None = None,
) -> BacktestResult:
    """Run one ESR backtest; asymptotic p-value, plus bootstrap p-value if ``n_boot > 0``."""
    raise NotImplementedError("M4")
