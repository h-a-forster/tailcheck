"""ES backtests: Acerbi-Szekely Z1 and Z2 with simulated null distributions, and the
McNeil-Frey exceedance-residual bootstrap test.

Formulas, edge cases and reference values: ``docs/DESIGN.md`` section 4.2. Milestone M3.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tailcheck.results import BacktestResult

# simulate(rng, n_paths) -> array of shape (n_paths, T): return paths drawn from the
# forecast distributions F_1..F_T that produced (q, e).
Simulator = Callable[[np.random.Generator, int], NDArray[np.float64]]


def acerbi_szekely_z1(r: ArrayLike, q: ArrayLike, e: ArrayLike) -> float:
    """Z1 = 1 - mean over hits of r_t / e_t. ``nan`` if there are no hits."""
    raise NotImplementedError("M3")


def acerbi_szekely_z2(r: ArrayLike, q: ArrayLike, e: ArrayLike, alpha: float) -> float:
    """Z2 = 1 - sum_t r_t 1{r_t <= q_t} / (T alpha e_t)."""
    raise NotImplementedError("M3")


def acerbi_szekely_test(
    r: ArrayLike,
    q: ArrayLike,
    e: ArrayLike,
    alpha: float,
    simulate: Simulator,
    statistic: Literal["z1", "z2"] = "z2",
    n_sim: int = 5000,
    seed: int | None = None,
    chunk: int = 500,
) -> BacktestResult:
    """One-sided simulation test; small Z rejects (risk underestimated)."""
    raise NotImplementedError("M3")


def mcneil_frey(
    r: ArrayLike,
    q: ArrayLike,
    e: ArrayLike,
    sigma: ArrayLike | None = None,
    n_boot: int = 1000,
    seed: int | None = None,
) -> BacktestResult:
    """McNeil and Frey (2000) bootstrap test that exceedance residuals have mean zero."""
    raise NotImplementedError("M3")
