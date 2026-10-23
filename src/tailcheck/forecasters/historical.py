"""Historical simulation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from tailcheck.forecasters.base import Arr, BlockForecast


def empirical_var_es(x: Arr, alpha: float) -> tuple[float, float]:
    """k = ceil(n alpha); VaR = k-th order statistic, ES = mean of the k smallest."""
    k = max(1, math.ceil(len(x) * alpha - 1e-12))
    part = np.partition(x, k - 1)[:k]
    return float(part.max()), float(part.mean())


class HistoricalSimulation:
    """Empirical alpha-quantile and tail mean of the last ``window`` returns.

    The window rolls daily (no parameters to refit); draws bootstrap the current window.
    """

    name = "hs"
    has_distribution = True

    def fit(self, window: Arr, alphas: Sequence[float], rng: np.random.Generator) -> Any:
        return len(window)

    def forecast_block(
        self,
        state: Any,
        r: Arr,
        start: int,
        t0: int,
        t1: int,
        alphas: Sequence[float],
        n_draws: int,
        rng: np.random.Generator,
    ) -> BlockForecast:
        w = int(state)
        days = t1 - t0
        q = np.empty((days, len(alphas)))
        e = np.empty_like(q)
        draws = np.empty((days, n_draws)) if n_draws > 0 else None
        for i, t in enumerate(range(t0, t1)):
            win = r[t - w : t]
            for j, a in enumerate(alphas):
                q[i, j], e[i, j] = empirical_var_es(win, a)
            if draws is not None:
                draws[i] = win[rng.integers(0, w, n_draws)]
        return BlockForecast(q, e, None, draws)
