"""Forecaster protocol, block forecast record and the rolling-window driver.

A forecaster is refit on the last ``window`` returns every ``refit_every`` days; between
refits its filter runs daily with fixed parameters on data up to t-1. ``forecast_block``
receives the full return array but may only read ``r[:t]`` for the forecast of day ``t``;
``tests/test_forecasters.py`` checks this by perturbing future data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


@dataclass(frozen=True)
class RiskForecast:
    """VaR and ES for one day on the return scale, plus an optional predictive sampler.

    ``draws`` (if present) are samples from the full predictive distribution, used by the
    Acerbi-Szekely simulation test; CAViaR has none.
    """

    q: float
    e: float
    draws: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class BlockForecast:
    """Forecasts for days ``t0..t1-1``: ``q``/``e`` have shape (days, n_alphas).

    ``sigma`` is the conditional volatility forecast (``None`` if the model has none);
    ``draws`` has shape (days, n_draws) or is ``None``.
    """

    q: Arr
    e: Arr
    sigma: Arr | None
    draws: Arr | None


class Forecaster(Protocol):
    """Rolling-window forecaster with periodic refits."""

    name: str
    has_distribution: bool

    def fit(self, window: Arr, alphas: Sequence[float], rng: np.random.Generator) -> Any: ...

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
        """Forecasts for days t0..t1-1 using ``r[start:t]`` for day t (``start`` is the
        first index of the estimation window used by ``fit``)."""
        ...


def rolling_forecast(
    model: Forecaster,
    r: Arr,
    oos_start: int,
    oos_end: int,
    window: int,
    refit_every: int,
    alphas: Sequence[float],
    n_draws: int = 0,
    seed: int | None = None,
    on_block: Any = None,
) -> dict[str, Any]:
    """Run ``model`` over days ``oos_start..oos_end-1`` (indices into ``r``).

    Returns ``q``/``e`` (days x alphas), ``sigma`` (or NaN) and, if ``n_draws > 0`` and the
    model has a predictive distribution, the ``draws`` per block via ``on_block(t0, t1,
    block)`` (draws are not kept in memory by this function).
    """
    if oos_start < window:
        raise ValueError("not enough history before the first forecast")
    rng = np.random.default_rng(seed)
    n = oos_end - oos_start
    na = len(alphas)
    q = np.full((n, na), np.nan)
    e = np.full((n, na), np.nan)
    sigma = np.full(n, np.nan)
    n_refits = 0
    for t0 in range(oos_start, oos_end, refit_every):
        t1 = min(t0 + refit_every, oos_end)
        start = t0 - window
        state = model.fit(r[start:t0], alphas, rng)
        n_refits += 1
        blk = model.forecast_block(state, r, start, t0, t1, alphas, n_draws, rng)
        q[t0 - oos_start : t1 - oos_start] = blk.q
        e[t0 - oos_start : t1 - oos_start] = blk.e
        if blk.sigma is not None:
            sigma[t0 - oos_start : t1 - oos_start] = blk.sigma
        if on_block is not None:
            on_block(t0, t1, blk)
    return {"q": q, "e": e, "sigma": sigma, "n_refits": n_refits}
