"""Forecaster protocol and the per-day forecast record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RiskForecast:
    """VaR and ES for one day on the return scale, plus an optional predictive sampler.

    ``draws`` (if present) are samples from the full predictive distribution, used by the
    Acerbi-Szekely simulation test; CAViaR has none.
    """

    q: float
    e: float
    draws: NDArray[np.float64] | None = None


class Forecaster(Protocol):
    """Rolling-window forecaster with periodic refits.

    ``fit`` sees only the estimation window (oldest first). ``forecast`` sees the fitted
    state and all returns up to and including day t-1, and returns the forecast for day t.
    """

    name: str

    def fit(self, window: NDArray[np.float64], alpha: float, rng: np.random.Generator) -> Any: ...

    def forecast(
        self,
        state: Any,
        history: NDArray[np.float64],
        alpha: float,
        n_draws: int,
        rng: np.random.Generator,
    ) -> RiskForecast: ...
