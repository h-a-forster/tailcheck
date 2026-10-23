"""CAViaR symmetric absolute value (Engle-Manganelli 2004) with a constant ES/VaR ratio."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize, signal

from tailcheck.forecasters.base import Arr, BlockForecast

N_INIT = 300


def sav_path(b: Arr, r: Arr, q0: float) -> Arr:
    """q_0 = q0; q_t = b0 + b1 q_{t-1} + b2 |r_{t-1}| for t = 1..len(r) (len(r) + 1 values)."""
    x = b[0] + b[2] * np.abs(r)
    rest = signal.lfilter([1.0], [1.0, -b[1]], x, zi=[b[1] * q0])[0]
    return np.concatenate([[q0], rest])


def check_loss(r: Arr, q: Arr, alpha: float) -> float:
    u = r - q
    return float(np.mean(u * (alpha - (u < 0))))


@dataclass(frozen=True)
class SAVFit:
    b: Arr
    q0: float
    es_ratio: float
    loss: float


def fit_sav(
    window: Arr, alpha: float, rng: np.random.Generator, n_random: int = 2000, n_best: int = 10
) -> SAVFit:
    """Check-loss estimate with random starts, the best ``n_best`` refined by Nelder-Mead."""
    q0 = float(np.quantile(window[:N_INIT], alpha))
    qemp = float(np.quantile(window, alpha))
    mabs = float(np.mean(np.abs(window)))

    def obj(b: Arr) -> float:
        if not 0.0 <= b[1] < 1.0:
            return 1e10
        q = sav_path(b, window[:-1], q0)
        val = check_loss(window, q, alpha)
        return val if np.isfinite(val) else 1e10

    b1 = rng.uniform(0.0, 1.0, n_random)
    w = rng.uniform(0.0, 1.0, n_random)
    b2 = (1 - b1) * w * qemp / mabs
    b0 = (1 - b1) * (1 - w) * qemp * rng.uniform(0.5, 1.5, n_random)
    starts = np.column_stack([b0, b1, b2])
    vals = np.array([obj(s) for s in starts])
    best = None
    for i in np.argsort(vals)[:n_best]:
        res = optimize.minimize(
            obj,
            starts[i],
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-10},
        )
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    b = np.asarray(best.x, dtype=np.float64)
    q = sav_path(b, window[:-1], q0)
    hit = window <= q
    ratio = float(np.mean(window[hit] / q[hit])) if hit.any() else 1.0
    return SAVFit(b, q0, max(1.0, ratio), float(best.fun))


class CaviarSAV:
    """q_t = b0 + b1 q_{t-1} + b2 |r_{t-1}|, fitted by check-loss minimisation; e = c q."""

    name = "caviar-sav"
    has_distribution = False

    def __init__(self, n_random: int = 2000, n_best: int = 10) -> None:
        self.n_random = n_random
        self.n_best = n_best

    def fit(self, window: Arr, alphas: Sequence[float], rng: np.random.Generator) -> Any:
        return {a: fit_sav(window, a, rng, self.n_random, self.n_best) for a in alphas}

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
        q = np.empty((t1 - t0, len(alphas)))
        e = np.empty_like(q)
        for j, a in enumerate(alphas):
            f: SAVFit = state[a]
            path = sav_path(f.b, r[start : t1 - 1], f.q0)  # path[i] is the forecast for start+i
            qq = path[t0 - start : t1 - start]
            q[:, j] = np.minimum(qq, -1e-8)
            e[:, j] = f.es_ratio * q[:, j]
        return BlockForecast(q, e, None, None)
