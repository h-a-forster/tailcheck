"""GJR-GARCH(1,1) based forecasters: parametric Student-t, filtered HS, EVT-POT.

Parameters are estimated with ``arch``; the variance filter is our own recursion, which
reproduces ``arch``'s conditional volatility (same backcast) and lets the filter run past the
estimation window with fixed parameters and no look-ahead.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from tailcheck.dist import std_t_var_es
from tailcheck.forecasters.base import Arr, BlockForecast
from tailcheck.forecasters.historical import empirical_var_es


@dataclass(frozen=True)
class GJRParams:
    mu: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    nu: float | None
    backcast: float


def fit_gjr(window: Arr, dist: str) -> GJRParams:
    """Fit a constant-mean GJR-GARCH(1,1) with ``arch`` (``dist`` 'normal' or 't')."""
    from arch import arch_model

    m = arch_model(window, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist=dist, rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = m.fit(disp="off", show_warning=False)
    p = res.params
    e0 = window - window.mean()  # arch computes the backcast from the starting residuals
    tau = min(75, len(e0))
    w = 0.94 ** np.arange(tau)
    w /= w.sum()
    return GJRParams(
        mu=float(p["mu"]),
        omega=float(p["omega"]),
        alpha=float(p["alpha[1]"]),
        gamma=float(p["gamma[1]"]),
        beta=float(p["beta[1]"]),
        nu=float(p["nu"]) if "nu" in p else None,
        backcast=float(np.sum(e0[:tau] ** 2 * w)),
    )


def gjr_filter(p: GJRParams, r: Arr) -> Arr:
    """Conditional variances sigma2[0..len(r)] (one step past the data) for returns ``r``."""
    e = r - p.mu
    n = len(e)
    s2 = np.empty(n + 1)
    s2[0] = p.omega + (p.alpha + 0.5 * p.gamma) * p.backcast + p.beta * p.backcast
    for t in range(1, n + 1):
        et = e[t - 1]
        s2[t] = p.omega + (p.alpha + (p.gamma if et < 0 else 0.0)) * et * et + p.beta * s2[t - 1]
    return s2


@dataclass
class _GarchState:
    params: GJRParams
    z: Arr  # standardized in-sample residuals
    extra: Any = None


class _GarchBase:
    name = "garch"
    has_distribution = True
    dist = "normal"

    def _fit_common(self, window: Arr) -> _GarchState:
        p = fit_gjr(window, self.dist)
        s2 = gjr_filter(p, window)[:-1]
        z = (window - p.mu) / np.sqrt(s2)
        return _GarchState(p, z)

    def _std_var_es(self, st: _GarchState, alpha: float) -> tuple[float, float]:
        raise NotImplementedError

    def _std_draws(self, st: _GarchState, rng: np.random.Generator, size: int) -> Arr:
        raise NotImplementedError

    def fit(self, window: Arr, alphas: Sequence[float], rng: np.random.Generator) -> Any:
        st = self._fit_common(window)
        st.extra = {a: self._std_var_es(st, a) for a in alphas}
        return st

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
        st: _GarchState = state
        s2 = gjr_filter(st.params, r[start : t1 - 1])
        sigma = np.sqrt(s2[t0 - start : t1 - start])
        mu = st.params.mu
        qz = np.array([st.extra[a][0] for a in alphas])
        ez = np.array([st.extra[a][1] for a in alphas])
        q = mu + sigma[:, None] * qz[None, :]
        e = mu + sigma[:, None] * ez[None, :]
        draws = None
        if n_draws > 0:
            draws = mu + sigma[:, None] * self._std_draws(st, rng, (t1 - t0) * n_draws).reshape(
                t1 - t0, n_draws
            )
        return BlockForecast(q, e, sigma, draws)


class GJRGarchT(_GarchBase):
    """GJR-GARCH(1,1) with standardized Student-t innovations (arch)."""

    name = "gjr-t"
    dist = "t"

    def _std_var_es(self, st: _GarchState, alpha: float) -> tuple[float, float]:
        assert st.params.nu is not None
        return std_t_var_es(max(st.params.nu, 2.05), alpha)

    def _std_draws(self, st: _GarchState, rng: np.random.Generator, size: int) -> Arr:
        nu = max(float(st.params.nu or 30.0), 2.05)
        return rng.standard_t(nu, size) * math.sqrt((nu - 2) / nu)


class FilteredHS(_GarchBase):
    """GJR-GARCH(1,1) QMLE volatility times empirical quantile/tail mean of residuals."""

    name = "fhs"
    dist = "normal"

    def _std_var_es(self, st: _GarchState, alpha: float) -> tuple[float, float]:
        return empirical_var_es(st.z, alpha)

    def _std_draws(self, st: _GarchState, rng: np.random.Generator, size: int) -> Arr:
        return st.z[rng.integers(0, len(st.z), size)]


@dataclass(frozen=True)
class GPDTail:
    u: float  # threshold on the loss scale (-z)
    xi: float
    beta: float
    k: int
    n: int
    method: str
    capped: bool


def _gpd_pwm(y: Arr) -> tuple[float, float]:
    """Hosking-Wallis (1987) probability-weighted moments estimates (xi, beta)."""
    ys = np.sort(y)
    n = len(ys)
    b0 = ys.mean()
    b1 = np.sum((np.arange(n)) / (n - 1) * ys) / n
    # a0 = b0, a1 = E[Y (1 - F)] = b0 - b1
    a0, a1 = b0, b0 - b1
    xi = 2 - a0 / (a0 - 2 * a1)
    beta = 2 * a0 * a1 / (a0 - 2 * a1)
    # Hosking-Wallis use k = -xi; convert to the scipy/McNeil sign convention
    return float(xi), float(beta)


def fit_gpd_tail(z: Arr, tail_fraction: float) -> GPDTail:
    """GPD on the k largest losses ``-z`` above the (k+1)-th largest (McNeil-Frey 2000)."""
    loss = np.sort(-z)[::-1]
    n = len(loss)
    k = max(10, int(round(tail_fraction * n)))
    u = float(loss[k])
    y = loss[:k] - u
    method = "mle"
    xi = beta = math.nan
    try:
        c, _, scale = stats.genpareto.fit(y, floc=0)
        xi, beta = float(c), float(scale)
    except Exception:  # noqa: BLE001 - scipy raises several types
        method = "pwm"
    if (
        method == "pwm"
        or not (math.isfinite(xi) and math.isfinite(beta))
        or xi <= -0.5
        or beta <= 0
    ):
        xi, beta = _gpd_pwm(y)
        method = "pwm"
    capped = False
    if xi >= 0.95:
        xi, capped = 0.95, True
    return GPDTail(u, xi, beta, k, n, method, capped)


def gpd_var_es(t: GPDTail, alpha: float) -> tuple[float, float]:
    """(VaR, ES) on the loss scale (positive numbers) at tail probability ``alpha``."""
    ratio = t.n * alpha / t.k
    if abs(t.xi) < 1e-6:
        var = t.u - t.beta * math.log(ratio)
        return var, var + t.beta
    var = t.u + t.beta / t.xi * (ratio ** (-t.xi) - 1)
    es = var / (1 - t.xi) + (t.beta - t.xi * t.u) / (1 - t.xi)
    return var, es


class EVTPot(_GarchBase):
    """McNeil-Frey (2000): GPD fitted to the worst residuals of a GJR-GARCH(1,1) QMLE fit."""

    name = "evt-pot"
    dist = "normal"

    def __init__(self, tail_fraction: float = 0.10) -> None:
        self.tail_fraction = tail_fraction

    def fit(self, window: Arr, alphas: Sequence[float], rng: np.random.Generator) -> Any:
        st = self._fit_common(window)
        tail = fit_gpd_tail(st.z, self.tail_fraction)
        out = {}
        for a in alphas:
            if a < tail.k / tail.n:
                var, es = gpd_var_es(tail, a)
                out[a] = (-var, -es)
            else:  # level inside the body: fall back to the empirical residual tail
                out[a] = empirical_var_es(st.z, a)
        st.extra = out
        st.extra_tail = tail  # type: ignore[attr-defined]
        return st

    def _std_draws(self, st: _GarchState, rng: np.random.Generator, size: int) -> Arr:
        tail: GPDTail = st.extra_tail  # type: ignore[attr-defined]
        body = st.z[-st.z <= tail.u]
        out = body[rng.integers(0, len(body), size)]
        in_tail = rng.random(size) < tail.k / tail.n
        m = int(in_tail.sum())
        if m:
            exc = stats.genpareto.rvs(tail.xi, scale=tail.beta, size=m, random_state=rng)
            out[in_tail] = -(tail.u + exc)
        return out
