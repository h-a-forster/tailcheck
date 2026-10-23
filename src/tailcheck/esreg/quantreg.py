"""Exact linear quantile regression by linear programming (scipy HiGHS)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, sparse, stats


def _prep(y: ArrayLike, x: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    ya = np.asarray(y, dtype=np.float64)
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim == 1:
        xa = xa[:, None]
    if ya.ndim != 1 or xa.shape[0] != ya.shape[0]:
        raise ValueError("y must be 1-D and x must have len(y) rows")
    if not (np.all(np.isfinite(ya)) and np.all(np.isfinite(xa))):
        raise ValueError("y and x must be finite")
    return ya, xa


def quantile_regression(y: ArrayLike, x: ArrayLike, tau: float) -> NDArray[np.float64]:
    """Koenker-Bassett (1978) estimate of ``beta`` minimising the check loss.

    ``x`` must already contain an intercept column if one is wanted. Solved exactly as the
    linear programme ``min tau 1'u+ + (1 - tau) 1'u-`` s.t. ``X b + u+ - u- = y``.
    """
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    ya, xa = _prep(y, x)
    n, k = xa.shape
    if k == 1 and np.all(xa[:, 0] == xa[0, 0]) and xa[0, 0] != 0:
        # Intercept only: an LP vertex solution is the order statistic y_(ceil(n tau)).
        idx = max(math.ceil(n * tau - 1e-12), 1) - 1
        return np.array([np.partition(ya, idx)[idx] / xa[0, 0]])
    c = np.concatenate([np.zeros(k), np.full(n, tau), np.full(n, 1.0 - tau)])
    eye = sparse.identity(n, format="csr")
    a_eq = sparse.hstack([sparse.csr_matrix(xa), eye, -eye], format="csr")
    bounds = [(None, None)] * k + [(0.0, None)] * (2 * n)
    res = optimize.linprog(c, A_eq=a_eq, b_eq=ya, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"quantile regression LP failed: {res.message}")
    return np.asarray(res.x[:k], dtype=np.float64)


def hall_sheather(n: int, tau: float, level: float = 0.05) -> float:
    """Hall-Sheather (1988) bandwidth for sparsity estimation at quantile ``tau``."""
    z = stats.norm.ppf(tau)
    core = 1.5 * stats.norm.pdf(z) ** 2 / (2 * z**2 + 1)
    return float(n ** (-1 / 3) * stats.norm.ppf(1 - level / 2) ** (2 / 3) * core ** (1 / 3))


def bofinger(n: int, tau: float) -> float:
    """Bofinger (1975) bandwidth."""
    z = stats.norm.ppf(tau)
    return float(n ** (-1 / 5) * (4.5 * stats.norm.pdf(z) ** 4 / (2 * z**2 + 1) ** 2) ** (1 / 5))


def chamberlain(n: int, tau: float, level: float = 0.05) -> float:
    """Chamberlain (1994) bandwidth."""
    return float(stats.norm.ppf(1 - level / 2) * math.sqrt(tau * (1 - tau) / n))


def quantile_regression_se_iid(
    y: ArrayLike, x: ArrayLike, tau: float, beta: ArrayLike
) -> NDArray[np.float64]:
    """IID-error standard errors (Koenker 2005, sec. 3.4), used only to scale perturbations.

    Sparsity ``s = (Q_u(tau + h) - Q_u(tau - h)) / (2h)`` from the empirical quantiles of the
    residuals with the Hall-Sheather bandwidth; ``Cov = tau (1 - tau) s^2 (X'X)^{-1}``.
    """
    ya, xa = _prep(y, x)
    n = len(ya)
    u = ya - xa @ np.asarray(beta, dtype=np.float64)
    h = hall_sheather(n, tau)
    lo, hi = max(tau - h, 1.0 / n), min(tau + h, 1.0 - 1.0 / n)
    s = (np.quantile(u, hi) - np.quantile(u, lo)) / (hi - lo)
    xtx_inv = np.linalg.pinv(xa.T @ xa)
    var = tau * (1 - tau) * s**2 * np.diag(xtx_inv)
    return np.sqrt(np.maximum(var, 0.0))
