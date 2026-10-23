"""M-estimator for the joint linear quantile / ES regression with iterated local search.

Specification: ``docs/DESIGN.md`` section 4.3.2 (Dimitriadis and Bayer 2019, sec. 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats

from tailcheck._validation import check_alpha
from tailcheck.esreg.gfuns import G1, G2, NEGATIVE_DOMAIN
from tailcheck.esreg.loss import esr_loss_raw
from tailcheck.esreg.quantreg import quantile_regression, quantile_regression_se_iid

_NM_OPTIONS = {"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-10}


@dataclass(frozen=True)
class ESRegFit:
    """Fitted joint regression. Coefficients are on the original ``y`` scale.

    ``xq``/``xe`` include the intercept column. ``y_shift`` is the constant subtracted from
    ``y`` before optimisation (``max(y)`` for G2 types 1-3, else 0); ``loss`` is the mean loss
    at the optimum on that shifted scale.
    """

    coef_q: NDArray[np.float64]
    coef_e: NDArray[np.float64]
    loss: float
    alpha: float
    g1: G1
    g2: G2
    y: NDArray[np.float64]
    xq: NDArray[np.float64]
    xe: NDArray[np.float64]
    n_local_searches: int
    y_shift: float = 0.0
    converged: bool = True

    @property
    def coef(self) -> NDArray[np.float64]:
        return np.concatenate([self.coef_q, self.coef_e])

    @property
    def n(self) -> int:
        return len(self.y)

    def fitted(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Fitted (quantile, ES) on the original scale."""
        return self.xq @ self.coef_q, self.xe @ self.coef_e


def alpha_tilde(alpha: float) -> float:
    """Level whose normal quantile equals the normal ES at ``alpha`` (starting values)."""
    z = stats.norm.ppf(alpha)
    return float(stats.norm.cdf(-stats.norm.pdf(z) / alpha))


def design(x: ArrayLike | None, n: int, name: str) -> NDArray[np.float64]:
    """Prepend an intercept to covariates ``x`` (``None`` means intercept only)."""
    if x is None:
        return np.ones((n, 1))
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim == 1:
        xa = xa[:, None]
    if xa.ndim != 2 or xa.shape[0] != n:
        raise ValueError(f"{name} must have {n} rows")
    if not np.all(np.isfinite(xa)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return np.column_stack([np.ones(n), xa])


class _Objective:
    def __init__(
        self,
        y: NDArray[np.float64],
        xq: NDArray[np.float64],
        xe: NDArray[np.float64],
        alpha: float,
        g1: G1,
        g2: G2,
    ) -> None:
        self.y, self.xq, self.xe, self.alpha, self.g1, self.g2 = y, xq, xe, alpha, g1, g2
        self.kq = xq.shape[1]
        self.neg = g2 in NEGATIVE_DOMAIN

    def __call__(self, b: NDArray[np.float64]) -> float:
        q = self.xq @ b[: self.kq]
        e = self.xe @ b[self.kq :]
        if self.neg and np.any(e >= -0.01):
            return np.inf
        val = float(np.mean(esr_loss_raw(self.y, q, e, self.alpha, self.g1, self.g2)))
        return val if np.isfinite(val) else np.inf


def _local_search(obj: _Objective, b0: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    res = optimize.minimize(obj, b0, method="Nelder-Mead", options=_NM_OPTIONS)
    return np.asarray(res.x, dtype=np.float64), float(res.fun)


def _fix_es_intercept(b: NDArray[np.float64], xe: NDArray[np.float64], kq: int, thr: float) -> None:
    m = float(np.max(xe @ b[kq:]))
    if m >= thr:
        b[kq] -= m + 0.1


def esreg(
    y: ArrayLike,
    xq: ArrayLike | None,
    xe: ArrayLike | None,
    alpha: float,
    g1: G1 = G1.ZERO,
    g2: G2 = G2.NEG_INV,
    early_stopping: int = 10,
    seed: int | None = None,
) -> ESRegFit:
    """Fit ``Q_alpha(y|xq) = xq'bq`` and ``ES_alpha(y|xe) = xe'be`` jointly.

    ``xq``/``xe`` are covariates without intercept (an intercept is always added);
    ``None`` means intercept only. Estimation follows DESIGN 4.3.2: quantile-regression
    starting values, Nelder-Mead, then an iterated local search that stops after
    ``early_stopping`` consecutive non-improving perturbations.
    """
    alpha = check_alpha(alpha)
    g1, g2 = G1(g1), G2(g2)
    ya = np.asarray(y, dtype=np.float64)
    if ya.ndim != 1 or ya.size == 0 or not np.all(np.isfinite(ya)):
        raise ValueError("y must be a finite, non-empty 1-D array")
    n = len(ya)
    xq_m = design(xq, n, "xq")
    xe_m = design(xe, n, "xe")
    kq, ke = xq_m.shape[1], xe_m.shape[1]
    if n * alpha < kq + ke + 1:
        raise ValueError(
            f"too few expected tail observations: n*alpha = {n * alpha:.1f} < {kq + ke + 1}"
        )
    if early_stopping < 0:
        raise ValueError("early_stopping must be >= 0")
    neg = g2 in NEGATIVE_DOMAIN
    shift = float(np.max(ya)) if neg else 0.0
    ys = ya - shift

    bq0 = quantile_regression(ys, xq_m, alpha)
    at = alpha_tilde(alpha)
    be0 = quantile_regression(ys, xe_m, at)
    b0 = np.concatenate([bq0, be0])
    if neg:
        _fix_es_intercept(b0, xe_m, kq, -0.1)
    se = np.concatenate(
        [
            quantile_regression_se_iid(ys, xq_m, alpha, bq0),
            quantile_regression_se_iid(ys, xe_m, at, be0),
        ]
    )
    se = np.where(np.isfinite(se) & (se > 0), se, 1e-3)

    obj = _Objective(ys, xq_m, xe_m, alpha, g1, g2)
    best_b, best_f = _local_search(obj, b0)
    n_searches = 1
    if early_stopping > 0:
        rng = np.random.default_rng(seed)
        fails = 0
        while fails < early_stopping:
            cand = best_b + rng.standard_normal(len(best_b)) * se
            if neg:
                _fix_es_intercept(cand, xe_m, kq, 0.0)
            b, f = _local_search(obj, cand)
            n_searches += 1
            if f < best_f - 1e-12:
                best_b, best_f = b, f
                fails = 0
            else:
                fails += 1
    coef_q = best_b[:kq].copy()
    coef_e = best_b[kq:].copy()
    coef_q[0] += shift
    coef_e[0] += shift
    return ESRegFit(
        coef_q=coef_q,
        coef_e=coef_e,
        loss=best_f,
        alpha=alpha,
        g1=g1,
        g2=g2,
        y=ya,
        xq=xq_m,
        xe=xe_m,
        n_local_searches=n_searches,
        y_shift=shift,
        converged=bool(np.isfinite(best_f)),
    )
