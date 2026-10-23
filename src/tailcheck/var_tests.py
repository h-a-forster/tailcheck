"""VaR backtests: Kupiec POF, Christoffersen independence and conditional coverage,
Engle-Manganelli dynamic quantile, Basel traffic light.

Formulas, edge cases and reference values: ``docs/DESIGN.md`` section 4.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from tailcheck._validation import as_float_1d, check_alpha, check_same_length
from tailcheck.results import BacktestResult

Zone = Literal["green", "yellow", "red"]


def _xlogy(x: float, y: float) -> float:
    """``x * log(y)`` with the convention ``0 * log(0) = 0``."""
    return 0.0 if x == 0 else x * math.log(y)


def hits(r: ArrayLike, q: ArrayLike) -> NDArray[np.bool_]:
    """VaR exceedance indicator ``r <= q`` (validated, same length)."""
    ra = as_float_1d(r, "r")
    qa = as_float_1d(q, "q")
    check_same_length([ra, qa], ["r", "q"])
    return ra <= qa


def _lr_pof(x: int, t: int, alpha: float) -> float:
    pi = x / t
    ll0 = _xlogy(t - x, 1 - alpha) + _xlogy(x, alpha)
    ll1 = _xlogy(t - x, 1 - pi) + _xlogy(x, pi)
    return max(0.0, -2.0 * (ll0 - ll1))


def kupiec_pof(r: ArrayLike, q: ArrayLike, alpha: float) -> BacktestResult:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test, chi2(1)."""
    alpha = check_alpha(alpha)
    h = hits(r, q)
    t, x = len(h), int(h.sum())
    lr = _lr_pof(x, t, alpha)
    return BacktestResult(
        test="kupiec_pof",
        statistic=lr,
        pvalue=float(stats.chi2.sf(lr, 1)),
        nobs=t,
        df=1,
        details={"hits": x, "hit_rate": x / t, "expected_hits": alpha * t},
    )


def _transition_counts(h: NDArray[np.bool_]) -> tuple[int, int, int, int]:
    a, b = h[:-1], h[1:]
    n00 = int(np.sum(~a & ~b))
    n01 = int(np.sum(~a & b))
    n10 = int(np.sum(a & ~b))
    n11 = int(np.sum(a & b))
    return n00, n01, n10, n11


def _lr_ind(h: NDArray[np.bool_]) -> tuple[float, tuple[int, int, int, int]]:
    n00, n01, n10, n11 = _transition_counts(h)
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (len(h) - 1)
    ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll1 = _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    return max(0.0, -2.0 * (ll0 - ll1)), (n00, n01, n10, n11)


def christoffersen_independence(r: ArrayLike, q: ArrayLike, alpha: float) -> BacktestResult:
    """Christoffersen (1998) first-order Markov independence LR test, chi2(1).

    ``alpha`` is validated but does not enter the statistic.
    """
    check_alpha(alpha)
    h = hits(r, q)
    if len(h) < 2:
        raise ValueError("need at least 2 observations")
    lr, counts = _lr_ind(h)
    details: dict[str, object] = {
        "n00": counts[0],
        "n01": counts[1],
        "n10": counts[2],
        "n11": counts[3],
        "hits": int(h.sum()),
    }
    if counts[1] + counts[3] == 0:
        details["reason"] = "no exceedances"
    return BacktestResult(
        test="christoffersen_ind",
        statistic=lr,
        pvalue=float(stats.chi2.sf(lr, 1)),
        nobs=len(h),
        df=1,
        details=details,
    )


def christoffersen_cc(r: ArrayLike, q: ArrayLike, alpha: float) -> BacktestResult:
    """Christoffersen (1998) conditional coverage test: LR_pof + LR_ind, chi2(2)."""
    alpha = check_alpha(alpha)
    h = hits(r, q)
    if len(h) < 2:
        raise ValueError("need at least 2 observations")
    lr_pof = _lr_pof(int(h.sum()), len(h), alpha)
    lr_ind, counts = _lr_ind(h)
    lr = lr_pof + lr_ind
    return BacktestResult(
        test="christoffersen_cc",
        statistic=lr,
        pvalue=float(stats.chi2.sf(lr, 2)),
        nobs=len(h),
        df=2,
        details={"lr_pof": lr_pof, "lr_ind": lr_ind, "hits": int(h.sum())},
    )


def dynamic_quantile(
    r: ArrayLike, q: ArrayLike, alpha: float, lags: int = 4, include_var: bool = True
) -> BacktestResult:
    """Engle and Manganelli (2004) dynamic quantile Wald test, chi2(rank of regressors).

    The statistic is ``Hit' X (X'X)^+ X' Hit / (alpha (1 - alpha))`` with
    ``X_t = [1, Hit_{t-1}, ..., Hit_{t-lags}, q_t]``. A pseudo-inverse is used, so collinear
    regressors (for example no hits at all) reduce the degrees of freedom to ``rank(X)``.
    """
    alpha = check_alpha(alpha)
    if lags < 0:
        raise ValueError("lags must be >= 0")
    qa = as_float_1d(q, "q")
    h = hits(r, qa).astype(np.float64) - alpha
    t = len(h)
    if t <= lags + 2:
        raise ValueError(f"need more than {lags + 2} observations")
    y = h[lags:]
    cols = [np.ones(t - lags)]
    cols += [h[lags - j : t - j] for j in range(1, lags + 1)]
    if include_var:
        cols.append(qa[lags:])
    x = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    rank = int(np.linalg.matrix_rank(x))
    dq = float(fitted @ fitted / (alpha * (1 - alpha)))
    details: dict[str, object] = {"lags": lags, "include_var": include_var, "rank": rank}
    if rank < x.shape[1]:
        details["warning"] = f"regressors are collinear (rank {rank} < {x.shape[1]})"
    return BacktestResult(
        test="dynamic_quantile",
        statistic=dq,
        pvalue=float(stats.chi2.sf(dq, rank)),
        nobs=len(y),
        df=rank,
        details=details,
    )


@dataclass(frozen=True)
class TrafficLight:
    """Basel (1996) traffic-light classification of a VaR exception count."""

    zone: Zone
    exceptions: int
    nobs: int
    cumulative_probability: float
    plus_factor: float


_PLUS_FACTOR = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


def traffic_light(
    r: ArrayLike,
    q: ArrayLike,
    alpha: float = 0.01,
    green_below: float = 0.95,
    red_from: float = 0.9999,
) -> TrafficLight:
    """Classify the exception count by its binomial cumulative probability.

    The plus factor follows BCBS (1996) Table 2 and is only defined for 250 observations at
    ``alpha = 0.01``; otherwise it is ``nan``.
    """
    alpha = check_alpha(alpha)
    h = hits(r, q)
    t, x = len(h), int(h.sum())
    c = float(stats.binom.cdf(x, t, alpha))
    zone: Zone = "green" if c < green_below else ("yellow" if c < red_from else "red")
    if t == 250 and alpha == 0.01:
        plus = 0.0 if zone == "green" else (1.0 if zone == "red" else _PLUS_FACTOR.get(x, math.nan))
    else:
        plus = math.nan
    return TrafficLight(zone, x, t, c, plus)
