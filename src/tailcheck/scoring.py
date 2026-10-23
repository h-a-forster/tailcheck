"""Forecast comparison: FZ0 loss, Diebold-Mariano test, model confidence set (via ``arch``).

Formulas: ``docs/DESIGN.md`` section 4.4.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from tailcheck._validation import as_float_1d, check_alpha, check_same_length
from tailcheck.results import BacktestResult


def fz0_loss(r: ArrayLike, q: ArrayLike, e: ArrayLike, alpha: float) -> NDArray[np.float64]:
    """Patton-Ziegel-Chen (2019) FZ0 loss per observation. Requires e <= q < 0."""
    alpha = check_alpha(alpha)
    ra, qa, ea = as_float_1d(r, "r"), as_float_1d(q, "q"), as_float_1d(e, "e")
    check_same_length([ra, qa, ea], ["r", "q", "e"])
    if np.any(qa >= 0) or np.any(ea > qa):
        raise ValueError("FZ0 requires e <= q < 0")
    hit = (ra <= qa).astype(np.float64)
    return -hit * (qa - ra) / (alpha * ea) + qa / ea + np.log(-ea) - 1.0


def newey_west_lrv(d: NDArray[np.float64], lags: int) -> float:
    """Bartlett-kernel long-run variance of ``d`` (demeaned)."""
    u = d - d.mean()
    t = len(u)
    lrv = float(u @ u) / t
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1)
        lrv += 2 * w * float(u[k:] @ u[:-k]) / t
    return lrv


def default_lags(t: int) -> int:
    return int(math.floor(4 * (t / 100) ** (2 / 9)))


def mean_with_ci(x: ArrayLike, lags: int | None = None, level: float = 0.95) -> dict[str, float]:
    """Mean with a Newey-West confidence interval."""
    xa = as_float_1d(x, "x")
    t = len(xa)
    lags = default_lags(t) if lags is None else lags
    se = math.sqrt(max(newey_west_lrv(xa, lags), 0.0) / t)
    z = float(stats.norm.ppf(0.5 + level / 2))
    m = float(xa.mean())
    return {"mean": m, "se": se, "lo": m - z * se, "hi": m + z * se}


def diebold_mariano(
    loss_a: ArrayLike,
    loss_b: ArrayLike,
    lags: int | None = None,
    small_sample: bool = True,
) -> BacktestResult:
    """DM test of equal expected loss with a Newey-West long-run variance.

    Negative statistic: model ``a`` has lower loss. ``small_sample`` applies the
    Harvey-Leybourne-Newbold (1997) correction and Student-t(T-1) reference.
    """
    la, lb = as_float_1d(loss_a, "loss_a"), as_float_1d(loss_b, "loss_b")
    t = check_same_length([la, lb], ["loss_a", "loss_b"])
    if t < 3:
        raise ValueError("need at least 3 observations")
    d = la - lb
    lags = default_lags(t) if lags is None else int(lags)
    if lags < 0:
        raise ValueError("lags must be >= 0")
    lrv = newey_west_lrv(d, lags)
    if not lrv > 0:
        return BacktestResult(
            "diebold_mariano",
            math.nan,
            math.nan,
            t,
            None,
            "two-sided",
            {"reason": "non-positive long-run variance", "mean_diff": float(d.mean())},
        )
    dm = float(d.mean()) / math.sqrt(lrv / t)
    if small_sample:
        h = 1
        dm *= math.sqrt((t + 1 - 2 * h + h * (h - 1) / t) / t)
        dist = stats.t(t - 1)
    else:
        dist = stats.norm
    return BacktestResult(
        "diebold_mariano",
        dm,
        float(2 * dist.sf(abs(dm))),
        t,
        None,
        "two-sided",
        {
            "mean_diff": float(d.mean()),
            "lags": lags,
            "pvalue_less": float(dist.cdf(dm)),  # H1: a has lower loss
            "pvalue_greater": float(dist.sf(dm)),
            "small_sample": small_sample,
        },
    )


@dataclass(frozen=True)
class MCSResult:
    """Model confidence set at level ``1 - size``."""

    included: tuple[str, ...]
    excluded: tuple[str, ...]
    pvalues: Mapping[str, float]
    size: float
    method: str
    reps: int
    block_size: int
    seed: int | None


def model_confidence_set(
    losses: pd.DataFrame,
    size: float = 0.10,
    reps: int = 10_000,
    block_size: int | None = None,
    method: Literal["R", "max"] = "R",
    seed: int | None = None,
) -> MCSResult:
    """Hansen-Lunde-Nason (2011) MCS on a T x M loss frame, wrapping ``arch.bootstrap.MCS``.

    Stationary bootstrap; ``block_size`` defaults to ``ceil(sqrt(T))``.
    """
    from arch.bootstrap import MCS

    if not isinstance(losses, pd.DataFrame) or losses.shape[1] < 2:
        raise ValueError("losses must be a DataFrame with at least 2 model columns")
    if not np.all(np.isfinite(losses.to_numpy(dtype=float))):
        raise ValueError("losses contain NaN or infinite values")
    if not 0 < size < 1:
        raise ValueError("size must be in (0, 1)")
    bs = int(math.ceil(math.sqrt(len(losses)))) if block_size is None else int(block_size)
    mcs = MCS(
        losses.astype(float),
        size=size,
        reps=reps,
        block_size=bs,
        method=method,
        bootstrap="stationary",
        seed=seed,
    )
    mcs.compute()
    pv = mcs.pvalues.iloc[:, 0]
    return MCSResult(
        included=tuple(str(c) for c in mcs.included),
        excluded=tuple(str(c) for c in mcs.excluded),
        pvalues={str(k): float(v) for k, v in pv.items()},
        size=size,
        method=method,
        reps=reps,
        block_size=bs,
        seed=seed,
    )
