"""Bayer-Dimitriadis (2022) ES regression backtests.

version 1 "strict ESR":       r ~ e | e,        H0: ES coefficients (0, 1), Wald chi2(2)
version 2 "auxiliary ESR":    r ~ q | e,        H0: ES coefficients (0, 1), Wald chi2(2)
version 3 "strict intercept": (r - e) ~ e | 1,  H0: ES intercept 0, t-test (one- and two-sided)

Specification: ``docs/DESIGN.md`` section 4.3.4.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from tailcheck._validation import as_float_1d, check_alpha, check_same_length
from tailcheck.esreg.covariance import SigmaEstimator, Sparsity, vcov_details
from tailcheck.esreg.estimator import ESRegFit, esreg
from tailcheck.esreg.gfuns import G1, G2
from tailcheck.results import BacktestResult

MIN_TAIL_OBS = 3


def _setup(
    r: NDArray[np.float64], e: NDArray[np.float64], q: NDArray[np.float64] | None, version: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """(response, quantile covariate, ES covariate or None for intercept only)."""
    if version == 1:
        return r, e, e
    if version == 2:
        if q is None:
            raise ValueError("version 2 needs the VaR forecasts q")
        return r, q, e
    return r - e, e, None


def _fit(
    y: NDArray[np.float64],
    xq: NDArray[np.float64],
    xe: NDArray[np.float64] | None,
    alpha: float,
    early_stopping: int,
    seed: int | None,
) -> ESRegFit:
    return esreg(y, xq, xe, alpha, G1.ZERO, G2.NEG_INV, early_stopping=early_stopping, seed=seed)


def _stat(version: int, coef_e: NDArray[np.float64], cov_e: NDArray[np.float64]) -> float:
    if version in (1, 2):
        d = coef_e - np.array([0.0, 1.0])
        return float(d @ np.linalg.solve(cov_e, d))
    return float(coef_e[0] / math.sqrt(cov_e[0, 0]))


def esr_backtest(
    r: ArrayLike,
    e: ArrayLike,
    alpha: float,
    version: Literal[1, 2, 3] = 1,
    q: ArrayLike | None = None,
    n_boot: int = 0,
    sparsity: Sparsity = "nid",
    sigma_est: SigmaEstimator = "scl_sp",
    misspec: bool = True,
    seed: int | None = None,
    early_stopping: int = 10,
    _esreg_compat_g2pp: bool = False,
    _also_esback_g2pp: bool = False,
    _keep_boot_stats: bool = False,
) -> BacktestResult:
    """Run one ESR backtest; asymptotic p-value, plus bootstrap p-value if ``n_boot > 0``.

    Primary p-value: two-sided Wald chi2(2) for versions 1 and 2; one-sided ("less") for
    version 3, where a negative intercept means risk is underestimated. Too few negative
    quantile residuals, a failed fit or a singular covariance give ``pvalue = nan`` with
    ``details["reason"]``. ``_keep_boot_stats`` stores the centred bootstrap statistics in
    ``details["bootstrap_stats"]`` (used by the warp-speed Monte Carlo of the bootstrap test).
    """
    alpha = check_alpha(alpha)
    if version not in (1, 2, 3):
        raise ValueError("version must be 1, 2 or 3")
    ra, ea = as_float_1d(r, "r"), as_float_1d(e, "e")
    qa = as_float_1d(q, "q") if q is not None else None
    check_same_length(
        [ra, ea] + ([qa] if qa is not None else []), ["r", "e", "q"][: 2 + (qa is not None)]
    )
    if n_boot < 0:
        raise ValueError("n_boot must be >= 0")
    name = f"esr_v{version}"
    alt: Literal["two-sided", "less"] = "less" if version == 3 else "two-sided"
    df = 2 if version in (1, 2) else None
    n = len(ra)
    y, xq, xe = _setup(ra, ea, qa, version)

    def undefined(reason: str, extra: dict[str, Any] | None = None) -> BacktestResult:
        return BacktestResult(
            name, math.nan, math.nan, n, df, alt, {"reason": reason, **(extra or {})}
        )

    kq, ke = 2, (2 if xe is not None else 1)
    if n * alpha < kq + ke + 1:
        return undefined(f"too few expected exceedances (n*alpha = {n * alpha:.1f})")
    try:
        fit = _fit(y, xq, xe, alpha, early_stopping, seed)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return undefined(f"fit failed: {exc}")
    xi, _ = fit.fitted()
    n_neg = int(np.sum(y - xi <= 0))
    if n_neg < MIN_TAIL_OBS:
        return undefined(f"fewer than {MIN_TAIL_OBS} negative quantile residuals ({n_neg})")
    try:
        cov, info = vcov_details(
            fit, sparsity, sigma_est, misspec, _esreg_compat_g2pp=_esreg_compat_g2pp
        )
        cov_e = cov[kq:, kq:]
        if np.linalg.eigvalsh(cov_e).min() <= 0:
            return undefined("ES covariance not positive definite", {"coef": fit.coef.tolist()})
        stat = _stat(version, fit.coef_e, cov_e)
    except (ValueError, RuntimeError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
        return undefined(f"covariance failed: {exc}", {"coef": fit.coef.tolist()})
    if not math.isfinite(stat):
        return undefined("non-finite statistic", {"coef": fit.coef.tolist()})
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    details: dict[str, Any] = {
        "coef_q": fit.coef_q.tolist(),
        "coef_e": fit.coef_e.tolist(),
        "se_q": se[:kq].tolist(),
        "se_e": se[kq:].tolist(),
        "n_tail": n_neg,
        "n_local_searches": fit.n_local_searches,
        "sparsity": info["sparsity"],
        "sigma_est": info["sigma_est"],
        "misspec": misspec,
    }
    if info["warnings"]:
        details["warnings"] = list(info["warnings"])
    if version in (1, 2):
        p = float(stats.chi2.sf(stat, 2))
        details["pvalue_two_sided"] = p
    else:
        p_less = float(stats.norm.cdf(stat))
        p_two = float(2 * stats.norm.sf(abs(stat)))
        details["pvalue_less"] = p_less
        details["pvalue_two_sided"] = p_two
        p = p_less

    if _also_esback_g2pp and not _esreg_compat_g2pp:
        # Same fit, covariance with esreg 0.6.2's G2'' (docs/esreg-covariance.md).
        try:
            cov_c, _ = vcov_details(fit, sparsity, sigma_est, misspec, _esreg_compat_g2pp=True)
            ce = cov_c[kq:, kq:]
            if np.linalg.eigvalsh(ce).min() <= 0:
                raise ValueError("not positive definite")
            st = _stat(version, fit.coef_e, ce)
            details["pvalue_esback_g2pp"] = (
                float(stats.chi2.sf(st, 2)) if version in (1, 2) else float(stats.norm.cdf(st))
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            details["pvalue_esback_g2pp"] = math.nan

    if n_boot > 0:
        details.update(
            _bootstrap(
                version,
                y,
                xq,
                xe,
                alpha,
                fit,
                stat,
                n_boot,
                seed,
                sparsity,
                sigma_est,
                misspec,
                keep_stats=_keep_boot_stats,
            )
        )
    return BacktestResult(name, stat, p, n, df, alt, details)


def _bootstrap(
    version: int,
    y: NDArray[np.float64],
    xq: NDArray[np.float64],
    xe: NDArray[np.float64] | None,
    alpha: float,
    fit: ESRegFit,
    stat0: float,
    n_boot: int,
    seed: int | None,
    sparsity: Sparsity,
    sigma_est: SigmaEstimator,
    misspec: bool,
    keep_stats: bool = False,
) -> dict[str, Any]:
    """iid row bootstrap; statistic centred at the full-sample estimate (DESIGN 4.3.4)."""
    rng = np.random.default_rng(None if seed is None else seed + 7919)
    n = len(y)
    kq = 2
    stats_b: list[float] = []
    failed = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            fb = _fit(
                y[idx],
                xq[idx],
                None if xe is None else xe[idx],
                alpha,
                0,
                int(rng.integers(1 << 31)),
            )
            cov_b, _ = vcov_details(fb, sparsity, sigma_est, misspec)
            d = fb.coef_e - fit.coef_e
            ce = cov_b[kq:, kq:]
            if np.linalg.eigvalsh(ce).min() <= 0:
                raise ValueError("bootstrap covariance not positive definite")
            t = (
                float(d @ np.linalg.solve(ce, d))
                if version in (1, 2)
                else float(d[0] / math.sqrt(ce[0, 0]))
            )
            if not math.isfinite(t):
                raise ValueError("non-finite bootstrap statistic")
            stats_b.append(t)
        except (ValueError, RuntimeError, np.linalg.LinAlgError, ZeroDivisionError):
            failed += 1
    out: dict[str, Any] = {"n_boot": n_boot, "n_boot_failed": failed}
    if keep_stats:
        out["bootstrap_stats"] = list(stats_b)
    if failed >= 0.05 * n_boot:
        out["bootstrap_reason"] = f"{failed} of {n_boot} bootstrap fits failed"
        out["pvalue_bootstrap"] = math.nan
        return out
    tb = np.asarray(stats_b)
    b = len(tb)
    if version in (1, 2):
        out["pvalue_bootstrap"] = (1 + int(np.sum(tb >= stat0))) / (b + 1)
    else:
        out["pvalue_bootstrap"] = (1 + int(np.sum(tb <= stat0))) / (b + 1)
        out["pvalue_bootstrap_two_sided"] = (1 + int(np.sum(np.abs(tb) >= abs(stat0)))) / (b + 1)
    return out
