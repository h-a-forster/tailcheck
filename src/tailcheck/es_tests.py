"""ES backtests: Acerbi-Szekely Z1 and Z2 with simulated null distributions, and the
McNeil-Frey exceedance-residual bootstrap test.

Formulas, edge cases and reference values: ``docs/DESIGN.md`` section 4.2.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from tailcheck._validation import as_float_1d, check_alpha, check_same_length
from tailcheck.results import BacktestResult

# simulate(rng, n_paths) -> array of shape (n_paths, T): return paths drawn from the
# forecast distributions F_1..F_T that produced (q, e).
Simulator = Callable[[np.random.Generator, int], NDArray[np.float64]]


def _rqe(
    r: ArrayLike, q: ArrayLike, e: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    ra, qa, ea = as_float_1d(r, "r"), as_float_1d(q, "q"), as_float_1d(e, "e")
    check_same_length([ra, qa, ea], ["r", "q", "e"])
    if np.any(ea >= 0):
        raise ValueError("ES forecasts must be negative (return convention, left tail)")
    if np.any(ea > qa + 1e-12 * np.abs(qa)):
        raise ValueError("ES must not exceed VaR (e <= q)")
    return ra, qa, ea


def acerbi_szekely_z1(r: ArrayLike, q: ArrayLike, e: ArrayLike) -> float:
    """Z1 = 1 - mean over hits of r_t / e_t. ``nan`` if there are no hits."""
    ra, qa, ea = _rqe(r, q, e)
    h = ra <= qa
    if not h.any():
        return math.nan
    return float(1.0 - np.mean(ra[h] / ea[h]))


def acerbi_szekely_z2(r: ArrayLike, q: ArrayLike, e: ArrayLike, alpha: float) -> float:
    """Z2 = 1 - sum_t r_t 1{r_t <= q_t} / (T alpha e_t)."""
    alpha = check_alpha(alpha)
    ra, qa, ea = _rqe(r, q, e)
    h = ra <= qa
    return float(1.0 - np.sum(np.where(h, ra / ea, 0.0)) / (len(ra) * alpha))


def _z_paths(
    x: NDArray[np.float64],
    q: NDArray[np.float64],
    e: NDArray[np.float64],
    alpha: float,
    statistic: str,
) -> tuple[NDArray[np.float64], int]:
    """Statistic for each simulated path (rows of ``x``); returns (values, n_undefined)."""
    h = x <= q
    ratio = np.where(h, x / e, 0.0)
    if statistic == "z2":
        return 1.0 - ratio.sum(axis=1) / (x.shape[1] * alpha), 0
    n = h.sum(axis=1)
    ok = n > 0
    z = 1.0 - ratio[ok].sum(axis=1) / n[ok]
    return z, int((~ok).sum())


def acerbi_szekely_test(
    r: ArrayLike,
    q: ArrayLike,
    e: ArrayLike,
    alpha: float,
    simulate: Simulator,
    statistic: Literal["z1", "z2"] = "z2",
    n_sim: int = 5000,
    seed: int | None = None,
    chunk: int = 500,
) -> BacktestResult:
    """One-sided simulation test; small Z rejects (risk underestimated).

    ``simulate(rng, k)`` must return ``k`` paths of length T from the forecast distributions.
    p-value: ``(1 + #{Z_sim <= Z_obs}) / (M + 1)`` over the M defined simulated statistics.
    """
    alpha = check_alpha(alpha)
    if statistic not in ("z1", "z2"):
        raise ValueError("statistic must be 'z1' or 'z2'")
    if n_sim < 1 or chunk < 1:
        raise ValueError("n_sim and chunk must be positive")
    ra, qa, ea = _rqe(r, q, e)
    t = len(ra)
    z_obs = (
        acerbi_szekely_z2(ra, qa, ea, alpha) if statistic == "z2" else acerbi_szekely_z1(ra, qa, ea)
    )
    name = f"acerbi_szekely_{statistic}"
    n_hits = int(np.sum(ra <= qa))
    if math.isnan(z_obs):
        return BacktestResult(
            name, math.nan, math.nan, t, None, "less", {"reason": "no exceedances", "hits": 0}
        )
    rng = np.random.default_rng(seed)
    sims: list[NDArray[np.float64]] = []
    n_undefined = 0
    remaining = n_sim
    while remaining > 0:
        k = min(chunk, remaining)
        x = np.asarray(simulate(rng, k), dtype=np.float64)
        if x.shape != (k, t):
            raise ValueError(f"simulate returned shape {x.shape}, expected {(k, t)}")
        z, und = _z_paths(x, qa, ea, alpha, statistic)
        sims.append(z)
        n_undefined += und
        remaining -= k
    zs = np.concatenate(sims)
    m = len(zs)
    if m == 0:
        return BacktestResult(
            name, z_obs, math.nan, t, None, "less", {"reason": "all simulated paths undefined"}
        )
    p = (1 + int(np.sum(zs <= z_obs))) / (m + 1)
    return BacktestResult(
        name,
        z_obs,
        p,
        t,
        None,
        "less",
        {
            "hits": n_hits,
            "n_sim": m,
            "n_undefined": n_undefined,
            "null_mean": float(zs.mean()),
            "null_q05": float(np.quantile(zs, 0.05)),
        },
    )


def acerbi_szekely_pvalue_from_null(z_obs: float, z_null: ArrayLike) -> float:
    """``(1 + #{Z_null <= z_obs}) / (M + 1)`` for a precomputed null sample."""
    zs = np.asarray(z_null, dtype=np.float64)
    zs = zs[np.isfinite(zs)]
    if math.isnan(z_obs) or zs.size == 0:
        return math.nan
    return (1 + int(np.sum(zs <= z_obs))) / (zs.size + 1)


def location_scale_simulator(
    mu: ArrayLike,
    sigma: ArrayLike,
    draw_standardized: Callable[[np.random.Generator, tuple[int, int]], NDArray[np.float64]],
) -> Simulator:
    """Simulator for forecasts ``r_t = mu_t + sigma_t z_t`` with a fixed standardized law."""
    mu_a = as_float_1d(mu, "mu")
    sig_a = as_float_1d(sigma, "sigma")
    check_same_length([mu_a, sig_a], ["mu", "sigma"])
    if np.any(sig_a <= 0):
        raise ValueError("sigma must be positive")

    def simulate(rng: np.random.Generator, n_paths: int) -> NDArray[np.float64]:
        z = draw_standardized(rng, (n_paths, len(mu_a)))
        return mu_a + sig_a * z

    return simulate


def mcneil_frey(
    r: ArrayLike,
    q: ArrayLike,
    e: ArrayLike,
    sigma: ArrayLike | None = None,
    n_boot: int = 1000,
    seed: int | None = None,
) -> BacktestResult:
    """McNeil and Frey (2000) bootstrap test that exceedance residuals have mean zero.

    Residuals ``eps_t = (r_t - e_t) / s_t`` on hit days. The primary p-value is one-sided
    ("less": realized tail returns worse than the ES forecast), from a bootstrap-t test on
    the mean-shifted residuals (Efron and Tibshirani 1993, ch. 16). ``details`` holds the
    two-sided bootstrap p-value and Student-t(m-1) asymptotic p-values.
    """
    ra, qa, ea = _rqe(r, q, e)
    if sigma is None:
        sa = np.ones_like(ra)
    else:
        sa = as_float_1d(sigma, "sigma")
        check_same_length([ra, sa], ["r", "sigma"])
        if np.any(sa <= 0):
            raise ValueError("sigma must be positive")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    h = ra <= qa
    eps = (ra[h] - ea[h]) / sa[h]
    m = len(eps)
    if m < 3:
        return BacktestResult(
            "mcneil_frey",
            math.nan,
            math.nan,
            len(ra),
            None,
            "less",
            {"reason": f"fewer than 3 exceedances ({m})", "hits": m},
        )
    sd = float(eps.std(ddof=1))
    if sd == 0.0:
        return BacktestResult(
            "mcneil_frey",
            math.nan,
            math.nan,
            len(ra),
            None,
            "less",
            {"reason": "exceedance residuals have zero variance", "hits": m},
        )
    t0 = math.sqrt(m) * float(eps.mean()) / sd
    rng = np.random.default_rng(seed)
    centred = eps - eps.mean()
    idx = rng.integers(0, m, size=(n_boot, m))
    bs = centred[idx]
    bsd = bs.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tb = math.sqrt(m) * bs.mean(axis=1) / bsd
    tb = tb[np.isfinite(tb)]
    b = len(tb)
    p_less = (1 + int(np.sum(tb <= t0))) / (b + 1)
    p_two = (1 + int(np.sum(np.abs(tb) >= abs(t0)))) / (b + 1)
    return BacktestResult(
        "mcneil_frey",
        t0,
        p_less,
        len(ra),
        None,
        "less",
        {
            "hits": m,
            "mean_residual": float(eps.mean()),
            "pvalue_two_sided": p_two,
            "pvalue_t_less": float(stats.t.cdf(t0, m - 1)),
            "pvalue_t_two_sided": float(2 * stats.t.sf(abs(t0), m - 1)),
            "n_boot": b,
        },
    )
