"""Asymptotic sandwich covariance of the joint estimator and its nuisance estimators.

``Cov(b) = Lambda^{-1} C Lambda^{-1} / n`` (Dimitriadis and Bayer 2019, Thm 2.6). The
terms, including the misspecification-robust ones, are derived in
``docs/esreg-covariance.md``; specification in ``docs/DESIGN.md`` section 4.3.3.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, special, stats

from tailcheck.esreg.estimator import ESRegFit
from tailcheck.esreg.gfuns import G2, g1_prime, g1_prime_prime, g2, g2_prime, g2_prime_prime
from tailcheck.esreg.quantreg import bofinger, chamberlain, hall_sheather, quantile_regression

Sparsity = Literal["iid", "nid"]
SigmaEstimator = Literal["ind", "scl_N", "scl_sp"]
Bandwidth = Literal["hall-sheather", "bofinger", "chamberlain"]

Arr = NDArray[np.float64]
_EPS23 = np.finfo(float).eps ** (2 / 3)


def bandwidth(n: int, alpha: float, kind: Bandwidth = "hall-sheather") -> tuple[float, str | None]:
    """Sparsity bandwidth, shrunk to ``0.9 min(alpha, 1 - alpha)`` if it leaves (0, 1)."""
    if kind == "hall-sheather":
        h = hall_sheather(n, alpha)
    elif kind == "bofinger":
        h = bofinger(n, alpha)
    elif kind == "chamberlain":
        h = chamberlain(n, alpha)
    else:
        raise ValueError(f"unknown bandwidth {kind!r}")
    limit = min(alpha, 1 - alpha)
    if h >= limit:
        return 0.9 * limit, f"bandwidth {h:.6f} >= {limit}; shrunk to {0.9 * limit:.6f}"
    return h, None


def _phi4(u: Arr) -> Arr:
    u2 = u * u
    return (u2 * u2 - 6 * u2 + 3) * np.exp(-u2 / 2) / math.sqrt(2 * math.pi)


def _phi6(u: Arr) -> Arr:
    u2 = u * u
    return (u2**3 - 15 * u2**2 + 45 * u2 - 15) * np.exp(-u2 / 2) / math.sqrt(2 * math.pi)


def _pair_distance_counts(xs: Arr, edges: Arr, block_elems: int = 4_000_000) -> NDArray[np.int64]:
    """Histogram of all ``n (n - 1) / 2`` pairwise distances of sorted ``xs`` over ``edges``.

    Rows are processed in blocks of at most ``block_elems`` differences, so memory is O(n)
    rather than O(n^2); the counts are identical to histogramming all differences at once.
    """
    n = len(xs)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    rows = max(1, block_elems // max(n, 1))
    for i0 in range(0, n - 1, rows):
        i1 = min(i0 + rows, n - 1)
        # differences x[j] - x[i] for i in [i0, i1), j > i (non-negative because sorted)
        cols = xs[i0 + 1 :]
        diff = cols[None, :] - xs[i0:i1, None]
        keep = np.arange(i0 + 1, n)[None, :] > np.arange(i0, i1)[:, None]
        c, _ = np.histogram(diff[keep], edges)
        counts += c
    return counts


def _binned_distance_counts(xs: Arr, nb: int) -> tuple[Arr, Arr]:
    """Approximate pair-distance counts from ``nb`` equal-width bins of the data.

    As in R's ``bw.SJ``: points are assigned to bins, and pairs are counted by bin
    separation ``k`` (distance ``k * width``), ``c_k = sum_i n_i n_{i+k}`` for ``k >= 1`` and
    ``sum_i n_i (n_i - 1) / 2`` at distance 0. O(n + nb log nb) time and O(nb) memory.
    """
    lo, hi = float(xs[0]), float(xs[-1])
    width = (hi - lo) / nb
    idx = np.minimum(((xs - lo) / width).astype(np.int64), nb - 1)
    m = np.bincount(idx, minlength=nb).astype(np.float64)
    size = 1 << int(np.ceil(np.log2(2 * nb)))
    f = np.fft.rfft(m, size)
    ac = np.fft.irfft(f * np.conj(f), size)[:nb]
    cnt = np.rint(ac).astype(np.float64)
    cnt[0] = float(np.sum(m * (m - 1) / 2))
    return cnt, np.arange(nb, dtype=np.float64) * width


def bandwidth_sj(x: ArrayLike, nbins: int = 20_000, max_exact: int = 10_000) -> float:
    """Sheather-Jones (1991) solve-the-equation bandwidth for a Gaussian KDE.

    For ``n <= max_exact`` the pairwise sums run over all ``n (n - 1) / 2`` absolute
    differences, counted once into ``nbins`` fine bins (width ``max|d| / nbins``) so that each
    functional evaluation is O(nbins); with 20,000 bins the result differs from the unbinned
    sum by about 1e-5 relative. The differences are formed in row blocks, so memory is
    O(n + nbins) and time O(n^2). Above ``max_exact`` the data themselves are first binned
    into ``nbins`` bins and pairs are counted by bin separation (as R's ``bw.SJ`` does with
    1,000 bins), which is O(n + nbins log nbins). The functional estimates include the
    diagonal terms and divide by ``n (n - 1)`` (Wand and Jones 1995, sec. 3.6).
    """
    xa = np.asarray(x, dtype=np.float64)
    xa = xa[np.isfinite(xa)]
    n = len(xa)
    if n < 2:
        raise ValueError("need at least 2 finite values")
    iqr = float(np.subtract(*np.percentile(xa, [75, 25])))
    sd = float(np.std(xa, ddof=1))
    scale = min(sd, iqr / 1.349) if iqr > 0 else sd
    if scale <= 0:
        raise ValueError("data have zero spread")
    xs = np.sort(xa)
    if n <= max_exact:
        dmax = float(xs[-1] - xs[0])  # equals max |x_i - x_j| exactly
        edges = np.linspace(0.0, dmax * (1 + 1e-12) + 1e-300, nbins + 1)
        counts = _pair_distance_counts(xs, edges)
        mid = 0.5 * (edges[1:] + edges[:-1])
    else:
        counts_f, mid = _binned_distance_counts(xs, nbins)
        counts = counts_f
    keep = counts > 0
    cnt, d = counts[keep].astype(np.float64), mid[keep]

    def sdh(h: float) -> float:  # estimate of R(f'') = int f''^2
        return (2 * (cnt * _phi4(d / h)).sum() + n * _phi4(np.zeros(1))[0]) / (n * (n - 1) * h**5)

    def tdh(h: float) -> float:  # estimate of -int f'' f''''
        return (2 * (cnt * _phi6(d / h)).sum() + n * _phi6(np.zeros(1))[0]) / (n * (n - 1) * h**7)

    a = 1.24 * scale * n ** (-1 / 7)
    b = 1.23 * scale * n ** (-1 / 9)
    c1 = 1 / (2 * math.sqrt(math.pi) * n)
    td = -tdh(b)
    if not td > 0:
        raise RuntimeError("Sheather-Jones: non-positive pilot estimate")
    alph2 = 1.357 * (sdh(a) / td) ** (1 / 7)

    def eq(h: float) -> float:
        return (c1 / sdh(alph2 * h ** (5 / 7))) ** (1 / 5) - h

    hmax = 1.144 * scale * n ** (-1 / 5)
    lo, hi = 0.1 * hmax, hmax
    for _ in range(20):
        if eq(lo) * eq(hi) < 0:
            break
        lo, hi = lo / 1.2, hi * 1.2
    else:  # pragma: no cover - pathological data
        raise RuntimeError("Sheather-Jones: no sign change found")
    return float(optimize.brentq(eq, lo, hi, xtol=0.1 * lo / 1e3))


def location_scale_fit(y: Arr, x: Arr) -> tuple[Arr, Arr]:
    """Gaussian PMLE of ``y = x'zeta + (x'phi) eps``; returns fitted (mu, sigma)."""
    k = x.shape[1]
    zeta0, *_ = np.linalg.lstsq(x, y, rcond=None)
    absres = np.abs(y - x @ zeta0)
    phi0, *_ = np.linalg.lstsq(x, absres, rcond=None)
    s0 = x @ phi0
    if np.min(s0) <= 0:
        phi0[0] += -np.min(s0) + 0.1 * max(float(np.mean(absres)), 1e-8)
    p0 = np.concatenate([zeta0, phi0])

    def nll(p: Arr) -> float:
        mu = x @ p[:k]
        s = x @ p[k:]
        if np.any(s <= 0):
            return 1e300
        return float(np.sum(np.log(s) + 0.5 * ((y - mu) / s) ** 2))

    best = p0
    for method in ("BFGS", "Nelder-Mead"):
        try:
            res = optimize.minimize(nll, best, method=method)
        except (ValueError, FloatingPointError):  # pragma: no cover
            continue
        if np.isfinite(res.fun) and res.fun < 1e299 and res.fun <= nll(best):
            best = res.x
            if res.success:
                break
    return x @ best[:k], x @ best[k:]


def _density_iid(u: Arr, kq: int, h: float) -> float:
    """Koenker (1994) iid density at the quantile from the smallest-|u| residuals."""
    n = len(u)
    pz = int(np.sum(np.abs(u) < _EPS23))
    hh = max(kq + 1, math.ceil(n * h))
    order = np.argsort(np.abs(u), kind="stable")
    ir = np.arange(pz + 1, min(hh + pz + 1, n - 1) + 1)  # 1-based ranks, as in the paper
    ord_resid = np.sort(u[order[ir - 1]])
    xt = ir / (n - kq)
    slope = quantile_regression(ord_resid, np.column_stack([np.ones(len(xt)), xt]), 0.5)[1]
    return 1.0 / slope if slope > 0 else math.nan


def _density_nid(y: Arr, xq: Arr, alpha: float, h: float) -> Arr:
    """Hendricks-Koenker (1992) difference-quotient density at each fitted quantile."""
    b_hi = quantile_regression(y, xq, alpha + h)
    b_lo = quantile_regression(y, xq, alpha - h)
    dq = xq @ (b_hi - b_lo)
    return np.maximum(0.0, 2 * h / (dq - _EPS23))


def _cdf_at_quantile(y: Arr, xq: Arr, xi: Arr) -> Arr:
    mu, sigma = location_scale_fit(y, xq)
    z = np.sort((y - mu) / sigma)
    return np.searchsorted(z, (xi - mu) / sigma, side="right") / len(z)


def _truncated_variance(
    y: Arr, xq: Arr, xi: Arr, kind: SigmaEstimator, info: dict[str, Any]
) -> Arr:
    u = y - xi
    neg = u[u <= 0]
    v_ind = float(np.var(neg, ddof=1)) if len(neg) > 1 else math.nan
    if kind == "ind":
        return np.full(len(y), v_ind)
    mu, sigma = location_scale_fit(u, xq)
    beta = np.maximum(-mu / sigma, -30.0)
    if kind == "scl_N":
        lam = stats.norm.pdf(beta) / stats.norm.cdf(beta)
        v = sigma**2 * (1 - beta * lam - lam**2)
    elif kind == "scl_sp":
        z = (u - mu) / sigma
        bw = bandwidth_sj(z)
        info["sj_bandwidth"] = bw
        # Exact truncated moments of the Gaussian KDE up to beta_i (closed form of the
        # integrals that a fine trapezoidal grid approximates).
        m0, m1, m2 = (np.empty(len(beta)) for _ in range(3))
        zz = z[None, :]
        for lo in range(0, len(beta), 256):  # row blocks bound memory to 256 x n
            sl = slice(lo, lo + 256)
            a = (beta[sl, None] - zz) / bw
            cdf = special.ndtr(a)
            pdf = np.exp(-0.5 * a * a) * (1 / math.sqrt(2 * math.pi))
            m0[sl] = cdf.mean(axis=1)
            m1[sl] = (zz * cdf - bw * pdf).mean(axis=1)
            m2[sl] = (zz**2 * cdf - 2 * zz * bw * pdf + bw**2 * (cdf - a * pdf)).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = sigma**2 * (m2 / m0 - (m1 / m0) ** 2)
    else:
        raise ValueError(f"unknown sigma estimator {kind!r}")
    bad = ~np.isfinite(v) | (v <= 0)
    if np.any(bad):
        info.setdefault("warnings", []).append(
            f"{int(bad.sum())} non-finite truncated variances replaced by the 'ind' estimate"
        )
        v = np.where(bad, v_ind, v)
    return v


def vcov_details(
    fit: ESRegFit,
    sparsity: Sparsity = "nid",
    sigma_est: SigmaEstimator = "scl_sp",
    misspec: bool = True,
    bandwidth_type: Bandwidth = "hall-sheather",
    _esreg_compat_g2pp: bool = False,
) -> tuple[Arr, dict[str, Any]]:
    """Covariance plus diagnostics (bandwidth, warnings, estimators actually used).

    ``_esreg_compat_g2pp`` reproduces the ``G2''`` used by R ``esreg`` 0.6.2 for G2 type 1
    (``-1/z^3`` instead of ``-2/z^3``); it exists only for the reference-fixture comparison
    documented in ``docs/esreg-covariance.md``.
    """
    alpha = fit.alpha
    y = fit.y - fit.y_shift
    xq, xe = fit.xq, fit.xe
    n, kq = xq.shape
    ke = xe.shape[1]
    bq = fit.coef_q.copy()
    be = fit.coef_e.copy()
    bq[0] -= fit.y_shift
    be[0] -= fit.y_shift
    xi = xq @ bq
    eta = xe @ be
    info: dict[str, Any] = {"warnings": []}
    if kq == 1 and ke == 1:
        sparsity, sigma_est = "iid", "ind"
    info["sparsity"], info["sigma_est"], info["misspec"] = sparsity, sigma_est, misspec
    h, warn = bandwidth(n, alpha, bandwidth_type)
    info["bandwidth"] = h
    if warn:
        info["warnings"].append(warn)

    if sparsity == "iid":
        f = np.full(n, _density_iid(y - xi, kq, h))
    elif sparsity == "nid":
        f = _density_nid(y, xq, alpha, h)
    else:
        raise ValueError(f"unknown sparsity estimator {sparsity!r}")
    cdf = _cdf_at_quantile(y, xq, xi) if misspec else np.full(n, alpha)
    v = _truncated_variance(y, xq, xi, sigma_est, info)

    g1p = g1_prime(xi, fit.g1)
    g1pp = g1_prime_prime(xi, fit.g1)
    g2v = g2(eta, fit.g2)
    g2p = g2_prime(eta, fit.g2)
    g2pp = g2_prime_prime(eta, fit.g2)
    if _esreg_compat_g2pp and fit.g2 == G2.NEG_INV:
        g2pp = -1.0 / eta**3

    a = alpha
    d = cdf - a
    w_core = a * g1p + g2v

    def wmean(xa: Arr, xb: Arr, w: Arr) -> Arr:
        return (xa * w[:, None]).T @ xb / n

    l11_w = f * (g1p + g2v / a)
    l22_w = g2p.copy()
    l12_w = np.zeros(n)
    c11_w = w_core**2 * (1 - a) / a
    c12_w = w_core * g2p * (xi - eta) * (1 - a) / a
    c22_w = g2p**2 * (v / a + (1 - a) / a * (xi - eta) ** 2)
    if misspec:
        l11_w = l11_w + g1pp * d
        l12_w = g2p * d / a
        l22_w = l22_w + g2pp * xi * d / a
        c11_w = c11_w * (1 + (1 - 2 * a) * d / (a * (1 - a)))
        c12_w = c12_w + w_core * g2p * ((1 - a) / a * xi * d / a - d / a * (xi - eta))
        c22_w = c22_w + g2p**2 * 2 * (xi - eta) * xi * (-d) / a

    lam = np.zeros((kq + ke, kq + ke))
    cmat = np.zeros_like(lam)
    lam[:kq, :kq] = wmean(xq, xq, l11_w)
    lam[:kq, kq:] = wmean(xq, xe, l12_w)
    lam[kq:, :kq] = lam[:kq, kq:].T
    lam[kq:, kq:] = wmean(xe, xe, l22_w)
    cmat[:kq, :kq] = wmean(xq, xq, c11_w)
    cmat[kq:, :kq] = wmean(xe, xq, c12_w)
    cmat[:kq, kq:] = cmat[kq:, :kq].T
    cmat[kq:, kq:] = wmean(xe, xe, c22_w)
    lam_inv = np.linalg.inv(lam)
    cov = lam_inv @ cmat @ lam_inv / n
    cov = (cov + cov.T) / 2
    info["density"] = f
    info["cdf"] = cdf
    info["trunc_var"] = v
    info["lambda"] = lam
    info["sigma"] = cmat
    return cov, info


def vcov_asymptotic(
    fit: ESRegFit,
    sparsity: Sparsity = "nid",
    sigma_est: SigmaEstimator = "scl_sp",
    misspec: bool = True,
    bandwidth: Bandwidth = "hall-sheather",
) -> NDArray[np.float64]:
    """``Lambda^{-1} C Lambda^{-1} / n`` for the stacked coefficients ``(bq, be)``."""
    cov, _ = vcov_details(fit, sparsity, sigma_est, misspec, bandwidth)
    return cov
