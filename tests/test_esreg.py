"""M4a/M4b: joint quantile/ES regression, loss, quantile regression and covariance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, stats

from tailcheck.esreg import G1, G2, esr_loss, esreg, gfuns, vcov_asymptotic
from tailcheck.esreg.covariance import bandwidth, bandwidth_sj, vcov_details
from tailcheck.esreg.estimator import ESRegFit, alpha_tilde
from tailcheck.esreg.quantreg import hall_sheather, quantile_regression
from tailcheck.scoring import fz0_loss

FIX = Path(__file__).parent / "fixtures" / "r"
REF = json.loads((FIX / "outputs" / "r_reference.json").read_text())


def _dgp2() -> tuple[np.ndarray, np.ndarray]:
    x, y = np.loadtxt(FIX / "inputs" / "dgp2.csv", delimiter=",", skiprows=1).T
    return x, y


def _fit_at(coef: list[float], y: np.ndarray, x: np.ndarray | None) -> ESRegFit:
    c = np.asarray(coef)
    xm = np.ones((len(y), 1)) if x is None else np.column_stack([np.ones_like(x), x])
    k = xm.shape[1]
    return ESRegFit(c[:k], c[k:], np.nan, 0.025, G1.ZERO, G2.NEG_INV, y, xm, xm, 0, float(y.max()))


# ---------------------------------------------------------------- G functions and loss


@pytest.mark.parametrize("kind", list(G2))
def test_g2_derivatives_by_finite_differences(kind: G2) -> None:
    z = np.array([-3.0, -1.2, -0.4]) if kind in gfuns.NEGATIVE_DOMAIN else np.array([-2, 0.3, 1.5])
    h = 1e-5
    np.testing.assert_allclose(
        (gfuns.curly_g2(z + h, kind) - gfuns.curly_g2(z - h, kind)) / (2 * h),
        gfuns.g2(z, kind),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        (gfuns.g2(z + h, kind) - gfuns.g2(z - h, kind)) / (2 * h),
        gfuns.g2_prime(z, kind),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        (gfuns.g2_prime(z + h, kind) - gfuns.g2_prime(z - h, kind)) / (2 * h),
        gfuns.g2_prime_prime(z, kind),
        rtol=1e-5,
    )


def test_g2_type1_second_derivative_is_minus_two_over_z_cubed() -> None:
    assert gfuns.g2_prime_prime(np.array([-2.0]), G2.NEG_INV)[0] == pytest.approx(0.25)


def test_g1() -> None:
    z = np.array([-1.0, 2.0])
    np.testing.assert_array_equal(gfuns.g1(z, G1.IDENTITY), z)
    np.testing.assert_array_equal(gfuns.g1_prime(z, G1.ZERO), 0.0)


def test_loss_equals_fz0() -> None:
    rng = np.random.default_rng(0)
    r = rng.standard_normal(500)
    q = -1.9 + 0.1 * rng.standard_normal(500)
    e = q - 0.4 - 0.1 * rng.random(500)
    np.testing.assert_allclose(esr_loss(r, q, e, 0.025), fz0_loss(r, q, e, 0.025), atol=1e-12)
    with pytest.raises(ValueError):
        esr_loss(r, q, -e, 0.025)


def test_loss_minimised_at_truth() -> None:
    rng = np.random.default_rng(1)
    r = rng.standard_normal(200_000)
    q0, e0 = stats.norm.ppf(0.025), -stats.norm.pdf(stats.norm.ppf(0.025)) / 0.025
    base = esr_loss(r, np.full_like(r, q0), np.full_like(r, e0), 0.025, G1.IDENTITY, G2.INV_SQRT)
    for dq, de in [(0.15, 0), (-0.15, 0), (0, 0.2), (0, -0.2)]:
        other = esr_loss(
            r, np.full_like(r, q0 + dq), np.full_like(r, e0 + de), 0.025, G1.IDENTITY, G2.INV_SQRT
        )
        assert other.mean() > base.mean()


# ---------------------------------------------------------------- quantile regression


@pytest.mark.parametrize("i", range(4))
def test_quantile_regression_matches_r(i: int) -> None:
    x, y = _dgp2()
    ref = REF["rq"][i]
    b = quantile_regression(y, np.column_stack([np.ones_like(x), x]), ref["tau"])
    np.testing.assert_allclose(b, ref["coef"], atol=1e-6)


def test_quantile_regression_intercept_only_is_order_statistic() -> None:
    y = np.random.default_rng(3).standard_normal(401)
    b = quantile_regression(y, np.ones((401, 1)), 0.1)
    assert b[0] == np.sort(y)[40]  # ceil(401 * 0.1) = 41st smallest


@pytest.mark.parametrize(
    ("n", "alpha", "h"),
    [(1000, 0.025, 0.01313628), (2000, 0.025, 0.01042628), (500, 0.01, 0.00884657)],
)
def test_hall_sheather(n: int, alpha: float, h: float) -> None:
    assert hall_sheather(n, alpha) == pytest.approx(h, abs=1e-8)


def test_bandwidth_shrink() -> None:
    assert hall_sheather(250, 0.01) == pytest.approx(0.011146, abs=1e-6)
    h, warn = bandwidth(250, 0.01)
    assert h == pytest.approx(0.009) and warn is not None


def test_alpha_tilde() -> None:
    assert alpha_tilde(0.025) == pytest.approx(0.0096987404, abs=1e-9)
    assert alpha_tilde(0.01) == pytest.approx(0.0038469647, abs=1e-9)


# ---------------------------------------------------------------- estimator


@pytest.mark.parametrize("i", range(3))
def test_esreg_matches_r_loss_and_coefficients(i: int) -> None:
    x, y = _dgp2()
    ref = REF["esreg"][i]
    fit = esreg(y, x, x, 0.025, G1(ref["g1"]), G2(ref["g2"]), seed=1)
    assert fit.loss <= ref["loss"] + 1e-7
    cov = vcov_asymptotic(_fit_at(REF["vcov_coef"], y, x), "nid", "scl_N", False)
    se = np.sqrt(np.diag(cov))
    assert np.all(np.abs(fit.coef - np.asarray(ref["coef"])) <= 0.05 * se)


def test_esreg_validation() -> None:
    with pytest.raises(ValueError):
        esreg(np.zeros(50), None, None, 0.025)  # 1.25 expected tail obs < 3
    with pytest.raises(ValueError):
        esreg(np.r_[np.zeros(9), np.nan], None, None, 0.5)


def test_esreg_scale_equivariance() -> None:
    x, y = _dgp2()
    a = esreg(y, x, x, 0.025, seed=2)
    b = esreg(3 * y, x, x, 0.025, seed=2)
    np.testing.assert_allclose(b.coef, 3 * a.coef, rtol=2e-3, atol=2e-3)


def test_esreg_consistency_dgp2() -> None:
    """DB 2019 DGP-(2) at n = 20,000: estimates within 4 estimated SEs of the truth."""
    rng = np.random.default_rng(11)
    n = 20_000
    x = rng.chisquare(1, n)
    y = -x + (1 + 0.5 * x) * rng.standard_normal(n)
    fit = esreg(y, x, x, 0.025, early_stopping=2, seed=1)
    truth = np.array([-1.959964, -1.979982, -2.337803, -2.168901])
    se = np.sqrt(np.diag(vcov_asymptotic(fit, "nid", "scl_N", True)))
    assert np.all(np.abs(fit.coef - truth) <= 4 * se), (fit.coef, se)


# ---------------------------------------------------------------- covariance


def _cases() -> list[int]:
    return list(range(len(REF["vcov"])))


@pytest.mark.parametrize("i", _cases())
def test_vcov_matches_r_at_r_coefficients(i: int) -> None:
    x, y = _dgp2()
    case = REF["vcov"][i]
    if case["sparsity"] == "iid":
        pytest.skip("iid sparsity differs from esreg (documented in docs/esreg-covariance.md)")
    fit = _fit_at(REF["vcov_coef"], y, x)
    r_cov = np.asarray(case["cov"]).reshape(4, 4)
    # scl_N: 1e-4 (the location-scale PMLE optimisers differ slightly); scl_sp: KDE binning
    tol = 1e-4 if case["sigma_est"] == "scl_N" else 2e-2
    compat, _ = vcov_details(
        fit, case["sparsity"], case["sigma_est"], case["misspec"], _esreg_compat_g2pp=True
    )
    np.testing.assert_allclose(compat, r_cov, rtol=tol)
    ours, _ = vcov_details(fit, case["sparsity"], case["sigma_est"], case["misspec"])
    if case["misspec"]:
        # the only difference is esreg's G2'' for type 1 (-1/z^3 instead of -2/z^3)
        assert not np.allclose(ours, r_cov, rtol=tol)
    else:
        np.testing.assert_allclose(ours, r_cov, rtol=tol)


def test_bandwidth_sj_matches_r() -> None:
    for i, ref in enumerate(REF["bw_sj"]):
        x = np.loadtxt(FIX / "inputs" / f"bw_{i}.csv", skiprows=1)
        assert bandwidth_sj(x) == pytest.approx(ref, rel=0.02)


def _remark_210(y: np.ndarray, fit: ESRegFit, f: float, v: float) -> np.ndarray:
    a = fit.alpha
    q, e = fit.coef_q[0], fit.coef_e[0]
    return np.array(
        [
            [a * (1 - a) / f**2, (1 - a) * (q - e) / f],
            [(1 - a) * (q - e) / f, v / a + (1 - a) / a * (q - e) ** 2],
        ]
    )


@pytest.mark.parametrize(("g1", "g2"), [(G1.ZERO, G2.NEG_INV), (G1.IDENTITY, G2.INV_SQRT)])
def test_intercept_only_covariance_formula(g1: G1, g2: G2) -> None:
    """DB 2019 Remark 2.10: n Cov equals Sigma(f, v, q, e) for any G1/G2 (exact algebra)."""
    y = np.random.default_rng(4).standard_normal(5000)
    fit = esreg(y, None, None, 0.025, g1, g2, early_stopping=0)
    cov, info = vcov_details(fit, misspec=False)
    sig = _remark_210(y, fit, float(info["density"][0]), float(info["trunc_var"][0]))
    np.testing.assert_allclose(len(y) * cov, sig, rtol=1e-8)


@pytest.mark.parametrize("law", ["normal", "t5"])
def test_intercept_only_covariance_oracle(law: str) -> None:
    """Remark 2.10 closed form at n = 200,000; the ES variance within 3% (normal). The density-based
    entries use the Koenker (1994) iid sparsity, whose sampling sd is about 5% at this n, so
    they get a 15% tolerance."""
    n, a = 200_000, 0.025
    rng = np.random.default_rng(5)
    if law == "normal":
        y = rng.standard_normal(n)
        dist = stats.norm
    else:
        y = rng.standard_t(5, n)
        dist = stats.t(5)
    fit = esreg(y, None, None, a, early_stopping=0)
    cov = vcov_asymptotic(fit, misspec=False)
    q = dist.ppf(a)
    e = integrate.quad(lambda t: t * dist.pdf(t), -np.inf, q)[0] / a
    m2 = integrate.quad(lambda t: t * t * dist.pdf(t), -np.inf, q)[0] / a
    f = dist.pdf(q)
    v = m2 - e**2
    sig = np.array(
        [
            [a * (1 - a) / f**2, (1 - a) * (q - e) / f],
            [(1 - a) * (q - e) / f, v / a + (1 - a) / a * (q - e) ** 2],
        ]
    )
    got = n * cov
    # t5 tail variances converge slowly (the truncated sample variance is noisy): 6%
    assert got[1, 1] == pytest.approx(sig[1, 1], rel=0.03 if law == "normal" else 0.06)
    assert got[0, 0] == pytest.approx(sig[0, 0], rel=0.15)
    assert got[0, 1] == pytest.approx(sig[0, 1], rel=0.15)


def test_intercept_only_forces_iid_ind() -> None:
    y = np.random.default_rng(6).standard_normal(2000)
    fit = esreg(y, None, None, 0.025, early_stopping=0)
    _, info = vcov_details(fit, "nid", "scl_sp")
    assert info["sparsity"] == "iid" and info["sigma_est"] == "ind"
