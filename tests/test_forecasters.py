"""M6: forecasters."""

from __future__ import annotations

import math

import numpy as np
import pytest
from arch import arch_model

from tailcheck.dist import std_t_var_es
from tailcheck.forecasters import MODELS, make_forecaster, rolling_forecast
from tailcheck.forecasters.caviar import fit_sav, sav_path
from tailcheck.forecasters.garch import fit_gjr, fit_gpd_tail, gjr_filter, gpd_var_es
from tailcheck.forecasters.historical import empirical_var_es


def _garch_series(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_t(5, n + 500) * math.sqrt(3 / 5)
    r = np.empty(n + 500)
    s2 = 1.0
    for t in range(n + 500):
        r[t] = math.sqrt(s2) * z[t]
        s2 = 0.05 + (0.05 + 0.1 * (r[t] < 0)) * r[t] ** 2 + 0.85 * s2
    return r[500:]


def test_empirical_var_es() -> None:
    x = np.arange(1.0, 101.0)
    assert empirical_var_es(x, 0.025) == (3.0, 2.0)  # k = 3
    assert empirical_var_es(x, 0.01) == (1.0, 1.0)


@pytest.mark.parametrize("dist", ["t", "normal"])
def test_gjr_recursion_matches_arch(dist: str) -> None:
    r = _garch_series(1000, 1)
    p = fit_gjr(r, dist)
    res = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist=dist, rescale=False).fit(
        disp="off"
    )
    s = np.sqrt(gjr_filter(p, r)[:-1])
    np.testing.assert_allclose(s, res.conditional_volatility, atol=1e-8)


def test_evt_recovers_t4_tail() -> None:
    rng = np.random.default_rng(2)
    z = rng.standard_t(4, 20_000) / math.sqrt(2)  # unit variance t4
    tail = fit_gpd_tail(z, 0.10)
    var, es = gpd_var_es(tail, 0.01)
    q, e = std_t_var_es(4.0, 0.01)
    assert -var == pytest.approx(q, rel=0.05)
    assert -es == pytest.approx(e, rel=0.08)
    assert 0.1 < tail.xi < 0.45  # true xi = 1/4


def test_gpd_exponential_limit() -> None:
    from tailcheck.forecasters.garch import GPDTail

    t = GPDTail(u=2.0, xi=0.0, beta=0.5, k=100, n=1000, method="mle", capped=False)
    var, es = gpd_var_es(t, 0.01)
    assert var == pytest.approx(2.0 - 0.5 * math.log(0.1))
    assert es == pytest.approx(var + 0.5)


def test_caviar_recovers_sav_parameters() -> None:
    rng = np.random.default_rng(3)
    n, a = 3000, 0.05
    b_true = np.array([-0.1, 0.8, -0.3])
    r = np.empty(n)
    q = -1.5
    from scipy import stats

    zq = stats.norm.ppf(a)
    for t in range(n):
        # returns whose alpha-quantile is q_t: scale a normal so that its quantile is q
        r[t] = q / zq * rng.standard_normal()
        q = b_true[0] + b_true[1] * q + b_true[2] * abs(r[t])
    fit = fit_sav(r, a, np.random.default_rng(0), n_random=500, n_best=5)
    np.testing.assert_allclose(fit.b, b_true, atol=0.12)
    path = sav_path(fit.b, r[:-1], fit.q0)
    assert np.mean(r <= path) == pytest.approx(a, abs=0.015)
    assert fit.es_ratio >= 1.0


@pytest.mark.parametrize("name", MODELS)
def test_no_look_ahead_and_shapes(name: str) -> None:
    r = _garch_series(700, 4)
    model = make_forecaster(name, caviar_random_starts=100, caviar_best_starts=2)
    alphas = [0.01, 0.025]
    kw = {"window": 500, "refit_every": 21, "alphas": alphas, "n_draws": 0, "seed": 1}
    base = rolling_forecast(model, r, 500, 560, **kw)
    t_check = 530  # forecast for day 530 may use r[:530] only
    r2 = r.copy()
    r2[t_check:] += 50 * np.random.default_rng(9).standard_normal(len(r) - t_check)
    pert = rolling_forecast(model, r2, 500, 560, **kw)
    i = t_check - 500
    np.testing.assert_allclose(pert["q"][: i + 1], base["q"][: i + 1])
    np.testing.assert_allclose(pert["e"][: i + 1], base["e"][: i + 1])
    assert base["q"].shape == (60, 2)
    assert np.all(base["e"] <= base["q"] + 1e-12) and np.all(base["q"] < 0)
    assert base["n_refits"] == 3


def test_gjr_t_forecast_formula_and_draws() -> None:
    r = _garch_series(600, 5)
    model = make_forecaster("gjr-t")
    blocks = []
    out = rolling_forecast(
        model,
        r,
        500,
        510,
        500,
        21,
        [0.025],
        n_draws=4000,
        seed=1,
        on_block=lambda t0, t1, b: blocks.append(b),
    )
    st = model.fit(r[:500], [0.025], np.random.default_rng(0))
    qz, ez = std_t_var_es(st.params.nu, 0.025)
    sig = out["sigma"]
    np.testing.assert_allclose(out["q"][:, 0], st.params.mu + sig * qz)
    np.testing.assert_allclose(out["e"][:, 0], st.params.mu + sig * ez)
    d = blocks[0].draws
    assert d.shape == (10, 4000)
    # draws come from the predictive law: their empirical 2.5% quantile is close to q
    emp = np.quantile(d, 0.025, axis=1)
    np.testing.assert_allclose(emp, out["q"][:, 0], rtol=0.1)


def test_draws_shapes_all_distribution_models() -> None:
    r = _garch_series(560, 6)
    for name in ("hs", "fhs", "evt-pot"):
        blocks = []
        rolling_forecast(
            make_forecaster(name),
            r,
            500,
            505,
            500,
            21,
            [0.01],
            n_draws=50,
            seed=0,
            on_block=lambda t0, t1, b, acc=blocks: acc.append(b),
        )
        assert blocks[0].draws is not None and blocks[0].draws.shape == (5, 50)
        assert np.all(np.isfinite(blocks[0].draws))
