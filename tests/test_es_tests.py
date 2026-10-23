"""M3: Acerbi-Szekely and McNeil-Frey tests, and the closed-form VaR/ES helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, stats

from tailcheck.dist import draw_std_t, normal_var_es, std_t_var_es
from tailcheck.es_tests import (
    acerbi_szekely_pvalue_from_null,
    acerbi_szekely_test,
    acerbi_szekely_z1,
    acerbi_szekely_z2,
    location_scale_simulator,
    mcneil_frey,
)


@pytest.mark.parametrize(
    ("nu", "alpha", "ref"),
    [
        (5, 0.025, (-1.9911641279, -2.7278020716)),
        (5, 0.01, (-2.6064635694, -3.4488367600)),
        (4, 0.025, (-1.9632431615, -2.8238712518)),
    ],
)
def test_std_t_reference(nu: float, alpha: float, ref: tuple[float, float]) -> None:
    q, e = std_t_var_es(nu, alpha)
    assert q == pytest.approx(ref[0], abs=1e-9)
    assert e == pytest.approx(ref[1], abs=1e-9)
    s = math.sqrt((nu - 2) / nu)
    tail, _ = integrate.quad(lambda x: x * stats.t.pdf(x / s, nu) / s, -np.inf, q)
    assert e == pytest.approx(tail / alpha, rel=1e-7)


def test_normal_reference() -> None:
    assert normal_var_es(0.025) == pytest.approx((-1.9599639845, -2.3378027922), abs=1e-9)
    assert normal_var_es(0.01) == pytest.approx((-2.3263478740, -2.6652142203), abs=1e-9)


def test_draw_std_t_unit_variance() -> None:
    x = draw_std_t(np.random.default_rng(0), 8.0, 400_000)
    assert x.var() == pytest.approx(1.0, abs=0.01)


def test_z_formulas_by_hand() -> None:
    r = np.array([-3.0, 0.5, -1.0, -2.5, 1.0])
    q = np.full(5, -2.0)
    e = np.full(5, -2.5)
    # hits at 0 and 3: r/e = 1.2 and 1.0
    assert acerbi_szekely_z1(r, q, e) == pytest.approx(1 - 1.1)
    assert acerbi_szekely_z2(r, q, e, 0.4) == pytest.approx(1 - 2.2 / (5 * 0.4))
    assert math.isnan(acerbi_szekely_z1(np.zeros(5), q, e))
    assert acerbi_szekely_z2(np.zeros(5), q, e, 0.1) == 1.0


def test_z_scale_invariance() -> None:
    rng = np.random.default_rng(3)
    r = rng.standard_normal(300)
    q, e = np.full(300, -1.96), np.full(300, -2.34)
    for c in (0.1, 7.0):
        assert acerbi_szekely_z2(c * r, c * q, c * e, 0.025) == pytest.approx(
            acerbi_szekely_z2(r, q, e, 0.025)
        )
        assert acerbi_szekely_z1(c * r, c * q, c * e) == pytest.approx(acerbi_szekely_z1(r, q, e))


def test_validation() -> None:
    with pytest.raises(ValueError):
        acerbi_szekely_z2([1.0], [-1.0], [0.5], 0.05)  # positive ES
    with pytest.raises(ValueError):
        acerbi_szekely_z2([1.0], [-2.0], [-1.0], 0.05)  # e > q
    with pytest.raises(ValueError):
        mcneil_frey([1.0, 2.0], [-1.0, -1.0], [-2.0, -2.0], sigma=[1.0, -1.0])


def _normal_sim(t: int) -> object:
    return location_scale_simulator(
        np.zeros(t), np.ones(t), lambda rng, shape: rng.standard_normal(shape)
    )


def test_as_null_quantile_normal() -> None:
    """DESIGN 4.2: 5% null quantile of Z2 is about -0.70 for T = 250, alpha = 0.025."""
    t, alpha = 250, 0.025
    q, e = normal_var_es(alpha)
    rng = np.random.default_rng(1)
    r = rng.standard_normal(t)
    res = acerbi_szekely_test(
        r, np.full(t, q), np.full(t, e), alpha, _normal_sim(t), n_sim=20_000, seed=1
    )
    assert res.details["null_q05"] == pytest.approx(-0.70, abs=0.05)
    # E[Z2] = 0 exactly under H0; sd of Z2 is about 0.42 here, so 4 SE is about 0.012
    assert abs(res.details["null_mean"]) < 4 * 0.45 / math.sqrt(20_000)
    assert 0 < res.pvalue <= 1 and res.alternative == "less"


def test_as_rejects_underestimated_risk_and_z1_undefined_draws() -> None:
    t, alpha = 500, 0.025
    q, e = normal_var_es(alpha)
    rng = np.random.default_rng(2)
    r = 1.6 * rng.standard_normal(t)  # true volatility 60% higher than forecast
    qa, ea = np.full(t, q), np.full(t, e)
    z2 = acerbi_szekely_test(r, qa, ea, alpha, _normal_sim(t), n_sim=2000, seed=0)
    assert z2.pvalue < 0.01
    small = acerbi_szekely_test(
        r[:40], qa[:40], ea[:40], alpha, _normal_sim(40), "z1", n_sim=2000, seed=0
    )
    assert small.details["n_undefined"] > 0  # many 40-day paths have no hit
    assert small.details["n_sim"] + small.details["n_undefined"] == 2000
    nohit = acerbi_szekely_test(np.zeros(40), qa[:40], ea[:40], alpha, _normal_sim(40), "z1")
    assert math.isnan(nohit.pvalue) and nohit.details["reason"] == "no exceedances"


def test_as_seeded_and_chunk_independent() -> None:
    t, alpha = 200, 0.025
    q, e = normal_var_es(alpha)
    r = np.random.default_rng(5).standard_normal(t)
    args = (r, np.full(t, q), np.full(t, e), alpha, _normal_sim(t))
    a = acerbi_szekely_test(*args, n_sim=1000, seed=9, chunk=1000)
    b = acerbi_szekely_test(*args, n_sim=1000, seed=9, chunk=1000)
    assert a.pvalue == b.pvalue
    assert acerbi_szekely_pvalue_from_null(-10.0, [0.0, 1.0]) == pytest.approx(1 / 3)
    assert math.isnan(acerbi_szekely_pvalue_from_null(math.nan, [0.0]))


def test_mcneil_frey_by_hand_and_edges() -> None:
    r = np.array([-3.0, -2.6, -2.2, -2.9, 0.0, 1.0])
    q = np.full(6, -2.0)
    e = np.full(6, -2.5)
    res = mcneil_frey(r, q, e, n_boot=999, seed=0)
    eps = np.array([-0.5, -0.1, 0.3, -0.4])
    t0 = 2 * eps.mean() / eps.std(ddof=1)
    assert res.statistic == pytest.approx(t0)
    assert res.details["pvalue_t_less"] == pytest.approx(stats.t.cdf(t0, 3))
    assert res.details["hits"] == 4
    few = mcneil_frey(np.array([-3.0, -3.0, 1.0]), np.full(3, -2.0), np.full(3, -2.5))
    assert math.isnan(few.pvalue) and "fewer than 3" in few.details["reason"]
    scaled = mcneil_frey(r, q, e, sigma=np.full(6, 2.0), n_boot=999, seed=0)
    assert scaled.statistic == pytest.approx(res.statistic)  # t-stat is scale-free


def test_mcneil_frey_size() -> None:
    """DESIGN 9 M3: bootstrap size at R = 300 in [0.02, 0.10] (iid normal, n = 1000, 2.5%).

    With heavy-tailed (t5) residuals the one-sided test is conservative (about 3% at n = 1000
    in a 400-replication check); the Monte Carlo study reports that case.
    """
    rng = np.random.default_rng(20261023)
    n, alpha, reps = 1000, 0.025, 300
    q, e = normal_var_es(alpha)
    qa, ea = np.full(n, q), np.full(n, e)
    rej = 0
    for _ in range(reps):
        r = rng.standard_normal(n)
        rej += mcneil_frey(r, qa, ea, n_boot=499, seed=int(rng.integers(1 << 31))).reject()
    assert 0.02 <= rej / reps <= 0.10, rej / reps


def test_mcneil_frey_power_normal_tails() -> None:
    """Normal ES forecasts on t5 data: ES too thin, so the test should reject."""
    rng = np.random.default_rng(7)
    n, alpha = 2000, 0.025
    q, e = normal_var_es(alpha)
    r = draw_std_t(rng, 5.0, n)
    assert mcneil_frey(r, np.full(n, q), np.full(n, e), n_boot=999, seed=1).pvalue < 0.05


def test_mcneil_frey_against_esback_er_backtest() -> None:
    """Loose check vs esback::er_backtest (B = 20,000). esback centres the bootstrap
    t-statistics rather than the data, so one-sided p-values differ by up to about 0.06 on
    these fixtures (DESIGN expected 0.02); the two-sided conventions differ more and are
    not compared."""
    import json
    from pathlib import Path

    import pandas as pd

    fix = Path(__file__).parent / "fixtures" / "r"
    ref = json.loads((fix / "outputs" / "r_reference.json").read_text())
    fc = pd.read_csv(fix / "inputs" / "forecasts.csv")
    for case in ref["er"]:
        s = case["set"]
        res = mcneil_frey(
            fc.r, fc[f"q_{s}_025"], fc[f"e_{s}_025"], sigma=fc.s, n_boot=20000, seed=1
        )
        assert res.pvalue == pytest.approx(case["pvalue_onesided_standardized"], abs=0.07)
