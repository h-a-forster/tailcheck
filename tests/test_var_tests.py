"""M2: VaR backtests against the closed-form reference values in DESIGN 4.1."""

from __future__ import annotations

import math

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats

from tailcheck.var_tests import (
    christoffersen_cc,
    christoffersen_independence,
    dynamic_quantile,
    hits,
    kupiec_pof,
    traffic_light,
)


def _from_hits(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns/VaR pair whose exceedance pattern is ``h``."""
    q = np.full(len(h), -1.0)
    r = np.where(np.asarray(h, dtype=bool), -2.0, 0.5)
    return r, q


def _kupiec_region(t: int, alpha: float) -> tuple[int, int]:
    acc = []
    for x in range(t + 1):
        h = np.zeros(t, dtype=bool)
        h[:x] = True
        if kupiec_pof(*_from_hits(h), alpha).pvalue > 0.05:
            acc.append(x)
    return min(acc), max(acc)


@pytest.mark.parametrize(
    ("alpha", "t", "region"),
    [
        (0.01, 250, (1, 6)),
        (0.01, 510, (2, 10)),
        (0.01, 1000, (5, 16)),
        (0.025, 250, (3, 11)),
        (0.025, 510, (7, 20)),
        (0.025, 1000, (16, 35)),
        (0.05, 250, (7, 19)),
        (0.05, 510, (17, 35)),
        (0.05, 1000, (38, 64)),
    ],
)
def test_kupiec_regions(alpha: float, t: int, region: tuple[int, int]) -> None:
    assert _kupiec_region(t, alpha) == region


def test_kupiec_edges() -> None:
    r, q = _from_hits(np.zeros(250))
    res = kupiec_pof(r, q, 0.01)
    assert res.statistic == pytest.approx(-2 * 250 * math.log(0.99))
    assert res.statistic == pytest.approx(5.025167, abs=1e-5)  # x = 0 rejected at 5%
    assert res.reject()
    r, q = _from_hits(np.ones(40))
    assert kupiec_pof(r, q, 0.05).statistic == pytest.approx(-2 * 40 * math.log(0.05))
    assert kupiec_pof(r, q, 0.05).details["hit_rate"] == 1.0


def test_christoffersen_short_example() -> None:
    h = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0])
    res = christoffersen_independence(*_from_hits(h), 0.3)
    assert (res.details["n00"], res.details["n01"], res.details["n10"], res.details["n11"]) == (
        2,
        2,
        2,
        3,
    )
    assert res.statistic == pytest.approx(0.0900142642, abs=1e-9)
    assert res.pvalue == pytest.approx(0.7641590225, abs=1e-9)


def test_christoffersen_clustered_example() -> None:
    h = np.zeros(250, dtype=bool)
    h[[10, 11, 100, 180, 181, 182]] = True
    r, q = _from_hits(h)
    ind = christoffersen_independence(r, q, 0.01)
    cc = christoffersen_cc(r, q, 0.01)
    pof = kupiec_pof(r, q, 0.01)
    assert (ind.details["n00"], ind.details["n01"], ind.details["n10"], ind.details["n11"]) == (
        240,
        3,
        3,
        3,
    )
    assert pof.statistic == pytest.approx(3.5553547711, abs=1e-9)
    assert pof.pvalue == pytest.approx(0.0593536190, abs=1e-9)
    assert ind.statistic == pytest.approx(15.9152966511, abs=1e-9)
    assert ind.pvalue == pytest.approx(6.62412e-5, rel=1e-5)
    assert cc.statistic == pytest.approx(19.4706514222, abs=1e-9)
    assert cc.pvalue == pytest.approx(5.91564e-5, rel=1e-5)
    assert cc.statistic == pytest.approx(pof.statistic + ind.statistic)


def test_christoffersen_no_hits_and_single_hit_at_end() -> None:
    res = christoffersen_independence(*_from_hits(np.zeros(100)), 0.01)
    assert res.statistic == 0.0 and res.pvalue == 1.0
    assert res.details["reason"] == "no exceedances"
    h = np.zeros(100)
    h[-1] = 1
    res = christoffersen_independence(*_from_hits(h), 0.01)
    assert res.details["n01"] == 1 and res.details["n11"] == 0
    assert math.isfinite(res.statistic) and 0 <= res.pvalue <= 1
    all_hits = christoffersen_independence(*_from_hits(np.ones(50)), 0.01)
    assert all_hits.statistic == pytest.approx(0.0, abs=1e-12)


def test_christoffersen_single_hit_on_first_day_reason() -> None:
    h = np.zeros(100)
    h[0] = 1
    res = christoffersen_independence(*_from_hits(h), 0.01)
    assert res.statistic == 0.0 and res.pvalue == 1.0
    assert res.details["hits"] == 1
    assert res.details["reason"] == "no transitions into a hit"


def test_var_tests_depend_only_on_hits() -> None:
    rng = np.random.default_rng(0)
    r = rng.standard_normal(300)
    q = np.full(300, -1.96)
    h = hits(r, q)
    r2, q2 = _from_hits(h)
    for f in (kupiec_pof, christoffersen_independence, christoffersen_cc):
        assert f(r, q, 0.025).statistic == pytest.approx(f(r2, q2, 0.025).statistic)


def test_dq_matches_ols_form() -> None:
    rng = np.random.default_rng(1)
    n, alpha, lags = 600, 0.025, 4
    sigma = np.exp(0.3 * np.sin(np.arange(n) / 40))
    r = sigma * rng.standard_normal(n)
    q = -1.96 * np.ones(n) * (1 + 0.1 * rng.standard_normal(n))
    res = dynamic_quantile(r, q, alpha, lags=lags)
    hit = (r <= q).astype(float) - alpha
    x = np.column_stack(
        [np.ones(n - lags)] + [hit[lags - j : n - j] for j in range(1, lags + 1)] + [q[lags:]]
    )
    ols = sm.OLS(hit[lags:], x).fit()
    assert res.statistic == pytest.approx(np.sum(ols.fittedvalues**2) / (alpha * (1 - alpha)))
    assert res.df == lags + 2 and res.nobs == n - lags
    no_var = dynamic_quantile(r, q, alpha, lags=lags, include_var=False)
    assert no_var.df == lags + 1


def test_dq_no_hits_is_rank_deficient() -> None:
    r, q = _from_hits(np.zeros(200))
    res = dynamic_quantile(r, q, 0.01)
    assert res.df == 1  # lag columns equal the intercept up to scale; q is constant too
    assert "warning" in res.details
    # with no hits Hit = -alpha everywhere: DQ = n alpha / (1 - alpha), the coverage part
    assert res.statistic == pytest.approx(196 * 0.01 / 0.99)


BASEL_CUM = [8.11, 28.58, 54.32, 75.81, 89.22, 95.88, 98.63, 99.60, 99.89, 99.97, 99.99]


@pytest.mark.parametrize("x", range(11))
def test_basel_table(x: int) -> None:
    h = np.zeros(250, dtype=bool)
    h[:x] = True
    tl = traffic_light(*_from_hits(h), 0.01)
    assert round(100 * tl.cumulative_probability, 2) == pytest.approx(BASEL_CUM[x], abs=0.005)
    expected = "green" if x <= 4 else ("yellow" if x <= 9 else "red")
    assert tl.zone == expected
    plus = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}.get(x, 0.0 if x <= 4 else 1.0)
    assert tl.plus_factor == pytest.approx(plus)


def test_basel_exact_and_non_standard() -> None:
    for x, c in [(4, 0.892188), (5, 0.958817), (9, 0.999750), (10, 0.999946)]:
        h = np.zeros(250, dtype=bool)
        h[:x] = True
        assert traffic_light(*_from_hits(h)).cumulative_probability == pytest.approx(c, abs=1e-6)
    tl = traffic_light(*_from_hits(np.zeros(500)), 0.01)
    assert math.isnan(tl.plus_factor) and tl.zone == "green"


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        kupiec_pof([1.0, 2.0], [1.0], 0.01)
    with pytest.raises(ValueError):
        kupiec_pof([1.0, np.nan], [1.0, 1.0], 0.01)
    with pytest.raises(ValueError):
        kupiec_pof([1.0], [1.0], 1.5)
    with pytest.raises(ValueError):
        dynamic_quantile(np.zeros(5), np.zeros(5), 0.01)


def test_fast_size_kupiec_christoffersen() -> None:
    """DESIGN 6: size at n = 500, alpha = 0.025, R = 400, iid normal with true VaR."""
    rng = np.random.default_rng(20261023)
    n, alpha, reps = 500, 0.025, 400
    q = np.full(n, stats.norm.ppf(alpha))
    rej = {"kupiec": 0, "ind": 0, "cc": 0}
    for _ in range(reps):
        r = rng.standard_normal(n)
        rej["kupiec"] += kupiec_pof(r, q, alpha).reject()
        rej["ind"] += christoffersen_independence(r, q, alpha).reject()
        rej["cc"] += christoffersen_cc(r, q, alpha).reject()
    assert 0.02 <= rej["kupiec"] / reps <= 0.09, rej
    assert 0.02 <= rej["cc"] / reps <= 0.09, rej
    # The independence LR is known to be conservative when hits are rare (about 12 expected
    # here, so n11 is usually 0 and the chi2(1) approximation is poor): only an upper bound.
    assert rej["ind"] / reps <= 0.09, rej
