"""M4c: ESR backtests against esback 0.3.1 reference p-values."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tailcheck.dist import draw_std_t, std_t_var_es
from tailcheck.esr_backtest import esr_backtest

FIX = Path(__file__).parent / "fixtures" / "r"
REF = json.loads((FIX / "outputs" / "r_reference.json").read_text())
FC = pd.read_csv(FIX / "inputs" / "forecasts.csv")
FAST = {("true", 0.025, 1, "scl_N"), ("normal", 0.025, 3, "scl_sp"), ("scale", 0.01, 2, "scl_N")}


def _case_id(c: dict) -> str:
    return f"{c['set']}-{c['alpha']}-v{c['version']}-{c['sigma_est']}"


def _run(case: dict, compat: bool) -> float:
    tag = "025" if case["alpha"] == 0.025 else "01"
    res = esr_backtest(
        FC.r,
        FC[f"e_{case['set']}_{tag}"],
        case["alpha"],
        case["version"],
        q=FC[f"q_{case['set']}_{tag}"],
        sigma_est=case["sigma_est"],
        seed=1,
        _esreg_compat_g2pp=compat,
    )
    return float(res.details["pvalue_two_sided"])


CASES = [
    pytest.param(
        c,
        id=_case_id(c),
        marks=[]
        if (c["set"], c["alpha"], c["version"], c["sigma_est"]) in FAST
        else [pytest.mark.slow],
    )
    for c in REF["esr"]
]


@pytest.mark.parametrize("case", CASES)
def test_esr_pvalues_match_esback_in_compat_mode(case: dict) -> None:
    """With esreg's G2'' reproduced, p-values match esback (DESIGN 7 tolerances)."""
    tol = 0.01  # observed maximum 0.006 over all 36 cases
    assert _run(case, compat=True) == pytest.approx(case["pvalue_twosided_asymptotic"], abs=tol)


def test_esr_default_differs_from_esback_through_g2pp() -> None:
    """Documented discrepancy: with the correct G2'' the strict ESR p-value moves (here by
    about 0.25 at alpha = 1%); see docs/esreg-covariance.md."""
    case = next(
        c
        for c in REF["esr"]
        if (c["set"], c["alpha"], c["version"], c["sigma_est"]) == ("true", 0.01, 1, "scl_N")
    )
    ours = _run(case, compat=False)
    assert abs(ours - case["pvalue_twosided_asymptotic"]) > 0.1


def test_esr_v3_one_sided_and_details() -> None:
    res = esr_backtest(FC.r, FC.e_normal_025, 0.025, 3, sigma_est="scl_N", seed=1)
    assert res.alternative == "less" and res.df is None
    assert res.pvalue == res.details["pvalue_less"]
    assert res.statistic < 0  # normal ES is too thin for t5 data
    assert len(res.details["coef_e"]) == 1


def test_esr_too_few_hits_is_nan() -> None:
    r = np.random.default_rng(0).standard_normal(100)
    res = esr_backtest(r, np.full(100, -2.3) + 0.01 * r, 0.025)
    assert math.isnan(res.pvalue) and "too few" in res.details["reason"]


def test_esr_version2_needs_q() -> None:
    with pytest.raises(ValueError):
        esr_backtest(FC.r, FC.e_true_025, 0.025, 2)
    with pytest.raises(ValueError):
        esr_backtest(FC.r, FC.e_true_025, 0.025, 4)  # type: ignore[arg-type]


def test_esr_bootstrap_seeded() -> None:
    n = 600
    a = esr_backtest(FC.r[:n], FC.e_true_025[:n], 0.025, 3, n_boot=9, sigma_est="scl_N", seed=3)
    b = esr_backtest(FC.r[:n], FC.e_true_025[:n], 0.025, 3, n_boot=9, sigma_est="scl_N", seed=3)
    assert a.details["pvalue_bootstrap"] == b.details["pvalue_bootstrap"]
    assert 0 < a.details["pvalue_bootstrap"] <= 1
    assert a.details["n_boot"] == 9


@pytest.mark.slow
def test_esr_v1_size_small_mc() -> None:
    """ESR v1 asymptotic size at R = 60, n = 1000, iid t5 with true forecasts: loose bound
    (the test is known to be oversized in small samples)."""
    rng = np.random.default_rng(20261023)
    n, a, reps = 1000, 0.025, 60
    q, e = std_t_var_es(5.0, a)
    rej = 0
    for _ in range(reps):
        s = np.exp(0.3 * rng.standard_normal(n))
        r = s * draw_std_t(rng, 5.0, n)
        rej += esr_backtest(r, e * s, a, 1, sigma_est="scl_N", seed=1).reject()
    assert rej / reps <= 0.25


# --- bootstrap p-values against esback's bootstrap (distributional check) ---------------

BOOT_REF_PATH = FIX / "outputs" / "r_esr_bootstrap.json"
BOOT_REF = json.loads(BOOT_REF_PATH.read_text()) if BOOT_REF_PATH.exists() else {"cases": []}


@pytest.mark.slow
@pytest.mark.parametrize(
    "case", BOOT_REF["cases"], ids=[f"{c['set']}-v{c['version']}" for c in BOOT_REF["cases"]]
)
def test_esr_bootstrap_pvalues_match_esback(case: dict) -> None:
    """Same data and B as esback's bootstrap; the random streams differ, so p-values agree
    only within Monte Carlo error (sd of a difference <= 0.032 at B = 499). The misspecified
    ``scale`` cases have small p-values that a centring or tail-direction error would push
    towards 0.5 or 1."""
    b = int(BOOT_REF["B"])
    res = esr_backtest(
        FC.r,
        FC[f"e_{case['set']}_025"],
        case["alpha"],
        case["version"],
        q=FC[f"q_{case['set']}_025"],
        sigma_est="scl_sp",
        n_boot=b,
        seed=20261024,
        _esreg_compat_g2pp=True,
    )
    d = res.details
    assert d["n_boot_failed"] < 0.05 * b
    if case["version"] == 3:
        assert d["pvalue_bootstrap"] == pytest.approx(case["pvalue_onesided_bootstrap"], abs=0.1)
        assert d["pvalue_bootstrap_two_sided"] == pytest.approx(
            case["pvalue_twosided_bootstrap"], abs=0.1
        )
    else:
        assert d["pvalue_bootstrap"] == pytest.approx(case["pvalue_twosided_bootstrap"], abs=0.1)
