"""M5: FZ0 loss, Diebold-Mariano and the MCS wrapper."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tailcheck.dist import normal_var_es
from tailcheck.scoring import (
    default_lags,
    diebold_mariano,
    fz0_loss,
    mean_with_ci,
    model_confidence_set,
    newey_west_lrv,
)


def test_fz0_by_hand() -> None:
    r, q, e, a = np.array([-3.0, 1.0]), np.array([-2.0, -2.0]), np.array([-2.5, -2.5]), 0.05
    expected = [
        -(1 / (a * -2.5)) * (-2 + 3) + 0.8 + math.log(2.5) - 1,
        0.8 + math.log(2.5) - 1,
    ]
    np.testing.assert_allclose(fz0_loss(r, q, e, a), expected)


def test_fz0_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        fz0_loss([0.0], [-1.0], [-0.5], 0.05)  # e > q
    with pytest.raises(ValueError):
        fz0_loss([0.0], [0.5], [-0.5], 0.05)  # q >= 0


def test_fz0_minimised_at_truth() -> None:
    rng = np.random.default_rng(0)
    r = rng.standard_normal(400_000)
    a = 0.025
    q0, e0 = normal_var_es(a)
    best = None
    for dq in (-0.2, 0.0, 0.2):
        for de in (-0.2, 0.0, 0.2):
            q, e = q0 + dq, e0 + de
            if e > q:
                continue
            m = fz0_loss(r, np.full_like(r, q), np.full_like(r, e), a).mean()
            if best is None or m < best[0]:
                best = (m, dq, de)
    assert best is not None and best[1:] == (0.0, 0.0)


def test_newey_west_by_hand() -> None:
    d = np.array([1.0, -1.0, 2.0, 0.0])
    u = d - d.mean()  # [0.5, -1.5, 1.5, -0.5]
    g0 = (u @ u) / 4
    g1 = (u[1:] @ u[:-1]) / 4
    assert newey_west_lrv(d, 1) == pytest.approx(g0 + 2 * 0.5 * g1)
    assert default_lags(1000) == math.floor(4 * 10 ** (2 / 9))


def test_dm_hand_computed() -> None:
    rng = np.random.default_rng(1)
    la = rng.standard_normal(300)
    lb = la + 0.2 + 0.5 * rng.standard_normal(300)
    res = diebold_mariano(la, lb, lags=3, small_sample=False)
    d = la - lb
    dm = d.mean() / math.sqrt(newey_west_lrv(d, 3) / 300)
    assert res.statistic == pytest.approx(dm)
    assert res.pvalue == pytest.approx(2 * stats.norm.sf(abs(dm)))
    assert res.statistic < 0 and res.details["pvalue_less"] < 0.05
    hln = diebold_mariano(la, lb, lags=3)
    assert hln.statistic == pytest.approx(dm * math.sqrt((300 - 1) / 300))


def test_dm_identical_losses() -> None:
    x = np.ones(10)
    assert math.isnan(diebold_mariano(x, x).pvalue)


def test_mean_with_ci() -> None:
    x = np.random.default_rng(2).standard_normal(5000)
    ci = mean_with_ci(x)
    assert ci["lo"] < 0 < ci["hi"] and ci["se"] == pytest.approx(1 / math.sqrt(5000), rel=0.1)


def test_mcs_keeps_best_model() -> None:
    rng = np.random.default_rng(3)
    t = 500
    base = rng.standard_normal(t)
    losses = pd.DataFrame(
        {
            "best": base + 1.0,
            "close": base + 1.02 + 0.3 * rng.standard_normal(t),
            "bad": base + 1.5 + 0.3 * rng.standard_normal(t),
            "worse": base + 2.0 + 0.3 * rng.standard_normal(t),
        }
    )
    res = model_confidence_set(losses, size=0.10, reps=500, block_size=5, seed=1)
    assert "best" in res.included
    assert "worse" in res.excluded and "bad" in res.excluded
    assert set(res.pvalues) == set(losses.columns)
    assert res.reps == 500 and res.seed == 1
    with pytest.raises(ValueError):
        model_confidence_set(losses[["best"]])
