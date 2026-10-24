"""M8: Monte Carlo DGPs, runner and report on a tiny configuration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tailcheck.dist import normal_var_es, std_t_var_es
from tailcheck.montecarlo.dgp import as_null, garch_t5, simulate
from tailcheck.montecarlo.report import collect_mc, wilson, write_mc_report
from tailcheck.montecarlo.runner import build_units, run_mc


def test_garch_unconditional_variance() -> None:
    r, s = garch_t5(200_000, np.random.default_rng(0))
    assert r.var() == pytest.approx(1.0, rel=0.08)
    assert np.all(s > 0)


def test_simulate_forecast_sets() -> None:
    rng = np.random.default_rng(1)
    qt, et = std_t_var_es(5.0, 0.025)
    qn, en = normal_var_es(0.025)
    s = simulate("scale-0.85", 100, 0.025, rng)
    np.testing.assert_allclose(s.q, qt * s.sigma)  # sigma is the (scaled) forecast sigma
    n = simulate("normal-tails", 100, 0.025, rng)
    np.testing.assert_allclose(n.e / n.sigma, en)
    assert n.law == "normal"
    v = simulate("unconditional", 100, 0.025, rng)
    assert np.all(v.q == qt) and np.all(v.sigma == 1)
    with pytest.raises(ValueError):
        simulate("nope", 10, 0.025, rng)


def test_as_null_mean_zero() -> None:
    z = as_null("t5", 500, 0.025, 20_000, np.random.default_rng(2))
    assert abs(z["z2"].mean()) < 4 * z["z2"].std() / np.sqrt(20_000)


def test_wilson() -> None:
    lo, hi = wilson(50, 1000)
    assert lo == pytest.approx(0.0381, abs=1e-3) and hi == pytest.approx(0.0653, abs=1e-3)


def _cfg(tmp: Path) -> Path:
    p = tmp / "mc.toml"
    p.write_text(f"""
name = "mc-test"
seed = 1
level = 0.05
chunk = 5
[cheap]
tests = ["kupiec", "christoffersen_ind", "christoffersen_cc", "dq", "as_z1", "as_z2", "mcneil_frey"]
n = [250]
alphas = [0.025]
dgps_size = ["iid-t5"]
dgps_power = ["scale-0.70"]
reps_size = 10
reps_power = 5
as_n_sim = 200
mcneil_frey_boot = 99
[esr]
versions = [1, 3]
n = [500]
alphas = [0.025]
extra_cells = []
dgps_size = ["garch-t5"]
dgps_power = []
reps_size = 2
reps_power = 0
chunk = 2
[run]
workers = 1
out_dir = "{(tmp / "runs").as_posix()}"
""")
    return p


def test_runner_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    from tailcheck.study.runner import load_config

    units = build_units(load_config(cfg))
    assert sum(1 for u, _ in units if u.startswith("esr|")) == 1
    assert sum(1 for u, _ in units if u.startswith("cheap|")) == 3  # 2 size chunks + 1 power
    res = run_mc(cfg, workers=1)
    assert res.complete and (res.done, res.expected, res.failed) == (4, 4, 0)
    df = collect_mc(cfg)
    assert set(df["test"]) >= {"kupiec", "as_z2", "esr_v1", "esr_v3", "esr_v1_esback_g2pp"}
    k = df[(df["test"] == "kupiec") & (df["dgp"] == "iid-t5")].iloc[0]
    assert k["reps"] == 10 and k["kind"] == "size"
    monkeypatch.chdir(tmp_path)
    write_mc_report(cfg)
    out = pd.read_csv(tmp_path / "results" / "mc-test" / "size_power.csv")
    assert len(out) == len(df)
    assert (tmp_path / "results" / "mc-test" / "size_power.md").exists()
    summary = tmp_path / "results" / "mc-test" / "summary.json"
    assert json.loads(summary.read_text())["complete"] is True
    # a partial run (unit missing) is refused unless allowed, and then marked incomplete
    from tailcheck.checkpoint import IncompleteRunError, store_for

    store_for(load_config(cfg)).path(units[-1][0]).unlink()
    with pytest.raises(IncompleteRunError, match="3/4 units done"):
        write_mc_report(cfg)
    write_mc_report(cfg, allow_partial=True)
    s = json.loads(summary.read_text())
    assert (s["complete"], s["expected"], s["done"]) == (False, 4, 3)
    md = (tmp_path / "results" / "mc-test" / "size_power.md").read_text()
    assert md.startswith("**INCOMPLETE RUN: 3 of 4 units done.**")


def test_as_null_seed_differs_per_chunk(tmp_path: Path) -> None:
    from tailcheck.study.runner import load_config

    units = [u for u in build_units(load_config(_cfg(tmp_path))) if u[0].startswith("cheap|")]
    cell = [a for uid, a in units if uid.startswith("cheap|iid-t5|250|0.025|")]
    assert len(cell) == 2
    assert cell[0]["null_seeds"]["t5"] != cell[1]["null_seeds"]["t5"]


def test_distortion_band_follows_level() -> None:
    from tailcheck.montecarlo.report import distortion_band

    assert distortion_band(0.05) == pytest.approx((0.03, 0.07))
    assert distortion_band(0.10) == pytest.approx((0.06, 0.14))


def test_cluster_interval_independent_and_clustered() -> None:
    from tailcheck.montecarlo.report import cluster_interval

    rng = np.random.default_rng(4)
    m = np.full(400, 50)
    k = rng.binomial(50, 0.05, size=400)
    lo, hi, deff = cluster_interval(k, m)
    assert deff == pytest.approx(1.0, abs=0.2)
    assert lo < k.sum() / m.sum() < hi
    # a shared per-chunk shift inflates the variance
    p_chunk = np.clip(0.05 + rng.normal(0, 0.04, 400), 0, 1)
    k2 = rng.binomial(50, p_chunk)
    assert cluster_interval(k2, m)[2] > 1.8
    assert np.isnan(cluster_interval(np.array([1]), np.array([50]))[0])
