"""M8: Monte Carlo DGPs, runner and report on a tiny configuration."""

from __future__ import annotations

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
    v = simulate("slow-vol", 100, 0.025, rng)
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
bootstrap = 0
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
    run_mc(cfg, workers=1)
    df = collect_mc(cfg)
    assert set(df["test"]) >= {"kupiec", "as_z2", "esr_v1", "esr_v3", "esr_v1_esback_g2pp"}
    k = df[(df["test"] == "kupiec") & (df["dgp"] == "iid-t5")].iloc[0]
    assert k["reps"] == 10 and k["kind"] == "size"
    monkeypatch.chdir(tmp_path)
    write_mc_report(cfg)
    out = pd.read_csv(tmp_path / "results" / "mc-test" / "size_power.csv")
    assert len(out) == len(df)
    assert (tmp_path / "results" / "mc-test" / "size_power.md").exists()
