"""Bootstrap ESR Monte Carlo (warp-speed and exact modes) and the run time budget."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest

from tailcheck.checkpoint import UnitStore, run_units
from tailcheck.esr_backtest import esr_backtest
from tailcheck.montecarlo.dgp import simulate
from tailcheck.montecarlo.esr_boot import (
    boot_unit,
    build_boot_units,
    collect_boot,
    run_boot_mc,
    warp_interval,
    warp_rate,
    write_boot_report,
)


def test_warp_rate_upper_and_lower_tail() -> None:
    rng = np.random.default_rng(0)
    t_star = rng.standard_normal(20_000)
    t0 = rng.standard_normal(20_000)
    rate, crit = warp_rate(t0, t_star, 0.05, lower_tail=True)
    assert crit == pytest.approx(-1.645, abs=0.05)
    assert rate == pytest.approx(0.05, abs=0.007)
    rate, crit = warp_rate(t0**2, t_star**2, 0.05, lower_tail=False)
    assert crit == pytest.approx(3.84, abs=0.2)
    assert rate == pytest.approx(0.05, abs=0.007)


def test_warp_rate_undefined_values() -> None:
    t0 = np.array([-5.0, np.nan, 0.0, 0.0])
    t_star = np.array([np.nan, -1.0, 1.0, 0.0])
    rate, crit = warp_rate(t0, t_star, 0.05, lower_tail=True)
    assert crit == pytest.approx(np.quantile([-1.0, 1.0, 0.0], 0.05))
    assert rate == 0.25  # the undefined T0 counts as a non-rejection
    assert math.isnan(warp_rate(t0, np.full(4, np.nan), 0.05, True)[0])


def test_warp_interval_covers_rate() -> None:
    rng = np.random.default_rng(1)
    t0, t_star = rng.standard_normal(400), rng.standard_normal(400)
    rate, _ = warp_rate(t0, t_star, 0.05, True)
    lo, hi = warp_interval(t0, t_star, 0.05, True, n_resample=300)
    assert lo <= rate <= hi
    assert hi - lo < 0.08


def _cfg(tmp_path: Path, cells: list[dict[str, object]], chunk: int = 2) -> Path:
    lines = ['name = "boot-test"', "seed = 3", "level = 0.05", ""]
    for c in cells:
        lines.append("[[cells]]")
        for k, v in c.items():
            lines.append(f"{k} = {v!r}".replace("'", '"'))
        lines.append("")
    out = (tmp_path / "run").as_posix()
    lines += ["[run]", "workers = 1", f"chunk = {chunk}", f'out_dir = "{out}"']
    p = tmp_path / "cfg.toml"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_build_boot_units_validation(tmp_path: Path) -> None:
    from tailcheck.study.runner import load_config

    base = {"dgp": "garch-t5", "n": 1000, "alpha": 0.025, "versions": [3], "reps": 5}
    cfg = load_config(
        _cfg(tmp_path, [{"mode": "warp", **base}, {"mode": "exact", "n_boot": 19, **base}])
    )
    units = build_boot_units(cfg)
    assert [u[1]["reps"] for u in units if u[1]["mode"] == "warp"] == [2, 2, 1]
    assert units[0][1]["mode"] == "exact"  # slow units first
    assert len({u[1]["seed"] for u in units}) == len(units)
    with pytest.raises(ValueError, match="mode"):
        build_boot_units(load_config(_cfg(tmp_path, [{"mode": "fast", **base}])))
    with pytest.raises(ValueError, match="n_boot"):
        build_boot_units(load_config(_cfg(tmp_path, [{"mode": "exact", "n_boot": 5, **base}])))
    with pytest.raises(ValueError, match="duplicate"):
        build_boot_units(load_config(_cfg(tmp_path, [{"mode": "warp", **base}] * 2)))


def test_keep_boot_stats_matches_pvalue() -> None:
    s = simulate("garch-t5", 1000, 0.025, np.random.default_rng(5))
    res = esr_backtest(s.r, s.e, 0.025, 3, n_boot=19, seed=2, _keep_boot_stats=True)
    tb = np.asarray(res.details["bootstrap_stats"])
    assert len(tb) == 19 - res.details["n_boot_failed"]
    expected = (1 + np.sum(tb <= res.statistic)) / (len(tb) + 1)
    assert res.details["pvalue_bootstrap"] == pytest.approx(expected)
    plain = esr_backtest(s.r, s.e, 0.025, 3, n_boot=19, seed=2)
    assert "bootstrap_stats" not in plain.details
    assert plain.details["pvalue_bootstrap"] == res.details["pvalue_bootstrap"]


def test_boot_unit_warp_one_replication() -> None:
    arg = {
        "mode": "warp",
        "dgp": "garch-t5",
        "n": 1000,
        "alpha": 0.025,
        "reps": 1,
        "versions": [3],
        "n_boot": 1,
        "seed": 11,
    }
    out = boot_unit(arg)
    v3 = out["results"]["esr_v3"]  # type: ignore[index]
    assert len(v3["t0"]) == 1 and math.isfinite(v3["t0"][0])
    assert math.isfinite(v3["t_star"][0])
    assert out["n_boot"] == 1


@pytest.mark.slow
def test_boot_run_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    base = {"dgp": "garch-t5", "n": 1000, "alpha": 0.025, "versions": [1, 3]}
    cfg = _cfg(
        tmp_path,
        [{"mode": "warp", "reps": 4, **base}, {"mode": "exact", "reps": 1, "n_boot": 19, **base}],
    )
    run_boot_mc(cfg)
    df = collect_boot(cfg)
    assert set(df["mode"]) == {"warp", "exact"}
    assert (df.loc[df["mode"] == "warp", "reps"] == 4).all()
    text = write_boot_report(cfg)
    assert "esr_v3" in text
    assert (tmp_path / "results" / "boot-test" / "size_power.csv").exists()


def _sleep(x: float) -> dict[str, object]:
    time.sleep(x)
    return {"x": x}


def test_run_units_time_budget(tmp_path: Path) -> None:
    store = UnitStore(tmp_path / "s", "h")
    units = [(f"u{i}", 0.2) for i in range(6)]
    stats = run_units(store, units, _sleep, workers=1, max_seconds=0.3)
    assert stats["stopped"] == 1
    assert 1 <= stats["ran"] < 6
    assert "time budget reached" in (tmp_path / "s" / "progress.log").read_text("utf-8")
    stats = run_units(store, units, _sleep, workers=1)  # resumes the rest
    assert stats["skipped"] + stats["ran"] == 6 and stats["stopped"] == 0


def test_run_units_time_budget_pool(tmp_path: Path) -> None:
    store = UnitStore(tmp_path / "p", "h")
    units = [(f"u{i}", 0.5) for i in range(12)]
    stats = run_units(store, units, _sleep, workers=2, max_seconds=0.1)
    assert stats["stopped"] == 1
    assert stats["ran"] < 12  # queued units were cancelled
    assert sum(store.done(u) for u, _ in units) == stats["ran"]
