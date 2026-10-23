"""M7/M9: study smoke test on synthetic data, including resume, and data parsing."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tailcheck.checkpoint import UnitStore
from tailcheck.data import (
    load_returns,
    log_returns_pct,
    parse_binance_zip,
    series_sha256,
    timestamp_unit,
)
from tailcheck.study import runner


def _write_synthetic(cache: Path, asset: str, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    r = 1.2 * rng.standard_t(5, n) * np.sqrt(3 / 5)
    dates = pd.bdate_range("2019-01-01", periods=n)
    out = cache / "processed" / f"{asset}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": dates, "r": r}).to_csv(out, index=False)


def _toml(tmp: Path) -> Path:
    cfg = f"""
name = "smoke"
seed = 7
[data]
cache_dir = "{(tmp / "data").as_posix()}"
end = "2030-01-01"
[[data.assets]]
id = "AAA"
source = "fred"
class = "equity"
[[data.assets]]
id = "BBB"
source = "fred"
class = "crypto"
[forecast]
alphas = [0.025]
window = 300
oos_start = "{pd.bdate_range("2019-01-01", periods=301)[-1].date()}"
oos_end = "2030-01-01"
refit_every = 20
models = ["hs", "gjr-t"]
n_draws = 200
evt_tail_fraction = 0.10
caviar_random_starts = 50
caviar_best_starts = 2
[backtest]
esr_versions = [1, 3]
esr_bootstrap = 0
mcneil_frey_boot = 199
dq_lags = 4
[compare]
mcs_size = [0.10]
mcs_reps = 200
mcs_method = "R"
mcs_block_size = 5
dm_small_sample = true
[regimes]
vol_lookback = 63
[run]
workers = 1
out_dir = "{(tmp / "runs" / "smoke").as_posix()}"
"""
    p = tmp / "smoke.toml"
    p.write_text(cfg)
    return p


def test_study_smoke_and_resume(tmp_path: Path) -> None:
    _smoke(tmp_path)


def _smoke(tmp_path: Path) -> None:
    for i, a in enumerate(("AAA", "BBB")):
        _write_synthetic(tmp_path / "data", a, 360, i)
    cfg_path = _toml(tmp_path)
    out = runner.run_study(cfg_path, workers=1, stages=("forecast",))
    store = UnitStore(out, runner.config_hash(runner.load_config(cfg_path)))
    done = set(store.completed())
    assert done == {f"forecast__{a}__{m}" for a in ("AAA", "BBB") for m in ("hs", "gjr-t")}
    # simulate a crash: drop one finished unit, rerun, and check only it is recomputed
    store.path("forecast__AAA__hs").unlink()
    before = store.read("forecast__BBB__gjr-t")["seconds"]
    runner.run_study(cfg_path, workers=1)
    assert store.read("forecast__BBB__gjr-t")["seconds"] == before
    assert store.done("forecast__AAA__hs")
    bt = store.read("backtest__AAA__gjr-t__0.025")
    assert bt["n"] == 60 and "esr_v1" in bt and "regime_calm" in bt
    assert "as_z2_full" in bt
    cmp_ = store.read("compare__BBB__0.025")
    assert set(cmp_["mean_loss"]) == {"hs", "gjr-t"}
    log = (out / "progress.log").read_text().splitlines()
    assert any("forecast__AAA__hs" in line for line in log)
    from tailcheck.study.report import collect_study

    tables = collect_study(cfg_path)
    assert len(tables["backtests"]) == 4
    assert {"esr_v1_p_holm", "esr_v1_q_bh"} <= set(tables["backtests"].columns)


def test_unit_seed_independent_of_order() -> None:
    assert runner.unit_seed(1, "a") == runner.unit_seed(1, "a")
    assert runner.unit_seed(1, "a") != runner.unit_seed(1, "b")


def _zip_csv(rows: list[list[object]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.csv", "\n".join(",".join(map(str, r)) for r in rows))
    return buf.getvalue()


def test_binance_timestamp_units() -> None:
    ms = 1704067200000  # 2024-01-01 in milliseconds
    us = 1735689600000000  # 2025-01-01 in microseconds
    assert timestamp_unit(np.array([ms])) == "ms"
    assert timestamp_unit(np.array([us])) == "us"
    s1 = parse_binance_zip(_zip_csv([[ms, 1, 2, 0.5, 42000.0, 10]]))
    s2 = parse_binance_zip(_zip_csv([[us, 1, 2, 0.5, 93000.0, 10]]))
    assert s1.index[0] == pd.Timestamp("2024-01-01") and s1.iloc[0] == 42000.0
    assert s2.index[0] == pd.Timestamp("2025-01-01") and s2.iloc[0] == 93000.0
    s3 = parse_binance_zip(
        _zip_csv([["open_time", "o", "h", "l", "close", "v"], [us, 1, 2, 0.5, 1.0, 3]])
    )
    assert s3.iloc[0] == 1.0


def test_log_returns_and_hash(tmp_path: Path) -> None:
    p = pd.Series([100.0, 110.0, 99.0], index=pd.date_range("2020-01-01", periods=3))
    r = log_returns_pct(p)
    assert r.iloc[0] == pytest.approx(100 * np.log(1.1))
    assert len(r) == 2
    assert series_sha256(r) == series_sha256(r.copy())
    with pytest.raises(ValueError):
        log_returns_pct(pd.Series([1.0, -1.0]))
    out = tmp_path / "processed" / "X.csv"
    out.parent.mkdir(parents=True)
    r.rename("r").to_frame().to_csv(out, index_label="date")
    np.testing.assert_allclose(load_returns(tmp_path, "X").to_numpy(), r.to_numpy())
    json.dumps({"ok": True})
