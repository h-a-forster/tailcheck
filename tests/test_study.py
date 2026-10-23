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


def _toml(tmp: Path, models: str = '["hs", "gjr-t"]') -> Path:
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
models = {models}
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


# --- compare stage only runs on the full model set (resume bug) -------------------------


def test_compare_waits_for_all_forecasts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for i, a in enumerate(("AAA", "BBB")):
        _write_synthetic(tmp_path / "data", a, 360, i)
    cfg_path = _toml(tmp_path, models='["hs", "gjr-t", "fhs"]')
    real = runner.forecast_unit

    def flaky(arg: dict[str, object]) -> dict[str, object]:
        if arg["asset"] == "AAA" and arg["model"] == "hs":
            raise RuntimeError("simulated failure")
        return real(arg)

    monkeypatch.setattr(runner, "forecast_unit", flaky)
    out = runner.run_study(cfg_path, workers=1, stages=("forecast", "compare"))
    store = UnitStore(out, runner.config_hash(runner.load_config(cfg_path)))
    assert not store.done("forecast__AAA__hs")
    assert not store.done("compare__AAA__0.025")  # not run on a partial model set
    assert store.read("compare__BBB__0.025")["models"] == ["fhs", "gjr-t", "hs"]
    assert "skipped until forecasts finish: hs" in (out / "progress.log").read_text()
    monkeypatch.setattr(runner, "forecast_unit", real)
    runner.run_study(cfg_path, workers=1, stages=("forecast", "compare"))
    cmp_ = store.read("compare__AAA__0.025")
    assert cmp_["models"] == ["fhs", "gjr-t", "hs"]
    assert set(cmp_["mean_loss"]) == {"hs", "gjr-t", "fhs"}


def test_run_study_rejects_unknown_stage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        runner.run_study(_toml(tmp_path), stages=("forcast",))


def test_resume_refuses_changed_data(tmp_path: Path) -> None:
    for i, a in enumerate(("AAA", "BBB")):
        _write_synthetic(tmp_path / "data", a, 360, i)
    cfg_path = _toml(tmp_path)
    cfg = runner.load_config(cfg_path)
    UnitStore(Path(cfg["run"]["out_dir"]), runner.config_hash(cfg), inputs=runner.study_inputs(cfg))
    _write_synthetic(tmp_path / "data", "AAA", 360, 99)  # re-fetched data differ
    from tailcheck.checkpoint import ConfigMismatchError

    with pytest.raises(ConfigMismatchError, match="different inputs"):
        runner.run_study(cfg_path, workers=1, stages=("forecast",))


def test_configured_workers() -> None:
    assert runner.configured_workers({"run": {"workers": 2}}, None) == 2
    assert runner.configured_workers({"run": {"workers": 2}}, 3) == 3
    assert runner.configured_workers({}, None) == 4


# --- reports never touch results/ without a finished run --------------------------------


def test_reports_without_run_write_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tailcheck.checkpoint import RunNotFoundError
    from tailcheck.montecarlo.report import write_mc_report
    from tailcheck.study.report import write_study_report

    monkeypatch.chdir(tmp_path)
    cfg_path = _toml(tmp_path)
    committed = tmp_path / "results" / "smoke" / "backtests.csv"
    committed.parent.mkdir(parents=True)
    committed.write_text("asset,model\nAAA,hs\n")
    with pytest.raises(RunNotFoundError):
        write_study_report(cfg_path)
    assert committed.read_text() == "asset,model\nAAA,hs\n"
    assert not (tmp_path / "runs").exists()
    # an existing but empty run directory: clear error, still nothing written
    cfg = runner.load_config(cfg_path)
    UnitStore(Path(cfg["run"]["out_dir"]), runner.config_hash(cfg))
    with pytest.raises(ValueError, match="no completed backtest units"):
        write_study_report(cfg_path)
    assert sorted(p.name for p in committed.parent.iterdir()) == ["backtests.csv"]
    mc = tmp_path / "mc.toml"
    mc.write_text(
        'name = "mc-x"\nseed = 1\n[cheap]\ndgps_size = []\n[esr]\ndgps_size = []\n'
        f'[run]\nout_dir = "{(tmp_path / "runs" / "mc-x").as_posix()}"\n'
    )
    with pytest.raises(RunNotFoundError):
        write_mc_report(mc)
    assert not (tmp_path / "results" / "mc-x").exists()
    assert not (tmp_path / "runs" / "mc-x").exists()


# --- Acerbi-Szekely Z2 path of the study ------------------------------------------------


def test_as_accumulators_match_direct_computation() -> None:
    rng = np.random.default_rng(3)
    days, m, alphas = 40, 300, (0.01, 0.025)
    sigma = np.exp(rng.normal(0, 0.3, days))
    from tailcheck.dist import normal_var_es

    q = np.column_stack([normal_var_es(a)[0] * sigma for a in alphas])
    e = np.column_stack([normal_var_es(a)[1] * sigma for a in alphas])
    draws = sigma[:, None] * rng.standard_normal((days, m))
    turb = (rng.random(days) < 0.4).astype(np.int8)
    acc_s = np.zeros((3, 2, m))
    acc_n = np.zeros_like(acc_s)
    # two blocks, as the rolling driver delivers them
    runner.accumulate_as_null(acc_s, acc_n, q[:25], e[:25], draws[:25], turb[:25])
    runner.accumulate_as_null(acc_s, acc_n, q[25:], e[25:], draws[25:], turb[25:])
    from tailcheck.es_tests import acerbi_szekely_z2

    masks = (np.ones(days, bool), turb == 0, turb == 1)
    for g, mask in enumerate(masks):
        for j, a in enumerate(alphas):
            # Z2 of each simulated path (one draw column), computed with the public statistic
            direct = np.array(
                [acerbi_szekely_z2(draws[mask, k], q[mask, j], e[mask, j], a) for k in range(m)]
            )
            z_null = 1.0 - acc_s[g, j] / (mask.sum() * a)
            np.testing.assert_allclose(z_null, direct, rtol=1e-12, atol=1e-12)
            hits = (draws[mask] <= q[mask, j][:, None]).sum(axis=0)
            np.testing.assert_array_equal(acc_n[g, j], hits)


def test_as_z2_regime_null_calibrated() -> None:
    # iid normal returns with the true normal forecasts and true predictive draws: the null
    # has mean about 0 and the p-values are roughly uniform
    from scipy import stats

    from tailcheck.dist import normal_var_es

    rng = np.random.default_rng(11)
    n, m, alpha, reps = 250, 400, 0.025, 300
    qn, en = normal_var_es(alpha)
    q, e = np.full(n, qn), np.full(n, en)
    turb = (np.arange(n) % 2).astype(np.int8)
    pvals, means = [], []
    for _ in range(reps):
        r = rng.standard_normal(n)
        acc_s = np.zeros((3, 1, m))
        acc_n = np.zeros_like(acc_s)
        runner.accumulate_as_null(
            acc_s, acc_n, q[:, None], e[:, None], rng.standard_normal((n, m)), turb
        )
        res = runner.as_z2_regime(r, q, e, alpha, np.ones(n, bool), acc_s[0, 0])
        pvals.append(res["pvalue"])
        means.append(res["null_mean"])
    assert abs(np.mean(means)) < 0.02
    assert stats.kstest(pvals, "uniform").pvalue > 0.01
    assert 0.02 < np.mean(np.asarray(pvals) < 0.05) < 0.09


# --- downloads: atomic, checksum retry, finite prices (offline) -------------------------


def _kline_zip(close: float = 100.0) -> bytes:
    return _zip_csv([[1704067200000, 1, 2, 0.5, close, 10]])


def test_binance_checksum_retry_and_malformed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from tailcheck import data

    good = _kline_zip()
    digest = hashlib.sha256(good).hexdigest()
    calls: list[str] = []
    served = {"zip": [b"corrupt", good], "sum": [b"", f"{digest}  x.zip\n".encode()]}

    def fake_get(url: str, retries: int = 3, timeout: float = 60.0) -> bytes:
        kind = "sum" if url.endswith(".CHECKSUM") else "zip"
        calls.append(kind)
        queue = served[kind]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(data, "_get", fake_get)
    with pytest.raises(data.DataError, match="malformed checksum"):
        data.fetch_binance_daily("XUSDT", "2024-01-01", "2024-01-01", tmp_path)
    # second call: valid checksum, first zip corrupt -> both refetched once, then accepted
    s = data.fetch_binance_daily("XUSDT", "2024-01-01", "2024-01-01", tmp_path)
    assert s.iloc[0] == 100.0
    assert calls.count("zip") == 2
    # a cached truncated checksum file is replaced, not trusted forever
    root = tmp_path / "raw" / "binance" / "XUSDT"
    (root / "XUSDT-1d-2024-01.zip.CHECKSUM").write_text("abc")
    s = data.fetch_binance_daily("XUSDT", "2024-01-01", "2024-01-01", tmp_path)
    assert (root / "XUSDT-1d-2024-01.zip.CHECKSUM").read_text().startswith(digest)


def test_binance_persistent_mismatch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tailcheck import data

    def fake_get(url: str, retries: int = 3, timeout: float = 60.0) -> bytes:
        return ("0" * 64).encode() if url.endswith(".CHECKSUM") else _kline_zip()

    monkeypatch.setattr(data, "_get", fake_get)
    with pytest.raises(data.DataError, match="checksum mismatch"):
        data.fetch_binance_daily("XUSDT", "2024-01-01", "2024-01-01", tmp_path)
    assert list((tmp_path / "raw" / "binance" / "XUSDT").glob("*")) == []


def test_non_finite_prices_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tailcheck import data

    with pytest.raises(data.DataError, match="non-finite"):
        parse_binance_zip(_zip_csv([[1704067200000, 1, 2, 0.5, "nan", 10]]))
    with pytest.raises(ValueError, match="finite"):
        log_returns_pct(pd.Series([1.0, np.nan, 2.0]))
    # FRED: a truncated download is rejected before it reaches the cache
    monkeypatch.setattr(data, "_get", lambda url, **_: b"observation_date,SP500\n2020-01-0")
    with pytest.raises(data.DataError):
        data.fetch_fred("SP500", tmp_path)
    assert not (tmp_path / "raw" / "fred" / "SP500.csv").exists()
    ok = b"observation_date,SP500\n2020-01-02,3257.85\n2020-01-03,.\n2020-01-06,3246.28\n"
    monkeypatch.setattr(data, "_get", lambda url, **_: ok)
    s = data.fetch_fred("SP500", tmp_path)
    assert len(s) == 2 and (tmp_path / "raw" / "fred" / "SP500.csv").read_bytes() == ok


def test_downloads_written_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tailcheck import checkpoint, data

    ok = b"observation_date,X\n2020-01-02,1.0\n2020-01-03,2.0\n"
    monkeypatch.setattr(data, "_get", lambda url, **_: ok)

    def crash(src: str, dst: str) -> None:
        raise OSError("interrupted")

    monkeypatch.setattr(checkpoint.os, "replace", crash)
    with pytest.raises(OSError):
        data.fetch_fred("X", tmp_path)
    assert list((tmp_path / "raw" / "fred").glob("*")) == []  # no truncated snapshot left
