"""M1: resumable unit store."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tailcheck import checkpoint
from tailcheck.checkpoint import ConfigMismatchError, UnitStore, config_hash, run_units, safe_id

_ENV_BEFORE = {v: os.environ.get(v) for v in checkpoint._THREAD_VARS}


def _square(x: int) -> dict[str, object]:
    return {"value": x * x}


def _fail_on_three(x: int) -> dict[str, object]:
    if x == 3:
        raise RuntimeError("boom")
    return {"value": x}


def test_write_read_done(tmp_path: Path) -> None:
    store = UnitStore(tmp_path, "abc")
    assert not store.done("u1")
    store.write("u1", {"x": 1.5, "arr": np.arange(3), "nan": float("nan")})
    assert store.done("u1")
    d = store.read("u1")
    assert d["x"] == 1.5 and d["arr"] == [0, 1, 2] and d["nan"] is None
    assert store.completed() == ["u1"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["config_hash"] == "abc" and "numpy" in manifest["versions"]


def test_arrays_roundtrip(tmp_path: Path) -> None:
    store = UnitStore(tmp_path, "abc")
    store.write_array("a/b", {"x": np.linspace(0, 1, 5)})
    np.testing.assert_allclose(store.read_array("a/b")["x"], np.linspace(0, 1, 5))


def test_hash_mismatch_refused(tmp_path: Path) -> None:
    UnitStore(tmp_path, "abc").write("u", {})
    with pytest.raises(ConfigMismatchError):
        UnitStore(tmp_path, "different")
    fresh = UnitStore(tmp_path, "different", force_new=True)
    assert not fresh.done("u")


def test_config_hash_stable() -> None:
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_safe_id_unique() -> None:
    assert safe_id("forecast__BTC__hs") == "forecast__BTC__hs"
    assert safe_id("a/b") != safe_id("a:b")


def test_atomic_write_survives_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = UnitStore(tmp_path, "abc")

    def crash(src: str, dst: str) -> None:
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(checkpoint.os, "replace", crash)
    with pytest.raises(OSError):
        store.write("u1", {"x": 1})
    monkeypatch.undo()
    assert not store.done("u1")
    assert list(store.units_dir.glob("*")) == []  # temp file cleaned, nothing half written


def test_truncated_file_counts_as_not_done(tmp_path: Path) -> None:
    store = UnitStore(tmp_path, "abc")
    store.path("u1").write_text('{"unit": "u1", "x"', encoding="utf-8")
    assert not store.done("u1")


def test_run_units_skip_and_resume(tmp_path: Path) -> None:
    store = UnitStore(tmp_path, "h")
    units = [(f"u{i}", i) for i in range(5)]
    stats = run_units(store, units[:2], _square)
    assert stats["ran"] == 2
    stats = run_units(store, units, _square)
    assert stats == {"total": 5, "skipped": 2, "ran": 3, "failed": 0, "stopped": 0}
    assert store.read("u4")["value"] == 16
    log = store.log_path.read_text().strip().splitlines()
    assert len(log) == 5 and log[-1].endswith("5/5")


def test_run_units_failure_logged_and_retried(tmp_path: Path) -> None:
    store = UnitStore(tmp_path, "h")
    units = [(f"u{i}", i) for i in range(5)]
    stats = run_units(store, units, _fail_on_three, on_error="log")
    assert stats["failed"] == 1 and stats["ran"] == 4
    assert not store.done("u3")
    assert "FAILED" in store.log_path.read_text()
    with pytest.raises(RuntimeError):
        run_units(store, units, _fail_on_three, on_error="raise")


def test_run_units_process_pool(tmp_path: Path) -> None:
    store = UnitStore(tmp_path, "h")
    units = [(f"u{i}", i) for i in range(6)]
    stats = run_units(store, units, _square, workers=99)  # capped at 4
    assert stats["ran"] == 6
    assert [store.read(f"u{i}")["value"] for i in range(6)] == [i * i for i in range(6)]
    assert checkpoint.MAX_WORKERS == 4


def _blas_threads(_: int) -> dict[str, object]:
    from threadpoolctl import threadpool_info

    return {"threads": [p["num_threads"] for p in threadpool_info()]}


def test_workers_use_one_blas_thread(tmp_path: Path) -> None:
    # numpy is already imported when the initializer runs, so setting the environment
    # variables there alone leaves OpenBLAS with its full pool
    store = UnitStore(tmp_path, "h")
    run_units(store, [("a", 0), ("b", 1)], _blas_threads, workers=2)
    for uid in ("a", "b"):
        threads = store.read(uid)["threads"]
        assert threads and all(t == 1 for t in threads)  # type: ignore[union-attr]
    for var in checkpoint._THREAD_VARS:  # the parent environment is restored
        assert var not in os.environ or os.environ[var] == _ENV_BEFORE.get(var)


def test_inputs_mismatch_refused(tmp_path: Path) -> None:
    UnitStore(tmp_path, "abc", inputs={"data": {"A": "1"}}).write("u", {})
    UnitStore(tmp_path, "abc", inputs={"data": {"A": "1"}})  # same inputs: fine
    with pytest.raises(ConfigMismatchError, match="different inputs"):
        UnitStore(tmp_path, "abc", inputs={"data": {"A": "2"}})
    fresh = UnitStore(tmp_path, "abc", inputs={"data": {"A": "2"}}, force_new=True)
    assert not fresh.done("u")


def test_changed_code_warns_on_resume(tmp_path: Path) -> None:
    UnitStore(tmp_path, "abc").write("u", {})
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["source_hash"] == checkpoint.source_hash()
    m["source_hash"] = "0" * 16
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.warns(UserWarning, match="changed code"):
        store = UnitStore(tmp_path, "abc")
    assert store.done("u")
    assert "changed code" in store.log_path.read_text()


def test_read_only_creates_nothing(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(checkpoint.RunNotFoundError):
        UnitStore(missing, "abc", read_only=True)
    assert not missing.exists()
    UnitStore(tmp_path, "abc").write("u", {})
    ro = UnitStore(tmp_path, "abc", read_only=True)
    assert ro.completed() == ["u"]
    with pytest.raises(ConfigMismatchError):
        UnitStore(tmp_path, "other", read_only=True)
