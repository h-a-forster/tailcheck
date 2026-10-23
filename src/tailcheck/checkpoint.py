"""Resumable unit store: one file per finished unit, atomic writes, skip-if-done.

Every long computation (Monte Carlo cells, study asset x model x alpha units) writes each
finished unit through this module and appends a line to a progress log. A rerun with the
same configuration skips completed units; a changed configuration hash refuses to mix.
Specification: ``docs/DESIGN.md`` section 4.6.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

MAX_WORKERS = 4
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


class ConfigMismatchError(RuntimeError):
    """The store was created for a different configuration or different inputs."""


class RunNotFoundError(FileNotFoundError):
    """A read-only store was opened on a directory without a run manifest."""


def config_hash(config: Mapping[str, Any]) -> str:
    """Stable SHA-256 (first 16 hex chars) of a JSON-serialisable configuration."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def safe_id(unit_id: str) -> str:
    """File-system safe version of a unit id (keeps letters, digits, ``-_.``)."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", unit_id)
    if s != unit_id:
        s = f"{s}-{hashlib.sha1(unit_id.encode()).hexdigest()[:8]}"
    return s


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
    return obj


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "pandas", "arch"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # pragma: no cover - optional
            out[mod] = "missing"
    try:
        from tailcheck import __version__

        out["tailcheck"] = __version__
    except Exception:  # pragma: no cover
        out["tailcheck"] = "unknown"
    return out


def source_hash() -> str:
    """SHA-256 (first 16 hex chars) of the installed tailcheck source files.

    Recorded in each run manifest so that a resume with changed code is visible.
    """
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.py")):
        h.update(f.relative_to(root).as_posix().encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


class UnitStore:
    """Directory of completed units keyed by a string id.

    Layout: ``root/manifest.json`` (config hash, versions, start time),
    ``root/units/<safe_id>.json`` (one per finished unit, written atomically) and
    ``root/progress.log`` (``ISO time | unit | seconds | done/total``).
    """

    def __init__(
        self,
        root: Path,
        config_hash: str,
        force_new: bool = False,
        inputs: Mapping[str, Any] | None = None,
        read_only: bool = False,
    ) -> None:
        """Open (or create) the store.

        ``inputs`` describes what the configuration hash does not cover (input data hashes,
        package version). It is stored in the manifest; reopening with different ``inputs``
        raises :class:`ConfigMismatchError` unless ``force_new`` is set, so a resume never
        mixes units computed on different data. The source hash of the code is recorded
        too; a change only produces a warning (and a ``progress.log`` line), because
        documentation edits should not invalidate a long run.

        ``read_only=True`` (for reports) creates nothing: it requires an existing manifest
        with the same configuration hash and raises :class:`RunNotFoundError` otherwise.
        """
        self.root = Path(root)
        self.config_hash = config_hash
        self.units_dir = self.root / "units"
        self.log_path = self.root / "progress.log"
        self.read_only = read_only
        manifest = self.root / "manifest.json"
        if read_only:
            if not manifest.exists():
                raise RunNotFoundError(f"no run found at {self.root} (manifest.json missing)")
            old = json.loads(manifest.read_text(encoding="utf-8"))
            if old.get("config_hash") != config_hash:
                raise ConfigMismatchError(
                    f"{self.root} holds results for config {old.get('config_hash')}, "
                    f"not {config_hash}"
                )
            self.manifest: dict[str, Any] = old
            return
        self.units_dir.mkdir(parents=True, exist_ok=True)
        inputs_j = _jsonable(dict(inputs)) if inputs is not None else None
        code = source_hash()
        if manifest.exists():
            old = json.loads(manifest.read_text(encoding="utf-8"))
            problem = None
            if old.get("config_hash") != config_hash:
                problem = (
                    f"{self.root} holds results for config {old.get('config_hash')}, "
                    f"not {config_hash}"
                )
            elif inputs_j is not None and old.get("inputs") != inputs_j:
                problem = (
                    f"{self.root} was computed on different inputs "
                    f"(stored {old.get('inputs')}, now {inputs_j})"
                )
            if problem is not None:
                if not force_new:
                    raise ConfigMismatchError(f"{problem}; use a new directory or force_new=True")
                for p in self.units_dir.glob("*"):
                    p.unlink()
                manifest.unlink()
            elif old.get("source_hash") != code:
                msg = (
                    f"resuming {self.root} with changed code (source hash "
                    f"{old.get('source_hash')} -> {code}); finished units are kept"
                )
                warnings.warn(msg, stacklevel=2)
                self._append_log(f"{datetime.now(UTC).isoformat(timespec='seconds')} | {msg}")
        if not manifest.exists():
            meta = {
                "config_hash": config_hash,
                "inputs": inputs_j,
                "source_hash": code,
                "versions": _versions(),
                "started": datetime.now(UTC).isoformat(timespec="seconds"),
                "platform": sys.platform,
            }
            _atomic_write_bytes(manifest, json.dumps(meta, indent=2).encode())
        self.manifest = json.loads(manifest.read_text(encoding="utf-8"))

    def path(self, unit_id: str) -> Path:
        return self.units_dir / f"{safe_id(unit_id)}.json"

    def done(self, unit_id: str) -> bool:
        p = self.path(unit_id)
        if not p.exists():
            return False
        try:
            return bool(json.loads(p.read_text(encoding="utf-8")).get("complete"))
        except (json.JSONDecodeError, OSError):
            return False

    def write(self, unit_id: str, payload: Mapping[str, object]) -> Path:
        body = {"unit": unit_id, **_jsonable(dict(payload)), "complete": True}
        p = self.path(unit_id)
        _atomic_write_bytes(p, json.dumps(body, allow_nan=False).encode())
        return p

    def read(self, unit_id: str) -> dict[str, object]:
        p = self.path(unit_id)
        data: dict[str, object] = json.loads(p.read_text(encoding="utf-8"))
        if not data.get("complete"):
            raise KeyError(f"unit {unit_id!r} is not complete")
        return data

    def write_array(self, unit_id: str, arrays: Mapping[str, np.ndarray]) -> Path:
        """Store arrays as ``units/<safe_id>.npz`` (written atomically)."""
        import io

        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        p = self.units_dir / f"{safe_id(unit_id)}.npz"
        _atomic_write_bytes(p, buf.getvalue())
        return p

    def read_array(self, unit_id: str) -> dict[str, np.ndarray]:
        with np.load(self.units_dir / f"{safe_id(unit_id)}.npz", allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def completed(self) -> list[str]:
        out: list[str] = []
        if not self.units_dir.is_dir():
            return out
        for p in sorted(self.units_dir.glob("*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if d.get("complete"):
                out.append(str(d["unit"]))
        return out

    def log(self, unit_id: str, seconds: float, n_done: int, n_total: int, note: str = "") -> None:
        line = (
            f"{datetime.now(UTC).isoformat(timespec='seconds')} | {unit_id} | "
            f"{seconds:.2f} | {n_done}/{n_total}"
        )
        if note:
            line += f" | {note}"
        self._append_log(line)

    def _append_log(self, line: str) -> None:
        if self.read_only:
            raise RuntimeError("store is read-only")
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


_THREAD_LIMITER: Any = None


def _init_worker() -> None:
    """Pin BLAS/OpenMP pools to one thread in a worker process.

    The environment variables are also set in the parent while the pool runs (see
    :func:`_single_thread_env`), so a spawned child sees them before numpy loads. This
    initializer runs after numpy is already imported (unpickling it imports this module),
    so it also limits the live pools through ``threadpoolctl``.
    """
    global _THREAD_LIMITER
    for var in _THREAD_VARS:
        os.environ[var] = "1"
    from threadpoolctl import threadpool_limits

    _THREAD_LIMITER = threadpool_limits(limits=1)


@contextlib.contextmanager
def _single_thread_env() -> Iterator[None]:
    old = {v: os.environ.get(v) for v in _THREAD_VARS}
    for v in _THREAD_VARS:
        os.environ[v] = "1"
    try:
        yield
    finally:
        for v, val in old.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


def _timed(fn: Callable[[Any], Mapping[str, object]], arg: Any) -> tuple[dict[str, object], float]:
    t0 = time.perf_counter()
    out = dict(fn(arg))
    return out, time.perf_counter() - t0


def run_units(
    store: UnitStore,
    units: Sequence[tuple[str, Any]],
    fn: Callable[[Any], Mapping[str, object]],
    workers: int = 1,
    on_error: str = "raise",
) -> dict[str, int]:
    """Run ``fn(arg)`` for every ``(unit_id, arg)`` not yet done and store each result.

    ``fn`` must be a picklable module-level function when ``workers > 1``. At most
    :data:`MAX_WORKERS` processes are used; each sets BLAS thread counts to 1. With
    ``on_error="log"`` a failing unit is logged and skipped (it stays not-done, so a rerun
    retries it); with ``"raise"`` the first failure propagates after finished units are saved.
    """
    workers = max(1, min(int(workers), MAX_WORKERS))
    total = len(units)
    todo = [(u, a) for u, a in units if not store.done(u)]
    n_done = total - len(todo)
    stats = {"total": total, "skipped": n_done, "ran": 0, "failed": 0}
    if not todo:
        return stats

    def handle(uid: str, result: dict[str, object], secs: float) -> None:
        nonlocal n_done
        store.write(uid, {**result, "seconds": secs})
        n_done += 1
        stats["ran"] += 1
        store.log(uid, secs, n_done, total)

    def fail(uid: str, exc: BaseException) -> None:
        stats["failed"] += 1
        store.log(uid, 0.0, n_done, total, note=f"FAILED {type(exc).__name__}: {exc}")
        if on_error == "raise":
            raise exc

    if workers == 1:
        for uid, arg in todo:
            try:
                res, secs = _timed(fn, arg)
            except Exception as exc:  # noqa: BLE001
                fail(uid, exc)
                continue
            handle(uid, res, secs)
        return stats

    with (
        _single_thread_env(),
        ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool,
    ):
        pending: dict[Future[tuple[dict[str, object], float]], str] = {}
        it: Iterable[tuple[str, Any]] = iter(todo)
        for uid, arg in it:
            pending[pool.submit(_timed, fn, arg)] = uid
            if len(pending) >= 2 * workers:
                break
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                uid = pending.pop(fut)
                try:
                    res, secs = fut.result()
                except Exception as exc:  # noqa: BLE001
                    try:
                        fail(uid, exc)
                    except BaseException:
                        for f in pending:
                            f.cancel()
                        raise
                    continue
                handle(uid, res, secs)
            for uid, arg in it:
                pending[pool.submit(_timed, fn, arg)] = uid
                if len(pending) >= 2 * workers:
                    break
    return stats
