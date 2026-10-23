"""Public daily data: Binance spot klines (data.binance.vision) and FRED index levels.

Raw downloads are cached under a gitignored directory and never committed.
Specification: ``docs/DESIGN.md`` section 5.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tailcheck.checkpoint import _atomic_write_bytes

BINANCE_URL = (
    "https://data.binance.vision/data/spot/monthly/klines/{symbol}/1d/{symbol}-1d-{ym}.zip"
)
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
_UA = {"User-Agent": "tailcheck (research; +https://github.com/)"}


class DataError(RuntimeError):
    """Download, checksum or parsing problem."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _check_prices(values: np.ndarray, what: str) -> None:
    if not np.all(np.isfinite(values)):
        raise DataError(f"{what}: {int(np.sum(~np.isfinite(values)))} non-finite prices")
    if np.any(values <= 0):
        raise DataError(f"{what}: non-positive prices")


def _read_checksum(path: Path) -> str | None:
    """The SHA-256 in a Binance ``.CHECKSUM`` file, or None if the file is malformed."""
    try:
        parts = path.read_text(encoding="ascii").split()
    except (OSError, UnicodeDecodeError):
        return None
    if not parts:
        return None
    digest = parts[0].strip().lower()
    return digest if _SHA256_RE.match(digest) else None


def _get(url: str, retries: int = 3, timeout: float = 60.0) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed hosts
                return bytes(resp.read())
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise DataError(f"download failed: {url}: {last}")


def _months(start: str, end: str) -> list[str]:
    s = pd.Timestamp(start).to_period("M")
    e = pd.Timestamp(end).to_period("M")
    return [str(p) for p in pd.period_range(s, e, freq="M")]


def timestamp_unit(values: np.ndarray) -> str:
    """Binance open times: milliseconds before 2025, microseconds from 2025 (by magnitude)."""
    v = float(np.max(values))
    if v > 1e17:
        raise DataError(f"unrecognised timestamp magnitude {v}")
    return "us" if v > 1e14 else "ms"


def parse_binance_zip(blob: bytes) -> pd.Series:
    """Close prices indexed by UTC date from one monthly 1d klines zip."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise DataError(f"expected one CSV in zip, got {names}")
        raw = zf.read(names[0]).decode()
    df = pd.read_csv(io.StringIO(raw), header=None)
    if not pd.api.types.is_numeric_dtype(df[0]):  # a header row is present
        df = pd.read_csv(io.StringIO(raw))
        df.columns = range(df.shape[1])
    t = df[0].to_numpy(dtype=np.int64)
    unit = timestamp_unit(t)
    dates = pd.to_datetime(t, unit=unit, utc=True).tz_convert(None).normalize()
    close = pd.to_numeric(df[4], errors="coerce").to_numpy(dtype=float)
    _check_prices(close, names[0])
    return pd.Series(close, index=pd.DatetimeIndex(dates), name="close")


def fetch_binance_daily(symbol: str, start: str, end: str, cache_dir: Path) -> pd.Series:
    """Daily close (UTC day) as a float Series indexed by date.

    Each monthly zip is verified against its published ``.CHECKSUM`` (SHA-256) and cached
    under ``cache_dir/raw/binance/<symbol>/``. Downloads are written atomically, so an
    interrupted download never leaves a truncated file. A malformed checksum file is
    re-downloaded; on a mismatch both files are fetched again once before giving up.
    """
    root = Path(cache_dir) / "raw" / "binance" / symbol
    root.mkdir(parents=True, exist_ok=True)
    parts = []
    for ym in _months(start, end):
        url = BINANCE_URL.format(symbol=symbol, ym=ym)
        zpath = root / f"{symbol}-1d-{ym}.zip"
        cpath = root / f"{symbol}-1d-{ym}.zip.CHECKSUM"
        for attempt in range(2):
            expected = _read_checksum(cpath) if cpath.exists() else None
            if expected is None:
                _atomic_write_bytes(cpath, _get(url + ".CHECKSUM"))
                expected = _read_checksum(cpath)
                if expected is None:
                    cpath.unlink()
                    raise DataError(f"malformed checksum file for {zpath.name}")
            if not zpath.exists():
                _atomic_write_bytes(zpath, _get(url))
            got = hashlib.sha256(zpath.read_bytes()).hexdigest()
            if got == expected:
                break
            zpath.unlink()
            cpath.unlink()
            if attempt == 1:
                raise DataError(f"checksum mismatch for {zpath.name}")
        parts.append(parse_binance_zip(zpath.read_bytes()))
    s = pd.concat(parts).sort_index()
    if s.index.has_duplicates:
        raise DataError(f"{symbol}: duplicate dates {s.index[s.index.duplicated()][:5].tolist()}")
    s = s.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    gaps = pd.date_range(s.index[0], s.index[-1], freq="D").difference(s.index)
    if len(gaps):
        raise DataError(f"{symbol}: {len(gaps)} missing days, first {gaps[:5].tolist()}")
    s.name = symbol
    return s


def fetch_fred(series: str, cache_dir: Path, refresh: bool = False) -> pd.Series:
    """Daily level from FRED's public CSV endpoint; missing days dropped.

    The first download is cached as ``cache_dir/raw/fred/<series>.csv`` and reused (a fixed
    snapshot) unless ``refresh`` is set. FRED publishes no checksum, so a download is parsed
    and validated before it is written (atomically) to the cache.
    """
    path = Path(cache_dir) / "raw" / "fred" / f"{series}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not path.exists():
        blob = _get(FRED_URL.format(series=series))
        parse_fred_csv(blob, series)
        _atomic_write_bytes(path, blob)
    return parse_fred_csv(path.read_bytes(), series)


def parse_fred_csv(blob: bytes, series: str) -> pd.Series:
    """Parse FRED's ``fredgraph.csv`` (date, value; ``.`` marks a missing day)."""
    try:
        df = pd.read_csv(io.BytesIO(blob), na_values=[".", ""])
    except (ValueError, pd.errors.ParserError) as exc:
        raise DataError(f"{series}: unreadable FRED CSV: {exc}") from exc
    if df.shape[1] != 2:
        raise DataError(f"unexpected FRED columns {list(df.columns)}")
    df.columns = ["date", "value"]
    df = df.dropna()
    if df.empty:
        raise DataError(f"{series}: FRED CSV has no observations")
    try:
        values = df["value"].to_numpy(dtype=float)
        index = pd.DatetimeIndex(pd.to_datetime(df["date"], format="%Y-%m-%d"))
    except (ValueError, TypeError) as exc:
        raise DataError(f"{series}: malformed FRED CSV: {exc}") from exc
    s = pd.Series(values, index=index, name=series).sort_index()
    if s.index.has_duplicates:
        raise DataError(f"{series}: duplicate dates")
    _check_prices(s.to_numpy(), series)
    return s


def log_returns_pct(prices: pd.Series) -> pd.Series:
    """100 * log(P_t / P_{t-1}), first row dropped."""
    p = prices.astype(float)
    v = p.to_numpy()
    if not np.all(np.isfinite(v)):
        raise ValueError("prices must be finite")
    if np.any(v <= 0):
        raise ValueError("prices must be positive")
    return (100.0 * np.log(p / p.shift(1))).iloc[1:]


def series_sha256(s: pd.Series) -> str:
    """Hash of a return series (dates and values rounded to 1e-10) for the data manifest."""
    buf = "\n".join(f"{d:%Y-%m-%d},{v:.10f}" for d, v in zip(s.index, s.to_numpy(), strict=True))
    return hashlib.sha256(buf.encode()).hexdigest()


def processed_path(cache_dir: Path, asset: str) -> Path:
    return Path(cache_dir) / "processed" / f"{asset}.csv"


def fetch_assets(
    assets: Sequence[Mapping[str, Any]], end: str, cache_dir: Path
) -> dict[str, dict[str, Any]]:
    """Download (or reuse cached) prices for each asset; write processed percent log returns
    to ``cache_dir/processed/<id>.csv``; return a manifest entry per asset."""
    manifest: dict[str, dict[str, Any]] = {}
    for a in assets:
        aid, src = str(a["id"]), str(a["source"])
        if src == "binance":
            prices = fetch_binance_daily(aid, str(a.get("start", "2017-08-17")), end, cache_dir)
        elif src == "fred":
            prices = fetch_fred(aid, cache_dir).loc[: pd.Timestamp(end)]
        else:
            raise ValueError(f"unknown source {src!r}")
        ret = log_returns_pct(prices)
        out = processed_path(cache_dir, aid)
        csv = ret.rename("r").to_frame().to_csv(index_label="date", float_format="%.12g")
        _atomic_write_bytes(out, csv.encode())
        stored = load_returns(cache_dir, aid)
        manifest[aid] = {
            "source": src,
            "class": a.get("class"),
            "first": f"{stored.index[0]:%Y-%m-%d}",
            "last": f"{stored.index[-1]:%Y-%m-%d}",
            "n_returns": len(stored),
            "sha256": series_sha256(stored),
        }
    return manifest


def load_returns(cache_dir: Path, asset: str) -> pd.Series:
    path = processed_path(cache_dir, asset)
    if not path.exists():
        raise DataError(f"{path} missing; run `tailcheck data fetch` first")
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df["r"].astype(float)


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
