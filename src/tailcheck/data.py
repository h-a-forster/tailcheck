"""Public daily data: Binance spot klines (data.binance.vision) and FRED index levels.

Raw downloads are cached under a gitignored directory and never committed.
Specification: ``docs/DESIGN.md`` section 5.1. Milestone M7.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BINANCE_URL = (
    "https://data.binance.vision/data/spot/monthly/klines/{symbol}/1d/{symbol}-1d-{ym}.zip"
)
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def fetch_binance_daily(symbol: str, start: str, end: str, cache_dir: Path) -> pd.Series:
    """Daily close (UTC day) as a float Series indexed by date."""
    raise NotImplementedError("M7")


def fetch_fred(series: str, cache_dir: Path) -> pd.Series:
    """Daily level from FRED's public CSV endpoint; missing days dropped."""
    raise NotImplementedError("M7")


def log_returns_pct(prices: pd.Series) -> pd.Series:
    """100 * log(P_t / P_{t-1}), first row dropped."""
    raise NotImplementedError("M7")
