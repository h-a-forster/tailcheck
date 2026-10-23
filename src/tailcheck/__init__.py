"""tailcheck: VaR and Expected Shortfall backtests with Monte Carlo-checked p-values.

Sign convention (used everywhere): ``r`` are returns, ``q`` the VaR forecast and ``e`` the ES
forecast, all on the return scale, so left-tail risk forecasts are negative and
``e <= q``. A VaR exceedance ("hit") is ``r <= q``. See ``docs/DESIGN.md``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from tailcheck.results import BacktestResult

try:
    __version__ = version("tailcheck")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"

__all__ = ["BacktestResult", "__version__"]
