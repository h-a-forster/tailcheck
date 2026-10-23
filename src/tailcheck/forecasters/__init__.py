"""One-day-ahead (VaR, ES) forecasters for the study. Specification: ``docs/DESIGN.md`` 4.5."""

from __future__ import annotations

from tailcheck.forecasters.base import BlockForecast, Forecaster, RiskForecast, rolling_forecast
from tailcheck.forecasters.caviar import CaviarSAV
from tailcheck.forecasters.garch import EVTPot, FilteredHS, GJRGarchT
from tailcheck.forecasters.historical import HistoricalSimulation

MODELS = ("hs", "gjr-t", "fhs", "evt-pot", "caviar-sav")


def make_forecaster(
    name: str,
    evt_tail_fraction: float = 0.10,
    caviar_random_starts: int = 2000,
    caviar_best_starts: int = 10,
) -> Forecaster:
    """Construct a forecaster by its study name."""
    if name == "hs":
        return HistoricalSimulation()
    if name == "gjr-t":
        return GJRGarchT()
    if name == "fhs":
        return FilteredHS()
    if name == "evt-pot":
        return EVTPot(evt_tail_fraction)
    if name == "caviar-sav":
        return CaviarSAV(caviar_random_starts, caviar_best_starts)
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")


__all__ = [
    "MODELS",
    "BlockForecast",
    "CaviarSAV",
    "EVTPot",
    "FilteredHS",
    "Forecaster",
    "GJRGarchT",
    "HistoricalSimulation",
    "RiskForecast",
    "make_forecaster",
    "rolling_forecast",
]
