"""One-day-ahead (VaR, ES) forecasters for the study. Specification: ``docs/DESIGN.md`` 4.5."""

from __future__ import annotations

from tailcheck.forecasters.base import Forecaster, RiskForecast

__all__ = ["Forecaster", "RiskForecast"]
