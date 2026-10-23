"""Result containers shared by all backtests."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

Alternative = Literal["two-sided", "less", "greater"]


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of one backtest.

    ``pvalue`` is the primary p-value of the test as documented in ``docs/DESIGN.md``.
    Secondary p-values (one-sided, bootstrap, standardized variants) go in ``details``.
    A ``pvalue`` of ``nan`` means the test is undefined for this input (for example no VaR
    exceedances); ``details["reason"]`` then says why.
    """

    test: str
    statistic: float
    pvalue: float
    nobs: int
    df: int | None = None
    alternative: Alternative = "two-sided"
    details: Mapping[str, Any] = field(default_factory=dict)

    def reject(self, level: float = 0.05) -> bool:
        """True if the null is rejected at ``level``. Undefined tests never reject."""
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        return (not math.isnan(self.pvalue)) and self.pvalue < level

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for JSON/CSV output."""
        return {
            "test": self.test,
            "statistic": self.statistic,
            "pvalue": self.pvalue,
            "nobs": self.nobs,
            "df": self.df,
            "alternative": self.alternative,
            **{f"details.{k}": v for k, v in self.details.items()},
        }
