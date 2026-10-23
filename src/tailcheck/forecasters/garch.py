"""GJR-GARCH(1,1) based forecasters: parametric Student-t, filtered HS, EVT-POT. Milestone M6."""

from __future__ import annotations


class GJRGarchT:
    """GJR-GARCH(1,1) with standardized Student-t innovations (arch)."""

    name = "gjr-t"


class FilteredHS:
    """GJR-GARCH(1,1) QMLE volatility times empirical quantile/tail mean of residuals."""

    name = "fhs"


class EVTPot:
    """McNeil-Frey (2000): GPD fitted to the worst residuals of a GJR-GARCH(1,1) QMLE fit."""

    name = "evt-pot"
