"""CAViaR symmetric absolute value (Engle-Manganelli 2004) with a constant ES/VaR ratio.
Milestone M6."""

from __future__ import annotations


class CaviarSAV:
    """q_t = b0 + b1 q_{t-1} + b2 |r_{t-1}|, fitted by check-loss minimisation."""

    name = "caviar-sav"
