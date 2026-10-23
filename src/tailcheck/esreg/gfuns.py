"""Specification functions G1 and G2 of the Fissler-Ziegel (2016) joint loss.

G1 types: 1 -> G1(z) = z, 2 -> G1(z) = 0.
G2 types (each with its antiderivative "curly" G2 and two derivatives):
1 -> G2(z) = -1/z (z < 0), 2 -> 0.5/sqrt(-z) (z < 0), 3 -> 1/z^2 (z < 0),
4 -> 1/(1 + exp(-z)), 5 -> exp(z).
"""

from __future__ import annotations

from enum import IntEnum


class G1(IntEnum):
    IDENTITY = 1
    ZERO = 2


class G2(IntEnum):
    NEG_INV = 1
    INV_SQRT = 2
    INV_SQUARE = 3
    LOGISTIC = 4
    EXP = 5
