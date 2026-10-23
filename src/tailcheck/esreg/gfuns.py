"""Specification functions G1 and G2 of the Fissler-Ziegel (2016) joint loss.

G1 types: 1 -> G1(z) = z, 2 -> G1(z) = 0.
G2 types (each with its antiderivative "curly" G2 and two derivatives):
1 -> G2(z) = -1/z (z < 0), 2 -> 0.5/sqrt(-z) (z < 0), 3 -> 1/z^2 (z < 0),
4 -> 1/(1 + exp(-z)), 5 -> exp(z).

Formulas from Dimitriadis and Bayer (2019), table in ``docs/DESIGN.md`` section 4.3.1.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


class G1(IntEnum):
    IDENTITY = 1
    ZERO = 2


class G2(IntEnum):
    NEG_INV = 1
    INV_SQRT = 2
    INV_SQUARE = 3
    LOGISTIC = 4
    EXP = 5


NEGATIVE_DOMAIN = frozenset({G2.NEG_INV, G2.INV_SQRT, G2.INV_SQUARE})


def g1(z: Arr, kind: G1) -> Arr:
    z = np.asarray(z, dtype=np.float64)
    return z.copy() if kind == G1.IDENTITY else np.zeros_like(z)


def g1_prime(z: Arr, kind: G1) -> Arr:
    z = np.asarray(z, dtype=np.float64)
    return np.ones_like(z) if kind == G1.IDENTITY else np.zeros_like(z)


def g1_prime_prime(z: Arr, kind: G1) -> Arr:
    return np.zeros_like(np.asarray(z, dtype=np.float64))


def curly_g2(z: Arr, kind: G2) -> Arr:
    z = np.asarray(z, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        if kind == G2.NEG_INV:
            return -np.log(-z)
        if kind == G2.INV_SQRT:
            return -np.sqrt(-z)
        if kind == G2.INV_SQUARE:
            return -1.0 / z
        if kind == G2.LOGISTIC:
            return np.logaddexp(0.0, z)
        return np.exp(z)


def g2(z: Arr, kind: G2) -> Arr:
    z = np.asarray(z, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        if kind == G2.NEG_INV:
            return -1.0 / z
        if kind == G2.INV_SQRT:
            return 0.5 / np.sqrt(-z)
        if kind == G2.INV_SQUARE:
            return 1.0 / z**2
        if kind == G2.LOGISTIC:
            return 1.0 / (1.0 + np.exp(-z))
        return np.exp(z)


def g2_prime(z: Arr, kind: G2) -> Arr:
    z = np.asarray(z, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        if kind == G2.NEG_INV:
            return 1.0 / z**2
        if kind == G2.INV_SQRT:
            return 0.25 / (-z) ** 1.5
        if kind == G2.INV_SQUARE:
            return -2.0 / z**3
        if kind == G2.LOGISTIC:
            ez = np.exp(-np.abs(z))
            return ez / (1.0 + ez) ** 2
        return np.exp(z)


def g2_prime_prime(z: Arr, kind: G2) -> Arr:
    z = np.asarray(z, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        if kind == G2.NEG_INV:
            return -2.0 / z**3
        if kind == G2.INV_SQRT:
            return 0.375 / (-z) ** 2.5
        if kind == G2.INV_SQUARE:
            return 6.0 / z**4
        if kind == G2.LOGISTIC:
            ez = np.exp(z)
            return -ez * (ez - 1.0) / (1.0 + ez) ** 3
        return np.exp(z)
