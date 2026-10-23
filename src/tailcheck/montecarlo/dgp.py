"""Data-generating processes and forecast sets for the Monte Carlo (DESIGN section 6).

Each call returns returns ``r`` together with the forecasts ``(q, e, sigma)`` handed to the
tests, and the standardized law of the forecast distribution (``"t5"`` or ``"normal"``),
which determines the Acerbi-Szekely null.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tailcheck.dist import draw_std_t, normal_var_es, std_t_var_es

Arr = NDArray[np.float64]
SIZE_DGPS = ("iid-t5", "garch-t5", "iid-normal")
POWER_DGPS = ("scale-0.85", "scale-0.70", "normal-tails", "unconditional")
OMEGA, A1, B1, NU = 0.05, 0.10, 0.85, 5.0
BURN = 500


@dataclass(frozen=True)
class Sample:
    r: Arr
    q: Arr
    e: Arr
    sigma: Arr
    law: str  # standardized law of the forecast distribution


def garch_t5(n: int, rng: np.random.Generator) -> tuple[Arr, Arr]:
    """GARCH(1,1) with omega 0.05, alpha 0.10, beta 0.85 and standardized t5 shocks."""
    z = draw_std_t(rng, NU, n + BURN)
    r = np.empty(n + BURN)
    s2 = np.empty(n + BURN)
    s2[0] = OMEGA / (1 - A1 - B1)
    r[0] = math.sqrt(s2[0]) * z[0]
    for t in range(1, n + BURN):
        s2[t] = OMEGA + A1 * r[t - 1] ** 2 + B1 * s2[t - 1]
        r[t] = math.sqrt(s2[t]) * z[t]
    return r[BURN:], np.sqrt(s2[BURN:])


def simulate(dgp: str, n: int, alpha: float, rng: np.random.Generator) -> Sample:
    qt, et = std_t_var_es(NU, alpha)
    if dgp == "iid-t5":
        ones = np.ones(n)
        return Sample(draw_std_t(rng, NU, n), qt * ones, et * ones, ones, "t5")
    if dgp == "iid-normal":
        qn, en = normal_var_es(alpha)
        ones = np.ones(n)
        return Sample(rng.standard_normal(n), qn * ones, en * ones, ones, "normal")
    r, s = garch_t5(n, rng)
    if dgp == "garch-t5":
        return Sample(r, qt * s, et * s, s, "t5")
    if dgp.startswith("scale-"):
        c = float(dgp.split("-")[1])
        return Sample(r, c * qt * s, c * et * s, c * s, "t5")
    if dgp == "normal-tails":
        qn, en = normal_var_es(alpha)
        return Sample(r, qn * s, en * s, s, "normal")
    if dgp == "unconditional":  # constant-sigma t5 forecasts on the same GARCH data
        ones = np.ones(n)
        return Sample(r, qt * ones, et * ones, ones, "t5")
    raise ValueError(f"unknown DGP {dgp!r}")


def as_null(law: str, n: int, alpha: float, n_sim: int, rng: np.random.Generator) -> dict[str, Arr]:
    """Null draws of Z1 and Z2 for location-scale forecasts with a fixed standardized law.

    ``r_t / e_t`` and the hits are scale-free, so the null depends only on (law, alpha, n).
    """
    if law == "t5":
        q, e = std_t_var_es(NU, alpha)
        x = draw_std_t(rng, NU, (n_sim, n))
    else:
        q, e = normal_var_es(alpha)
        x = rng.standard_normal((n_sim, n))
    hit = x <= q
    s = np.where(hit, x / e, 0.0).sum(axis=1)
    k = hit.sum(axis=1)
    z2 = 1.0 - s / (n * alpha)
    with np.errstate(invalid="ignore", divide="ignore"):
        z1 = np.where(k > 0, 1.0 - s / k, np.nan)
    return {"z1": z1, "z2": z2}
