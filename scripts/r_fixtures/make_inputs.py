"""Write the input CSVs for the R reference fixtures (DESIGN section 7, layer 3).

Run from the repository root: ``uv run python scripts/r_fixtures/make_inputs.py``.
Deterministic (fixed seeds); independent of the tailcheck package except for nothing at all,
so the inputs do not change when the package changes.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path("tests/fixtures/r/inputs")


def std_t_var_es(nu: float, a: float) -> tuple[float, float]:
    ta = stats.t.ppf(a, nu)
    s = math.sqrt((nu - 2) / nu)
    return s * ta, -s * (nu + ta**2) / (nu - 1) * stats.t.pdf(ta, nu) / a


def normal_var_es(a: float) -> tuple[float, float]:
    z = stats.norm.ppf(a)
    return z, -stats.norm.pdf(z) / a


def garch_t5(n: int, rng: np.random.Generator, burn: int = 500) -> tuple[np.ndarray, np.ndarray]:
    omega, a1, b1, nu = 0.05, 0.10, 0.85, 5.0
    z = rng.standard_t(nu, size=n + burn) * math.sqrt((nu - 2) / nu)
    s2 = np.empty(n + burn)
    r = np.empty(n + burn)
    s2[0] = omega / (1 - a1 - b1)
    for t in range(n + burn):
        if t > 0:
            s2[t] = omega + a1 * r[t - 1] ** 2 + b1 * s2[t - 1]
        r[t] = math.sqrt(s2[t]) * z[t]
    return r[burn:], np.sqrt(s2[burn:])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261023)
    n = 1000
    x = rng.chisquare(1, n)
    y = -x + (1 + 0.5 * x) * rng.standard_normal(n)
    np.savetxt(
        OUT / "dgp2.csv",
        np.column_stack([x, y]),
        delimiter=",",
        header="x,y",
        comments="",
        fmt="%.17g",
    )

    r, s = garch_t5(1500, rng)
    cols, names = [r, s], ["r", "s"]
    for a, tag in ((0.025, "025"), (0.01, "01")):
        qt, et = std_t_var_es(5.0, a)
        qn, en = normal_var_es(a)
        cols += [qt * s, et * s, 0.85 * qt * s, 0.85 * et * s, qn * s, en * s]
        names += [
            f"q_true_{tag}",
            f"e_true_{tag}",
            f"q_scale_{tag}",
            f"e_scale_{tag}",
            f"q_normal_{tag}",
            f"e_normal_{tag}",
        ]
    np.savetxt(
        OUT / "forecasts.csv",
        np.column_stack(cols),
        delimiter=",",
        header=",".join(names),
        comments="",
        fmt="%.17g",
    )

    v = [
        rng.standard_normal(500),
        rng.standard_t(3, 800),
        np.concatenate([rng.normal(-2, 0.5, 300), rng.normal(1, 1, 700)]),
    ]
    for i, a in enumerate(v):
        np.savetxt(OUT / f"bw_{i}.csv", a, header="x", comments="", fmt="%.17g")


if __name__ == "__main__":
    main()
