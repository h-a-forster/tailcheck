"""Why is the asymptotic ESR v3 test conservative? Null study of its ES-intercept SE.

Under the ``garch-t5`` size DGP of the Monte Carlo (true forecasts, n = 1000,
alpha = 2.5%), fit the v3 regression ``r - e = b0 + error`` (quantile on ``e``) and compare
the empirical standard deviation of the ES intercept ``b0`` with its estimated standard
error under three covariance options:

- ``default``: nid sparsity, ``scl_sp`` conditional ES variance, misspecification terms
  (the tailcheck and esback default);
- ``no_misspec``: the same without the misspecification terms;
- ``scl_N_no_misspec``: normal-scale ``scl_N`` variance, no misspecification terms.

Writes ``results/mc-quick/esr_v3_null_se.json``. About 3 minutes with 3 processes.

    uv run python scripts/esr_v3_null_se.py
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SEED, N, ALPHA, REPS, WORKERS = 20261023, 1000, 0.025, 200, 3
OPTIONS = {
    "default": ("nid", "scl_sp", True),
    "no_misspec": ("nid", "scl_sp", False),
    "scl_N_no_misspec": ("nid", "scl_N", False),
}
OUT = Path("results/mc-quick/esr_v3_null_se.json")


def one(i: int) -> dict[str, float]:
    from tailcheck.esr_backtest import _fit
    from tailcheck.esreg.covariance import vcov_details
    from tailcheck.montecarlo.dgp import simulate

    s = simulate("garch-t5", N, ALPHA, np.random.default_rng([SEED, i]))
    fit = _fit(s.r - s.e, s.e, None, ALPHA, 10, i)
    out = {"b0": float(fit.coef_e[0])}
    for name, (sp, se, ms) in OPTIONS.items():
        cov, _ = vcov_details(fit, sp, se, ms)  # type: ignore[arg-type]
        out[name] = float(np.sqrt(cov[2, 2]))
    return out


def main() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    with ProcessPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(one, range(REPS)))
    b = np.array([r["b0"] for r in rows])
    summary: dict[str, object] = {
        "dgp": "garch-t5",
        "n": N,
        "alpha": ALPHA,
        "reps": REPS,
        "seed": SEED,
        "b0_mean": float(b.mean()),
        "b0_median": float(np.median(b)),
        "b0_sd": float(b.std(ddof=1)),
    }
    for name in OPTIONS:
        se = np.array([r[name] for r in rows])
        t = b / se
        summary[name] = {
            "se_mean": float(se.mean()),
            "se_median": float(np.median(se)),
            "t_mean": float(t.mean()),
            "t_sd": float(t.std(ddof=1)),
            "reject_5pct_one_sided": float(np.mean(t < -1.6448536269514722)),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
