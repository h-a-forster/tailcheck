"""Size of esback's asymptotic ESR tests on simulated null data, next to tailcheck's.

Steps (run from the repository root):

    uv run python scripts/esback_null/esback_null.py inputs     # tmp/esback_null/inputs
    node scripts/r_fixtures/run_r.mjs scripts/esback_null/esback_null.R tmp/esback_null \\
        esreg esback jsonlite                                    # esback (resumable)
    uv run python scripts/esback_null/esback_null.py ours       # tailcheck, same data
    uv run python scripts/esback_null/esback_null.py summary    # results/esback-null/

Optional trace of the largest strict-ESR differences (writes tmp/esback_null/trace.txt, then
compares the esreg optimum with tailcheck's at the same loss; results/esback-null/trace_v1.json):

    uv run python scripts/esback_null/esback_null.py trace-list
    node scripts/r_fixtures/run_r.mjs scripts/esback_null/trace_v1.R tmp/esback_null \
        esreg jsonlite
    uv run python scripts/esback_null/esback_null.py trace

The data are the Monte Carlo ``garch-t5`` size DGP (true forecasts) with n = 1000 and
alpha = 2.5%; replication i uses ``np.random.default_rng([SEED, i])``. tailcheck p-values are
computed with the correct G2'' and with esreg 0.6.2's (``_esreg_compat_g2pp``, which is what
esback uses). Every finished replication is a file on disk, so each step resumes.
"""

from __future__ import annotations

import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SEED, N, ALPHA = 20261024, 1000, 0.025
WORKERS = int(os.environ.get("ESBACK_NULL_WORKERS", "3"))
REPS = int(os.environ.get("ESBACK_NULL_REPS", "250"))
IO = Path("tmp/esback_null")
OURS = IO / "ours"
OUT = Path("results/esback-null")


def write_inputs() -> None:
    from tailcheck.montecarlo.dgp import simulate

    (IO / "inputs").mkdir(parents=True, exist_ok=True)
    (IO / "alpha.txt").write_text(f"{ALPHA}\n", encoding="utf-8", newline="\n")
    for i in range(REPS):
        path = IO / "inputs" / f"rep_{i:03d}.csv"
        if path.exists():
            continue
        s = simulate("garch-t5", N, ALPHA, np.random.default_rng([SEED, i]))
        rows = zip(s.r.tolist(), s.q.tolist(), s.e.tolist(), strict=True)
        lines = ["r,q,e"] + [f"{a!r},{b!r},{c!r}" for a, b, c in rows]
        part = path.with_suffix(".part")
        part.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        part.replace(path)


def _ours_one(i: int) -> None:
    from tailcheck.esr_backtest import esr_backtest
    from tailcheck.montecarlo.dgp import simulate

    path = OURS / f"rep_{i:03d}.json"
    if path.exists():
        return
    s = simulate("garch-t5", N, ALPHA, np.random.default_rng([SEED, i]))
    row: dict[str, float] = {}
    for v in (1, 3):
        for tag, compat in (("", False), ("_compat", True)):
            res = esr_backtest(s.r, s.e, ALPHA, v, q=s.q, seed=1, _esreg_compat_g2pp=compat)  # type: ignore[arg-type]
            row[f"v{v}{tag}"] = float(res.pvalue)
            if v == 3:
                row[f"v3{tag}_two"] = float(res.details.get("pvalue_two_sided", math.nan))
    part = path.with_suffix(".part")
    part.write_text(json.dumps(row) + "\n", encoding="utf-8", newline="\n")
    part.replace(path)


def run_ours() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    OURS.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(WORKERS) as ex:
        list(ex.map(_ours_one, range(REPS)))


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [math.nan, math.nan]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, c - h), min(1.0, c + h)]


def rate(p: np.ndarray, level: float = 0.05) -> dict[str, object]:
    ok = ~np.isnan(p)
    k = int(np.sum(p[ok] < level))
    n = int(p.size)  # an undefined p-value counts as a non-rejection
    return {
        "reject": k / n,
        "wilson95": wilson(k, n),
        "undefined": int(np.sum(~ok)),
        "p_median": float(np.nanmedian(p)),
        "p_q05": float(np.nanquantile(p, 0.05)),
    }


def summary() -> None:
    r_rows = {f.stem: json.loads(f.read_text("utf-8")) for f in (IO / "out").glob("rep_*.json")}
    o_rows = {f.stem: json.loads(f.read_text("utf-8")) for f in OURS.glob("rep_*.json")}
    keys = sorted(set(r_rows) & set(o_rows))
    if not keys:
        sys.exit("no replication finished in both tmp/esback_null/out and tmp/esback_null/ours")

    def rp(k: str, v: str, field: str) -> float:
        x = r_rows[k].get(v, {}).get(field)
        return math.nan if x is None or isinstance(x, (dict, list)) else float(x)

    cols = {
        "esback_v1": np.array([rp(k, "v1", "pvalue_twosided_asymptotic") for k in keys]),
        "esback_v3_less": np.array([rp(k, "v3", "pvalue_onesided_asymptotic") for k in keys]),
        "esback_v3_two": np.array([rp(k, "v3", "pvalue_twosided_asymptotic") for k in keys]),
    }
    for c in ("v1", "v1_compat", "v3", "v3_compat", "v3_two", "v3_compat_two"):
        cols[f"tailcheck_{c}"] = np.array([o_rows[k][c] for k in keys])
    res: dict[str, object] = {
        "dgp": "garch-t5 (true VaR/ES forecasts)",
        "n": N,
        "alpha": ALPHA,
        "reps": len(keys),
        "seed": SEED,
        "level": 0.05,
        "r_versions": (IO / "r_versions.txt").read_text("utf-8").splitlines(),
        "esback_errors": sum(
            1 for k in keys for v in ("v1", "v3") if "error" in r_rows[k].get(v, {})
        ),
        "esback_seconds_median": float(np.median([r_rows[k]["seconds"] for k in keys])),
        "size": {c: rate(p) for c, p in cols.items()},
    }
    agree = {}
    for a, b in (("esback_v1", "tailcheck_v1_compat"), ("esback_v3_less", "tailcheck_v3_compat")):
        d = np.abs(cols[a] - cols[b])
        agree[f"{a} vs {b}"] = {
            "max_abs_diff": float(np.nanmax(d)),
            "median_abs_diff": float(np.nanmedian(d)),
            "same_decision_5pct": float(np.mean((cols[a] < 0.05) == (cols[b] < 0.05))),
        }
    res["agreement"] = agree
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(res, indent=2) + "\n"
    (OUT / "summary.json").write_text(text, encoding="utf-8", newline="\n")
    with (OUT / "pvalues.csv").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("rep," + ",".join(cols) + "\n")
        for j, k in enumerate(keys):
            fh.write(k + "," + ",".join(f"{cols[c][j]:.6g}" for c in cols) + "\n")
    print(text)


def trace_list(k: int = 4) -> None:
    """Replications with the largest |esback v1 - tailcheck v1 (esreg G2'')|, plus rep_000."""
    import csv

    with (OUT / "pvalues.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: -abs(float(r["esback_v1"]) - float(r["tailcheck_v1_compat"])))
    reps = [r["rep"] for r in rows[:k]]
    if "rep_000" not in reps:
        reps.append("rep_000")
    (IO / "trace.txt").write_text("\n".join(reps) + "\n", encoding="utf-8", newline="\n")
    print(reps)


def trace() -> None:
    """Mean loss at esreg's and at tailcheck's strict-ESR optimum (same objective, same data)."""
    from tailcheck.esr_backtest import _fit
    from tailcheck.esreg.gfuns import G1, G2
    from tailcheck.esreg.loss import esr_loss_raw
    from tailcheck.montecarlo.dgp import simulate

    r_fits = json.loads((IO / "trace_out.json").read_text("utf-8"))
    out = {}
    for rep, v in r_fits.items():
        i = int(rep.split("_")[1])
        s = simulate("garch-t5", N, ALPHA, np.random.default_rng([SEED, i]))
        fit = _fit(s.r, s.e, s.e, ALPHA, 10, 1)
        x = np.column_stack([np.ones(N), s.e])

        def loss(b: np.ndarray, x: np.ndarray = x, s: object = s, fit: object = fit) -> float:
            sh = fit.y_shift  # type: ignore[attr-defined]
            y = s.r - sh  # type: ignore[attr-defined]
            q, e = x @ b[:2] - sh, x @ b[2:] - sh
            return float(np.mean(esr_loss_raw(y, q, e, ALPHA, G1.ZERO, G2.NEG_INV)))

        b_r = np.asarray(v["coef"], dtype=float)
        out[rep] = {
            "coef_esreg": b_r.tolist(),
            "coef_tailcheck": fit.coef.tolist(),
            "loss_reported_by_esreg": float(v["loss"]),
            "loss_at_esreg_coef": loss(b_r),
            "loss_at_tailcheck_coef": loss(fit.coef),
            "loss_difference_esreg_minus_tailcheck": loss(b_r) - loss(fit.coef),
            "max_abs_coef_difference": float(np.max(np.abs(b_r - fit.coef))),
        }
    text = json.dumps(out, indent=2) + "\n"
    (OUT / "trace_v1.json").write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    steps = {
        "inputs": write_inputs,
        "ours": run_ours,
        "summary": summary,
        "trace-list": trace_list,
        "trace": trace,
    }
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step not in steps:
        sys.exit("usage: esback_null.py inputs|ours|summary|trace-list|trace")
    steps[step]()
