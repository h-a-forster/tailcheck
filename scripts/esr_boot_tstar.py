"""Why is the v3 bootstrap statistic T* narrower than T0? Paired null study.

``garch-t5``, n = 1000, alpha = 2.5% (the warp-speed null cell of ``configs/mc_esr_boot.toml``,
where sd(T*) / sd(T0) = 0.85). Every replication draws one sample and one resample and
computes T* on that same resample several ways, so the arms differ only by the option named:

- ``es0``: the bootstrap refit of ``esr_backtest``, ``early_stopping=0``;
- ``es10``: the same resample refitted with ``early_stopping=10``, as the full-sample fit is;
- ``jitter``: ``early_stopping=0`` with Gaussian noise of 5% of sd(y) added to the resampled
  ``y``, which removes the repeated rows the nuisance estimators would otherwise see.

The ``iid-rows`` design keeps the marginal law of (r, e) but draws rows nearly independently
(a random n rows of a path 30 times as long), to see whether the mismatch needs the serial
dependence that the row bootstrap discards.

The summary also reports the tails of T0 and T* in the 1,000- and 400-replication null cells
of ``runs/mc-esr-boot`` and the warp-speed size of the two-sided v3 test there.

Replications go through ``UnitStore`` in ``runs/esr-boot-tstar/`` (rerun to resume); the
summary goes to ``results/mc-esr-boot/tstar_diagnostic.json``.

    uv run python scripts/esr_boot_tstar.py --workers 4 --max-minutes 9
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from tailcheck.checkpoint import UnitStore, run_units, store_for
from tailcheck.montecarlo.esr_boot import warp_interval, warp_rate
from tailcheck.study.runner import load_config

SEED, N, ALPHA, CHUNK = 20261107, 1000, 0.025, 10
REPS = {"garch": 400, "iid-rows": 400}
JITTER = 0.05
RUN = Path("runs/esr-boot-tstar")
OUT = Path("results/mc-esr-boot/tstar_diagnostic.json")


def _arm(y: Any, xq: Any, b0: float, early_stopping: int, seed: int) -> dict[str, float]:
    from tailcheck.esr_backtest import _fit
    from tailcheck.esreg.covariance import vcov_details

    try:
        fit = _fit(y, xq, None, ALPHA, early_stopping, seed)
        cov, _ = vcov_details(fit, "nid", "scl_sp", True)
        se = math.sqrt(cov[2, 2])
        b = float(fit.coef_e[0])
    except (ValueError, RuntimeError, np.linalg.LinAlgError, ZeroDivisionError):
        return {"b": math.nan, "se": math.nan, "t": math.nan, "loss": math.nan}
    return {"b": b, "se": se, "t": (b - b0) / se, "loss": float(fit.loss)}


def unit(arg: dict[str, Any]) -> dict[str, object]:
    from tailcheck.dist import std_t_var_es
    from tailcheck.montecarlo.dgp import NU, garch_t5, simulate

    rng = np.random.default_rng([SEED, arg["design"] == "garch", arg["chunk"]])
    rows: list[dict[str, float]] = []
    for _ in range(arg["reps"]):
        if arg["design"] == "garch":
            s = simulate("garch-t5", N, ALPHA, rng)
            r, e = s.r, s.e
        else:
            r_long, s_long = garch_t5(30 * N, rng)
            pick = rng.choice(30 * N, N, replace=False)
            r, e = r_long[pick], std_t_var_es(NU, ALPHA)[1] * s_long[pick]
        y, seed = r - e, int(rng.integers(1 << 31))
        full = _arm(y, e, 0.0, 10, seed)
        idx = rng.integers(0, N, N)
        row = {f"full_{k}": v for k, v in full.items()}
        row |= {f"es0_{k}": v for k, v in _arm(y[idx], e[idx], full["b"], 0, seed + 1).items()}
        if arg["design"] == "garch":
            es10 = _arm(y[idx], e[idx], full["b"], 10, seed + 1)
            noisy = y[idx] + JITTER * y.std() * rng.standard_normal(N)
            jitter = _arm(noisy, e[idx], full["b"], 0, seed + 1)
            row |= {f"es10_{k}": v for k, v in es10.items()}
            row |= {f"jitter_{k}": v for k, v in jitter.items()}
        rows.append(row)
    return {"design": arg["design"], "rows": rows}


def ratio_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> list[float]:
    """Point estimate and 95% interval of sd(a) / sd(b), resampling replications jointly."""
    idx = rng.integers(0, a.size, (4000, a.size))
    boot = a[idx].std(axis=1, ddof=1) / b[idx].std(axis=1, ddof=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return [float(a.std(ddof=1) / b.std(ddof=1)), float(lo), float(hi)]


def corr_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> list[float]:
    """Point estimate and 95% interval of corr(a, b), resampling replications jointly."""
    idx = rng.integers(0, a.size, (4000, a.size))
    x, y = a[idx] - a[idx].mean(axis=1, keepdims=True), b[idx] - b[idx].mean(axis=1, keepdims=True)
    boot = (x * y).sum(axis=1) / np.sqrt((x**2).sum(axis=1) * (y**2).sum(axis=1))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return [float(np.corrcoef(a, b)[0, 1]), float(lo), float(hi)]


def describe(t: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(t.mean()),
        "sd": float(t.std(ddof=1)),
        "q05": float(np.quantile(t, 0.05)),
        "q95": float(np.quantile(t, 0.95)),
    }


def summarise(rows: list[dict[str, float]]) -> dict[str, Any]:
    keys = rows[0].keys()
    a = {k: np.array([r.get(k, math.nan) for r in rows], dtype=float) for k in keys}
    ok = np.all([np.isfinite(v) for v in a.values()], axis=0)
    a = {k: v[ok] for k, v in a.items()}
    rng = np.random.default_rng(SEED)
    t0 = a["full_b"] / a["full_se"]
    out: dict[str, Any] = {
        "reps": int(ok.sum()),
        "dropped": int((~ok).sum()),
        "T0": describe(t0),
        "b_hat_sd": float(a["full_b"].std(ddof=1)),
        "se_full_median": float(np.median(a["full_se"])),
        "corr_b_se_full": corr_ci(a["full_b"], a["full_se"], rng),
        # Pairing each estimate with another replication's SE removes their correlation.
        "T0_sd_se_shuffled": float((a["full_b"] / rng.permutation(a["full_se"])).std(ddof=1)),
    }
    for arm in ("es0", "es10", "jitter"):
        if f"{arm}_t" not in a:
            continue
        d = a[f"{arm}_b"] - a["full_b"]
        out[arm] = {
            "Tstar": describe(a[f"{arm}_t"]),
            "sd_Tstar_over_sd_T0": ratio_ci(a[f"{arm}_t"], t0, rng),
            "numerator_sd": float(d.std(ddof=1)),
            "numerator_sd_over_b_hat_sd": ratio_ci(d, a["full_b"], rng),
            "se_median": float(np.median(a[f"{arm}_se"])),
            "corr_num_se": corr_ci(d, a[f"{arm}_se"], rng),
            # within a sample: does a low b* come with a raised SE, relative to that sample's?
            "corr_num_log_se_ratio": corr_ci(d, np.log(a[f"{arm}_se"] / a["full_se"]), rng),
            "Tstar_sd_se_shuffled": float((d / rng.permutation(a[f"{arm}_se"])).std(ddof=1)),
            "Tstar_sd_full_sample_se": float((d / a["full_se"]).std(ddof=1)),
        }
    if "es10_t" in a:
        gain = a["es0_loss"] - a["es10_loss"]
        out["es10_vs_es0"] = {
            "sd_Tstar_es10_over_es0": ratio_ci(a["es10_t"], a["es0_t"], rng),
            "sd_Tstar_jitter_over_es0": ratio_ci(a["jitter_t"], a["es0_t"], rng),
            "corr_Tstar": float(np.corrcoef(a["es0_t"], a["es10_t"])[0, 1]),
            "median_abs_diff_Tstar": float(np.median(np.abs(a["es10_t"] - a["es0_t"]))),
            "p99_abs_diff_Tstar": float(np.quantile(np.abs(a["es10_t"] - a["es0_t"]), 0.99)),
            "median_loss_gain": float(np.median(gain)),
            "share_es10_strictly_better": float(np.mean(gain > 1e-12)),
        }
    return out


def stored_null_cells() -> dict[str, Any]:
    """Tails and two-sided warp-speed size from the stored v3 null cells of the main run."""
    cfg = load_config(Path("configs/mc_esr_boot.toml"))
    store = store_for(cfg, read_only=True)
    t0: dict[int, list[float]] = {1000: [], 2000: []}
    t_star: dict[int, list[float]] = {1000: [], 2000: []}
    for uid in store.completed():
        d: Any = store.read(uid)
        if d["mode"] == "warp" and d["dgp"] == "garch-t5":
            t0[d["n"]] += d["results"]["esr_v3"]["t0"]
            t_star[d["n"]] += d["results"]["esr_v3"]["t_star"]
    out: dict[str, Any] = {}
    for n in t0:
        a, b = np.array(t0[n], dtype=float), np.array(t_star[n], dtype=float)
        rate, crit = warp_rate(np.abs(a), np.abs(b), 0.05, False)
        lo, hi = warp_interval(np.abs(a), np.abs(b), 0.05, False, seed=n)
        out[str(n)] = {
            "reps": int(a.size),
            "T0": describe(a) | {"skew": float(stats.skew(a))},
            "Tstar": describe(b) | {"skew": float(stats.skew(b))},
            "two_sided_warp": {"rate": rate, "lo": lo, "hi": hi, "crit": crit},
            "two_sided_asymptotic": float(np.mean(np.abs(a) > 1.959963984540054)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--only", choices=sorted(REPS), default=None)
    ap.add_argument("--reps", type=int, default=None, help="override, for timing a small sample")
    args = ap.parse_args()
    units = []
    for design, total in REPS.items():
        if args.only in (None, design):
            total = args.reps or total
            for k, start in enumerate(range(0, total, CHUNK)):
                arg = {"design": design, "chunk": k, "reps": min(CHUNK, total - start)}
                units.append((f"{design}|{k}", arg))
    store = UnitStore(RUN, f"tstar-{SEED}")
    run_units(store, units, unit, args.workers, max_seconds=(args.max_minutes or 1e6) * 60)
    pooled: dict[str, list[dict[str, float]]] = {}
    for uid in store.completed():
        d = store.read(uid)
        pooled.setdefault(str(d["design"]), []).extend(d["rows"])  # type: ignore[arg-type]
    summary = {
        "dgp": "garch-t5",
        "n": N,
        "alpha": ALPHA,
        "version": 3,
        "seed": SEED,
        "jitter_share_of_sd": JITTER,
        **{design: summarise(rows) for design, rows in sorted(pooled.items())},
        "stored_null_cells": stored_null_cells(),
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
