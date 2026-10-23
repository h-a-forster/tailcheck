"""Size and power of the bootstrap ESR tests (DESIGN section 6.1).

Two modes per cell:

- ``warp``: the warp-speed method of Giacomini, Politis and White (2013). Each Monte Carlo
  replication computes the statistic ``T0`` and a single bootstrap statistic ``T*`` (one
  resample, same centring and covariance as :func:`tailcheck.esr_backtest.esr_backtest`).
  The bootstrap critical value is the empirical quantile of the pooled ``T*``, so a cell
  costs about two fits per replication instead of ``B + 1``.
- ``exact``: every replication runs the full bootstrap test with ``n_boot`` resamples and
  its p-value is compared with the level. Expensive; used as a cross-check of ``warp``.

Both modes also record the asymptotic p-value of the same replication, so asymptotic and
bootstrap rates are paired. Unit ids are ``<mode>|<dgp>|<n>|<alpha>|<chunk>`` and every unit
goes through :class:`tailcheck.checkpoint.UnitStore`.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tailcheck.checkpoint import UnitStore, config_hash, run_units
from tailcheck.montecarlo.dgp import SIZE_DGPS, simulate
from tailcheck.montecarlo.report import wilson
from tailcheck.study.runner import configured_workers, load_config, unit_seed

MODES = ("warp", "exact")


def boot_unit(arg: dict[str, Any]) -> dict[str, object]:
    """Run ``reps`` replications of one cell chunk; returns raw statistics and p-values."""
    from tailcheck.esr_backtest import esr_backtest

    mode, dgp, n, alpha = arg["mode"], arg["dgp"], arg["n"], arg["alpha"]
    versions = [int(v) for v in arg["versions"]]
    n_boot = 1 if mode == "warp" else int(arg["n_boot"])
    rng = np.random.default_rng(arg["seed"])
    out: dict[str, dict[str, list[float]]] = {
        f"esr_v{v}": {"t0": [], "t_star": [], "p_asym": [], "p_boot": [], "boot_failed": []}
        for v in versions
    }
    reasons: dict[str, int] = {}
    t_start = time.perf_counter()
    for _ in range(int(arg["reps"])):
        s = simulate(dgp, n, alpha, rng)
        fit_seed = int(rng.integers(1 << 31))
        for v in versions:
            slot = out[f"esr_v{v}"]
            try:
                res = esr_backtest(
                    s.r,
                    s.e,
                    alpha,
                    v,  # type: ignore[arg-type]
                    q=s.q,
                    n_boot=n_boot,
                    seed=fit_seed,
                    _keep_boot_stats=True,
                )
                t0, p = float(res.statistic), float(res.pvalue)
                d = res.details
                stats_b = list(d.get("bootstrap_stats", []))
                t_star = float(stats_b[0]) if (mode == "warp" and stats_b) else math.nan
                p_boot = float(d.get("pvalue_bootstrap", math.nan))
                failed = float(d.get("n_boot_failed", math.nan))
                if math.isnan(p):
                    reason = str(d.get("reason", "nan"))[:60]
                    reasons[reason] = reasons.get(reason, 0) + 1
            except Exception as exc:  # noqa: BLE001 - counted as failure
                t0 = p = t_star = p_boot = failed = math.nan
                reasons[type(exc).__name__] = reasons.get(type(exc).__name__, 0) + 1
            slot["t0"].append(t0)
            slot["p_asym"].append(p)
            slot["t_star"].append(t_star)
            slot["p_boot"].append(p_boot)
            slot["boot_failed"].append(failed)
    return {
        "family": "esr_boot",
        "mode": mode,
        "dgp": dgp,
        "n": n,
        "alpha": alpha,
        "reps": int(arg["reps"]),
        "n_boot": n_boot,
        "results": out,
        "failure_reasons": reasons,
        "seconds_per_rep": (time.perf_counter() - t_start) / max(1, int(arg["reps"])),
    }


def build_boot_units(cfg: dict[str, Any]) -> list[tuple[str, Any]]:
    gseed = int(cfg["seed"])
    chunk = int(cfg["run"].get("chunk", 10))
    units: list[tuple[str, Any]] = []
    seen: set[tuple[str, str, int, float]] = set()
    for cell in cfg["cells"]:
        mode = str(cell["mode"])
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        key = (mode, str(cell["dgp"]), int(cell["n"]), float(cell["alpha"]))
        if key in seen:
            raise ValueError(f"duplicate cell {key}")
        seen.add(key)
        if mode == "exact" and int(cell.get("n_boot", 0)) < 19:
            raise ValueError("exact cells need n_boot >= 19")
        reps = int(cell["reps"])
        for k, start in enumerate(range(0, reps, chunk)):
            uid = f"{mode}|{key[1]}|{key[2]}|{key[3]}|{k}"
            units.append(
                (
                    uid,
                    {
                        "mode": mode,
                        "dgp": key[1],
                        "n": key[2],
                        "alpha": key[3],
                        "reps": min(chunk, reps - start),
                        "versions": sorted(int(v) for v in cell["versions"]),
                        "n_boot": int(cell.get("n_boot", 1)),
                        "seed": unit_seed(gseed, uid),
                    },
                )
            )
    # exact cells are the slowest units: start them first so the pool stays busy
    return sorted(units, key=lambda u: u[1]["mode"] != "exact")


def run_boot_mc(
    cfg_path: Path,
    workers: int | None = None,
    force_new: bool = False,
    max_minutes: float | None = None,
) -> Path:
    from tailcheck import __version__

    cfg = load_config(cfg_path)
    out_dir = Path(cfg["run"]["out_dir"])
    store = UnitStore(
        out_dir, config_hash(cfg), force_new=force_new, inputs={"tailcheck": __version__}
    )
    run_units(
        store,
        build_boot_units(cfg),
        boot_unit,
        configured_workers(cfg, workers),
        on_error="log",
        max_seconds=None if max_minutes is None else 60.0 * max_minutes,
    )
    return out_dir


def warp_rate(
    t0: np.ndarray, t_star: np.ndarray, level: float, lower_tail: bool
) -> tuple[float, float]:
    """Warp-speed rejection rate and the bootstrap critical value.

    ``lower_tail``: reject when ``T0`` is below the ``level`` quantile of the pooled ``T*``
    (one-sided v3); otherwise when it is above the ``1 - level`` quantile (Wald v1/v2). An
    undefined ``T0`` counts as a non-rejection; undefined ``T*`` are dropped from the pool.
    """
    pool = t_star[np.isfinite(t_star)]
    if pool.size == 0 or t0.size == 0:
        return math.nan, math.nan
    if lower_tail:
        crit = float(np.quantile(pool, level))
        rej = np.isfinite(t0) & (t0 < crit)
    else:
        crit = float(np.quantile(pool, 1.0 - level))
        rej = np.isfinite(t0) & (t0 > crit)
    return float(rej.mean()), crit


def warp_interval(
    t0: np.ndarray,
    t_star: np.ndarray,
    level: float,
    lower_tail: bool,
    n_resample: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% percentile interval for the warp-speed rate, resampling replications jointly.

    Resampling the pairs ``(T0_i, T*_i)`` carries both the binomial error of the rejection
    count and the Monte Carlo error of the critical value.
    """
    rng = np.random.default_rng(seed)
    m = t0.size
    rates = np.empty(n_resample)
    for b in range(n_resample):
        idx = rng.integers(0, m, m)
        rates[b] = warp_rate(t0[idx], t_star[idx], level, lower_tail)[0]
    lo, hi = np.nanquantile(rates, [0.025, 0.975])
    return float(lo), float(hi)


def collect_boot(cfg_path: Path) -> pd.DataFrame:
    cfg = load_config(cfg_path)
    level = float(cfg.get("level", 0.05))
    store = UnitStore(Path(cfg["run"]["out_dir"]), config_hash(cfg), read_only=True)
    pooled: dict[tuple[str, str, str, int, float], dict[str, Any]] = {}
    for uid in store.completed():
        d = store.read(uid)
        for test, res in d["results"].items():
            key = (str(d["mode"]), test, str(d["dgp"]), int(d["n"]), float(d["alpha"]))
            slot = pooled.setdefault(
                key, {f: [] for f in ("t0", "t_star", "p_asym", "p_boot")} | {"secs": []}
            )
            for f in ("t0", "t_star", "p_asym", "p_boot"):
                slot[f].extend(np.nan if v is None else float(v) for v in res[f])
            slot["secs"].append(float(d["seconds_per_rep"]))
            slot["n_boot"] = int(d["n_boot"])
    rows = []
    for (mode, test, dgp, n, alpha), slot in sorted(pooled.items()):
        arr = {f: np.asarray(slot[f], dtype=float) for f in ("t0", "t_star", "p_asym", "p_boot")}
        reps = arr["t0"].size
        lower = test == "esr_v3"
        p_asym = arr["p_asym"]
        k_asym = int(np.sum(p_asym[~np.isnan(p_asym)] < level))
        lo_a, hi_a = wilson(k_asym, reps)
        row: dict[str, Any] = {
            "mode": mode,
            "test": test,
            "dgp": dgp,
            "kind": "size" if dgp in SIZE_DGPS else "power",
            "n": n,
            "alpha": alpha,
            "reps": reps,
            "n_boot": slot["n_boot"],
            "undefined": int(np.sum(~np.isfinite(arr["t0"]))),
            "asym_rate": k_asym / reps,
            "asym_lo": lo_a,
            "asym_hi": hi_a,
            "seconds_per_rep": float(np.median(slot["secs"])),
        }
        if mode == "warp":
            rate, crit = warp_rate(arr["t0"], arr["t_star"], level, lower)
            lo, hi = warp_interval(arr["t0"], arr["t_star"], level, lower, seed=n)
            asym_crit = -1.6448536269514722 if lower else 5.991464547107979
            row |= {
                "boot_rate": rate,
                "boot_lo": lo,
                "boot_hi": hi,
                "boot_crit": crit,
                "asym_crit": asym_crit if test in ("esr_v1", "esr_v2", "esr_v3") else math.nan,
                "t_star_undefined": int(np.sum(~np.isfinite(arr["t_star"]))),
            }
        else:
            pb = arr["p_boot"]
            k = int(np.sum(pb[~np.isnan(pb)] < level))
            lo, hi = wilson(k, reps)
            row |= {
                "boot_rate": k / reps,
                "boot_lo": lo,
                "boot_hi": hi,
                "boot_crit": math.nan,
                "asym_crit": math.nan,
                "t_star_undefined": int(np.sum(np.isnan(pb))),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def write_boot_report(cfg_path: Path) -> str:
    cfg = load_config(cfg_path)
    level = float(cfg.get("level", 0.05))
    df = collect_boot(cfg_path)
    if not len(df):
        raise ValueError(
            f"no completed units in {cfg['run']['out_dir']}; run `tailcheck mc boot-run` first"
        )
    out = Path("results") / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "size_power.csv", index=False, float_format="%.6g")
    lines = [
        f"Rejection rates at {100 * level:g}% with 95% intervals (asymptotic: Wilson; "
        "bootstrap: Wilson for `exact`, joint resampling of replications for `warp`).",
        "",
        "| mode | test | dgp | n | alpha | reps | asymptotic | bootstrap | crit (boot / asym) "
        "| undefined T0 / T* |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in df.itertuples():
        crit = f"{r.boot_crit:.3f} / {r.asym_crit:.3f}" if r.mode == "warp" else ""
        boot_label = "" if r.mode == "warp" else f" (B={r.n_boot})"
        lines.append(
            f"| {r.mode}{boot_label} | {r.test} | {r.dgp} | {r.n} | {r.alpha:g} | {r.reps} "
            f"| {r.asym_rate:.3f} [{r.asym_lo:.3f}, {r.asym_hi:.3f}] "
            f"| {r.boot_rate:.3f} [{r.boot_lo:.3f}, {r.boot_hi:.3f}] | {crit} "
            f"| {r.undefined} / {r.t_star_undefined} |"
        )
    (out / "size_power.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    summary: dict[str, Any] = {
        "cells": len(df),
        "replications": int(df.groupby(["mode", "dgp", "n", "alpha"])["reps"].max().sum()),
        "config_hash": config_hash(cfg),
    }
    manifest = Path(cfg["run"]["out_dir"]) / "manifest.json"
    if manifest.exists():
        summary["run_manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return "\n".join(lines)
