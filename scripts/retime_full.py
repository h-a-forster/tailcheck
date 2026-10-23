"""Re-time samples of the full-size profiles and project their total CPU cost.

Single process (BLAS threads 1); about 10 minutes. Nothing here is a measurement of the full
runs themselves: it times a few reduced units and multiplies out.

- ``configs/mc_full.toml``: one ESR unit per (n, alpha) cell type with ``REPS_ESR``
  replications and one cheap unit per n with ``REPS_CHEAP`` replications, all on
  ``garch-t5`` (the size DGP; power DGPs cost about the same per replication).
- ``configs/study_full.toml``: per model, ``DAYS`` days of daily-refit forecasting on one
  asset (needs ``tailcheck data fetch`` first), and the per-resample cost of the bootstrap
  ESR at the out-of-sample lengths.

Writes ``results/retime-full.json``.

    uv run python scripts/retime_full.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

REPS_ESR, REPS_CHEAP, DAYS, BOOT = 2, 20, 15, 4
OUT = Path("results/retime-full.json")


def time_mc_full() -> dict[str, object]:
    from tailcheck.montecarlo.runner import build_units, cheap_unit, esr_unit
    from tailcheck.study.runner import load_config

    cfg = load_config(Path("configs/mc_full.toml"))
    units = build_units(cfg)
    per_rep: dict[str, float] = {}
    sampled: set[str] = set()
    for _uid, arg in units:
        fam = "esr" if "versions" in arg else "cheap"
        key = f"{fam}|{arg['n']}|{arg['alpha']}" if fam == "esr" else f"cheap|{arg['n']}"
        if key in sampled or arg["dgp"] != "garch-t5":
            continue
        sampled.add(key)
        small = dict(arg, reps=REPS_ESR if fam == "esr" else REPS_CHEAP)
        t0 = time.perf_counter()
        (esr_unit if fam == "esr" else cheap_unit)(small)
        per_rep[key] = (time.perf_counter() - t0) / small["reps"]
        print(f"mc-full {key}: {per_rep[key]:.2f} s/rep", flush=True)
    total = {"esr": 0.0, "cheap": 0.0}
    for _, arg in units:
        fam = "esr" if "versions" in arg else "cheap"
        key = f"{fam}|{arg['n']}|{arg['alpha']}" if fam == "esr" else f"cheap|{arg['n']}"
        total[fam] += per_rep[key] * arg["reps"]
    return {
        "seconds_per_replication": per_rep,
        "projected_cpu_hours": {k: v / 3600 for k, v in total.items()},
        "projected_cpu_hours_total": sum(total.values()) / 3600,
        "sampled_reps": {"esr": REPS_ESR, "cheap": REPS_CHEAP},
    }


def time_study_full() -> dict[str, object]:
    from tailcheck.esr_backtest import esr_backtest
    from tailcheck.forecasters import make_forecaster, rolling_forecast
    from tailcheck.montecarlo.dgp import simulate
    from tailcheck.study.runner import asset_frame, load_config

    cfg = load_config(Path("configs/study_full.toml"))
    f = cfg["forecast"]
    alphas = [float(a) for a in f["alphas"]]
    assets = [a["id"] for a in cfg["data"]["assets"]]
    n_oos = {}
    for a in assets:
        _, s, e_idx, _ = asset_frame(cfg, a)
        n_oos[a] = int(e_idx - s)
    r, s, e_idx, _ = asset_frame(cfg, assets[0])
    vals = r.to_numpy()
    per_day: dict[str, float] = {}
    for m in f["models"]:
        model = make_forecaster(
            m,
            evt_tail_fraction=float(f.get("evt_tail_fraction", 0.10)),
            caviar_random_starts=int(f.get("caviar_random_starts", 2000)),
            caviar_best_starts=int(f.get("caviar_best_starts", 10)),
        )
        n_draws = int(f.get("n_draws", 0)) if model.has_distribution else 0
        t0 = time.perf_counter()
        rolling_forecast(
            model, vals, s, s + DAYS, int(f["window"]), 1, alphas, n_draws=n_draws, seed=1
        )
        per_day[m] = (time.perf_counter() - t0) / DAYS
        print(f"study-full forecast {m}: {per_day[m]:.2f} s/day", flush=True)
    forecast_s = sum(per_day.values()) * sum(n_oos.values())
    per_boot: dict[str, float] = {}
    for n in sorted(set(n_oos.values())):
        smp = simulate("garch-t5", n, 0.025, np.random.default_rng(n))
        for v in (1, 2, 3):
            t0 = time.perf_counter()
            esr_backtest(smp.r, smp.e, 0.025, v, q=smp.q, seed=1)  # type: ignore[arg-type]
            t_asym = time.perf_counter() - t0
            t0 = time.perf_counter()
            esr_backtest(smp.r, smp.e, 0.025, v, q=smp.q, seed=1, n_boot=BOOT)  # type: ignore[arg-type]
            per_boot[f"{n}|v{v}"] = max(0.0, (time.perf_counter() - t0 - t_asym) / BOOT)
        print(f"study-full bootstrap n={n}: {per_boot}", flush=True)
    b = int(cfg["backtest"]["esr_bootstrap"])
    boot_s = sum(
        per_boot[f"{n_oos[a]}|v{v}"] * b
        for a in assets
        for v in (1, 2, 3)
        for _ in f["models"]
        for _ in alphas
    )
    return {
        "n_oos": n_oos,
        "forecast_seconds_per_day": per_day,
        "bootstrap_seconds_per_resample": per_boot,
        "projected_cpu_hours": {"forecast": forecast_s / 3600, "esr_bootstrap": boot_s / 3600},
        "projected_cpu_hours_total": (forecast_s + boot_s) / 3600,
        "sampled": {"days": DAYS, "bootstrap_resamples": BOOT},
        "note": "excludes the non-ESR backtests and the MCS/DM stage (minutes in the quick run)",
    }


def main() -> None:
    from tailcheck import __version__

    out = {
        "date": time.strftime("%Y-%m-%d"),
        "tailcheck": __version__,
        "method": "single process, BLAS threads 1, reduced units timed and multiplied out",
        "mc_full": time_mc_full(),
        "study_full": time_study_full(),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
