"""Aggregate Monte Carlo units into ``results/<name>/size_power.csv`` and a markdown table."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tailcheck.checkpoint import UnitStore, config_hash
from tailcheck.study.runner import load_config


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def distortion_band(level: float) -> tuple[float, float]:
    """Size band ``[0.6 level, 1.4 level]`` ([3%, 7%] at the 5% level).

    A size cell is flagged as distorted when its whole Wilson interval lies outside it.
    """
    return 0.6 * level, 1.4 * level


def collect_mc(cfg_path: Path) -> pd.DataFrame:
    cfg = load_config(cfg_path)
    level = float(cfg.get("level", 0.05))
    band = distortion_band(level)
    store = UnitStore(Path(cfg["run"]["out_dir"]), config_hash(cfg), read_only=True)
    size_dgps = set(cfg["cheap"]["dgps_size"]) | set(cfg["esr"]["dgps_size"])
    pooled: dict[tuple[str, str, int, float], dict[str, Any]] = {}
    for uid in store.completed():
        d = store.read(uid)
        for test, pv in d["pvalues"].items():
            key = (test, d["dgp"], int(d["n"]), float(d["alpha"]))
            slot = pooled.setdefault(key, {"p": [], "secs": []})
            slot["p"].extend(np.nan if v is None else float(v) for v in pv)
            slot["secs"].append(
                float(d.get("seconds_per_rep", d.get("median_seconds_per_test", 0)))
            )
    rows = []
    for (test, dgp, n, alpha), slot in sorted(pooled.items()):
        p = np.asarray(slot["p"], dtype=float)
        reps = len(p)
        fails = int(np.isnan(p).sum())
        rej = int(np.sum(p[~np.isnan(p)] < level))
        lo, hi = wilson(rej, reps)
        kind = "size" if dgp in size_dgps else "power"
        rows.append(
            {
                "test": test,
                "dgp": dgp,
                "kind": kind,
                "n": n,
                "alpha": alpha,
                "reps": reps,
                "failures": fails,
                "rejections": rej,
                "rate": rej / reps if reps else math.nan,
                "lo": lo,
                "hi": hi,
                "distorted": bool(kind == "size" and (hi < band[0] or lo > band[1])),
                "median_seconds": float(np.median(slot["secs"])),
            }
        )
    return pd.DataFrame(rows)


def write_mc_report(cfg_path: Path) -> str:
    cfg = load_config(cfg_path)
    level = float(cfg.get("level", 0.05))
    lo_band, hi_band = distortion_band(level)
    df = collect_mc(cfg_path)  # read-only; raises if the run directory is missing
    if not len(df):
        raise ValueError(
            f"no completed Monte Carlo units in {cfg['run']['out_dir']}; "
            "run `tailcheck mc run` first (nothing was written)"
        )
    out = Path("results") / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "size_power.csv", index=False, float_format="%.6g")
    lines = []
    for kind in ("size", "power"):
        sub = df[df["kind"] == kind]
        if not len(sub):
            continue
        lines.append(
            f"### {kind.capitalize()}: rejection rate at {100 * level:g}% (95% Wilson interval)\n"
        )
        sub = sub.assign(
            cell=sub["dgp"] + " n=" + sub["n"].astype(str) + " a=" + sub["alpha"].astype(str)
        )
        tab = (
            sub.assign(
                v=[
                    f"{r:.3f} [{lo:.3f}, {hi:.3f}]"
                    + ("*" if dist else "")
                    + (f" f={f}" if f else "")
                    for r, lo, hi, dist, f in zip(
                        sub["rate"],
                        sub["lo"],
                        sub["hi"],
                        sub["distorted"],
                        sub["failures"],
                        strict=True,
                    )
                ]
            )
            .pivot(index="cell", columns="test", values="v")
            .fillna("")
        )
        cols = list(tab.columns)
        lines.append("| cell | " + " | ".join(cols) + " |")
        lines.append("|---|" + "---|" * len(cols))
        for cell, r in tab.iterrows():
            lines.append(f"| {cell} | " + " | ".join(str(r[c]) for c in cols) + " |")
        lines.append("")
    lines.append(
        f"`*` size distorted (Wilson interval entirely outside [{lo_band:.3g}, {hi_band:.3g}]); "
        "`f=k`: k replications where the test was undefined or failed (counted as "
        "non-rejections)."
    )
    (out / "size_power.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    summary = {
        "cells": len(df),
        "total_replications": int(df["reps"].sum()),
        "distorted_size_cells": df.loc[
            df["distorted"], ["test", "dgp", "n", "alpha", "rate"]
        ].to_dict(orient="records"),
        "config_hash": config_hash(cfg),
    }
    manifest = Path(cfg["run"]["out_dir"]) / "manifest.json"
    if manifest.exists():
        summary["run_manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return json.dumps({k: v for k, v in summary.items() if k != "run_manifest"}, indent=2)
