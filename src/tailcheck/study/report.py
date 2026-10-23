"""Aggregate finished study units into tables (CSV + markdown) under ``results/<name>/``."""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tailcheck.checkpoint import UnitStore, config_hash
from tailcheck.multiple import benjamini_hochberg, holm
from tailcheck.study.runner import load_config


def _p(d: dict[str, Any] | None, key: str = "pvalue") -> float:
    if not isinstance(d, dict):
        return math.nan
    v = d.get(key)
    return math.nan if v is None else float(v)


def collect_study(cfg_path: Path) -> dict[str, pd.DataFrame]:
    cfg = load_config(cfg_path)
    store = UnitStore(Path(cfg["run"]["out_dir"]), config_hash(cfg), read_only=True)
    classes = {a["id"]: a.get("class", "") for a in cfg["data"]["assets"]}
    rows, regimes, compares, forecasts = [], [], [], []
    for uid in store.completed():
        d = store.read(uid)
        if uid.startswith("forecast__"):
            forecasts.append({k: d[k] for k in ("asset", "model", "n_oos", "n_refits", "seconds")})
        elif uid.startswith("backtest__"):
            row: dict[str, Any] = {
                "asset": d["asset"],
                "class": classes.get(str(d["asset"]), ""),
                "model": d["model"],
                "alpha": d["alpha"],
                "n": d["n"],
                "hits": d["hits"],
                "hit_rate": d["hit_rate"],
                "hit_rate_lo": d["hit_rate_lo"],
                "hit_rate_hi": d["hit_rate_hi"],
                "kupiec_p": _p(d["kupiec"]),
                "chr_ind_p": _p(d["christoffersen_ind"]),
                "chr_cc_p": _p(d["christoffersen_cc"]),
                "dq_p": _p(d["dq"]),
                "traffic_light": d["traffic_light"]["zone"],
                "mf_p": _p(d["mcneil_frey"]),
                "as_z2": _p(d.get("as_z2_full"), "statistic"),
                "as_z2_p": _p(d.get("as_z2_full")),
                "fz0": d["fz0"]["mean"],
                "fz0_lo": d["fz0"]["lo"],
                "fz0_hi": d["fz0"]["hi"],
            }
            for v in (1, 2, 3):
                if f"esr_v{v}" in d:
                    row[f"esr_v{v}_p"] = _p(d[f"esr_v{v}"])
                    row[f"esr_v{v}_reason"] = d[f"esr_v{v}"].get("details.reason")
            row["esr_v1_esback_p"] = _p(d.get("esr_v1_esback_g2pp"))
            rows.append(row)
            for name in ("calm", "turbulent"):
                g = d[f"regime_{name}"]
                regimes.append(
                    {
                        "asset": d["asset"],
                        "class": row["class"],
                        "model": d["model"],
                        "alpha": d["alpha"],
                        "regime": name,
                        "n": g["n"],
                        "hits": g["hits"],
                        "hit_rate": g["hit_rate"],
                        "hit_rate_lo": g["hit_rate_lo"],
                        "hit_rate_hi": g["hit_rate_hi"],
                        "kupiec_p": _p(g["kupiec"]),
                        "esr_v1_p": _p(g["esr_v1"]),
                        "esr_v3_p": _p(g["esr_v3"]),
                        "fz0": g["fz0"]["mean"],
                        "as_z2_p": _p(d.get(f"as_z2_{name}")),
                    }
                )
        elif uid.startswith("compare__"):
            for key in [k for k in d if str(k).startswith("mcs_")]:
                mcs = d[key]
                for model, pv in mcs["pvalues"].items():
                    compares.append(
                        {
                            "asset": d["asset"],
                            "alpha": d["alpha"],
                            "mcs": key,
                            "model": model,
                            "mcs_p": pv,
                            "included": model in mcs["included"],
                            "mean_fz0": d["mean_loss"][model],
                            "dm_vs_gjr_t": _p(d.get("dm", {}).get(model), "statistic"),
                            "dm_p": _p(d.get("dm", {}).get(model)),
                        }
                    )
    bt = pd.DataFrame(rows)
    if len(bt):
        bt = bt.sort_values(["alpha", "asset", "model"]).reset_index(drop=True)
        bt["esr_v1_p_holm"] = holm(bt["esr_v1_p"])
        bt["esr_v1_q_bh"] = benjamini_hochberg(bt["esr_v1_p"])
    return {
        "backtests": bt,
        "regimes": pd.DataFrame(regimes),
        "compare": pd.DataFrame(compares),
        "forecasts": pd.DataFrame(forecasts),
    }


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, float | np.floating):
        return f"{x:.{nd}f}" if abs(x) >= 10 ** (-nd) or x == 0 else f"{x:.0e}"
    return str(x)


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_study_report(cfg_path: Path) -> str:
    """Write ``results/<name>/`` CSVs, ``summary.json`` and ``tables.md``; return a summary."""
    cfg = load_config(cfg_path)
    t = collect_study(cfg_path)  # read-only; raises if the run directory is missing
    if not len(t["backtests"]):
        raise ValueError(
            f"no completed backtest units in {cfg['run']['out_dir']}; "
            "run `tailcheck study run` first (nothing was written)"
        )
    out = Path("results") / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    for name, df in t.items():
        df.to_csv(out / f"{name}.csv", index=False, float_format="%.6g")
    bt, rg, cp = t["backtests"], t["regimes"], t["compare"]
    md: list[str] = []
    for alpha in sorted(bt["alpha"].unique()):
        sub = bt[bt["alpha"] == alpha]
        md.append(f"### Strict ESR (v1) p-values, alpha = {alpha} (BH q in brackets)\n")
        piv = sub.pivot(index="model", columns="asset", values="esr_v1_p")
        qv = sub.pivot(index="model", columns="asset", values="esr_v1_q_bh")
        cells = piv.copy().astype(object)
        for i in piv.index:
            for c in piv.columns:
                cells.loc[i, c] = f"{_fmt(piv.loc[i, c])} [{_fmt(qv.loc[i, c])}]"
        md.append(_md(cells.reset_index()) + "\n")
        md.append(f"### Hit rates (95% Wilson interval) and VaR-test p-values, alpha = {alpha}\n")
        cols = [
            "asset",
            "model",
            "hits",
            "n",
            "hit_rate",
            "hit_rate_lo",
            "hit_rate_hi",
            "kupiec_p",
            "chr_cc_p",
            "dq_p",
            "mf_p",
            "as_z2_p",
            "esr_v3_p",
            "fz0",
        ]
        md.append(_md(sub[cols]) + "\n")
    if len(cp):
        for key in sorted(cp["mcs"].unique()):
            sub = cp[cp["mcs"] == key]
            md.append(f"### Model confidence set {key[4:]}% (FZ0 loss): members marked x\n")
            piv = sub.pivot_table(
                index="model", columns=["alpha", "asset"], values="included", aggfunc="first"
            )
            piv = piv.map(lambda v: "x" if v else "")
            piv.columns = [f"{a} {s}" for a, s in piv.columns]
            md.append(_md(piv.reset_index()) + "\n")
    if len(rg):
        md.append("### Calm vs turbulent regimes (pooled over assets)\n")
        agg = (
            rg.groupby(["alpha", "regime", "model"])
            .agg(
                hits=("hits", "sum"),
                n=("n", "sum"),
                fz0=("fz0", "mean"),
                esr_v1_rej=("esr_v1_p", lambda s: int((s < 0.05).sum())),
                cells=("esr_v1_p", "size"),
            )
            .reset_index()
        )
        agg["hit_rate"] = agg["hits"] / agg["n"]
        md.append(_md(agg) + "\n")
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    n_rej_raw = int((bt["esr_v1_p"] < 0.05).sum())
    n_rej_holm = int((bt["esr_v1_p_holm"] < 0.05).sum())
    n_rej_bh = int((bt["esr_v1_q_bh"] < 0.05).sum())
    summary = {
        "config": str(cfg_path),
        "config_hash": config_hash(cfg),
        "seed": cfg["seed"],
        "n_backtest_cells": len(bt),
        "esr_v1_rejections_5pct": {"raw": n_rej_raw, "holm": n_rej_holm, "bh": n_rej_bh},
        "esr_v1_undefined": int(bt["esr_v1_p"].isna().sum()),
        "forecast_cpu_seconds": float(t["forecasts"]["seconds"].sum())
        if len(t["forecasts"])
        else 0.0,
        "python": platform.python_version(),
    }
    manifest = Path(cfg["run"]["out_dir"]) / "manifest.json"
    if manifest.exists():
        summary["run_manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return json.dumps(summary, indent=2)
