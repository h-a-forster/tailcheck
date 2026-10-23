"""Monte Carlo runner: size and power of every test, checkpointed per (cell, chunk).

Unit ids: ``cheap|<dgp>|<n>|<alpha>|<chunk>`` (VaR tests, Acerbi-Szekely, McNeil-Frey) and
``esr|<dgp>|<n>|<alpha>|<chunk>`` (ESR versions for that cell). Each unit stores the raw
p-values of its replications, so the report can recompute any rejection rate. A test that
returns NaN or raises counts as a failure (never silently dropped).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from tailcheck.checkpoint import UnitStore, config_hash, run_units
from tailcheck.montecarlo.dgp import as_null, simulate
from tailcheck.study.runner import load_config, unit_seed


def _chunks(reps: int, chunk: int) -> list[int]:
    out = [chunk] * (reps // chunk)
    if reps % chunk:
        out.append(reps % chunk)
    return out


def cheap_unit(arg: dict[str, Any]) -> dict[str, object]:
    from tailcheck.es_tests import acerbi_szekely_pvalue_from_null, mcneil_frey
    from tailcheck.var_tests import (
        christoffersen_cc,
        christoffersen_independence,
        dynamic_quantile,
        kupiec_pof,
    )

    dgp, n, alpha, reps, seed = arg["dgp"], arg["n"], arg["alpha"], arg["reps"], arg["seed"]
    tests = list(arg["tests"])
    rng = np.random.default_rng(seed)
    null_cache: dict[str, dict[str, np.ndarray]] = {}
    pv: dict[str, list[float]] = {t: [] for t in tests}
    t0 = time.perf_counter()
    for _ in range(reps):
        s = simulate(dgp, n, alpha, rng)
        h = s.r <= s.q
        for t in tests:
            try:
                if t == "kupiec":
                    p = kupiec_pof(s.r, s.q, alpha).pvalue
                elif t == "christoffersen_ind":
                    p = christoffersen_independence(s.r, s.q, alpha).pvalue
                elif t == "christoffersen_cc":
                    p = christoffersen_cc(s.r, s.q, alpha).pvalue
                elif t == "dq":
                    p = dynamic_quantile(s.r, s.q, alpha).pvalue
                elif t in ("as_z1", "as_z2"):
                    if s.law not in null_cache:
                        null_cache[s.law] = as_null(
                            s.law,
                            n,
                            alpha,
                            arg["as_n_sim"],
                            np.random.default_rng(arg["null_seeds"][s.law]),
                        )
                    ratio = np.where(h, s.r / s.e, 0.0)
                    if t == "as_z2":
                        z = 1.0 - ratio.sum() / (n * alpha)
                    else:
                        z = 1.0 - ratio.sum() / h.sum() if h.any() else math.nan
                    p = acerbi_szekely_pvalue_from_null(z, null_cache[s.law][t[3:]])
                elif t == "mcneil_frey":
                    p = mcneil_frey(
                        s.r,
                        s.q,
                        s.e,
                        sigma=s.sigma,
                        n_boot=arg["mf_boot"],
                        seed=int(rng.integers(1 << 31)),
                    ).pvalue
                else:
                    raise ValueError(f"unknown test {t}")
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                p = math.nan
            pv[t].append(float(p))
    return {
        "family": "cheap",
        "dgp": dgp,
        "n": n,
        "alpha": alpha,
        "reps": reps,
        "pvalues": pv,
        "seconds_per_rep": (time.perf_counter() - t0) / reps,
    }


def esr_unit(arg: dict[str, Any]) -> dict[str, object]:
    from tailcheck.esr_backtest import esr_backtest

    dgp, n, alpha, reps, seed = arg["dgp"], arg["n"], arg["alpha"], arg["reps"], arg["seed"]
    versions = [int(v) for v in arg["versions"]]
    rng = np.random.default_rng(seed)
    pv: dict[str, list[float]] = {f"esr_v{v}": [] for v in versions}
    if arg.get("compat") and 1 in versions:
        pv["esr_v1_esback_g2pp"] = []
    reasons: dict[str, int] = {}
    secs: list[float] = []
    for _ in range(reps):
        s = simulate(dgp, n, alpha, rng)
        fit_seed = int(rng.integers(1 << 31))
        for v in versions:
            also = v == 1 and "esr_v1_esback_g2pp" in pv
            t0 = time.perf_counter()
            p_compat = math.nan
            try:
                res = esr_backtest(
                    s.r,
                    s.e,
                    alpha,
                    v,  # type: ignore[arg-type]
                    q=s.q,
                    seed=fit_seed,
                    _also_esback_g2pp=also,
                )
                p = res.pvalue
                p_compat = float(res.details.get("pvalue_esback_g2pp", math.nan))
                if math.isnan(p):
                    reason = str(res.details.get("reason", "nan"))
                    reasons[reason[:60]] = reasons.get(reason[:60], 0) + 1
            except Exception as exc:  # noqa: BLE001 - counted as failure
                p = math.nan
                reasons[type(exc).__name__] = reasons.get(type(exc).__name__, 0) + 1
            secs.append(time.perf_counter() - t0)
            pv[f"esr_v{v}"].append(float(p))
            if also:
                pv["esr_v1_esback_g2pp"].append(p_compat)
    return {
        "family": "esr",
        "dgp": dgp,
        "n": n,
        "alpha": alpha,
        "reps": reps,
        "pvalues": pv,
        "failure_reasons": reasons,
        "median_seconds_per_test": float(np.median(secs)),
    }


def build_units(cfg: dict[str, Any]) -> list[tuple[str, Any]]:
    gseed = int(cfg["seed"])
    chunk = int(cfg.get("chunk", 50))
    units: list[tuple[str, Any]] = []
    ch = cfg["cheap"]
    for dgp in list(ch["dgps_size"]) + list(ch["dgps_power"]):
        reps = int(ch["reps_size"] if dgp in ch["dgps_size"] else ch["reps_power"])
        for n in ch["n"]:
            for a in ch["alphas"]:
                null_seeds = {
                    law: unit_seed(gseed, f"asnull|{law}|{n}|{a}") for law in ("t5", "normal")
                }
                for k, r in enumerate(_chunks(reps, chunk)):
                    uid = f"cheap|{dgp}|{n}|{a}|{k}"
                    units.append(
                        (
                            uid,
                            {
                                "dgp": dgp,
                                "n": int(n),
                                "alpha": float(a),
                                "reps": r,
                                "seed": unit_seed(gseed, uid),
                                "tests": ch["tests"],
                                "as_n_sim": int(ch["as_n_sim"]),
                                "mf_boot": int(ch["mcneil_frey_boot"]),
                                "null_seeds": null_seeds,
                            },
                        )
                    )
    es = cfg["esr"]
    combos: dict[tuple[int, float], set[int]] = {}
    for n in es["n"]:
        for a in es["alphas"]:
            combos.setdefault((int(n), float(a)), set()).update(int(v) for v in es["versions"])
    for extra in es.get("extra_cells", []):
        combos.setdefault((int(extra["n"]), float(extra["alpha"])), set()).add(
            int(extra["version"])
        )
    esr_units = []
    for dgp in list(es["dgps_size"]) + list(es["dgps_power"]):
        size = dgp in es["dgps_size"]
        reps = int(es["reps_size"] if size else es["reps_power"])
        for (n, a), vers in sorted(combos.items()):
            for k, r in enumerate(_chunks(reps, int(es.get("chunk", 10)))):
                uid = f"esr|{dgp}|{n}|{a}|{k}"
                esr_units.append(
                    (
                        uid,
                        {
                            "dgp": dgp,
                            "n": n,
                            "alpha": a,
                            "reps": r,
                            "versions": sorted(vers),
                            "compat": size,
                            "seed": unit_seed(gseed, uid),
                        },
                    )
                )
    # ESR units are slow: run them first so the pool stays busy
    return esr_units + units


def _dispatch(arg: dict[str, Any]) -> dict[str, object]:
    return esr_unit(arg) if "versions" in arg else cheap_unit(arg)


def run_mc(cfg_path: Path, workers: int = 4, force_new: bool = False) -> Path:
    cfg = load_config(cfg_path)
    out_dir = Path(cfg["run"]["out_dir"])
    store = UnitStore(out_dir, config_hash(cfg), force_new=force_new)
    run_units(store, build_units(cfg), _dispatch, workers=workers, on_error="log")
    return out_dir
