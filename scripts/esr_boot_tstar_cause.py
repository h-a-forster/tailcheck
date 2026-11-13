"""Which part of the v3 standard error makes the bootstrap statistic T* narrower than T0?

Paired null study on the warp-speed cell of ``configs/mc_esr_boot.toml`` (n = 1000,
alpha = 2.5%). Each replication draws one sample and one row-bootstrap resample, fits both
once (``early_stopping=0``; ``scripts/esr_boot_tstar.py`` showed it does not matter) and
computes the ES-intercept standard error of both fits with every estimator below, so the arms
differ only by the standard error:

- ``base``: the package default (``nid`` sparsity, ``scl_sp`` truncated variance, misspec);
- ``sp_nomis``: ``scl_sp`` without the misspecification terms;
- ``ind``, ``ind_nomis``: the sample variance of the negative quantile residuals;
- ``scl_N``: the location-scale truncated-normal variance;
- ``iid_sp``: ``iid`` sparsity with ``scl_sp``;
- ``emp``: the plain influence-function SE of the ES intercept, ``sd(psi) / sqrt(n)`` with
  ``psi = xi - (xi - y) 1{y <= xi} / alpha`` (the v3 FOC; no smoothing, no model).

Each arm also gets the warp-speed size of the bootstrap test at 5% (one-sided "less", as
``esr_backtest`` reports it, and two-sided on ``|T|``) from the pooled ``T*`` of its design,
and ``sd(T*) / sd(T0)`` relative to ``base`` with a paired interval. ``shift_sensitivity``
refits 8 fresh samples and moves the constant subtracted from ``y`` before fitting
(``max(y)``) by +5: the v3 estimate does not depend on it, so neither should its SE.
``tail_mean_baseline`` is the same comparison without any regression: the bootstrap-t of the
plain empirical ES (mean of the ``ceil(n alpha)`` smallest values, delta-method SE) on iid
data, to show whether a narrower ``T*`` is generic to tail means.

Designs: ``garch-t5`` (the cell itself) and ``garch-normal`` (same GARCH, normal shocks).

    uv run python scripts/esr_boot_tstar_cause.py --workers 4   # 72 CPU-minutes, 18 min wall

Replications go through ``UnitStore`` in ``runs/esr-boot-tstar-cause/`` (rerun to resume);
the summary goes to ``results/mc-esr-boot/tstar_cause.json``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tailcheck.checkpoint import UnitStore, run_units

SEED, N, ALPHA, CHUNK = 20261113, 1000, 0.025, 20
REPS = {"garch-t5": 800, "garch-normal": 400}
SE_ARMS: dict[str, tuple[str, str, bool]] = {
    "base": ("nid", "scl_sp", True),
    "sp_nomis": ("nid", "scl_sp", False),
    "ind": ("nid", "ind", True),
    "ind_nomis": ("nid", "ind", False),
    "scl_N": ("nid", "scl_N", True),
    "iid_sp": ("iid", "scl_sp", True),
}
RUN = Path("runs/esr-boot-tstar-cause")
OUT = Path("results/mc-esr-boot/tstar_cause.json")


def _ses(y: Any, xq: Any, seed: int) -> dict[str, float]:
    from tailcheck.esr_backtest import _fit
    from tailcheck.esreg.covariance import vcov_details

    out: dict[str, float] = {}
    try:
        fit = _fit(y, xq, None, ALPHA, 0, seed)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return {"b": math.nan}
    out["b"] = float(fit.coef_e[0])
    for arm, (sparsity, sigma, misspec) in SE_ARMS.items():
        try:
            cov, _ = vcov_details(fit, sparsity, sigma, misspec)  # type: ignore[arg-type]
            out[arm] = math.sqrt(cov[2, 2]) if cov[2, 2] > 0 else math.nan
        except (ValueError, RuntimeError, np.linalg.LinAlgError, ZeroDivisionError):
            out[arm] = math.nan
    xi, _ = fit.fitted()
    psi = xi - (xi - y) * (y <= xi) / ALPHA
    out["emp"] = float(psi.std(ddof=1) / math.sqrt(len(y)))
    return out


def _sample(design: str, rng: np.random.Generator) -> tuple[Any, Any]:
    from tailcheck.dist import normal_var_es
    from tailcheck.montecarlo.dgp import A1, B1, BURN, OMEGA, simulate

    if design == "garch-t5":
        s = simulate("garch-t5", N, ALPHA, rng)
        return s.r - s.e, s.e
    z = rng.standard_normal(N + BURN)
    r = np.empty(N + BURN)
    s2 = np.empty(N + BURN)
    s2[0] = OMEGA / (1 - A1 - B1)
    r[0] = math.sqrt(s2[0]) * z[0]
    for t in range(1, N + BURN):
        s2[t] = OMEGA + A1 * r[t - 1] ** 2 + B1 * s2[t - 1]
        r[t] = math.sqrt(s2[t]) * z[t]
    e = normal_var_es(ALPHA)[1] * np.sqrt(s2[BURN:])
    return r[BURN:] - e, e


def unit(arg: dict[str, Any]) -> dict[str, object]:
    rng = np.random.default_rng([SEED, len(arg["design"]), arg["chunk"]])
    rows = []
    for _ in range(arg["reps"]):
        y, e = _sample(arg["design"], rng)
        seed = int(rng.integers(1 << 31))
        idx = rng.integers(0, N, N)
        full = _ses(y, e, seed)
        boot = _ses(y[idx], e[idx], seed + 1)
        rows.append(
            {
                **{f"full_{k}": v for k, v in full.items()},
                **{f"boot_{k}": v for k, v in boot.items()},
            }
        )
    return {"design": arg["design"], "rows": rows}


def _ratio_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> list[float]:
    idx = rng.integers(0, a.size, (4000, a.size))
    boot = a[idx].std(axis=1, ddof=1) / b[idx].std(axis=1, ddof=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    point = float(a.std(ddof=1) / b.std(ddof=1))
    return [round(point, 4), round(float(lo), 4), round(float(hi), 4)]


def _warp(t0: np.ndarray, ts: np.ndarray, lower_tail: bool) -> list[float]:
    """Warp-speed size at 5% with its 95% interval (``montecarlo.esr_boot``)."""
    from tailcheck.montecarlo.esr_boot import warp_interval, warp_rate

    rate, _ = warp_rate(t0, ts, 0.05, lower_tail)
    lo, hi = warp_interval(t0, ts, 0.05, lower_tail, seed=SEED)
    return [round(rate, 4), round(lo, 4), round(hi, 4)]


def _describe(t: np.ndarray) -> dict[str, float]:
    q05, q95 = np.quantile(t, [0.05, 0.95])
    return {
        "mean": round(float(t.mean()), 3),
        "sd": round(float(t.std(ddof=1)), 3),
        "q05": round(float(q05), 3),
        "q95": round(float(q95), 3),
    }


def summarise(rows: list[dict[str, float]]) -> dict[str, Any]:
    keys = rows[0].keys()
    a = {k: np.array([r.get(k, math.nan) for r in rows], dtype=float) for k in keys}
    ok = np.all([np.isfinite(v) for v in a.values()], axis=0)
    a = {k: v[ok] for k, v in a.items()}
    rng = np.random.default_rng(SEED)
    d = a["boot_b"] - a["full_b"]
    out: dict[str, Any] = {
        "reps": int(ok.sum()),
        "dropped": int((~ok).sum()),
        "sd_b_hat": round(float(a["full_b"].std(ddof=1)), 4),
        "sd_bstar_minus_b_hat": round(float(d.std(ddof=1)), 4),
    }
    for arm in [*SE_ARMS, "emp"]:
        se0, se1 = a[f"full_{arm}"], a[f"boot_{arm}"]
        t0, ts = a["full_b"] / se0, d / se1
        out[arm] = {
            "T0": _describe(t0),
            "Tstar": _describe(ts),
            "sd_Tstar_over_sd_T0": _ratio_ci(ts, t0, rng),
            # SE against the spread it should estimate, in the real and the bootstrap world
            "sd_b_hat_over_rms_se": round(float(a["full_b"].std() / np.sqrt(np.mean(se0**2))), 4),
            "sd_d_over_rms_se_star": round(float(d.std() / np.sqrt(np.mean(se1**2))), 4),
            "median_se_star_over_se": round(float(np.median(se1 / se0)), 4),
            "sd_d_over_se_full": round(float((d / se0).std(ddof=1)), 4),
            "size_one_sided_asym": round(float(np.mean(t0 < -1.6448536269514722)), 4),
            "size_warp_less": _warp(t0, ts, True),
            "size_warp_two_sided": _warp(np.abs(t0), np.abs(ts), False),
        }
    # Paired: (sd(T*)/sd(T0) of the arm) / (the same for base), resampling replications jointly
    base_t0, base_ts = a["full_b"] / a["full_base"], d / a["boot_base"]
    idx = rng.integers(0, d.size, (4000, d.size))

    def ratio(t0: np.ndarray, ts: np.ndarray) -> np.ndarray:
        return ts[idx].std(axis=1, ddof=1) / t0[idx].std(axis=1, ddof=1)

    base_r = ratio(base_t0, base_ts)
    for arm in [*SE_ARMS, "emp"]:
        if arm == "base":
            continue
        rel = ratio(a["full_b"] / a[f"full_{arm}"], d / a[f"boot_{arm}"]) / base_r
        lo, hi = np.quantile(rel, [0.025, 0.975])
        out[arm]["ratio_over_base"] = [round(float(np.median(rel)), 4), round(lo, 4), round(hi, 4)]
    return out


def tail_mean_baseline(reps: int = 4000) -> dict[str, Any]:
    """sd(T*) / sd(T0) for the bootstrap-t of the plain empirical ES on iid normal and t5."""
    from scipy import stats

    rng = np.random.default_rng([SEED, 7])
    k = math.ceil(N * ALPHA)

    def es_se(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.sort(x, axis=-1)
        tail, q = s[..., :k], s[..., k - 1]
        es = tail.mean(-1)
        var = tail.var(-1, ddof=1) + (1 - ALPHA) * (q - es) ** 2
        return es, np.sqrt(var / (N * ALPHA))

    out: dict[str, Any] = {"reps": reps, "tail_obs": k}
    for law in ("normal", "t5"):
        if law == "normal":
            x = rng.standard_normal((reps, N))
            true = -stats.norm.pdf(stats.norm.ppf(ALPHA)) / ALPHA
        else:
            x = rng.standard_t(5.0, (reps, N))
            qt = stats.t.ppf(ALPHA, 5.0)
            true = -stats.t.pdf(qt, 5.0) / ALPHA * (5.0 + qt**2) / 4.0
        es, se = es_se(x)
        esb, seb = es_se(np.take_along_axis(x, rng.integers(0, N, (reps, N)), 1))
        t0, ts = (es - true) / se, (esb - es) / seb
        out[law] = {
            "T0": _describe(t0),
            "Tstar": _describe(ts),
            "sd_Tstar_over_sd_T0": _ratio_ci(ts, t0, rng),
        }
    return out


def shift_sensitivity(samples: int = 8, shift: float = 5.0) -> dict[str, Any]:
    """Relative change of the v3 ES-intercept SE when the internal shift grows by ``shift``."""
    import dataclasses

    from tailcheck.esr_backtest import _fit
    from tailcheck.esreg.covariance import vcov_details

    out: dict[str, Any] = {"samples": samples, "shift_added": shift}
    for misspec in (True, False):
        changes = []
        for k in range(samples):
            y, e = _sample("garch-t5", np.random.default_rng([SEED, 99, k]))
            fit = _fit(y, e, None, ALPHA, 0, k)
            moved = dataclasses.replace(fit, y_shift=fit.y_shift + shift)
            se0, se1 = (
                math.sqrt(vcov_details(f, "nid", "scl_sp", misspec)[0][2, 2]) for f in (fit, moved)
            )
            changes.append(round(se1 / se0 - 1, 4))
        out["misspec" if misspec else "no_misspec"] = changes
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()
    units = []
    for design, total in REPS.items():
        for k, start in enumerate(range(0, total, CHUNK)):
            units.append(
                (f"{design}|{k}", {"design": design, "chunk": k, "reps": min(CHUNK, total - start)})
            )
    store = UnitStore(RUN, f"tstar-cause-{SEED}")
    run_units(store, units, unit, args.workers, max_seconds=(args.max_minutes or 1e6) * 60)
    pooled: dict[str, list[dict[str, float]]] = {}
    for uid in store.completed():
        d = store.read(uid)
        pooled.setdefault(str(d["design"]), []).extend(d["rows"])  # type: ignore[arg-type]
    summary = {
        "n": N,
        "alpha": ALPHA,
        "version": 3,
        "seed": SEED,
        **{k: summarise(v) for k, v in sorted(pooled.items())},
        "shift_sensitivity": shift_sensitivity(),
        "tail_mean_baseline": tail_mean_baseline(),
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
