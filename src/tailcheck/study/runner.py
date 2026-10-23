"""Study runner: forecast, backtest and compare stages, each checkpointed per unit.

Units (DESIGN 5): ``forecast__<asset>__<model>`` (NPZ arrays + JSON marker),
``backtest__<asset>__<model>__<alpha>`` and ``compare__<asset>__<alpha>`` (JSON).
Every finished unit is written atomically through :class:`tailcheck.checkpoint.UnitStore`
and logged to ``<out_dir>/progress.log``; a rerun skips finished units.
"""

from __future__ import annotations

import math
import time
import tomllib
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tailcheck.checkpoint import UnitStore, _atomic_write_bytes, config_hash, run_units, safe_id
from tailcheck.data import load_returns

REGIMES = ("full", "calm", "turbulent")


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def unit_seed(global_seed: int, unit_id: str) -> int:
    """Seed for a unit, independent of run order (DESIGN 3)."""
    return int(
        np.random.SeedSequence([int(global_seed), zlib.crc32(unit_id.encode())]).generate_state(1)[
            0
        ]
    )


def asset_frame(cfg: Mapping[str, Any], asset: str) -> tuple[pd.Series, int, int, np.ndarray]:
    """Returns, OOS start/end indices and the ex-ante regime label (1 = turbulent) per OOS day."""
    f = cfg["forecast"]
    r = load_returns(Path(cfg["data"]["cache_dir"]), asset)
    r = r.loc[: pd.Timestamp(f["oos_end"])]
    idx = r.index
    oos_start = int(idx.searchsorted(pd.Timestamp(f["oos_start"])))
    oos_end = len(r)
    if oos_start < int(f["window"]):
        raise ValueError(f"{asset}: only {oos_start} observations before the OOS start")
    lb = int(cfg.get("regimes", {}).get("vol_lookback", 63))
    vals = r.to_numpy()
    rv = np.array([vals[t - lb : t].std(ddof=1) for t in range(oos_start, oos_end)])
    turb = (rv > np.median(rv)).astype(np.int8)
    return r, oos_start, oos_end, turb


def forecast_unit(arg: dict[str, Any]) -> dict[str, object]:
    """Rolling forecasts for one (asset, model) over all alphas; writes the NPZ itself."""
    from tailcheck.forecasters import make_forecaster, rolling_forecast

    cfg, asset, model_name, npz_path, seed = (
        arg["cfg"],
        arg["asset"],
        arg["model"],
        Path(arg["npz"]),
        arg["seed"],
    )
    f = cfg["forecast"]
    alphas = [float(a) for a in f["alphas"]]
    r, s, e_idx, turb = asset_frame(cfg, asset)
    vals = r.to_numpy()
    model = make_forecaster(
        model_name,
        evt_tail_fraction=float(f.get("evt_tail_fraction", 0.10)),
        caviar_random_starts=int(f.get("caviar_random_starts", 2000)),
        caviar_best_starts=int(f.get("caviar_best_starts", 10)),
    )
    n_draws = int(f.get("n_draws", 0)) if model.has_distribution else 0
    acc_s = np.zeros((len(REGIMES), len(alphas), max(n_draws, 1)))
    acc_n = np.zeros_like(acc_s)

    def on_block(t0: int, t1: int, blk: Any) -> None:
        if blk.draws is None:
            return
        reg = turb[t0 - s : t1 - s]
        for j in range(len(alphas)):
            qj = blk.q[:, j][:, None]
            ej = blk.e[:, j][:, None]
            hit = blk.draws <= qj
            contrib = np.where(hit, blk.draws / ej, 0.0)
            for g, mask in enumerate((np.ones_like(reg, bool), reg == 0, reg == 1)):
                if mask.any():
                    acc_s[g, j] += contrib[mask].sum(axis=0)
                    acc_n[g, j] += hit[mask].sum(axis=0)

    t_start = time.perf_counter()
    out = rolling_forecast(
        model,
        vals,
        s,
        e_idx,
        int(f["window"]),
        int(f["refit_every"]),
        alphas,
        n_draws=n_draws,
        seed=seed,
        on_block=on_block,
    )
    import io

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        dates=r.index[s:e_idx].to_numpy().astype("datetime64[D]").astype(np.int64),
        r=vals[s:e_idx],
        q=out["q"],
        e=out["e"],
        sigma=out["sigma"],
        turbulent=turb,
        alphas=np.array(alphas),
        as_sum=acc_s if n_draws else np.zeros((0,)),
        as_hits=acc_n if n_draws else np.zeros((0,)),
    )
    _atomic_write_bytes(npz_path, buf.getvalue())
    return {
        "asset": asset,
        "model": model_name,
        "n_oos": int(e_idx - s),
        "n_refits": int(out["n_refits"]),
        "n_draws": n_draws,
        "forecast_seconds": time.perf_counter() - t_start,
    }


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def backtest_unit(arg: dict[str, Any]) -> dict[str, object]:
    """All backtests for one (asset, model, alpha), plus per-regime summaries."""
    from tailcheck.es_tests import acerbi_szekely_pvalue_from_null, acerbi_szekely_z2, mcneil_frey
    from tailcheck.esr_backtest import esr_backtest
    from tailcheck.scoring import fz0_loss, mean_with_ci
    from tailcheck.var_tests import (
        christoffersen_cc,
        christoffersen_independence,
        dynamic_quantile,
        kupiec_pof,
        traffic_light,
    )

    cfg, npz, j, alpha, seed = arg["cfg"], Path(arg["npz"]), arg["j"], arg["alpha"], arg["seed"]
    bt = cfg["backtest"]
    with np.load(npz) as z:
        r, q, e, sigma, turb = z["r"], z["q"][:, j], z["e"][:, j], z["sigma"], z["turbulent"]
        as_sum = z["as_sum"]
    n = len(r)
    hit = r <= q
    k = int(hit.sum())
    out: dict[str, Any] = {"asset": arg["asset"], "model": arg["model"], "alpha": alpha, "n": n}
    lo, hi = _wilson(k, n)
    out["hits"] = k
    out["hit_rate"] = k / n
    out["hit_rate_lo"], out["hit_rate_hi"] = lo, hi
    out["kupiec"] = kupiec_pof(r, q, alpha).to_dict()
    out["christoffersen_ind"] = christoffersen_independence(r, q, alpha).to_dict()
    out["christoffersen_cc"] = christoffersen_cc(r, q, alpha).to_dict()
    out["dq"] = dynamic_quantile(r, q, alpha, lags=int(bt.get("dq_lags", 4))).to_dict()
    tl = traffic_light(r, q, alpha)
    out["traffic_light"] = {"zone": tl.zone, "cumulative_probability": tl.cumulative_probability}
    nb = int(bt.get("esr_bootstrap", 0))
    for v in bt.get("esr_versions", [1, 2, 3]):
        res = esr_backtest(r, e, alpha, int(v), q=q, n_boot=nb, seed=seed + int(v))
        out[f"esr_v{v}"] = res.to_dict()
    out["esr_v1_esback_g2pp"] = esr_backtest(
        r, e, alpha, 1, q=q, seed=seed + 1, _esreg_compat_g2pp=True
    ).to_dict()
    has_sigma = bool(np.all(np.isfinite(sigma)))
    out["mcneil_frey"] = mcneil_frey(
        r,
        q,
        e,
        sigma=sigma if has_sigma else None,
        n_boot=int(bt.get("mcneil_frey_boot", 2000)),
        seed=seed,
    ).to_dict()
    out["mcneil_frey"]["details.scaled_by_sigma"] = has_sigma
    loss = fz0_loss(r, q, e, alpha)
    out["fz0"] = mean_with_ci(loss)
    masks = {"full": np.ones(n, bool), "calm": turb == 0, "turbulent": turb == 1}
    if as_sum.size:
        for g, name in enumerate(REGIMES):
            m = masks[name]
            z_obs = acerbi_szekely_z2(r[m], q[m], e[m], alpha)
            z_null = 1.0 - as_sum[g, j] / (m.sum() * alpha)
            out[f"as_z2_{name}"] = {
                "statistic": z_obs,
                "pvalue": acerbi_szekely_pvalue_from_null(z_obs, z_null),
                "n_sim": int(z_null.size),
                "null_q05": float(np.quantile(z_null, 0.05)),
            }
    else:
        out["as_z2_full"] = {
            "statistic": math.nan,
            "pvalue": math.nan,
            "reason": "model has no predictive distribution",
        }
    for name in ("calm", "turbulent"):
        m = masks[name]
        kk = int(hit[m].sum())
        lo, hi = _wilson(kk, int(m.sum()))
        out[f"regime_{name}"] = {
            "n": int(m.sum()),
            "hits": kk,
            "hit_rate": kk / max(int(m.sum()), 1),
            "hit_rate_lo": lo,
            "hit_rate_hi": hi,
            "kupiec": kupiec_pof(r[m], q[m], alpha).to_dict(),
            "esr_v1": esr_backtest(r[m], e[m], alpha, 1, seed=seed + 11).to_dict(),
            "esr_v3": esr_backtest(r[m], e[m], alpha, 3, seed=seed + 13).to_dict(),
            "fz0": mean_with_ci(loss[m]),
        }
    return out


def compare_unit(arg: dict[str, Any]) -> dict[str, object]:
    """MCS (both sizes) and DM vs the benchmark on FZ0 losses for one (asset, alpha)."""
    from tailcheck.scoring import diebold_mariano, fz0_loss, model_confidence_set

    cfg, j, alpha, seed = arg["cfg"], arg["j"], arg["alpha"], arg["seed"]
    c = cfg["compare"]
    losses = {}
    for model, npz in arg["npz"].items():
        with np.load(npz) as z:
            losses[model] = fz0_loss(z["r"], z["q"][:, j], z["e"][:, j], alpha)
    frame = pd.DataFrame(losses)
    out: dict[str, Any] = {
        "asset": arg["asset"],
        "alpha": alpha,
        "mean_loss": frame.mean().to_dict(),
    }
    for size in c.get("mcs_size", [0.10, 0.25]):
        res = model_confidence_set(
            frame,
            size=float(size),
            reps=int(c.get("mcs_reps", 10000)),
            block_size=int(c.get("mcs_block_size", 10)),
            method=c.get("mcs_method", "R"),
            seed=seed,
        )
        out[f"mcs_{int(round(100 * (1 - float(size))))}"] = {
            "included": list(res.included),
            "pvalues": dict(res.pvalues),
            "reps": res.reps,
            "block_size": res.block_size,
            "seed": res.seed,
        }
    bench = "gjr-t"
    out["dm_vs"] = bench
    if bench in frame:
        out["dm"] = {
            m: diebold_mariano(
                frame[m], frame[bench], small_sample=bool(c.get("dm_small_sample", True))
            ).to_dict()
            for m in frame.columns
            if m != bench
        }
    return out


def run_study(
    cfg_path: Path,
    workers: int = 4,
    force_new: bool = False,
    stages: tuple[str, ...] = ("forecast", "backtest", "compare"),
) -> Path:
    """Run (or resume) all stages; returns the run directory."""
    cfg = load_config(cfg_path)
    out_dir = Path(cfg["run"]["out_dir"])
    store = UnitStore(out_dir, config_hash(cfg), force_new=force_new)
    gseed = int(cfg["seed"])
    assets = [a["id"] for a in cfg["data"]["assets"]]
    models = list(cfg["forecast"]["models"])
    alphas = [float(a) for a in cfg["forecast"]["alphas"]]

    def npz(asset: str, model: str) -> str:
        return str(store.units_dir / f"{safe_id(f'forecast__{asset}__{model}')}.npz")

    if "forecast" in stages:
        units = []
        for asset in assets:
            for model in models:
                uid = f"forecast__{asset}__{model}"
                units.append(
                    (
                        uid,
                        {
                            "cfg": cfg,
                            "asset": asset,
                            "model": model,
                            "npz": npz(asset, model),
                            "seed": unit_seed(gseed, uid),
                        },
                    )
                )
        # CAViaR units are the slowest; start them first for better packing
        units.sort(key=lambda u: u[1]["model"] != "caviar-sav")
        run_units(store, units, forecast_unit, workers=workers, on_error="log")
    if "backtest" in stages:
        units = []
        for asset in assets:
            for model in models:
                if not store.done(f"forecast__{asset}__{model}"):
                    continue
                for j, a in enumerate(alphas):
                    uid = f"backtest__{asset}__{model}__{a}"
                    units.append(
                        (
                            uid,
                            {
                                "cfg": cfg,
                                "asset": asset,
                                "model": model,
                                "j": j,
                                "alpha": a,
                                "npz": npz(asset, model),
                                "seed": unit_seed(gseed, uid) % (2**31 - 100),
                            },
                        )
                    )
        run_units(store, units, backtest_unit, workers=workers, on_error="log")
    if "compare" in stages:
        units = []
        for asset in assets:
            avail = {m: npz(asset, m) for m in models if store.done(f"forecast__{asset}__{m}")}
            if len(avail) < 2:
                continue
            for j, a in enumerate(alphas):
                uid = f"compare__{asset}__{a}"
                units.append(
                    (
                        uid,
                        {
                            "cfg": cfg,
                            "asset": asset,
                            "j": j,
                            "alpha": a,
                            "npz": avail,
                            "seed": unit_seed(gseed, uid) % (2**31 - 100),
                        },
                    )
                )
        run_units(store, units, compare_unit, workers=workers, on_error="log")
    return out_dir
