"""Command-line interface: ``tailcheck backtest | mc | study | data``."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from tailcheck import __version__

STAGES = ("forecast", "backtest", "compare")
WORKERS_HELP = "worker processes (default: [run] workers in the config, else 4; capped at 4)"


def _stages(text: str) -> tuple[str, ...]:
    parts = tuple(p.strip() for p in text.split(",") if p.strip())
    bad = [p for p in parts if p not in STAGES]
    if bad or not parts:
        raise argparse.ArgumentTypeError(
            f"unknown stage(s) {bad or [text]}; choose from {','.join(STAGES)}"
        )
    return parts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tailcheck",
        description="VaR and Expected Shortfall backtests with Monte Carlo-checked p-values.",
    )
    parser.add_argument("--version", action="version", version=f"tailcheck {__version__}")
    sub = parser.add_subparsers(dest="command")

    bt = sub.add_parser("backtest", help="run all backtests on a CSV with columns r, q, e[, sigma]")
    bt.add_argument("csv", type=Path)
    bt.add_argument("--alpha", type=float, default=0.025)
    bt.add_argument("--n-boot", type=int, default=2000, help="McNeil-Frey bootstrap size")
    bt.add_argument("--seed", type=int, default=1)
    bt.add_argument("--no-esr", action="store_true", help="skip the (slower) ESR regressions")

    data = sub.add_parser("data", help="download and cache public data")
    data_sub = data.add_subparsers(dest="action", required=True)
    fetch = data_sub.add_parser("fetch")
    fetch.add_argument("--config", type=Path, required=True)

    study = sub.add_parser("study", help="forecaster study: run | report")
    study_sub = study.add_subparsers(dest="action", required=True)
    sr = study_sub.add_parser("run")
    sr.add_argument("--config", type=Path, required=True)
    sr.add_argument("--workers", type=int, default=None, help=WORKERS_HELP)
    sr.add_argument("--force-new", action="store_true")
    sr.add_argument(
        "--stages",
        type=_stages,
        default=STAGES,
        help="comma-separated subset of " + ",".join(STAGES) + " (default: all)",
    )
    srep = study_sub.add_parser("report")
    srep.add_argument("--config", type=Path, required=True)

    mc = sub.add_parser("mc", help="Monte Carlo size/power: run | report | boot-run | boot-report")
    mc_sub = mc.add_subparsers(dest="action", required=True)
    mr = mc_sub.add_parser("run")
    mr.add_argument("--config", type=Path, required=True)
    mr.add_argument("--workers", type=int, default=None, help=WORKERS_HELP)
    mr.add_argument("--force-new", action="store_true")
    mrep = mc_sub.add_parser("report")
    mrep.add_argument("--config", type=Path, required=True)
    mbr = mc_sub.add_parser("boot-run", help="bootstrap ESR size/power (configs/mc_esr_boot.toml)")
    mbr.add_argument("--config", type=Path, required=True)
    mbr.add_argument("--workers", type=int, default=None, help=WORKERS_HELP)
    mbr.add_argument("--force-new", action="store_true")
    mbr.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="start no new unit after this many minutes (rerun to resume)",
    )
    mbrep = mc_sub.add_parser("boot-report")
    mbrep.add_argument("--config", type=Path, required=True)
    return parser


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{x:.4f}" if abs(x) >= 1e-4 else f"{x:.2e}"


def cmd_backtest(args: argparse.Namespace) -> int:
    import numpy as np
    import pandas as pd

    from tailcheck._validation import as_float_1d, check_alpha, check_same_length
    from tailcheck.es_tests import mcneil_frey
    from tailcheck.esr_backtest import esr_backtest
    from tailcheck.scoring import fz0_loss, mean_with_ci
    from tailcheck.var_tests import (
        christoffersen_cc,
        christoffersen_independence,
        dynamic_quantile,
        kupiec_pof,
        traffic_light,
    )

    df = pd.read_csv(args.csv)
    missing = {"r", "q", "e"} - set(df.columns)
    if missing:
        print(f"error: CSV lacks columns {sorted(missing)}", file=sys.stderr)
        return 2
    cols = ["r", "q", "e"] + (["sigma"] if "sigma" in df else [])
    arrays = {}
    for c in cols:
        try:
            arrays[c] = pd.to_numeric(df[c], errors="raise").to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"column {c!r} is not numeric: {exc}") from exc
        arrays[c] = as_float_1d(arrays[c], c)  # rejects NaN / infinite values
    check_same_length([arrays[c] for c in cols], cols)
    r, q, e = arrays["r"], arrays["q"], arrays["e"]
    sigma = arrays.get("sigma")
    a = check_alpha(args.alpha)
    # compute everything before printing, so bad input fails with a clean error
    results = [
        kupiec_pof(r, q, a),
        christoffersen_independence(r, q, a),
        christoffersen_cc(r, q, a),
        dynamic_quantile(r, q, a),
        mcneil_frey(r, q, e, sigma=sigma, n_boot=args.n_boot, seed=args.seed),
    ]
    if not args.no_esr:
        results += [esr_backtest(r, e, a, v, q=q, seed=args.seed) for v in (1, 2, 3)]  # type: ignore[arg-type]
    tl = traffic_light(r, q, a)
    hits = int(np.sum(r <= q))
    print(f"n = {len(r)}, alpha = {a}, hits = {hits} (expected {a * len(r):.1f})")
    print(f"{'test':<22}{'statistic':>12}{'p-value':>12}  alternative")
    for res in results:
        note = f"  ({res.details['reason']})" if "reason" in res.details else ""
        print(
            f"{res.test:<22}{_fmt(res.statistic):>12}{_fmt(res.pvalue):>12}"
            f"  {res.alternative}{note}"
        )
    print(f"traffic light: {tl.zone} (binomial CDF {tl.cumulative_probability:.4f})")
    if np.all(e <= q) and np.all(q < 0):
        ci = mean_with_ci(fz0_loss(r, q, e, a))
        print(f"mean FZ0 loss: {ci['mean']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}] (95% NW)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    from tailcheck.checkpoint import ConfigMismatchError
    from tailcheck.data import DataError

    try:
        return _dispatch(args)
    except (ValueError, OSError, DataError, ConfigMismatchError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "backtest":
        return cmd_backtest(args)
    if args.command == "data":
        from tailcheck.data import fetch_assets, write_manifest
        from tailcheck.study.runner import load_config

        cfg = load_config(args.config)
        manifest = fetch_assets(
            cfg["data"]["assets"], cfg["data"]["end"], Path(cfg["data"]["cache_dir"])
        )
        out = Path("results") / cfg["name"] / "data_manifest.json"
        write_manifest(out, manifest)
        for k, v in manifest.items():
            print(
                f"{k:<10} {v['first']} .. {v['last']}  n={v['n_returns']}"
                f"  sha256={v['sha256'][:12]}"
            )
        print(f"manifest: {out}")
        return 0
    if args.command == "study":
        if args.action == "run":
            from tailcheck.study.runner import run_study

            out = run_study(
                args.config,
                workers=args.workers,
                force_new=args.force_new,
                stages=args.stages,
            )
            print(f"run directory: {out}")
            return 0
        from tailcheck.study.report import write_study_report

        print(write_study_report(args.config))
        return 0
    if args.command == "mc":
        if args.action == "boot-run":
            from tailcheck.montecarlo.esr_boot import run_boot_mc

            out = run_boot_mc(
                args.config,
                workers=args.workers,
                force_new=args.force_new,
                max_minutes=args.max_minutes,
            )
            print(f"run directory: {out}")
            return 0
        if args.action == "boot-report":
            from tailcheck.montecarlo.esr_boot import write_boot_report

            print(write_boot_report(args.config))
            return 0
        if args.action == "run":
            from tailcheck.montecarlo.runner import run_mc

            out = run_mc(args.config, workers=args.workers, force_new=args.force_new)
            print(f"run directory: {out}")
            return 0
        from tailcheck.montecarlo.report import write_mc_report

        print(write_mc_report(args.config))
        return 0
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
