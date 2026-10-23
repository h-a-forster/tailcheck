"""Command-line interface: ``tailcheck backtest | mc | study | data``."""

from __future__ import annotations

import argparse
import sys

from tailcheck import __version__

_PLANNED = {
    "backtest": "run all backtests on a CSV with columns r, q, e[, sigma] (M10)",
    "mc": "Monte Carlo size/power validation: run | report (M8)",
    "study": "forecaster bake-off: run | report (M7, M9)",
    "data": "download and cache public data: fetch (M7)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tailcheck",
        description="VaR and Expected Shortfall backtests with Monte Carlo-checked p-values.",
    )
    parser.add_argument("--version", action="version", version=f"tailcheck {__version__}")
    sub = parser.add_subparsers(dest="command")
    for name, help_text in _PLANNED.items():
        sub.add_parser(name, help=help_text)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    print(f"tailcheck {args.command}: not implemented yet", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
