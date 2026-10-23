"""M10: command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from tailcheck.cli import main

EXAMPLE = Path(__file__).parent.parent / "examples" / "garch_t5.csv"


def test_backtest_prints_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["backtest", str(EXAMPLE), "--alpha", "0.025", "--n-boot", "199"]) == 0
    out = capsys.readouterr().out
    for name in (
        "kupiec_pof",
        "christoffersen_cc",
        "dynamic_quantile",
        "mcneil_frey",
        "esr_v1",
        "esr_v3",
        "traffic light",
        "mean FZ0",
    ):
        assert name in out


def test_backtest_missing_columns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "x.csv"
    p.write_text("r,q\n1,2\n")
    assert main(["backtest", str(p)]) == 2
