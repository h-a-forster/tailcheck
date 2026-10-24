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


@pytest.mark.parametrize(
    ("content", "needle"),
    [
        ("r,q,e\n,-1,-2\n" + "0.1,-1,-2\n" * 20, "NaN"),
        ("r,q,e\n0.1,-1,-2\n0.2,-1,-2\n0.3,-1,-2\n", "observations"),
        ("r,q,e\nabc,-1,-2\n" + "0.1,-1,-2\n" * 20, "not numeric"),
    ],
)
def test_backtest_bad_input_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str, needle: str
) -> None:
    p = tmp_path / "x.csv"
    p.write_text(content)
    assert main(["backtest", str(p), "--no-esr", "--n-boot", "99"]) == 2
    cap = capsys.readouterr()
    assert cap.out == ""  # nothing printed before validation
    assert cap.err.startswith("error:") and needle in cap.err
    assert "Traceback" not in cap.err


def test_backtest_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["backtest", str(tmp_path / "missing.csv")]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_unknown_stage_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["study", "run", "--config", "configs/study.toml", "--stages", "forcast"])
    assert exc.value.code == 2
    assert "unknown stage" in capsys.readouterr().err


def test_report_without_run_is_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = Path(__file__).parent.parent / "configs" / "study.toml"
    monkeypatch.chdir(tmp_path)
    assert main(["study", "report", "--config", str(cfg)]) == 2
    assert "no run found" in capsys.readouterr().err
    assert not (tmp_path / "results").exists() and not (tmp_path / "runs").exists()


def test_mc_boot_subcommands_parse() -> None:
    from tailcheck.cli import build_parser

    args = build_parser().parse_args(
        ["mc", "boot-run", "--config", "c.toml", "--workers", "2", "--max-minutes", "5"]
    )
    assert args.action == "boot-run" and args.max_minutes == 5.0 and args.workers == 2
    assert build_parser().parse_args(["mc", "boot-report", "--config", "c.toml"]).action == (
        "boot-report"
    )
