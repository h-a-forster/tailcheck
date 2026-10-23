"""Scaffold checks: the package imports, exposes its version, and the CLI starts."""

from __future__ import annotations

import importlib
import math

import pytest

import tailcheck
from tailcheck.cli import main
from tailcheck.results import BacktestResult

MODULES = [
    "tailcheck.var_tests",
    "tailcheck.es_tests",
    "tailcheck.esreg",
    "tailcheck.esr_backtest",
    "tailcheck.scoring",
    "tailcheck.forecasters",
    "tailcheck.forecasters.historical",
    "tailcheck.forecasters.garch",
    "tailcheck.forecasters.caviar",
    "tailcheck.data",
    "tailcheck.checkpoint",
    "tailcheck.montecarlo",
    "tailcheck.study",
]


@pytest.mark.parametrize("name", MODULES)
def test_modules_import(name: str) -> None:
    importlib.import_module(name)


def test_version_is_a_string() -> None:
    assert isinstance(tailcheck.__version__, str) and tailcheck.__version__


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "tailcheck" in capsys.readouterr().out


def test_cli_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_backtest_result_reject() -> None:
    res = BacktestResult(test="x", statistic=1.0, pvalue=0.01, nobs=10)
    assert res.reject(0.05)
    assert not res.reject(0.001)
    undefined = BacktestResult(test="x", statistic=math.nan, pvalue=math.nan, nobs=10)
    assert not undefined.reject(0.05)
    with pytest.raises(ValueError):
        res.reject(1.5)
    assert res.to_dict()["pvalue"] == 0.01
