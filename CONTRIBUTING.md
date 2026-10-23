# Contributing

Bug reports, new backtests, forecasters and reference fixtures are welcome.

## Setup

You need [uv](https://docs.astral.sh/uv/) and git.

```sh
uv sync --group dev
uv run tailcheck --version
```

## Checks

```sh
uv run pytest              # fast suite; add --runslow for long Monte Carlo checks
uv run ruff check .
uv run ruff format --check .
```

CI runs these on pull requests (Ubuntu with Python 3.11 and 3.13, Windows with Python 3.13).

Test rules:

- No network. Tests use synthetic data or committed fixtures.
- Use `tmp_path` for every file.
- Tests must pass on Windows: use `pathlib` and do not assume a POSIX shell.
- Fix every random seed. Statistical tests must be deterministic, not flaky.
- Mark anything over a few seconds with `@pytest.mark.slow`.

## Adding a backtest

1. Put the formula, its reference and its edge cases in `docs/DESIGN.md` first.
2. Return a `tailcheck.results.BacktestResult`. Return `nan` with `details["reason"]` for
   statistically undefined input; raise `ValueError` for invalid input.
3. Add an oracle test (closed form, published table or R fixture) and a Monte Carlo size
   check.

## R reference fixtures

`scripts/r_fixtures/run_r.mjs` runs R scripts in webR (R compiled to WebAssembly) under Node,
so no R installation is needed:

```sh
npm ci --ignore-scripts --prefix scripts/r_fixtures
node scripts/r_fixtures/run_r.mjs scripts/r_fixtures/versions.R tests/fixtures/r esreg esback
```

Commit only the small JSON outputs and their CSV inputs. Tests read the JSON and never call
Node or R. The R packages are GPL-3: use them as black-box oracles and do not port their
source line by line.

## Code

- Python 3.11+, typed. Runtime dependencies: numpy, scipy, pandas, arch. Open an issue before
  adding one.
- Match the surrounding style; ruff enforces the rest (line length 100).
- Long computations must go through `tailcheck.checkpoint.UnitStore`, so they can resume
  after a crash.
