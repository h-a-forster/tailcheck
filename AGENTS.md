# AGENTS.md

Guide for coding agents (and humans) working on tailcheck. Read `docs/DESIGN.md` before
changing any statistics. It holds the formulas, reference values and milestones this file
points to.

## What this is

A Python package of VaR/ES backtests (Kupiec, Christoffersen, dynamic quantile, Basel traffic
light, Acerbi-Szekely, McNeil-Frey, Bayer-Dimitriadis ESR on a joint quantile/ES regression),
FZ0/DM/MCS comparisons, a Monte Carlo size/power validation, and a forecaster study on crypto
and equity indices.

## Conventions

- uv + hatchling, `src/` layout, Python >= 3.11 (local dev pinned to 3.13 by
  `.python-version`), typed code, `ruff check` + `ruff format` (line length 100), pytest.
- Sign convention: `r`, `q` (VaR) and `e` (ES) on the return scale; `e <= q < 0` in the left
  tail; hit = `r <= q`. Returns in percent log returns.
- Every test returns `BacktestResult`. Statistically undefined input gives `pvalue = nan` plus
  `details["reason"]`; invalid input raises `ValueError`.
- Simulation and bootstrap p-values use `(1 + count)/(B + 1)`. Every stochastic function takes
  `seed`.
- No LLM or model API calls. CPU only. At most 4 worker processes, with BLAS threads set to 1
  in each worker.
- Long jobs: every finished unit goes through `tailcheck.checkpoint.UnitStore` (atomic file per
  unit, skip on rerun, `progress.log`). Never keep results only in memory.
- Data: never commit raw downloads (`data/`, `cache/` and `runs/` are gitignored). Commit only
  small derived results under `results/`.
- `esreg`/`esback` are GPL-3. Implement from the papers; use the R packages only as
  black-box oracles through `scripts/r_fixtures/` (webR). Do not translate their source.
- Scratch files go in `tmp/` (gitignored). Do not create `.bak` copies; use git.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do not push and
  do not create remote repositories; publishing is done separately after review.
- Statistical honesty: report sizes, seeds, versions and intervals; adjust for multiple
  testing where DESIGN says so; never overstate. Before calling anything a library bug,
  produce a minimal repro and rule out an adapter mistake.

## Architecture map

| Path | Role | Milestone |
|---|---|---|
| `src/tailcheck/results.py` | `BacktestResult` | done |
| `src/tailcheck/_validation.py` | input checks | done |
| `src/tailcheck/checkpoint.py` | resumable unit store + progress log | M1 |
| `src/tailcheck/var_tests.py` | Kupiec, Christoffersen, DQ, traffic light | M2 |
| `src/tailcheck/es_tests.py` | Acerbi-Szekely Z1/Z2, McNeil-Frey | M3 |
| `src/tailcheck/esreg/` | joint quantile/ES regression: G functions, loss, LP quantile regression, estimator, covariance | M4a, M4b |
| `src/tailcheck/esr_backtest.py` | ESR backtests v1-v3 | M4c |
| `src/tailcheck/scoring.py` | FZ0, Diebold-Mariano, MCS (`arch`) | M5 |
| `src/tailcheck/forecasters/` | hs, gjr-t, fhs, evt-pot, caviar-sav | M6 |
| `src/tailcheck/data.py` | Binance + FRED download, cache, returns | M7 |
| `src/tailcheck/study/` | study runner + report | M7, M9 |
| `src/tailcheck/montecarlo/` | DGPs, MC runner + report | M8 |
| `src/tailcheck/cli.py` | `tailcheck backtest / mc / study / data` | M10 |
| `configs/` | study and Monte Carlo configs (quick + full) | - |
| `scripts/reference_values.py` | independent recomputation of the reference values in DESIGN | done |
| `scripts/r_fixtures/` | webR runner (`run_r.mjs`) for R oracle fixtures | runner done; R scripts M4 |
| `tests/fixtures/r/` | committed R inputs (CSV) and outputs (JSON) | M4 |

## How to test

```sh
uv sync --group dev
uv run pytest                 # fast suite (CI)
uv run pytest --runslow       # plus slow Monte Carlo checks
uv run ruff check .
uv run ruff format --check .
```

R fixtures (only when (re)generating them; needs Node 20+):

```sh
npm ci --ignore-scripts --prefix scripts/r_fixtures
node scripts/r_fixtures/run_r.mjs scripts/r_fixtures/<script>.R tests/fixtures/r esreg esback quantreg
```

The runner patches a Windows path bug in webR 0.6.0's worker inside the gitignored
`node_modules` copy. The first run downloads the R packages from repo.r-wasm.org (about 1
minute).

## How to run the study and the Monte Carlo (after M7-M9)

```sh
uv run tailcheck data fetch  --config configs/study.toml
uv run tailcheck study run   --config configs/study.toml --workers 4      # about 16 CPU-min, 4.5 min wall
uv run tailcheck study report --config configs/study.toml                 # results/study-v1/ (REPORT.md is written by hand)
uv run tailcheck mc run      --config configs/mc_quick.toml --workers 4   # about 45 CPU-min, 12 min wall
uv run tailcheck mc report   --config configs/mc_quick.toml
```

Full size: `configs/study_full.toml` (daily refits, bootstrap ESR; projected 20-30
CPU-hours) and `configs/mc_full.toml` (projected 35-45 CPU-hours); projections, not
measurements (see the config headers). Both resume from `runs/<name>/units/`. A run directory
refuses to resume when the config, the processed data or the package version changed (use
`--force-new` or a new `out_dir`); changed code only warns. Reports are read-only and fail
cleanly without a finished run. Before a
long run, validate on a small sample, wrap it in a resume-on-failure loop, and watch
`runs/<name>/progress.log` for stalls.

## Status and next steps

Updated: 2026-10-23.

Done:
- Scaffold, `docs/DESIGN.md`, research checks (FRED instead of Stooq; Binance ms->us
  timestamps in 2025; webR runs esreg/esback).
- M1: `checkpoint.UnitStore` (atomic JSON/NPZ units, manifest, `progress.log`) and
  `run_units` (skip done units, <= 4 processes, BLAS threads = 1).
- M2: `var_tests` with all DESIGN 4.1 reference values. Christoffersen ind is conservative
  with rare hits (1.25% at 5%, n=500, alpha=0.025).
- M3: `es_tests` (AS Z1/Z2, McNeil-Frey bootstrap-t) and `dist`. MF one-sided is
  conservative with t5 residuals; vs `esback::er_backtest` one-sided p within 0.07.
- M4a/b: `esreg` (loss, G functions, LP quantile regression, ILS estimator, sandwich
  covariance with iid/nid, ind/scl_N/scl_sp, SJ bandwidth). R fixtures generated
  (`scripts/r_fixtures/make_inputs.py`, `fixtures.R`, outputs in `tests/fixtures/r/`).
  QR matches R to 1e-14; our esreg loss <= R loss; vcov matches vcovA to 4e-6 (scl_N) /
  0.4% (scl_sp). Derivation and checks: `docs/esreg-covariance.md`.
- **Confirmed esreg 0.6.2 discrepancy**: `G2_prime_prime(z, 1)` returns `-1/z^3` (correct:
  `-2/z^3`); repro in `scripts/r_fixtures/g2pp_repro.R`. Affects misspec=TRUE covariance;
  private flag `_esreg_compat_g2pp` reproduces R. Default is the correct formula.
- iid sparsity differs from esreg's by ~22% (ours verified unbiased in simulation); not
  used by the backtests.
- M4c: `esr_backtest` v1-v3, asymptotic + bootstrap. In compat mode all 36 esback p-values
  agree within 0.006 (3 fast cases in CI, the rest `slow`).
- M5: `scoring` (FZ0, DM with NW/HLN, `mean_with_ci`, MCS wrapper on arch).

- M6: forecasters (`hs` rolling daily, `gjr-t`, `fhs`, `evt-pot` with own GJR filter
  matching arch to 1e-8, `caviar-sav`) and `rolling_forecast` driver; no-look-ahead test
  per model. ESR backtest returns nan when the ES covariance is not positive definite.

- M7: `data` (Binance zips verified against .CHECKSUM, ms/us timestamp detection, FRED
  CSV snapshot cache, processed returns + SHA-256 manifest in
  `results/study-v1/data_manifest.json`), `study.runner` (forecast/backtest/compare units
  through UnitStore), `study.report` (CSV + markdown tables, Holm/BH), CLI (`backtest`,
  `data fetch`, `study run|report`, `mc run|report`), `multiple` (Holm, BH).
  Timings: forecast units 3-60 s each (CAViaR slowest), backtest unit about 25 s.

- M8: `montecarlo` (DGPs, runner per (cell, chunk), report with Wilson intervals). Quick
  run (regenerated at `d6a4d86`): 231 cells, 296,280 replications, 11.6 min wall / 45.4
  CPU-min, 0 failures; results in `results/mc-quick/`. ESR v3 asymptotic is conservative
  (0-1% at 5%; cause analysed with `scripts/esr_v3_null_se.py`); Christoffersen ind
  undersized; DQ oversized at 1%; McNeil-Frey below 5% everywhere.
- Study quick run (regenerated at `d6a4d86`): 50 cells, 4.4 min wall / 15.9 CPU-min, 0
  failures; results in `results/study-v1/`. Strict ESR rejections 11 raw / 6 BH / 4 Holm
  (all too conservative).
- Pre-release audit (2026-10-23) fixed: BLAS threads per worker (were 16; now 1, via
  parent env + threadpoolctl), resume key now covers data hashes and version (code change
  warns), compare stage waits for all forecasters, read-only reports, O(n)-memory SJ
  bandwidth, atomic/validated downloads, CLI error handling and `--stages` validation,
  per-chunk Acerbi-Szekely null in the Monte Carlo, `slow-vol` renamed `unconditional`,
  ignored config keys removed, README tolerance claims, REPORT wording (Kupiec/MF size,
  ESR v3 cause, FZ0 objective change in `d097f0f`). Each fix has a test.
- M9/M10: `REPORT.md`, README (install, quick start, CLI example), CHANGELOG 0.1.0
  (version bumped, not tagged), DESIGN section 12 (implementation notes).
- Tests: see the latest commit message for counts (`uv run pytest`; `--runslow` also
  runs the slow Monte Carlo and R-fixture checks; 2 documented skips: iid sparsity vs
  esreg).

Next (not done):
- Full-size profiles (`configs/study_full.toml`, `configs/mc_full.toml`; projected 20-30 and
  35-45 CPU-hours). Re-time a few units first: the projections predate the BLAS fix and
  may be pessimistic.
- ESR bootstrap size study (no Monte Carlo runner path yet; would answer whether the
  bootstrap fixes the conservative v3).
- Running `esback` itself on simulated null data (webR) to confirm directly that its v3 is
  conservative; currently inferred from matching p-values on fixtures.
- Acerbi-Szekely Wilson intervals still treat the replications of a chunk as independent
  although they share one null sample (documented in REPORT section 3).
- Tagging v0.1.0 and publishing (needs the owner's go-ahead). Re-running anything: resume
from `runs/<name>/units/`; delete a run directory only to start over.
