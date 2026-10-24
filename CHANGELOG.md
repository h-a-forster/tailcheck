# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `tailcheck.montecarlo.esr_boot` and `tailcheck mc boot-run | boot-report`: size and power
  of the bootstrap ESR tests, by the warp-speed method (one resample per replication) and
  by the full bootstrap, paired with the asymptotic p-values of the same replications.
  Config `configs/mc_esr_boot.toml`; results in `results/mc-esr-boot/`.
- `run_units(max_seconds=...)` and `--max-minutes` on `mc boot-run`: stop starting units
  after a time budget; a rerun resumes.
- Monte Carlo report: chunk-clustered intervals (`lo_chunk`, `hi_chunk`) and the design
  effect per cell in `size_power.csv`.
- `scripts/esback_null/`: `esback`'s asymptotic ESR tests on simulated null data through
  webR, next to tailcheck on the same samples (`results/esback-null/`).
- `scripts/retime_full.py`: re-times samples of the full-size profiles
  (`results/retime-full.json`).

### Changed

- Run sessions (`study run`, `mc run`, `mc boot-run`) print units done/expected, failures
  and whether they stopped at the time budget, and exit with code 3 when the run is
  incomplete. The Python functions return `RunResult` instead of the run directory.
- Reports (`study report`, `mc report`, `mc boot-report`) refuse a partial run unless
  `--allow-partial`; `summary.json` now records `complete`, `expected` and `done`.
- The run config hash ignores `[run] workers`; runs started under the old hash still resume.
  Hashes quoted in REPORT.md and the committed `summary.json` files are the old
  whole-config hashes; a regenerated report prints the new one.
- The sdist now includes `examples/` (used by the CLI test and the quick start) and
  `REPORT.md`.
- Tighter R-fixture tolerances (quantile regression 1e-10, covariance 1e-5 / 1%, ESR
  p-values 0.01, `bw.SJ` 1%).

### Fixed

- `christoffersen_independence`: a single hit on the first day no longer reports
  `reason = "no exceedances"`; it reports `"no transitions into a hit"`.

### Findings

- The asymptotic ESR v3 test is conservative because the ES-intercept standard error is
  negatively correlated with the estimate (about -0.5), not because of bias (0.09 sd) or a
  uniformly inflated SE; an oracle-SE test has correct size (REPORT section 3).
- The bootstrap restores the size of the one-sided ESR v3 test (4.3% at 5%, against 1.3%
  asymptotically) and nearly doubles its power; `esback`'s asymptotic v3 is equally
  conservative (0 of 250 null samples). See REPORT section 3.1.

## [0.1.0] - unreleased

First feature-complete version. Not tagged and not on PyPI.

### Added

- `tailcheck.var_tests`: Kupiec POF, Christoffersen independence and conditional coverage,
  Engle-Manganelli dynamic quantile (rank-aware degrees of freedom), Basel traffic light with
  the BCBS (1996) plus factors.
- `tailcheck.es_tests`: Acerbi-Szekely Z1 and Z2 with a simulated null
  (`acerbi_szekely_test`, `location_scale_simulator`) and the McNeil-Frey exceedance-residual
  bootstrap-t test.
- `tailcheck.esreg`: joint linear quantile/ES regression (Dimitriadis and Bayer 2019) written
  from the paper: Fissler-Ziegel losses for all G1/G2 choices, exact LP quantile regression
  (HiGHS), iterated-local-search M-estimator, and the sandwich covariance with iid/nid
  sparsity, ind/scl_N/scl_sp truncated-variance estimators, a Sheather-Jones bandwidth and
  the misspecification-robust terms (derivation in `docs/esreg-covariance.md`).
- `tailcheck.esr_backtest`: Bayer-Dimitriadis (2022) strict, auxiliary and strict-intercept
  ESR backtests with asymptotic and bootstrap p-values.
- `tailcheck.scoring`: FZ0 loss, Diebold-Mariano with Newey-West variance and the
  Harvey-Leybourne-Newbold correction, model confidence set wrapper around `arch`.
- `tailcheck.forecasters`: historical simulation, GJR-GARCH-t, filtered HS, EVT-POT and
  CAViaR-SAV, with a rolling-window driver and no-look-ahead tests.
- `tailcheck.data`: Binance spot klines (SHA-256 verified, millisecond/microsecond timestamp
  detection) and FRED downloads with a cached snapshot and a hash manifest.
- `tailcheck.checkpoint`: resumable unit store with atomic writes and a progress log; capped
  process pool (4 workers, single-threaded BLAS).
- Study (`tailcheck study run|report`) and Monte Carlo (`tailcheck mc run|report`) runners,
  `tailcheck backtest <csv>`, and `tailcheck data fetch`.
- R reference fixtures from `esreg` 0.6.2, `esback` 0.3.1 and `quantreg` 6.1 generated with
  webR (`scripts/r_fixtures/`), committed as JSON; the tests never need R.
- `REPORT.md` with the quick-profile study and Monte Carlo results.

### Fixed (pre-release audit)

- Worker processes really run with one BLAS thread: the thread variables are set in the
  parent while the pool runs and `threadpoolctl` limits the live pools (before, OpenBLAS
  kept its full pool in every worker). Adds the `threadpoolctl` dependency.
- Resume safety: run manifests record the input-data SHA-256s and package version, and a
  resume on different inputs is refused; a changed source hash warns. The study's compare
  unit waits for every configured forecaster, so a forecaster that only succeeds on a rerun
  is no longer left out of the MCS/DM results.
- `study report` / `mc report` open the run read-only and write nothing when there is no
  finished run (before, they truncated `results/<name>/*.csv` and crashed).
- `bandwidth_sj` uses O(n) memory (blocked pair differences, identical results) and bins
  the data above n = 10,000 (before, O(n^2) memory, about 8 GB at n = 20,000).
- Downloads are written atomically; malformed Binance checksum files are refetched and a
  mismatch is retried once; non-finite prices are rejected; FRED CSVs are validated before
  caching.
- CLI: bad input gives `error: ...` and exit code 2 instead of a traceback; unknown
  `--stages` values are rejected; `[run] workers` is the default for `--workers`.
- Monte Carlo: the Acerbi-Szekely null sample is drawn per chunk instead of once per cell;
  the size-distortion band follows `level`; the `slow-vol` DGP is renamed `unconditional`
  (it is constant-sigma forecasts on the GARCH data). Ignored config keys removed.
- Study and quick Monte Carlo regenerated with the fixed code (`REPORT.md`).

### Found

- `esreg` 0.6.2 returns `-1/z^3` for the second derivative of G2 type 1 (correct: `-2/z^3`).
  It changes misspecification-robust ESR p-values (by up to 0.25 on the reference fixtures).
  Minimal repro: `scripts/r_fixtures/g2pp_repro.R`; analysis in `docs/esreg-covariance.md`.
  tailcheck uses the correct formula.

### Known limitations

- In the quick Monte Carlo the one-sided ESR v3 test (asymptotic) is conservative (0-1%
  rejections at 5%), Christoffersen independence is undersized and the dynamic quantile
  test is oversized at alpha = 1%. See `REPORT.md` section 3.
- Only the quick study and Monte Carlo profiles have been run.
