# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### Found

- `esreg` 0.6.2 returns `-1/z^3` for the second derivative of G2 type 1 (correct: `-2/z^3`).
  It changes misspecification-robust ESR p-values (by up to 0.25 on the reference fixtures).
  Minimal repro: `scripts/r_fixtures/g2pp_repro.R`; analysis in `docs/esreg-covariance.md`.
  tailcheck uses the correct formula.
