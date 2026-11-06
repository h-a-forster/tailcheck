# tailcheck 0.1.0 release notes

Prepared 2026-10-30, updated 2026-11-06; the release date is the tag date. First release;
install from GitHub (not on PyPI): `uv add git+https://github.com/h-a-forster/tailcheck`. Python 3.11+.

## What is in it

- VaR backtests: Kupiec POF, Christoffersen independence and conditional coverage,
  Engle-Manganelli dynamic quantile, Basel traffic light.
- ES backtests: Acerbi-Szekely Z1 and Z2 (simulated null), McNeil-Frey (bootstrap-t).
- ES regression: a Python implementation of the Dimitriadis-Bayer joint quantile/ES
  regression and the Bayer-Dimitriadis strict, auxiliary and strict-intercept ESR backtests,
  with asymptotic and bootstrap p-values.
- Comparison: FZ0 loss, Diebold-Mariano (Newey-West, HLN), model confidence set (`arch`).
- Five forecasters, a resumable study runner and Monte Carlo runner, and the `tailcheck` CLI.

## Validation (details in REPORT.md)

- All 36 ESR p-values on the `esback` 0.3.1 reference fixtures agree within 0.006 once a
  discrepancy in `esreg` 0.6.2 is reproduced; quantile regression agrees with `quantreg` to
  1e-14.
- `esreg` 0.6.2 returns `-1/z^3` for the second derivative of its default G2 function
  (correct: `-2/z^3`); tailcheck uses the correct formula. Draft upstream report:
  `docs/upstream/esreg-g2pp.md`.
- Quick Monte Carlo: 231 cells, 296,280 replications, with Wilson intervals for the size and
  power of every test.
- The asymptotic one-sided ESR v3 test is conservative (here and in `esback`); the bootstrap
  version has 4.3% size at 5% and nearly twice the power. Use `n_boot > 0` for v3.

## Fixed in the 2026-11-06 audit

- ESR tests and `esreg` were wrong on small-scale data: from a daily sd of about 0.001 down,
  the absolute ES barrier made the true ES infeasible and correct models were rejected
  (`examples/garch_t5.csv` at 1/1000 of its scale, commit `b6b715c`: strict ESR p = 0.003
  instead of 0.215, no warning). Very large scales (1e7 times the same data) also drifted
  (p = 0.28). The statistics now do not depend on the unit, and results on percent data are
  unchanged. `esback` 0.3.1 depends on the unit too (`scripts/r_fixtures/scale_repro.R`).
- `esr_backtest` with constant forecasts returns `nan` with a reason in every version. Version
  3 used to return a number from a singular quantile design. The ESR section of
  `configs/mc_full.toml` drops the two constant-forecast DGPs.
- Invalid options raise `ValueError` instead of being swallowed or ignored: `esr_backtest`
  `sparsity`, `sigma_est` and `early_stopping` (they gave `nan`), `model_confidence_set`
  `method` (any other value was accepted), `mean_with_ci` `lags` and `level`.
- `--force-new` on an unchanged config was ignored and the run resumed; it now starts over,
  as documented.
- A worker process that dies ends the session with exit code 3, so a resume loop continues
  (it was a traceback with exit code 1). A config without a required key is a clean error.
- The private `_esreg_compat_g2pp` flag now also reaches the bootstrap covariance.
- CI actions are pinned to commit SHAs. The description no longer says FZ0 scoring and model
  confidence sets have Monte Carlo size checks; the backtests do.

## Known limitations

- Only the quick study and Monte Carlo profiles and the bootstrap ESR study have been run;
  the full-size profiles (estimated 23-42 and 55-105 CPU-hours) have not.
- Christoffersen independence is undersized, the dynamic quantile test is oversized at
  alpha = 1%, and McNeil-Frey is conservative (REPORT section 3).

Full list of changes: `CHANGELOG.md`.
