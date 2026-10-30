# tailcheck 0.1.0 release notes

Date: 2026-10-30. First release; install from GitHub (not on PyPI):
`uv add git+https://github.com/h-a-forster/tailcheck`. Python 3.11+.

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

## Known limitations

- Only the quick study and Monte Carlo profiles and the bootstrap ESR study have been run;
  the full-size profiles (estimated 23-42 and 55-105 CPU-hours) have not.
- Christoffersen independence is undersized, the dynamic quantile test is oversized at
  alpha = 1%, and McNeil-Frey is conservative (REPORT section 3).

Full list of changes: `CHANGELOG.md`.
