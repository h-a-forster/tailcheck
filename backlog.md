# Backlog

## In progress

## Waiting on me

- Branch `polish-2026-10-30` is merged through a pull request with a green first CI run (no workflow has ever run on GitHub).
- The repository description no longer claims Monte Carlo checks for FZ0 scoring and model confidence sets (text in `pyproject.toml`).
- v0.1.0 is tagged, with `CHANGELOG.md` and the release notes carrying the tag date.
- The `esreg` G2'' issue is filed (`docs/upstream/esreg-g2pp.md`).

## Decided

- 2026-10-23: GitHub-only release; no PyPI until the project has users.
- 2026-10-23: FRED replaces Stooq for equity indices; the out-of-sample period starts in 2021.
- 2026-10-24: CI runs on pull requests and manual dispatch only.

## Proposed

- `_wilson` in `study/runner.py` can fall about 1e-17 outside [0, 1] at k = 0; clamp it the next time results are regenerated.
- The `esback` unit dependence (`scripts/r_fixtures/scale_repro.R`) is reported upstream with the G2'' issue.
- Bootstrap ESR size and power are measured at alpha = 1%, at more DGPs and on estimated forecasts, including a full-size boot profile (about 25 CPU-hours).
- The two-sided v3 bootstrap p-value holds its size (8.1% at n = 1000), or `esr_backtest` stops returning it.
- The cause of the v3 T* upper-tail mismatch is settled, including a block-bootstrap comparison for serial dependence.
- The `early_stopping` mismatch is measured for strict ESR (v1) on paired resamples, as it was for v3.
- The study reports bootstrap ESR p-values, so its v3 column is no longer conservative.
- The full-size study and Monte Carlo profiles have run on an idle machine, after a fresh re-timing.
- `esr_backtest` supports constant forecasts in v3 through an intercept-only quantile equation, instead of returning `nan`.
