# tailcheck

VaR and Expected Shortfall backtests for Python, with p-values checked by Monte Carlo.

> **Status: pre-alpha scaffold.** The API in `src/tailcheck/` is stubbed and the design is in
> [docs/DESIGN.md](docs/DESIGN.md). Nothing here is ready to use yet. Progress is tracked in
> [AGENTS.md](AGENTS.md#status-and-next-steps).

Basel FRTB made Expected Shortfall the standard market-risk measure. The standard ES
backtests are available in R (`esreg`, `esback`) and in the commercial MATLAB Risk Management
Toolbox. tailcheck aims to provide them as a tested Python package:

| Family | Tests |
|---|---|
| VaR | Kupiec POF, Christoffersen independence and conditional coverage, Engle-Manganelli dynamic quantile, Basel traffic light |
| ES | Acerbi-Szekely Z1 and Z2 (simulated null), McNeil-Frey exceedance residuals (bootstrap) |
| ES regression | Bayer-Dimitriadis strict, auxiliary and strict-intercept ESR backtests, built on a Python implementation of joint quantile/ES regression (Dimitriadis-Bayer M-estimator with sandwich covariance) |
| Comparison | FZ0 loss, Diebold-Mariano, model confidence set (via `arch`) |

Every test will ship with a Monte Carlo size and power table, so you can see whether its
p-values hold at your sample size. The package also includes a reproducible study:
historical simulation, GJR-GARCH-t, filtered HS, EVT-POT and CAViaR forecasting one-day 1% and
2.5% VaR/ES for BTC, ETH and three equity indices, out of sample over 2021-2026.

## Conventions

Returns `r`, VaR `q` and ES `e` are all on the return scale, so left-tail forecasts are
negative and `e <= q`. A VaR exceedance is `r <= q`.

## Development

```sh
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/DESIGN.md](docs/DESIGN.md).

## Licence

MIT. The R packages `esreg` and `esback` (GPL-3) are not dependencies and no code is taken
from them. They are run separately, through webR, only to produce numerical reference
fixtures for the tests.
