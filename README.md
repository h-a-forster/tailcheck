# tailcheck

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Value-at-Risk and Expected Shortfall backtests for Python, with p-values checked by Monte
Carlo and the ES regression tests checked against the reference R implementation.

Basel FRTB made Expected Shortfall the standard market-risk measure. The established ES
backtests live in R (`esreg`, `esback`) and in commercial toolboxes. tailcheck provides them as
a typed, tested Python package:

| Family | Tests |
|---|---|
| VaR | Kupiec POF, Christoffersen independence and conditional coverage, Engle-Manganelli dynamic quantile, Basel traffic light |
| ES | Acerbi-Szekely Z1 and Z2 (simulated null), McNeil-Frey exceedance residuals (bootstrap-t) |
| ES regression | Bayer-Dimitriadis strict, auxiliary and strict-intercept ESR backtests, on a Python implementation of joint quantile/ES regression (Dimitriadis-Bayer M-estimator with sandwich covariance) |
| Comparison | FZ0 loss, Diebold-Mariano (Newey-West, HLN correction), model confidence set (via `arch`) |

To our knowledge, no other Python package implements the ESR backtests or the
Acerbi-Szekely tests (searched 2026-10-23).

## What has been checked

- **Reference values.** Tested against independent computations
  (`scripts/reference_values.py`): Kupiec non-rejection regions (exact integer bounds),
  Christoffersen worked examples and standardized-t and normal VaR/ES (to 1e-9), Hall-Sheather
  bandwidths (to 1e-8), and the Basel cumulative probabilities (the published two-decimal
  table, 8.11% ... 99.99%, plus exact values to 1e-6).
- **Against R.** On committed fixtures from `esreg` 0.6.2 / `esback` 0.3.1 (run through
  webR): quantile regression agrees to 1e-14; our joint-regression loss is never above R's;
  the sandwich covariance agrees to 4e-6 (`scl_N`) and 0.4% (`scl_sp`) at R's coefficients;
  all 36 `esr_backtest` p-values agree within 0.006 once one discrepancy in `esreg` is
  reproduced (next point).
- **A discrepancy in `esreg` 0.6.2.** Its second derivative of the default G2 function
  returns `-1/z^3` instead of `-2/z^3`. The term enters the misspecification-robust
  covariance used by `esback`'s default ESR tests and moved strict-ESR p-values by up to
  0.25 on the fixtures. tailcheck uses the correct formula. Minimal repro and analysis:
  [docs/esreg-covariance.md](docs/esreg-covariance.md).
- **Size and power.** A Monte Carlo study reports the rejection rate of every test at the
  sample sizes used in practice, with Wilson intervals. See [REPORT.md](REPORT.md).
- **ESR v3: use the bootstrap.** The asymptotic one-sided intercept test (v3) is
  conservative, in tailcheck and in `esback` alike (about 0-2% rejections at 5% under the
  null; `esback` 0 of 250). With `n_boot > 0` the bootstrap p-value rejected 4.3% of 1,000
  null samples and had nearly twice the power (REPORT section 3.1). It costs about 0.3-0.5 s
  per resample at n = 1000-2000.

## Install

Not on PyPI. Install from GitHub with uv (or pip):

```sh
uv add git+https://github.com/h-a-forster/tailcheck
# or, for the command-line tool only
uv tool install git+https://github.com/h-a-forster/tailcheck
```

Python 3.11+. Runtime dependencies: numpy, scipy, pandas, arch.

## Quick start

Conventions: returns `r`, VaR `q` and ES `e` are on the return scale, so left-tail forecasts
are negative and `e <= q`. A VaR exceedance is `r <= q`. Every test returns a
`BacktestResult` with `statistic`, `pvalue`, `alternative` and `details`; an undefined test
(for example no exceedances) has `pvalue = nan` and a `details["reason"]`.

```python
import pandas as pd
from tailcheck.var_tests import kupiec_pof, christoffersen_cc, dynamic_quantile
from tailcheck.es_tests import mcneil_frey
from tailcheck.esr_backtest import esr_backtest
from tailcheck.scoring import fz0_loss, diebold_mariano

df = pd.read_csv("examples/garch_t5.csv")  # columns r, q, e, sigma; alpha = 2.5%
r, q, e, s = df.r, df.q, df.e, df.sigma

kupiec_pof(r, q, 0.025).pvalue
christoffersen_cc(r, q, 0.025).pvalue
dynamic_quantile(r, q, 0.025, lags=4).pvalue
mcneil_frey(r, q, e, sigma=s, n_boot=2000, seed=1).pvalue
esr_backtest(r, e, 0.025, version=1, seed=1).pvalue  # strict ESR, Wald chi2(2)
esr_backtest(r, e, 0.025, version=3, seed=1).pvalue  # one-sided intercept test (conservative)
esr_backtest(r, e, 0.025, version=3, n_boot=499, seed=1).details["pvalue_bootstrap"]  # preferred
fz0_loss(r, q, e, 0.025).mean()
```

From the command line:

```console
$ tailcheck backtest examples/garch_t5.csv --alpha 0.025
n = 1000, alpha = 0.025, hits = 24 (expected 25.0)
test                     statistic     p-value  alternative
kupiec_pof                  0.0416      0.8384  two-sided
christoffersen_ind          1.1817      0.2770  two-sided
christoffersen_cc           1.2232      0.5425  two-sided
dynamic_quantile            2.8109      0.8322  two-sided
mcneil_frey                 2.2968      0.9200  less
esr_v1                      3.0744      0.2150  two-sided
esr_v2                      3.0751      0.2149  two-sided
esr_v3                      1.5024      0.9335  less
traffic light: green (binomial CDF 0.4724)
mean FZ0 loss: 0.8074 [0.6889, 0.9260] (95% NW)
```

The Acerbi-Szekely test needs the forecast distribution, passed as a simulator:

```python
import numpy as np
from tailcheck.dist import draw_std_t
from tailcheck.es_tests import acerbi_szekely_test, location_scale_simulator

sim = location_scale_simulator(np.zeros(len(r)), s, lambda rng, shape: draw_std_t(rng, 5.0, shape))
acerbi_szekely_test(r, q, e, 0.025, sim, statistic="z2", n_sim=5000, seed=1)
```

## Study and Monte Carlo

The repository includes a reproducible study (five forecasters, BTC, ETH, S&P 500, Nasdaq
Composite and Nikkei 225, 1% and 2.5% VaR/ES, out of sample 2021-01 to 2026-09) and the
Monte Carlo size/power validation. Results and limitations: [REPORT.md](REPORT.md).

```sh
uv run tailcheck data fetch   --config configs/study.toml   # Binance + FRED, cached in data/
uv run tailcheck study run    --config configs/study.toml --workers 4
uv run tailcheck study report --config configs/study.toml   # writes results/study-v1/
uv run tailcheck mc run       --config configs/mc_quick.toml --workers 4
uv run tailcheck mc report    --config configs/mc_quick.toml
uv run tailcheck mc boot-run    --config configs/mc_esr_boot.toml --max-minutes 36  # bootstrap ESR size/power
uv run tailcheck mc boot-report --config configs/mc_esr_boot.toml
```

Every long run is checkpointed per unit under `runs/<name>/units/` and resumes where it
stopped; progress goes to `runs/<name>/progress.log`. Full-size configurations
(`configs/study_full.toml`, `configs/mc_full.toml`) take many CPU-hours.

## How to validate

```sh
uv sync --group dev
uv run pytest                 # fast suite (as in CI)
uv run pytest --runslow       # plus the slow checks (all 36 ESR fixtures)
uv run ruff check . && uv run ruff format --check .
```

To regenerate the R fixtures (needs Node 20+, downloads R packages into webR's in-memory
library; nothing is installed on the host), see `scripts/r_fixtures/` and
[AGENTS.md](AGENTS.md#how-to-test).

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md): formulas, reference values, edge cases, study and
  Monte Carlo design.
- [docs/esreg-covariance.md](docs/esreg-covariance.md): covariance derivation and the R
  comparison.
- [REPORT.md](REPORT.md): study and Monte Carlo results.

## Citation

See [CITATION.cff](CITATION.cff). Please also cite the papers behind the tests you use, in
particular Dimitriadis and Bayer (2019) and Bayer and Dimitriadis (2022) for the ESR tests.

## Licence

MIT. The R packages `esreg` and `esback` (GPL-3) are not dependencies and no code is taken
from them. They are run separately, through webR, only to produce numerical reference
fixtures for the tests.
