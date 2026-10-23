# tailcheck v0.1 report: validation and forecaster study

Generated 2026-10-23 from the committed files under `results/`. Every number below comes from
those files; the commands that rebuild them are at the end.

**Summary.**

- The ES regression backtests reproduce the R reference implementation (`esback` 0.3.1) to
  within 0.006 in p-value once one discrepancy in `esreg` 0.6.2 is reproduced. That
  discrepancy (a wrong second derivative, `-1/z^3` instead of `-2/z^3`) changes strict-ESR
  p-values materially: by up to 0.25 on the reference fixtures and by up to 0.76 (median
  0.03) across the 50 study cells. tailcheck uses the correct formula.
- Monte Carlo (231 cells, 296,280 replications): Kupiec and both Acerbi-Szekely
  statistics keep their 5% size in every cell except Kupiec at n = 250, alpha = 1%
  (8.6% and 10.1%, binomial discreteness). The dynamic quantile test is oversized at 1%
  (9-11%), Christoffersen independence is undersized everywhere (0.8-2.5%), and McNeil-Frey
  is undersized at n = 250 (1.2-1.7%). The strict ESR test rejects 5.5-8% of
  200 null replications per cell (every Wilson interval reaches 5%); the one-sided ESR v3 test is
  conservative (0-1%).
- In the study (5 assets x 5 forecasters x 2 levels, out of sample 2021-01 to 2026-09), 11 of
  50 strict ESR tests reject at 5% before adjustment, 6 after Benjamini-Hochberg and 4 after
  Holm. All 6 BH rejections come with exceedance rates below the nominal level and one-sided
  intercept-test p-values of 0.94 or more: they flag ES forecasts that are too conservative,
  not too aggressive. Five of the six are historical simulation.
- Historical simulation is excluded from the 90% model confidence set in 8 of 10
  asset-level cells. The other four forecasters cannot be separated on FZ0 loss in most
  cells (each is in the 90% set in 9 of 10).

## 1. What was run

| Item | Value |
|---|---|
| Code | commit `c08d4f3` for the study (the next commit only speeds up the Sheather-Jones bandwidth, changing it by about 1e-5 relative); commit `d097f0f` for the Monte Carlo |
| Software | Python 3.13.1, numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, arch 8.0.0; Windows 11, CPU only, 4 worker processes |
| Study config | `configs/study.toml` (hash `8a7e477162200e84`), seed 20261023 |
| Monte Carlo config | `configs/mc_quick.toml`, seed 20261023 |
| R references | `esreg` 0.6.2, `esback` 0.3.1, `quantreg` 6.1 on R 4.6.0 (webR 0.6.0) |

**Data** (`results/study-v1/data_manifest.json` holds the SHA-256 of each processed return
series). Daily percent log returns: BTCUSDT and ETHUSDT from Binance spot daily klines (UTC
close; every monthly file verified against its published SHA-256), SP500, NASDAQCOM and
NIKKEI225 from FRED (trading days only, so returns span weekends and holidays). Snapshot end
2026-09-30.

**Design.** Rolling window of 1,000 observations, parameters refit every 21 observations and
filters updated daily, out-of-sample period 2021-01-01 to 2026-09-30: 2,099 days per crypto
asset, 1,442 for SP500 and NASDAQCOM, 1,403 for NIKKEI225. Levels alpha = 1% and 2.5%.
Forecasters: historical simulation (`hs`); GJR-GARCH(1,1) with Student-t innovations
(`gjr-t`); filtered historical simulation on a Gaussian-QMLE GJR fit (`fhs`); EVT-POT on the
same fit, with a GPD on the worst 10% of standardized residuals (`evt-pot`); and CAViaR-SAV
with ES as a constant multiple of VaR (`caviar-sav`). Acerbi-Szekely uses 2,000 predictive
draws per day; it is not available for CAViaR, which has no predictive distribution.

**Compute.** About 16 minutes of wall time: 13 CPU-minutes of forecasting (CAViaR 10 of
them), 43 of backtests and 2 of model confidence sets, 57 CPU-minutes in total against a plan
of 45. No unit failed and no strict ESR test was undefined.

## 2. Validation of the implementation

### 2.1 Closed-form and published reference values

All pass in `uv run pytest` (values in `docs/DESIGN.md` section 4): Kupiec 5%
non-rejection regions for 9 (alpha, T) pairs; both Christoffersen worked examples (to 1e-9);
the BCBS (1996) cumulative probabilities and zones for 0-10 exceptions; standardized-t and
normal VaR/ES (to 1e-9, also checked by numerical integration); Hall-Sheather bandwidths;
the dynamic quantile statistic against an OLS fit; the intercept-only covariance identity of
Dimitriadis and Bayer (2019, Remark 2.10) to 1e-8; the Acerbi-Szekely Z2 null 5% quantile
(-0.70 for T = 250, alpha = 2.5%); consistency of the joint regression on DGP-(2) of DB 2019
at n = 20,000 (within 4 standard errors); recovery of a t4 tail by EVT-POT; and recovery of
simulated CAViaR-SAV parameters. Each forecaster passes a no-look-ahead test (forecasts up to
day t are unchanged when data from day t on are perturbed).

### 2.2 Against the R packages

Fixtures from `esreg`, `esback` and `quantreg`, generated with
`scripts/r_fixtures/fixtures.R` and committed as JSON
(`tests/fixtures/r/outputs/r_reference.json`):

| Check | Result |
|---|---|
| `quantreg::rq` at 4 levels | agree to 1e-14 |
| `esreg` fits, (G1, G2) in {(2,1), (1,2), (2,3)} | our loss at or below R's in all three; coefficients within 0.05 standard errors |
| `vcovA`, nid/scl_N, at R's coefficients | 4e-6 relative with misspec = FALSE; with misspec = TRUE only after reproducing esreg's G2'' |
| `vcovA`, nid/scl_sp | 0.4% relative (same proviso) |
| `bw.SJ` on 3 samples | within 0.5% |
| `esr_backtest`: 3 forecast sets x 2 levels x 3 versions x 2 covariance options | all 36 p-values within 0.006 with esreg's G2'' reproduced |
| `er_backtest` (B = 20,000) | one-sided p-values within 0.06 (the bootstrap is centred differently) |
| iid sparsity | about 22% different from esreg's; ours is unbiased in simulation; not used by the backtests |

**The esreg discrepancy.** `esreg:::G2_prime_prime(-2, 1)` returns 0.125; calculus and a
finite difference of esreg's own `G2_prime_fun` both give 0.25 (repro:
`scripts/r_fixtures/g2pp_repro.R`). The term only enters the misspecification-robust
covariance that `esback` uses by default. We ruled out an error on our side: every other
covariance term matches R to 4e-6, and switching only this term reproduces R in all cases.
On the study data the strict-ESR p-values with and without the correction differ by a
median of 0.034 and at most 0.76 (`results/study-v1/backtests.csv`, columns `esr_v1_p` and
`esr_v1_esback_p`); 6 cells are BH-significant with the correct formula and 4 with esreg's.
Details: `docs/esreg-covariance.md`.

## 3. Monte Carlo size and power

Configuration `configs/mc_quick.toml` (hash `e2a5d2f1aeea242b`, seed 20261023); full
table with Wilson 95% intervals in `results/mc-quick/size_power.md`, raw rates in
`size_power.csv`. Run time 22.5 minutes of wall time, 87.5 CPU-minutes (ESR 81, everything
else 7) against a calibrated projection of 124 (`results/mc-quick/calibration.json`). No
unit failed. Size and power are rejection rates at the 5% level; a test that is undefined
in a replication (for example no exceedances) counts as a non-rejection, and the number of
such replications is reported.

**DGPs** (DESIGN section 6). Size: returns from a GARCH(1,1) with standardized t5
innovations (`garch-t5`) or iid t5 (`iid-t5`), with the true VaR/ES as forecasts. Power: the
true forecasts scaled by 0.85 (`scale-0.85`, risk understated); normal-quantile forecasts
for t5 data with the right volatility (`normal-tails`, tail shape wrong); constant forecasts
for data with slowly varying volatility (`slow-vol`, no conditioning). VaR and ES tests use
2,000 null replications per size cell and 1,000 per power cell; the ESR tests 200 and 40.

**Size** (rejection rate of a true model; selected cells, `garch-t5`):

| test | n=250, 1% | n=250, 2.5% | n=1000, 2.5% | n=2000, 1% | n=2000, 2.5% |
|---|---|---|---|---|---|
| Kupiec | 0.086 | 0.072 | 0.051 | 0.045 | 0.048 |
| Christoffersen ind | 0.015 | 0.008 | 0.013 | 0.021 | 0.025 |
| Christoffersen cc | 0.005 | 0.023 | 0.040 | 0.034 | 0.042 |
| DQ | 0.091 | 0.079 | 0.060 | 0.089 | 0.058 |
| Acerbi-Szekely Z1 | 0.045 (149 undefined) | 0.059 | 0.054 | 0.051 | 0.040 |
| Acerbi-Szekely Z2 | 0.055 | 0.050 | 0.043 | 0.050 | 0.051 |
| McNeil-Frey | 0.012 (1049 undefined) | 0.017 (95) | 0.036 | 0.035 | 0.039 |
| strict ESR (v1) | | | 0.080 [0.050, 0.126] | 0.055 [0.031, 0.096] | 0.080 [0.050, 0.126] |
| strict ESR, esback G2'' | | | 0.090 [0.058, 0.138] | 0.060 [0.035, 0.102] | 0.065 [0.038, 0.108] |
| ESR v3 (one-sided) | | | 0.000 [0, 0.019] | 0.000 [0, 0.019] | 0.010 [0.003, 0.036] |

Wilson half-widths are about 0.01 for the VaR/ES tests and 0.03-0.04 for the ESR tests.
The `iid-t5` cells look the same (Kupiec 10.1% at n = 250, 1%; DQ 9.2-11.1% at 1%).

- **Distorted cells.** 29 of the 93 size cells have a Wilson interval entirely outside
  [3%, 7%]: Christoffersen independence (11 of 12 cells, all undersized) and conditional
  coverage at 1% with n <= 1000 (3), DQ at 1% (6, oversized at 9-11%), Kupiec at n = 250,
  1% (2), McNeil-Frey at small n (5, undersized) and ESR v3 (2). These are properties of the
  tests with few exceedances (the Christoffersen and DQ distortions with rare hits are
  known from the literature), not implementation failures; the closed-form checks in
  section 2 pin down the statistics themselves.
- **Strict ESR.** Close to nominal or slightly oversized; with 200 replications the data
  cannot distinguish 5% from 8%. The esback G2'' variant has a similar size: the
  discrepancy changes individual p-values (section 2.2) more than the rejection rate.
- **ESR v3** is conservative: under the null the 5% quantile of its p-value is 0.10-0.18
  and its median 0.56-0.65, so the statistic is shifted towards "too conservative". We have
  not established whether `esback`'s asymptotic v3 behaves the same way under this DGP; the
  R comparison checks p-values on fixed forecast sets, not their null distribution. The
  bootstrap versions (not run here) may behave differently.

**Power** (rejection rate of a wrong model; selected cells):

| test | scale-0.85, n=1000, 2.5% | scale-0.85, n=2000, 2.5% | normal-tails, n=1000, 2.5% | normal-tails, n=2000, 2.5% | slow-vol, n=1000, 2.5% |
|---|---|---|---|---|---|
| Kupiec | 0.75 | 0.96 | 0.05 | 0.05 | 0.24 |
| Christoffersen cc | 0.72 | 0.94 | 0.04 | 0.06 | 0.33 |
| DQ | 0.66 | 0.91 | 0.07 | 0.09 | 0.55 |
| Acerbi-Szekely Z1 | 0.08 | 0.10 | 0.94 | 1.00 | 0.19 |
| Acerbi-Szekely Z2 | 0.85 | 0.98 | 0.29 | 0.43 | 0.18 |
| McNeil-Frey | 0.13 | 0.19 | 0.87 | 0.99 | 0.18 |
| strict ESR (v1) | 0.20 | 0.45 | 0.15 | 0.53 | |
| ESR v3 | 0.40 | 0.73 | 0.30 | 0.63 | |

ESR power cells have 40 replications (Wilson half-width up to 0.15). The pattern matches
what the tests are built for: VaR tests and Z2 detect understated scale, Z1 and McNeil-Frey
detect a wrong tail shape at a correct VaR (where VaR tests have no power by construction),
and DQ is best against unconditional forecasts. At these sample sizes the ESR tests are
less powerful than Z2 against understated scale and than Z1/McNeil-Frey against wrong tail
shape, but they need no predictive distribution and no volatility input.

## 4. Study results

Primary family: the strict ESR test (version 1, asymptotic, nid/scl_sp covariance with the
misspecification terms, as in `esback`) over the 50 asset x model x level cells, with Holm
(family-wise error) and Benjamini-Hochberg (false discovery rate) adjustments. All other
tables are descriptive and unadjusted; with about 7 tests per cell, some raw rejections at
5% are expected by chance alone.

### 4.1 Strict ESR (two-sided Wald test that the ES coefficients are (0, 1))

p-values, with BH q-values in brackets (`results/study-v1/tables.md`).

alpha = 1%:

| model | BTCUSDT | ETHUSDT | NASDAQCOM | NIKKEI225 | SP500 |
|---|---|---|---|---|---|
| caviar-sav | 0.030 [0.172] | 0.150 [0.357] | 0.620 [0.719] | 0.322 [0.505] | 0.793 [0.826] |
| evt-pot | 0.009 [0.063] | 0.520 [0.642] | 0.170 [0.386] | 0.314 [0.505] | 0.843 [0.848] |
| fhs | 0.005 [0.044] | 0.354 [0.506] | 0.213 [0.463] | 0.413 [0.568] | 0.848 [0.848] |
| gjr-t | 0.066 [0.235] | 0.307 [0.505] | 0.344 [0.505] | 0.265 [0.505] | 0.233 [0.465] |
| hs | 0.005 [0.044] | 0.119 [0.313] | 6e-05 [0.001] | 0.526 [0.642] | 0.001 [0.012] |

alpha = 2.5%:

| model | BTCUSDT | ETHUSDT | NASDAQCOM | NIKKEI225 | SP500 |
|---|---|---|---|---|---|
| caviar-sav | 0.038 [0.175] | 0.736 [0.783] | 0.327 [0.505] | 0.064 [0.235] | 0.288 [0.505] |
| evt-pot | 0.093 [0.299] | 0.647 [0.719] | 0.106 [0.299] | 0.480 [0.632] | 0.642 [0.719] |
| fhs | 0.108 [0.299] | 0.678 [0.737] | 0.101 [0.299] | 0.525 [0.642] | 0.647 [0.719] |
| gjr-t | 0.341 [0.505] | 0.420 [0.568] | 0.038 [0.175] | 0.126 [0.314] | 0.058 [0.235] |
| hs | 0.031 [0.172] | 0.230 [0.465] | 9e-04 [0.012] | 0.326 [0.505] | 3e-09 [2e-07] |

Rejections at 5%: 11 raw, 6 BH, 4 Holm. The six BH rejections:

| cell | exceedances / n | rate | strict ESR p | one-sided intercept p (v3) |
|---|---|---|---|---|
| BTCUSDT fhs 1% | 14 / 2099 | 0.67% | 0.0048 | 0.94 |
| BTCUSDT hs 1% | 13 / 2099 | 0.62% | 0.0052 | 0.97 |
| NASDAQCOM hs 1% | 9 / 1442 | 0.62% | 6e-5 | 1.00 |
| SP500 hs 1% | 6 / 1442 | 0.42% | 0.0010 | 1.00 |
| NASDAQCOM hs 2.5% | 26 / 1442 | 1.80% | 0.0009 | 1.00 |
| SP500 hs 2.5% | 20 / 1442 | 1.39% | 3e-9 | 1.00 |

The strict test is two-sided, so it also rejects forecasts that are too cautious. The
one-sided strict-intercept test (version 3), which looks only for underestimated risk,
rejects in 1 of 50 cells (`gjr-t` on NIKKEI225 at 2.5%, p = 0.011, unadjusted).

### 4.2 Exceedances and the other tests

Averaged over the 10 cells per model, the ratio of the observed to the nominal exceedance
rate is 0.80 for `hs`, 0.94 for `fhs`, 0.95 for `evt-pot`, 1.02 for `caviar-sav` and 1.29
for `gjr-t`. Unadjusted rejections at 5%, out of 10 cells per model:

| model | Kupiec | Christoffersen cc | DQ | McNeil-Frey | Acerbi-Szekely Z2 | strict ESR | ESR v3 |
|---|---|---|---|---|---|---|---|
| hs | 2 | 7 | 8 | 0 | 1 | 6 | 0 |
| gjr-t | 3 | 2 | 4 | 2 | 5 | 1 | 1 |
| fhs | 0 | 0 | 0 | 2 | 0 | 1 | 0 |
| evt-pot | 0 | 0 | 0 | 1 | 1 | 1 | 0 |
| caviar-sav | 1 | 0 | 5 | 3 | n/a | 2 | 0 |

`gjr-t` has too many exceedances on the equity indices (for example 26 / 1403 = 1.9% at 1% on
NIKKEI225, Wilson 95% interval 1.3% to 2.7%). The Acerbi-Szekely test, which uses the whole
predictive distribution, flags it in 5 cells; strict ESR flags it in 1. `hs` adapts slowly
to volatility changes, so its exceedances cluster and the Christoffersen and DQ tests reject.
The per-cell table with Wilson intervals is in `results/study-v1/tables.md`.

### 4.3 Model comparison (FZ0 loss)

Model confidence sets (range statistic, stationary bootstrap with mean block length 10,
10,000 replications). Number of the 10 asset x level cells in which each model is included:

| model | 90% MCS | 75% MCS |
|---|---|---|
| caviar-sav | 9 | 6 |
| evt-pot | 9 | 9 |
| fhs | 9 | 8 |
| gjr-t | 9 | 8 |
| hs | 2 | 2 |

Pairwise Diebold-Mariano tests against `gjr-t` (HLN-corrected, unadjusted) reject at 5% only
for `hs` on ETHUSDT, at both levels (`hs` worse). The differences among the four
volatility-based models are small relative to their sampling error.

### 4.4 Crypto vs equity, calm vs turbulent

Strict-ESR rejections (raw / BH): crypto 6 / 2 of 20 cells, equity 5 / 4 of 30 cells. Four
of the six raw crypto rejections are on BTCUSDT at 1% (all four models other than `gjr-t`).

Regimes split the out-of-sample days by the ex-ante 63-day realized volatility, above or
below its per-asset median. Pooled exceedance rates (about 4,240 days per regime and level):

| model | 1% calm | 1% turbulent | 2.5% calm | 2.5% turbulent |
|---|---|---|---|---|
| hs | 0.40% | 1.11% | 1.46% | 2.71% |
| gjr-t | 1.30% | 1.27% | 3.28% | 2.76% |
| fhs | 0.97% | 0.92% | 2.31% | 2.19% |
| evt-pot | 0.97% | 0.97% | 2.45% | 2.17% |
| caviar-sav | 1.04% | 1.13% | 2.78% | 2.24% |

`hs` overstates risk in calm periods and is close to nominal in turbulent ones, as expected
for an unconditional window. The conditional models are roughly stable across regimes.
Per-regime tests (Kupiec, strict ESR, ESR v3, FZ0, Acerbi-Szekely) are in
`results/study-v1/regimes.csv`.

## 5. Limitations

- **Few exceedances.** At 1% the equity series have 6 to 26 exceedances. ESR, McNeil-Frey and
  Acerbi-Szekely have little power there, and the ESR asymptotics are least reliable
  (section 3).
- **Quick profile only.** Parameters are refit every 21 days, ESR p-values are asymptotic (no
  bootstrap), and the Monte Carlo uses 200 replications per ESR size cell, enough to detect
  only gross size distortions (Wilson half-width about 0.03 at a 5% rate). The full-size
  configurations were not run.
- **One sample period.** 2021-2026 contains several stress episodes but is one realisation;
  the results are not a general ranking of the forecasters.
- **Data revisions.** FRED serves only the most recent 10 years of SP500, so a rerun after
  about 2027-01 cannot rebuild the first SP500 windows. The manifest hashes make any
  difference visible.
- **Multiple testing.** Only the strict ESR family is adjusted; the other tables are
  descriptive.
- **Specification choices** (constant mean, GJR(1,1), 10% EVT tail fraction, CAViaR-SAV with
  a constant ES/VaR ratio) were fixed in advance and not tuned.
- **R comparison** covers `esreg` and `esback` only. No R oracle was available for the VaR
  tests or Acerbi-Szekely (`rugarch` and `GAS` do not run in webR); those rely on
  closed-form and published values.
- **Budget.** The quick runs took longer than planned (57 instead of 45 CPU-minutes for the
  study; see section 3 for the Monte Carlo), mainly because an ESR fit with the default
  covariance takes 2-4 s at n = 2000.

## 6. Reproduce

```sh
uv sync
uv run tailcheck data fetch   --config configs/study.toml
uv run tailcheck study run    --config configs/study.toml --workers 4
uv run tailcheck study report --config configs/study.toml
uv run tailcheck mc run       --config configs/mc_quick.toml --workers 4
uv run tailcheck mc report    --config configs/mc_quick.toml
```

Runs resume from `runs/<name>/units/`. The download depends on the sources serving the same
history (see the FRED limitation above).
