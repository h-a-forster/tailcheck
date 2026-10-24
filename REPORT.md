# tailcheck v0.1 report: validation and forecaster study

Generated 2026-10-23 from the committed files under `results/`. Every number below comes from
those files; the commands that rebuild them are at the end.

**Summary.**

- The ES regression backtests reproduce the R reference implementation (`esback` 0.3.1) to
  within 0.006 in p-value once one discrepancy in `esreg` 0.6.2 is reproduced. That
  discrepancy (a wrong second derivative, `-1/z^3` instead of `-2/z^3`) changes strict-ESR
  p-values materially: by up to 0.25 on the reference fixtures and by up to 0.76 (median
  0.03) across the 50 study cells. tailcheck uses the correct formula.
- Monte Carlo (231 cells, 296,280 replications). Acerbi-Szekely Z2 has a Wilson interval
  covering 5% in all 12 size cells, Z1 in 10 of 12 (4.0% and 6.1% in the other two, both at
  alpha = 1%; about 0.6 such cells per test are expected by chance). Kupiec is oversized at
  n = 250 (8.6% and 10.1% at alpha = 1%, 7.3% at 2.5%; binomial discreteness); at
  alpha = 1% two larger-n `iid-t5` cells also miss 5% narrowly (6.1% at n = 1000, 3.8% at
  n = 2000), and its other 6 cells cover 5%. The dynamic quantile test is oversized at
  1% (9-11%), Christoffersen independence is undersized everywhere (0.8-2.5%), and
  McNeil-Frey is below 5% in every size cell (1.2-4.3%), clearly so at n = 250 and at
  n = 1000, alpha = 1% (1.2-2.9%). The strict ESR test rejects 5.5-8% of 200 null
  replications per cell (every Wilson interval reaches 5%); the one-sided ESR v3 test is
  conservative (0-1%), mainly through a small positive finite-sample bias of the ES
  intercept and an inflated standard error (section 3). `esback`'s asymptotic v3 rejects 0
  of 250 null samples too. The bootstrap restores the size of v3 (4.3% in 1,000 null
  replications, interval 2.2-6.1%) and nearly doubles its power (0.65 against 0.35-0.37);
  the strict ESR test is mildly oversized at n = 1000 asymptotically (7.3%) and 6.2% with
  the bootstrap (section 3.1).
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
| Code | commit `d6a4d86` for both the study and the Monte Carlo (regenerated after the pre-release audit fixes; see the note below) |
| Software | Python 3.13.1, numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, arch 8.0.0; Windows 11, CPU only, 4 worker processes |
| Study config | `configs/study.toml` (hash `b5e6bab664e5e3e2`), seed 20261023 |
| Monte Carlo config | `configs/mc_quick.toml`, seed 20261023 |
| R references | `esreg` 0.6.2, `esback` 0.3.1, `quantreg` 6.1 on R 4.6.0 (webR 0.6.0) |

**Earlier versions of these results.** The first published numbers came from commit
`c08d4f3` (study) and `d097f0f` (Monte Carlo). Commit `d097f0f` did not only speed up the
Sheather-Jones bandwidth: it also wrote the FZ0 estimator objective out explicitly. That is
the same loss, but it changes Nelder-Mead paths, and the location-scale covariance step is
numerically sensitive, so the old study ESR p-values were not exactly reproducible at later
commits. The regenerated study differs from the `c08d4f3` run only in the ESR p-values: the
largest change is NASDAQCOM `hs` at 1% (strict ESR p 5.8e-5 -> 3.7e-5, 35%), then SP500 `hs`
at 1% (0.0010 -> 0.0007, 29%) and NIKKEI225 `hs` at 1% (0.526 -> 0.508, 3.5%); every other
cell moved by less than 0.3% (strict ESR; up to 22% for one auxiliary-ESR p-value), and no
rejection count changes. The regenerated Monte Carlo reproduces every
ESR rate exactly; the Acerbi-Szekely rates change by up to 3 percentage points because the
null sample is now drawn per chunk (section 3), and the renamed `unconditional` power DGP
has new seeds.

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

**Compute.** 4.4 minutes of wall time: 6.6 CPU-minutes of forecasting (CAViaR 5.4 of
them), 8.3 of backtests and 1.0 of model confidence sets, 15.9 CPU-minutes in total against a
plan of 45. No unit failed and no strict ESR test was undefined. The first run took 16
minutes and 57 CPU-minutes because every worker process still ran a 16-thread OpenBLAS pool
(4 workers on a shared CPU); fixing the per-worker thread limit roughly tripled throughput.

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

Configuration `configs/mc_quick.toml` (hash `c7b1a7514ee396dd`, seed 20261023); full
table with Wilson 95% intervals in `results/mc-quick/size_power.md`, raw rates in
`size_power.csv`. Run time 11.6 minutes of wall time, 45.4 CPU-minutes (ESR 41, everything
else 4.4; the first run, with oversubscribed BLAS threads, took 22.5 and 87.5) against a
calibrated projection of 124 (`results/mc-quick/calibration.json`). No unit failed. Size and power are rejection rates at the 5% level; a test that is undefined
in a replication (for example no exceedances) counts as a non-rejection, and the number of
such replications is reported.

**DGPs** (DESIGN section 6). Size: returns from a GARCH(1,1) with standardized t5
innovations (`garch-t5`) or iid t5 (`iid-t5`), with the true VaR/ES as forecasts. Power: the
true forecasts scaled by 0.85 (`scale-0.85`, risk understated); normal-quantile forecasts
for t5 data with the right volatility (`normal-tails`, tail shape wrong); unconditional
constant-sigma t5 forecasts on the same GARCH(1,1) data (alpha 0.10, beta 0.85, volatility
half-life about 13.5 days; `unconditional`, no conditioning, called `slow-vol` in the first
run). VaR and ES tests use 2,000 null replications per size cell and 1,000 per power cell;
the ESR tests 200 and 40.

**Acerbi-Szekely null.** The null distribution of Z1/Z2 is simulated with 2,000 paths. It is
drawn afresh for each chunk of 50 replications, so its Monte Carlo error is shared within a
chunk only and averages out over the 40 chunks of a size cell. In the first run one null
sample served a whole cell, which could shift that cell's true size by about +-0.5
percentage points (1 sd), an error the Wilson interval did not include. The Wilson
intervals treat replications as independent. `size_power.csv` now also has a
chunk-clustered interval (`lo_chunk`, `hi_chunk`; chunks as sampling units) and the design
effect (clustered over binomial variance). Over the 30 cells per test the mean design effect
is 1.05 for Z1 and 1.06 for Z2, against 0.99-1.09 for the VaR tests and McNeil-Frey, whose
replications share nothing; the per-cell estimates are noisy (0.5-2.2 with 20-40 chunks).
So the shared null sample does not measurably widen the Acerbi-Szekely intervals, and the
clustered intervals change no conclusion.

**Size** (rejection rate of a true model; selected cells, `garch-t5`):

| test | n=250, 1% | n=250, 2.5% | n=1000, 2.5% | n=2000, 1% | n=2000, 2.5% |
|---|---|---|---|---|---|
| Kupiec | 0.086 | 0.072 | 0.051 | 0.045 | 0.048 |
| Christoffersen ind | 0.015 | 0.008 | 0.013 | 0.021 | 0.025 |
| Christoffersen cc | 0.005 | 0.023 | 0.040 | 0.034 | 0.042 |
| DQ | 0.091 | 0.079 | 0.060 | 0.089 | 0.058 |
| Acerbi-Szekely Z1 | 0.045 (149 undefined) | 0.055 (4) | 0.051 | 0.053 | 0.046 |
| Acerbi-Szekely Z2 | 0.052 | 0.051 | 0.048 | 0.049 | 0.050 |
| McNeil-Frey | 0.012 (1049 undefined) | 0.017 (95) | 0.036 | 0.035 | 0.039 |
| strict ESR (v1) | | | 0.080 [0.050, 0.126] | 0.055 [0.031, 0.096] | 0.080 [0.050, 0.126] |
| strict ESR, esback G2'' | | | 0.090 [0.058, 0.138] | 0.060 [0.035, 0.102] | 0.065 [0.038, 0.108] |
| ESR v3 (one-sided) | | | 0.000 [0, 0.019] | 0.000 [0, 0.019] | 0.010 [0.003, 0.036] |

Wilson half-widths are about 0.01 for the VaR/ES tests and 0.03-0.04 for the ESR tests.
The `iid-t5` cells look the same (Kupiec 10.1% at n = 250, 1%, 7.3% at n = 250, 2.5% and
6.1% at n = 1000, 1%; DQ 9.2-11.1% at 1%).

- **Distorted cells.** 29 of the 93 size cells have a Wilson interval entirely outside
  [3%, 7%]: Christoffersen independence (11 of 12 cells, all undersized) and conditional
  coverage at 1% with n <= 1000 (3), DQ at 1% (6, oversized at 9-11%), Kupiec at n = 250,
  1% (2), McNeil-Frey at n = 250 and at `garch-t5`, n = 1000, 1% (5, undersized) and ESR v3
  (2). This [3%, 7%] rule is lenient: a stricter reading (interval excludes 5%) also flags
  Kupiec at n = 250, 2.5% (7.3% in both DGPs) and at `iid-t5`, alpha = 1%, n = 1000 and 2000
  (6.1%, 3.8%), Z1 at alpha = 1% in two cells (4.0%, 6.1%), and every McNeil-Frey cell except
  `iid-t5`, n = 2000, 2.5%. These are properties of the
  tests with few exceedances (the Christoffersen and DQ distortions with rare hits are
  known from the literature), not implementation failures; the closed-form checks in
  section 2 pin down the statistics themselves.
- **Strict ESR.** Close to nominal or slightly oversized; with 200 replications the data
  cannot distinguish 5% from 8%. The esback G2'' variant has a similar size: the
  discrepancy changes individual p-values (section 2.2) more than the rejection rate.
- **ESR v3** is conservative: under the null the 5% quantile of its p-value is 0.10-0.18
  and its median 0.56-0.65. A separate null study (`scripts/esr_v3_null_se.py`, 200
  replications of `garch-t5`, n = 1000, alpha = 2.5%; `results/mc-quick/esr_v3_null_se.json`)
  shows two causes. (i) The ES intercept has a small positive finite-sample bias (mean 0.021,
  median 0.041, against a standard deviation of 0.231), which pushes the t-statistic towards
  "too conservative" (mean t about +0.3). (ii) The default covariance (nid sparsity,
  `scl_sp`, misspecification terms) overstates the intercept's standard error on average
  (mean 0.267, median 0.227, against the empirical 0.231), with a long right tail. Dropping
  the misspecification terms gives a mean SE of 0.214 but rejects only 2.0% (t mean 0.40,
  sd 1.23); the default rejects 1.5% in this study (0 of 200 in the Monte Carlo cell, within
  sampling error); `scl_N` without the misspecification terms understates the SE (0.159)
  and rejects 7%. So the misspecification terms explain part, not all, of the
  conservativeness. `esback`'s own asymptotic v3 is just as conservative when run directly
  on simulated null data (0 of 250), and the bootstrap restores its size; both are in
  section 3.1.

**Power** (rejection rate of a wrong model; selected cells):

| test | scale-0.85, n=1000, 2.5% | scale-0.85, n=2000, 2.5% | normal-tails, n=1000, 2.5% | normal-tails, n=2000, 2.5% | unconditional, n=1000, 2.5% |
|---|---|---|---|---|---|
| Kupiec | 0.75 | 0.96 | 0.05 | 0.05 | 0.22 |
| Christoffersen cc | 0.72 | 0.94 | 0.04 | 0.06 | 0.34 |
| DQ | 0.66 | 0.91 | 0.07 | 0.09 | 0.56 |
| Acerbi-Szekely Z1 | 0.08 | 0.12 | 0.93 | 1.00 | 0.17 |
| Acerbi-Szekely Z2 | 0.86 | 0.98 | 0.29 | 0.40 | 0.17 |
| McNeil-Frey | 0.13 | 0.19 | 0.87 | 0.99 | 0.16 |
| strict ESR (v1) | 0.20 | 0.45 | 0.15 | 0.53 | |
| ESR v3 | 0.40 | 0.73 | 0.30 | 0.63 | |

ESR power cells have 40 replications (Wilson half-width up to 0.15). The pattern matches
what the tests are built for: VaR tests and Z2 detect understated scale, Z1 and McNeil-Frey
detect a wrong tail shape at a correct VaR (where VaR tests have no power by construction),
and DQ is best against unconditional forecasts. At these sample sizes the ESR tests are
less powerful than Z2 against understated scale and than Z1/McNeil-Frey against wrong tail
shape, but they need no predictive distribution and no volatility input.

### 3.1 Bootstrap ESR tests and esback on null data (follow-up)

Run 2026-10-23/24 at tailcheck 0.1.0 with the code of commit `d23ea84` (the run started
from the working tree just before that commit; the one later source change, to the
Monte Carlo report aggregation, does not touch the units, which the run log records as a
source-hash warning on resume); config
`configs/mc_esr_boot.toml`, hash `c35185f6c450a45f`, seed 20261024; method in DESIGN
section 6.1). Full table: `results/mc-esr-boot/size_power.md`. 2,320 replications, 305
CPU-minutes over four resumed sessions of at most 40 minutes each on 2-3 worker processes
(the config header projected about 180; the machine was shared with other jobs). No
replication failed and no statistic was undefined.

Most cells use the warp-speed method (one bootstrap resample per replication, critical
value from the pooled bootstrap statistics); one cell runs the full bootstrap test
(B = 99) per replication as a cross-check. Asymptotic and bootstrap rates come from the same
replications. Intervals are 95% (Wilson for the asymptotic and exact rates; joint resampling
of replications for warp-speed rates, which is wider because the critical value is itself
estimated).

| test | DGP | n | reps | asymptotic | bootstrap |
|---|---|---|---|---|---|
| ESR v3 (one-sided) | garch-t5 (size) | 1000 | 1000 | 0.013 [0.008, 0.022] | 0.043 [0.022, 0.061] |
| ESR v3, exact bootstrap B = 99 | garch-t5 (size) | 1000 | 120 | 0.000 [0, 0.031] | 0.033 [0.013, 0.083] |
| ESR v3 | garch-t5 (size) | 2000 | 400 | 0.020 [0.010, 0.039] | 0.043 [0.020, 0.068] |
| strict ESR (v1) | garch-t5 (size) | 1000 | 1000 | 0.073 [0.058, 0.091] | 0.062 [0.041, 0.083] |
| strict ESR (v1) | garch-t5 (size) | 2000 | 400 | 0.055 [0.037, 0.082] | 0.055 [0.033, 0.083] |
| ESR v3 | scale-0.85 (power) | 1000 | 400 | 0.372 [0.327, 0.421] | 0.645 [0.593, 0.748] |
| ESR v3 | normal-tails (power) | 1000 | 400 | 0.345 [0.300, 0.393] | 0.645 [0.580, 0.725] |
| strict ESR (v1) | scale-0.85 (power) | 1000 | 400 | 0.185 [0.150, 0.226] | 0.145 [0.062, 0.323] |
| strict ESR (v1) | normal-tails (power) | 1000 | 400 | 0.193 [0.157, 0.234] | 0.147 [0.080, 0.250] |

- **The bootstrap fixes the conservative v3.** With 1,000 null replications the asymptotic
  one-sided v3 test rejects 1.3%; the bootstrap rejects 4.3%, and its interval covers 5%. The
  exact bootstrap cell agrees (3.3%, 120 replications). The bootstrap 5% critical value of
  the t-statistic is about -1.06 at n = 1000 and -1.28 at n = 2000, against -1.645 for the
  normal: the null distribution of the v3 statistic is shifted right and narrower than
  N(0, 1), consistent with the bias and inflated standard error in section 3.
- **Power of v3 nearly doubles** at the corrected size: 0.65 against 0.37 (understated
  scale) and 0.35 (normal tails) for the asymptotic test, at n = 1000.
- **Strict ESR (v1).** The asymptotic test is mildly oversized at n = 1000 (7.3%, interval
  5.8-9.1%; the quick run's 200-replication cells could not resolve this) and close to 5% at
  n = 2000. The bootstrap brings n = 1000 to 6.2% (interval covers 5%). Its power is not
  measurably different from the asymptotic test's (the warp-speed power intervals are wide).
- **Multiple testing.** Nine cells are reported; the statements above use each interval on
  its own. None of them rests on a single interval barely excluding 5%.

**esback directly.** `scripts/esback_null/` ran `esback::esr_backtest` (version 0.3.1, esreg
0.6.2, R 4.6.0 in webR; asymptotic, package defaults) on 250 `garch-t5` null samples
(n = 1000, alpha = 2.5%, seed 20261024; `results/esback-null/`), and tailcheck on the same
samples. No esback call failed.

| p-value | rejects at 5% | 5% quantile of p | median p |
|---|---|---|---|
| esback v3, one-sided | 0.000 [0, 0.015] | 0.171 | 0.58 |
| tailcheck v3, one-sided | 0.000 [0, 0.015] | 0.160 | 0.57 |
| esback v3, two-sided | 0.032 [0.016, 0.062] | 0.085 | 0.56 |
| esback v1 | 0.080 [0.052, 0.120] | 0.040 | 0.61 |
| tailcheck v1, esreg's G2'' | 0.080 [0.052, 0.120] | 0.040 | 0.61 |
| tailcheck v1, correct G2'' | 0.064 [0.040, 0.101] | 0.028 | 0.60 |

So `esback`'s asymptotic one-sided v3 test is conservative in the same way (no rejection in
250 null samples); this is now measured, not inferred from fixture agreement. On these 250
samples tailcheck (with esreg's G2'') and esback agree to a median absolute difference of
0.0005 (v3) and 0.0013 (v1) in p-value, and reach the same decision at 5% in every
replication; the largest v1 difference is 0.057 (3 replications differ by more than 0.01),
larger than the 0.006 seen on the fixtures. The cause was not traced per replication; a
different local optimum of the M-estimator (both use random starts) is the likely reason.

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
| evt-pot | 0.009 [0.063] | 0.520 [0.640] | 0.170 [0.386] | 0.314 [0.505] | 0.843 [0.848] |
| fhs | 0.005 [0.043] | 0.354 [0.506] | 0.213 [0.463] | 0.413 [0.568] | 0.848 [0.848] |
| gjr-t | 0.066 [0.235] | 0.307 [0.505] | 0.344 [0.505] | 0.265 [0.505] | 0.233 [0.465] |
| hs | 0.005 [0.043] | 0.119 [0.313] | 4e-05 [9e-04] | 0.508 [0.640] | 7e-04 [0.011] |

alpha = 2.5%:

| model | BTCUSDT | ETHUSDT | NASDAQCOM | NIKKEI225 | SP500 |
|---|---|---|---|---|---|
| caviar-sav | 0.038 [0.175] | 0.736 [0.783] | 0.327 [0.505] | 0.064 [0.235] | 0.288 [0.505] |
| evt-pot | 0.093 [0.299] | 0.647 [0.719] | 0.106 [0.299] | 0.480 [0.632] | 0.642 [0.719] |
| fhs | 0.108 [0.299] | 0.678 [0.737] | 0.101 [0.299] | 0.525 [0.640] | 0.647 [0.719] |
| gjr-t | 0.341 [0.505] | 0.420 [0.568] | 0.038 [0.175] | 0.126 [0.314] | 0.058 [0.235] |
| hs | 0.031 [0.172] | 0.230 [0.465] | 9e-04 [0.011] | 0.326 [0.505] | 3e-09 [2e-07] |

Rejections at 5%: 11 raw, 6 BH, 4 Holm. The six BH rejections:

| cell | exceedances / n | rate | strict ESR p | one-sided intercept p (v3) |
|---|---|---|---|---|
| BTCUSDT fhs 1% | 14 / 2099 | 0.67% | 0.0048 | 0.94 |
| BTCUSDT hs 1% | 13 / 2099 | 0.62% | 0.0052 | 0.97 |
| NASDAQCOM hs 1% | 9 / 1442 | 0.62% | 4e-5 | 1.00 |
| SP500 hs 1% | 6 / 1442 | 0.42% | 0.0007 | 1.00 |
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
- **Quick profile only.** Parameters are refit every 21 days, the study's ESR p-values are
  asymptotic (no bootstrap), and the main Monte Carlo uses 200 replications per ESR size
  cell, enough to detect only gross size distortions (Wilson half-width about 0.03 at a 5%
  rate). The bootstrap ESR study (section 3.1) covers one level (2.5%), one null DGP and
  n = 1000/2000; given its result, the study's v3 column understates evidence against the
  forecasts and its strict-ESR column may slightly overstate it. The full-size
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
- **Budget.** The study now fits its 45 CPU-minute plan (15.9). The Monte Carlo needed
  cuts to the ESR replication counts at calibration time, when an ESR fit with the default
  covariance took 2-4 s at n = 2000 (partly BLAS oversubscription, since fixed); see
  section 3.
- **Full-size profiles** are projected, not measured. A re-timing of sampled units on
  2026-10-24 (`scripts/retime_full.py`, `results/retime-full.json`) measured 59 CPU-hours for
  `configs/study_full.toml` (ESR bootstrap 50, at 0.8-1.6 s per resample) and 145 for
  `configs/mc_full.toml`, on a machine that was heavily loaded by other jobs: a reference fit
  took 1.4-2.6 times as long as on a quieter machine. Scaled by that factor the estimates are
  roughly 23-42 and 55-105 CPU-hours; the 0.1.0 projections (20-30 and 35-45) were too low
  for the Monte Carlo profile.

## 6. Reproduce

```sh
uv sync
uv run tailcheck data fetch   --config configs/study.toml
uv run tailcheck study run    --config configs/study.toml --workers 4
uv run tailcheck study report --config configs/study.toml
uv run tailcheck mc run       --config configs/mc_quick.toml --workers 4
uv run tailcheck mc report    --config configs/mc_quick.toml
# section 3.1 (about 5 CPU-hours; sessions of at most --max-minutes, rerun to resume)
uv run tailcheck mc boot-run    --config configs/mc_esr_boot.toml --workers 3 --max-minutes 36
uv run tailcheck mc boot-report --config configs/mc_esr_boot.toml
# esback on null data (Node 20+; steps in scripts/esback_null/esback_null.py)
```

Runs resume from `runs/<name>/units/`. The download depends on the sources serving the same
history (see the FRED limitation above).
