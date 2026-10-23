# tailcheck design

Status: v0.1 plan, written at scaffold time (2026-10-23). The module stubs in `src/tailcheck/`
follow this document. When code and this document disagree, fix one of them in the same commit.

## 1. Goal and claims

tailcheck is a typed, tested Python package for backtesting Value-at-Risk (VaR) and Expected
Shortfall (ES) forecasts, plus a reproducible study that applies it to five standard
forecasters on crypto and equity index returns.

v0.1 claims, and how each is supported:

| Claim | Evidence |
|---|---|
| The tests compute the published statistics | closed-form oracles, published tables, R reference fixtures (section 7) |
| The p-values are trustworthy at the sample sizes used | Monte Carlo size table with Wilson intervals (section 6) |
| The tests detect relevant misspecification | Monte Carlo power table (section 6) |
| Study findings | `REPORT.md`, generated from committed derived results, with sizes, seeds, versions and multiple-testing adjustments (section 5) |

At scaffold time (2026-10-23), searches found no Python port of the R packages `esreg`/`esback`
and no Python package with the Acerbi-Szekely tests. Say "to our knowledge" whenever this
appears in user-facing text.

Out of scope for v0.1: time-series foundation models, fine-tuning, intraday or realized-volatility
forecasting, multivariate or portfolio risk, ES-CAViaR, GPU work, the Nolde-Ziegel
conditional calibration test, a PyPI release.

## 2. Scope review (decisions taken at scaffold time)

Implementation budget: about 5 hours of agent time. What was cut or changed, and why:

1. **Stooq replaced by FRED.** On 2026-10-23, `stooq.com/q/d/l/` returned an HTML page with a
   JavaScript bot check instead of CSV. That cannot be automated without bypassing bot
   detection. FRED's public CSV endpoint
   (`https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>`) works without a key.
   Equity assets: `SP500`, `NASDAQCOM`, `NIKKEI225`. FRED only serves the last 10 years of
   `SP500` (2016-10-24 onward on 2026-10-23), so the out-of-sample (OOS) period starts in 2021.
2. **OOS period 2021-01-01 to 2026-09-30, the same for every asset.** Binance daily klines
   start 2017-08-17, so the 1000-observation window makes 2020-05 the earliest possible crypto
   OOS start. FRED `SP500` pushes the common start to 2021. COVID-2020 is therefore out of
   sample for no asset. The study covers the 2021 crypto boom and crashes, the 2022 bear
   market and rate hikes, and the 2025 tariff shock.
3. **Refit every 21 observations, filter daily** (quick run). Daily refits are the
   full-size configuration (`configs/study_full.toml`).
4. **CAViaR: SAV only, with ES = c * VaR**, where c is the in-sample mean of r/q over VaR
   exceedances (clipped at >= 1). This forecaster has no predictive distribution, so the
   Acerbi-Szekely simulation test is reported as not available for it.
5. **ESR bootstrap p-values** are implemented, but not used in the quick study or quick
   Monte Carlo (each replication refits the model). They are covered in the full run and a
   small dedicated check.
6. **Regimes**: an ex-ante volatility split (trailing 63-day realized volatility, known at
   t-1, above or below its per-asset OOS median) plus the crypto/equity split. Calendar-year
   splits have too few exceedances at 1% for the ESR tests, which need at least 3 negative
   quantile residuals and in practice far more.
7. **Acerbi-Szekely in the study** is a "should" item in M9: drop it before risking the
   ESR/MCS headline.
8. **R reference fixtures through webR.** No local R install. R 4.6.0 compiled to
   WebAssembly runs under Node (`scripts/r_fixtures/run_r.mjs`, webR 0.6.0, verified on
   2026-10-23 with `esreg` 0.6.2, `esback` 0.3.1, `quantreg` 6.1). `rugarch` (missing
   dependency `mclust`) and `GAS` are not in the webR 4.6 repository, so the VaR tests and
   Acerbi-Szekely use closed-form and published oracles instead.

## 3. Conventions

- **Sign convention.** Returns `r`, VaR `q` and ES `e` are on the return scale, so left-tail
  forecasts are negative and `e <= q`. Level `alpha` is the tail probability (0.01, 0.025).
  A hit (exceedance) is `I_t = 1{r_t <= q_t}`, as in `esback`. For continuous data the choice of
  `<` or `<=` has probability zero of mattering. Loss-convention users negate.
- **Returns in percent**: `100 * log(P_t / P_{t-1})`. This keeps `arch` well scaled and puts
  `esreg` data in a sensible range.
- **Results**: every test returns `BacktestResult(test, statistic, pvalue, nobs, df,
  alternative, details)`. A `pvalue` of `nan` means undefined for this input, with
  `details["reason"]`. Tests never raise on statistically degenerate input (no hits). They
  raise `ValueError` on invalid input (NaN, length mismatch, alpha outside (0,1), `e > q`
  where a test requires `e <= q`).
- **Randomness**: every stochastic function takes `seed: int | None` and builds
  `np.random.default_rng(seed)`. Study and Monte Carlo units derive their seed from
  `(global seed, zlib.crc32(unit_id))`, so a unit is reproducible regardless of run order.
- **p-values from simulation/bootstrap**: `(1 + #{T* at least as extreme as T0}) / (B + 1)`.
  This is never 0 and is valid in finite samples.
- **0 * log 0 = 0** in all likelihood ratios.
- **Parallelism**: at most 4 worker processes (`ProcessPoolExecutor`, spawn-safe). Workers set
  `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 1.

## 4. Module specifications

### 4.1 `var_tests` (M2)

Notation: `T` observations, `x = sum I_t` hits.

**Kupiec (1995) POF.**
`LR_pof = -2 [ (T-x) ln(1-alpha) + x ln(alpha) - (T-x) ln(1-x/T) - x ln(x/T) ]`, chi2(1).
Edge cases: `x = 0` gives `-2 T ln(1-alpha)`; `x = T` gives `-2 T ln(alpha)`.
`details`: `hits`, `hit_rate`, `expected_hits`.

Reference values (5% level, non-rejection regions of x computed from the formula; regression
fixtures):

| alpha | T=250 | T=510 | T=1000 |
|---|---|---|---|
| 0.01 | 1..6 | 2..10 | 5..16 |
| 0.025 | 3..11 | 7..20 | 16..35 |
| 0.05 | 7..19 | 17..35 | 38..64 |

Note: some published tables list "N < 7" for alpha=0.01, T=250, with no lower bound. The LR
test does reject x=0 there (LR = 5.03 > 3.84). Test against the formula, not the table.

**Christoffersen (1998) independence.** Transition counts `n_ij` over t = 2..T (i = I_{t-1},
j = I_t). `pi01 = n01/(n00+n01)`, `pi11 = n11/(n10+n11)`, `pi = (n01+n11)/(T-1)`.
`LR_ind = -2 [ (n00+n10) ln(1-pi) + (n01+n11) ln(pi) - n00 ln(1-pi01) - n01 ln(pi01)
- n10 ln(1-pi11) - n11 ln(pi11) ]`, chi2(1). A term whose denominator is 0 contributes 0.
With no hits: `LR_ind = 0`, p = 1, `details["reason"] = "no exceedances"`.
**Conditional coverage**: `LR_cc = LR_pof + LR_ind`, chi2(2) (Christoffersen's decomposition,
with LR_pof on all T observations).

Reference values (independent computation, `scripts/reference_values.py`):
- hits `0110011100`: `(n00,n01,n10,n11) = (2,2,2,3)`, `LR_ind = 0.0900142642`, p = 0.7641590225.
- T=250, alpha=0.01, hits at 0-based positions {10, 11, 100, 180, 181, 182}:
  counts (240,3,3,3); `LR_pof = 3.5553547711` (p 0.0593536190), `LR_ind = 15.9152966511`
  (p 6.62412e-5), `LR_cc = 19.4706514222` (p 5.91564e-5).

**Dynamic quantile (Engle-Manganelli 2004).** `Hit_t = I_t - alpha`. For t = L+1..T,
regressors `X_t = [1, Hit_{t-1}, ..., Hit_{t-L}, q_t]` (drop `q_t` if `include_var=False`),
L = 4 by default. `DQ = Hit' X (X'X)^+ X' Hit / (alpha (1-alpha))`, chi2(rank(X)). Use
`lstsq`/pseudo-inverse and report the rank: with no hits the lag columns are collinear
with the intercept. Oracle: equals `sum(fitted^2) / (alpha(1-alpha))` from an OLS fit of Hit
on X (statsmodels, transitively available through arch).

**Basel traffic light (BCBS 1996).** `c = BinomCDF(x; T, alpha)`. Green if `c < 0.95`,
yellow if `c < 0.9999`, red otherwise. The plus factor is defined only for T=250 and alpha=0.01:
green 0.00; x = 5..9 gives 0.40, 0.50, 0.65, 0.75, 0.85; red 1.00. Otherwise it is `nan`.
Reference (BCBS 1996 Table 2): cumulative probabilities for x=0..10 at T=250, alpha=0.01 are
8.11, 28.58, 54.32, 75.81, 89.22, 95.88, 98.63, 99.60, 99.89, 99.97, 99.99 %. So green is 0-4,
yellow 5-9 and red 10 or more (exact: c(4)=0.892188, c(5)=0.958817, c(9)=0.999750,
c(10)=0.999946).

### 4.2 `es_tests` (M3)

**Acerbi-Szekely (2014)** in the return convention (e < 0):
- `Z1 = 1 - (1/N) sum_{t: I_t=1} r_t / e_t`, where N = number of hits; `nan` if N = 0.
- `Z2 = 1 - (1/(T alpha)) sum_t I_t r_t / e_t`.
Under H0 (forecasts are the true conditional distribution) `E[Z2] = 0` exactly and
`E[Z1 | N>0] = 0`. Small Z means risk is underestimated, so the test is one-sided (left).
p-value: simulate M paths `X^(i)_t ~ F_t` from the forecast distributions (callable
`simulate(rng, n_paths) -> (n_paths, T)`, chunked to bound memory), compute `Z^(i)` with the
same (q, e), and set `p = (1 + #{Z^(i) <= Z_obs}) / (M + 1)`. For Z1, draws with N^(i)=0 are
dropped and counted in `details["n_undefined"]`.

Helper: `location_scale_simulator(mu, sigma, draw_standardized)`. For location-scale forecasts
with a fixed standardized law, `r_t/e_t` and `I_t` are scale-free, so the null law of Z
depends only on (law, alpha, T). The Monte Carlo runner exploits this to simulate the null
once per cell. The library function does not assume it.

Reference values (`scripts/reference_values.py`, M = 200,000, seed 1, T = 250, alpha = 0.025;
MC standard error of the 5% quantile about 0.003): 5% quantile of Z2 under H0 is -0.704
(normal), -0.738 (standardized t5), -0.708 (t10), -0.810 (t3). Acerbi and Szekely report
that the 5% threshold of about -0.70 is nearly distribution-free. Test: our normal-case 5%
quantile within -0.70 +/- 0.05; null mean of Z2 within 4 MC standard errors of 0.

**McNeil-Frey (2000) exceedance residuals.** `eps_t = (r_t - e_t) / s_t` for hits, where s_t
is the volatility forecast (1 if not given); m = number of hits. H0: E[eps] = 0. The primary
alternative is one-sided "less" (realized tail losses worse than ES). `t0 = sqrt(m) mean/sd`.
Bootstrap (Efron and Tibshirani 1993, ch. 16, one-sample bootstrap-t test): shift to `eps - mean(eps)`, resample B times,
`t*_b = sqrt(m) mean*/sd*`, then `p_less = (1 + #{t*_b <= t0})/(B+1)` and
`p_two = (1 + #{|t*_b| >= |t0|})/(B+1)`. `details` also holds the Student-t(m-1) asymptotic
p-values. m < 3 gives `nan`. Note: `esback::er_backtest` centres the bootstrap t-statistics
instead of the data. With the same data and B = 20,000, both versions should agree within
about 0.02 (loose fixture check, section 7).

### 4.3 `esreg` joint regression and `esr_backtest` (M4): the core port

Implementation rule: implement from the papers (Dimitriadis and Bayer 2019, Bayer and
Dimitriadis 2022) and the formulas below. `esreg` and `esback` are GPL-3. Do not translate
their source line by line; use them only as black-box oracles through webR.

#### 4.3.1 Model and loss

`Q_alpha(Y|X) = X_q' b_q`, `ES_alpha(Y|X) = X_e' b_e` (both include an intercept).
Fissler-Ziegel loss (DB 2019 eq. 2.2, with a(Y) = 0):

```
rho(y, q, e) = (1{y<=q} - alpha) G1(q) - 1{y<=q} G1(y)
             + G2(e) (e - q + (q - y) 1{y<=q} / alpha) - curlyG2(e)
```

| type | curlyG2(z) | G2(z) = curlyG2' | G2'(z) | G2''(z) | domain |
|---|---|---|---|---|---|
| 1 | -log(-z) | -1/z | 1/z^2 | -2/z^3 | z < 0 |
| 2 | -sqrt(-z) | 1/(2 sqrt(-z)) | 1/(4 (-z)^{3/2}) | 3/(8 (-z)^{5/2}) | z < 0 |
| 3 | -1/z | 1/z^2 | -2/z^3 | 6/z^4 | z < 0 |
| 4 | log(1+e^z) | 1/(1+e^{-z}) | e^z/(1+e^z)^2 | -e^z(e^z-1)/(1+e^z)^3 | all z |
| 5 | e^z | e^z | e^z | e^z | all z |

G1: type 1 is `G1(z) = z` (G1' = 1, G1'' = 0); type 2 is `G1(z) = 0`. Defaults are
`g1 = 2, g2 = 1`, as used by `esback`. With the defaults, `rho` equals the FZ0 loss exactly
(section 4.4). Unit test: `esr_loss(..., G1.ZERO, G2.NEG_INV) == fz0_loss(...)` to 1e-12.

**Suspected discrepancy in esreg 0.6.2 (verify in M4 before reporting):** in
`src/target_functions.cpp`, `G2_prime_prime` returns `-1/z^3` for type 1, while its own
documentation and calculus give `-2/z^3`. It enters only the misspecification term of
Lambda_22 (below), which is close to 0 when F(xi) is close to alpha. M4 must:
(a) check G-derivatives by finite differences;
(b) compare our `vcov` with R at R's coefficients with `misspec=FALSE` (should match) and
`misspec=TRUE` (expected to differ only through this term);
(c) confirm by switching a private `_esreg_compat_g2pp` flag that reproduces R to tolerance.
Only after (a)-(c), and with a minimal repro, describe it as a bug in docs/CHANGELOG. The
public default stays mathematically correct.

#### 4.3.2 Estimator (`esreg.estimator.esreg`)

1. Validate: finite data, same lengths, alpha in (0,1), and at least `kq + ke + 1` hits
   expected. Covariates exclude the intercept; it is always prepended.
2. For g2 in {1, 2, 3} the ES must be negative. Shift `y <- y - max(y)` and add `max(y)` back
   to both intercepts at the end. FZ-type losses are not translation invariant, so this shift is
   part of the estimator definition (DB 2019 sec. 3.1). Keep it for comparability with esreg.
3. Starting values: solve `Phi^{-1}(alpha_tilde) = -phi(Phi^{-1}(alpha))/alpha` for
   alpha_tilde in (0, alpha) (alpha=0.025 gives 0.0096987404; 0.01 gives 0.0038469647).
   `b_q0 = QR(y, X_q, alpha)`, `b_e0 = QR(y, X_e, alpha_tilde)`. If g2 in {1,2,3} and
   `m = max(X_e b_e0) >= -0.1`, subtract `m + 0.1` from the ES intercept.
4. Objective: mean loss. Return `+inf` if g2 in {1,2,3} and any `X_e b_e >= -0.01`.
5. Nelder-Mead (`scipy.optimize.minimize`, maxiter 2000, `xatol=1e-8`, `fatol=1e-10`).
6. Iterated local search: perturb the best point by `N(0, diag(se^2))`, with se the iid
   quantile-regression standard errors of the two starting fits; re-apply the intercept shift
   in step 3 when `max(X_e b) >= 0`; rerun Nelder-Mead; accept if the loss is lower. Stop after
   `early_stopping` (default 10) consecutive non-improvements. `early_stopping=0` means one
   local search.
7. Return `ESRegFit` with coefficients on the original scale, the loss and the number of local
   searches.

`quantreg.quantile_regression`: exact LP via `scipy.optimize.linprog(method="highs")`, with
`min tau 1'u+ + (1-tau) 1'u-` subject to `X b + u+ - u- = y` and `u+, u- >= 0`. Sparse
constraint matrix. Measured at scaffold time: about 140 ms for n=2000, k=2.
`quantile_regression_se_iid`: Koenker (2005) iid SEs, sparsity from the Hall-Sheather bandwidth.
These are used only to scale perturbations.

#### 4.3.3 Covariance (`esreg.covariance.vcov_asymptotic`)

`Cov(b_hat) = Lambda^{-1} C Lambda^{-1} / n` (DB 2019 Thm 2.6, eqs. 2.4-2.10), with the
sample averages below. Notation per observation i: `xi_i = x_q' b_q`, `eta_i = x_e' b_e`
(on the shifted scale when g2 in {1,2,3}), `f_i` the conditional density of y at xi_i, `F_i`
the conditional CDF at xi_i, and `v_i = Var(y - xi | y <= xi, x)` the conditional truncated
variance.

Correct specification (`misspec=False`), DB 2019 eqs. 2.6-2.10:
```
Lambda11 = mean[ x_q x_q' f_i (G1'(xi) + G2(eta)/alpha) ]
Lambda12 = 0
Lambda22 = mean[ x_e x_e' G2'(eta) ]
C11 = mean[ x_q x_q' (alpha G1'(xi) + G2(eta))^2 ] (1-alpha)/alpha
C12 = mean[ x_e x_q' (alpha G1'(xi) + G2(eta)) G2'(eta) (xi - eta) ] (1-alpha)/alpha
C22 = mean[ x_e x_e' G2'(eta)^2 ( v_i/alpha + (1-alpha)/alpha (xi - eta)^2 ) ]
```
(DB 2019 writes Lambda11 as `(1/alpha) E[XX' f (alpha G1' + G2)]`, which is the same thing.
C11 is `Var(psi1)` with `psi1 = (1/alpha)(1{y<=xi} - alpha)(alpha G1' + G2) x_q` (DB eq. 2.3),
which gives the `(1-alpha)/alpha` factor. Pin all scalings with the intercept-only oracle
below before trusting them.)

Misspecification-robust variant (`misspec=True`, the `esback` default, used by Bayer and
Dimitriadis 2022). It keeps the terms in `(F_i - alpha)` that vanish under correct
specification:
```
Lambda11 += mean[ x_q x_q' G1''(xi) (F_i - alpha) ]
Lambda12  = mean[ x_q x_e' G2'(eta) (F_i - alpha)/alpha ]
Lambda22 += mean[ x_e x_e' G2''(eta) xi (F_i - alpha)/alpha ]
C11 *= [1 + (1 - 2 alpha)(F_i - alpha) / (alpha (1 - alpha))]   (inside the mean)
C12 += mean[ x_e x_q' (alpha G1' + G2) G2' ((1-alpha)/alpha xi (F_i-alpha)/alpha
                                            - (F_i-alpha)/alpha (xi - eta)) ]
C22 += mean[ x_e x_e' G2'^2 2 (xi - eta) xi (alpha - F_i)/alpha ]
```
Before implementing, M4 derives these from `E[d psi / d theta]` and `E[psi psi']` without
imposing `F = alpha` (and with `E[y 1{y<=xi}] = alpha eta` for the ES part), and records the
derivation in `docs/esreg-covariance.md`. Then it checks them against R fixtures at R's
coefficients.

Nuisance estimators:
- **Bandwidth** (Hall-Sheather 1988, tau = 0.05):
  `h = n^{-1/3} z_{1-tau/2}^{2/3} [1.5 phi(z_alpha)^2 / (2 z_alpha^2 + 1)]^{1/3}`.
  Also Bofinger: `h = n^{-1/5} [4.5 phi(z_alpha)^4 / (2 z_alpha^2 + 1)^2]^{1/5}`, and
  Chamberlain: `h = z_{1-alpha/2} sqrt(tau(1-tau)/n)`.
  Reference: n=1000, alpha=0.025 gives h = 0.01313628; n=2000, alpha=0.025 gives 0.01042628;
  n=500, alpha=0.01 gives 0.00884657. Edge case: n=250, alpha=0.01 gives h = 0.011146 > alpha,
  so `alpha - h < 0`. Then shrink h to `0.9 min(alpha, 1-alpha)` and add a warning to
  `details`. R would fail here.
- **Density `f_i`**. `iid` (Koenker 1994): sparsity from a local linear fit of the ordered
  smallest-|u| residuals, constant across i. `nid` (Hendricks-Koenker 1992):
  `f_i = max(0, 2h / (x_q'(b(alpha+h) - b(alpha-h)) - eps))`, where `eps = machine_eps^{2/3}`
  and b(tau) are exact QR fits.
- **`F_i`**: fit the location-scale model `y = x_q' zeta + (x_q' phi) eps` by Gaussian PMLE
  (BFGS, falling back to Nelder-Mead and then to OLS starting values: OLS of y, then OLS of
  |residual|, with the intercept adjusted so that all scales are positive). F_i is the
  empirical CDF of the standardized residuals evaluated at `(xi_i - mu_i)/sigma_i`.
- **`v_i`**: `ind` is the sample variance of the negative quantile residuals (constant).
  `scl_N` applies the same location-scale fit to u = y - xi with
  `beta = -mu/sigma` (floored at -30) and
  `v = sigma^2 (1 - beta lambda - lambda^2)`, `lambda = phi(beta)/Phi(beta)`.
  `scl_sp` uses a Gaussian KDE of the standardized u-residuals with the Sheather-Jones (1991)
  solve-the-equation bandwidth, the truncated first and second moments up to beta by trapezoidal
  integration on a 1000-point grid, and `v = sigma^2 (m2/m0 - (m1/m0)^2)`. Any non-finite
  value falls back to `ind` with a warning in `details`.
- **Intercept-only case** (kq = ke = 1): force `iid` + `ind`, as esreg does. Oracle: DB 2019
  Remark 2.10. For an iid sample, `n Cov = Sigma` with
  `Sigma11 = alpha(1-alpha)/f(q)^2`, `Sigma12 = (1-alpha)(q - e)/f(q)`,
  `Sigma22 = v/alpha + (1-alpha)/alpha (q - e)^2`, independent of G1/G2.
  Check at n = 200,000 against the normal and t5 closed forms (within 3%).
- `bandwidth_sj`: implement from Sheather and Jones (1991) with exact (unbinned) pairwise
  sums, O(n^2) on at most a few thousand points. R's `bw.SJ` bins to 1000 points. Tolerance
  vs R: 2% relative.

#### 4.3.4 ESR backtests (`esr_backtest`)

Fit with `g1=2, g2=1` and default covariance `nid / scl_sp / misspec=True` (the esback
defaults).

| version | response | quantile eq. | ES eq. | H0 | statistic |
|---|---|---|---|---|---|
| 1 strict | r | 1, e | 1, e | b_e = (0, 1) | Wald, chi2(2) |
| 2 auxiliary | r | 1, q | 1, e | b_e = (0, 1) | Wald, chi2(2) |
| 3 strict intercept | r - e | 1, e | 1 | b_e0 = 0 | t; `p_less = Phi(t)`, `p_two = 2(1-Phi(abs(t)))` |

The primary p-value is two-sided for v1/v2 and one-sided ("less") for v3. In v3 a negative
intercept means the realized ES of r - e is below 0, so risk is underestimated. `details`
holds the coefficients, standard errors, both p-values, and the bootstrap p-values when
`n_boot > 0`. Bootstrap: iid row resampling, refit with `early_stopping=0`, and statistic
centred at the full-sample estimate (`b* - b_hat`) with the bootstrap covariance. Then
`p = (1 + #{T* >= T0})/(B+1)` (v1/v2); `|t*| >= |t0|` and `t* <= t0` (v3). If 5% or more of
the replications fail, the result is `nan` with a reason.

### 4.4 `scoring` (M5)

- **FZ0** (Patton, Ziegel and Chen 2019):
  `L = -(1/(alpha e)) 1{r<=q} (q - r) + q/e + log(-e) - 1`, requiring `e <= q < 0`
  (otherwise `ValueError`). Lower is better. Strictly consistent for (VaR, ES). Oracle: for
  iid normal data the expected loss is minimized at the true (q, e); test by grid search on
  simulated data.
- **Diebold-Mariano**: `d_t = L_a - L_b`, `DM = mean(d) / sqrt(LRV/T)`. Newey-West LRV with
  Bartlett weights and `lags = floor(4 (T/100)^{2/9})` by default. `small_sample=True`
  multiplies by `sqrt((T + 1 - 2h + h(h-1)/T)/T)` with h = 1 (Harvey, Leybourne and Newbold
  1997) and uses Student-t(T-1). Two-sided primary; `details` holds one-sided p-values.
- **MCS**: wrap `arch.bootstrap.MCS(losses, size, reps, block_size, method, bootstrap=
  "stationary", seed=...)`, then `.compute()` and read `.included`, `.excluded` and `.pvalues`
  (verified with arch 8.0.0 on 2026-10-23). Model names come from DataFrame columns. Record
  reps, block size and seed in `MCSResult`.

### 4.5 `forecasters` (M6)

All forecasters see a window of the last `window` returns (default 1000) ending at t-1.
Parameters are refit every `refit_every` observations; between refits, the filters run daily
with fixed parameters on data up to t-1. Never use data at or after t.

| name | VaR / ES | predictive draws |
|---|---|---|
| `hs` | k = ceil(n alpha); q = k-th order statistic of the window; e = mean of the k smallest | bootstrap from window |
| `gjr-t` | GJR-GARCH(1,1), constant mean, standardized t (arch). q = mu + sigma_t q_t(nu), e = mu + sigma_t ES_t(nu) | mu + sigma_t standardized-t draws |
| `fhs` | GJR-GARCH(1,1) Gaussian QMLE; z = standardized residuals; q, e = mu + sigma_t times (empirical k-th order statistic, tail mean) of z | mu + sigma_t bootstrap of z |
| `evt-pot` | same QMLE fit; GPD on the k = 0.10 n largest losses -z (McNeil-Frey 2000) | semi-parametric: empirical body, GPD tail |
| `caviar-sav` | `q_t = b0 + b1 q_{t-1} + b2 abs(r_{t-1})`; e = c q | none |

Formulas:
- Standardized t (unit variance), with `t_a = t_nu^{-1}(alpha)` and `s = sqrt((nu-2)/nu)`:
  `q_t(nu) = s t_a` and `ES_t(nu) = -s (nu + t_a^2)/(nu - 1) f_nu(t_a)/alpha`.
  Reference: nu=5 at alpha 0.025 gives (-1.9911641279, -2.7278020716); at alpha 0.01,
  (-2.6064635694, -3.4488367600). nu=4 at 0.025 gives (-1.9632431615, -2.8238712518).
  Normal at 0.025: (-1.9599639845, -2.3378027922); at 0.01: (-2.3263478740, -2.6652142203).
  Cross-check against numerical integration and `arch`'s `StudentsT.ppf`.
- EVT-POT: losses `l = -z`, threshold u = (k+1)-th largest loss, excesses `y = l - u` for the
  k largest. GPD MLE (`scipy.stats.genpareto.fit(y, floc=0)`), with a probability-weighted
  moments fallback (Hosking and Wallis 1987) if the MLE fails or gives xi <= -0.5.
  `VaR_l = u + (beta/xi)((n alpha / k)^{-xi} - 1)` and
  `ES_l = VaR_l/(1 - xi) + (beta - xi u)/(1 - xi)`. As |xi| -> 0 use the exponential limit
  (`VaR = u - beta ln(n alpha/k)`, `ES = VaR + beta`). If xi >= 1, ES is infinite: cap xi at
  0.95 and flag. Return `q = mu - sigma_t VaR_l` and `e = mu - sigma_t ES_l`.
- GARCH filtering with fixed parameters: either `arch_model(...).fix(params)` on the history
  (window start to t-1), or our own GJR recursion checked against `res.conditional_volatility`
  to 1e-8. Variance initialization must not look ahead (arch's backcast uses the start of the
  sample).
- CAViaR: q_0 = empirical alpha-quantile of the first 300 observations of the window (Engle
  and Manganelli 2004). Vectorize the recursion with `scipy.signal.lfilter`
  (`a = [1, -b1]`, input `b0 + b2 abs(r_{t-1})`). Estimate by minimizing the check loss:
  2000 random starts (b1 in (0, 1); b0 and b2 scaled by the window quantile), keep the 10 best,
  Nelder-Mead each, take the best. Constraint `0 <= b1 < 1` by penalty. ES ratio
  `c = max(1, mean(r_t/q_t | r_t <= q_t))` in-sample.

### 4.6 `checkpoint` (M1)

`UnitStore(root, config_hash)`: `root/manifest.json` holds the config hash, the package
version, the Python/numpy/scipy/arch versions and the start time. A mismatched hash raises
unless `--force-new`. Each unit is written to `root/units/<safe_id>.json` (or `.npz` for
arrays) through a temporary file and `os.replace`. `done(id)` checks existence and a
`complete: true` field. A progress line is appended to `root/progress.log`:
`ISO time | unit | seconds | done/total`. Runners take the todo list, skip units that are
done, and submit the rest to the pool.

## 5. Study design (M7, M9)

- **Data** (`data.py`). Binance spot monthly 1d klines for BTCUSDT and ETHUSDT from
  `data.binance.vision`. Verify each zip against its published `.CHECKSUM` (SHA-256). Column 0
  is the open time: milliseconds before 2025, microseconds from 2025-01-01 (seen in the files),
  so detect the unit by magnitude. Use column 4 (close). Index by UTC date. Check for gaps and
  duplicates. FRED: `SP500`, `NASDAQCOM`, `NIKKEI225`; `.` or empty values are dropped. Cache
  raw files under `data/raw/` (gitignored). Write a manifest with the SHA-256 of each
  processed return series to `results/<study>/data_manifest.json`. Snapshot end: 2026-09-30.
  Reproducibility caveat: FRED serves only the most recent 10 years of SP500, so a rerun after
  about 2027-01 cannot rebuild the first SP500 windows. The manifest hashes make any
  difference visible.
- **Design**: 5 assets x 5 models x alpha in {0.01, 0.025}; window 1000; OOS 2021-01-01 to
  2026-09-30 (about 2,099 crypto days and about 1,440 equity days); refit every 21
  observations (quick run).
- **Units and checkpointing**: the forecast unit is (asset, model), covering both alphas because
  the GARCH fits are shared. Output: `runs/<study>/units/forecast__<asset>__<model>.npz`
  (dates, r, q and e per alpha, AS accumulators). The backtest unit is
  (asset, model, alpha), stored as JSON. The compare unit is (asset, alpha), for MCS and DM.
- **Backtests per unit**: hit rate with Wilson 95% CI, Kupiec, Christoffersen ind/cc, DQ, and
  the traffic light (scaled to the full OOS length; zone reported as descriptive); ESR v1, v2,
  v3 (asymptotic); McNeil-Frey (s = model sigma where available); Acerbi-Szekely Z2 (should);
  mean FZ0 with Newey-West 95% CI.
- **Comparisons**: per (asset, alpha), MCS at 90% and 75% on FZ0 losses (R statistic,
  stationary bootstrap, block 10, 10,000 reps); pairwise DM vs `gjr-t` as the benchmark.
- **Acerbi-Szekely accumulators**: for each OOS day, draw `n_draws` from the model's
  predictive distribution and add `X 1{X<=q}/e` and `1{X<=q}` to per-regime accumulator
  arrays (full, calm, turbulent) of length n_draws. Z^(i) for any regime is then available
  without storing the draws.
- **Regimes**: calm/turbulent by the ex-ante 63-day realized volatility vs its OOS median, per
  asset. Report hit rate, Kupiec, ESR v1/v3 and mean FZ0 per regime. No MCS per regime
  (the series are not contiguous).
- **Multiple testing**: the primary family is the ESR v1 asymptotic test over 5 assets x 5
  models x 2 alphas = 50 hypotheses. Report raw p, Holm-adjusted p (FWER) and BH q-values
  (FDR). Treat all other tables as descriptive and say so in REPORT.md.
- **Headline outputs**: (1) a pass/fail matrix of strict ESR (BH q >= 0.05 counts as "not
  rejected") by model x asset, at each alpha; (2) MCS membership by model x asset; (3) a
  crypto vs equity summary; (4) a calm vs turbulent summary.
- **Budget**: about 33 CPU-minutes estimated (GARCH fits about 200 s, CAViaR about 1,200 s,
  ESR fits about 180 s, MCS about 300 s). Hard target: 45 CPU-minutes; check after M7 with
  timings in the progress log.
- **Commands**:
  `uv run tailcheck data fetch --config configs/study.toml`,
  `uv run tailcheck study run --config configs/study.toml --workers 4`,
  `uv run tailcheck study report --config configs/study.toml` (writes `results/study-v1/` and
  `REPORT.md`). Full size: the same commands with `configs/study_full.toml` (daily refits,
  bootstrap ESR with B = 999; roughly 10-20 CPU-hours, resumable).

## 6. Monte Carlo validation (M8)

**DGPs.** Under H0, forecasts are the true conditional (q, e) (and the law, for AS):
- `iid-t5`: r = z with z standardized t5.
- `garch-t5`: GARCH(1,1), omega 0.05, a 0.10, b 0.85 (unconditional variance 1), standardized
  t5 innovations, 500-observation burn-in.
- `iid-normal` (full profile only).

**Alternatives.** Data from `garch-t5`; the forecasts are wrong:
- `scale-0.85` / `scale-0.70`: forecasts are 0.85 / 0.70 times the true (q, e), so risk is
  underestimated.
- `normal-tails`: true sigma_t with normal quantile and ES. VaR is nearly right at 2.5%
  (-1.960 vs -1.991), ES is too thin (-2.338 vs -2.728). VaR tests should have little power
  here and ES tests should have power. This is the key diagnostic contrast.
- `slow-vol`: constant sigma = 1 with the right t5 shape. Unconditional coverage is right but
  clustering is ignored, so DQ, Christoffersen and strict ESR should have power and Kupiec
  should not.

**Output.** For each (test, DGP, n, alpha): rejection rate at 5% with a Wilson 95% interval,
the number of replications, the failure count (fits that errored count as failures, never
silently dropped), and median runtime. Size is flagged "distorted" if the Wilson interval
lies entirely outside [0.03, 0.07]. The ESR asymptotic tests are known to be oversized in
small samples (Bayer and Dimitriadis 2022 recommend the bootstrap there). Reproducing that is
expected, not a bug.

**Profiles.** `configs/mc_quick.toml` (default; at most 45 CPU-minutes, about 12 minutes on 4
workers) and `configs/mc_full.toml` (roughly 15-25 CPU-hours). Estimated quick budget:
ESR about 2,300 fits at about 1 s (Nelder-Mead ILS plus 4 LPs measured at about 1.3 s for
n = 2000 at scaffold time); cheap tests about 120 s, since all cheap tests share each
replication and the AS null is simulated once per cell. M8 starts with a timing calibration
on 10 replications per cell type. If the projection exceeds 45 CPU-minutes, lower
`reps_power` first and never take `reps_size` below 200. The results file records the reps
actually run. Units: (cell, chunk of 50 replications) through `UnitStore`.

**Power of the size check.** With R = 250 the Wilson half-width at 5% is about 0.028, so the
quick run detects gross distortions (size > 0.10) but not 0.06 vs 0.05. With R = 1000 (full)
the half-width is about 0.014; with R = 10,000 (cheap tests, full) about 0.004. State this
in REPORT.md.

**Commands.** `uv run tailcheck mc run --config configs/mc_quick.toml --workers 4` and
`uv run tailcheck mc report --config configs/mc_quick.toml` (writes
`results/mc-quick/size_power.csv` and a markdown table included in REPORT.md).

**Fast subset in pytest** (deterministic seeds, under 20 s total): Kupiec and Christoffersen
size at n = 500, alpha = 0.025, R = 400, iid normal; rejection rate in [0.02, 0.09]. AS Z2
null mean is about 0. McNeil-Frey size at R = 200. One ESR v1 size check at R = 60, n = 1000,
marked `slow`.

## 7. Validation and test strategy

Layers, each in `tests/`:

1. **Closed-form oracles**: Kupiec and Christoffersen worked examples (section 4.1), Basel
   cumulative probabilities and zones, standardized-t and normal VaR/ES values, Hall-Sheather
   bandwidths, alpha_tilde, the FZ0 = esr_loss identity, G-derivatives by finite differences,
   DQ vs the OLS form.
2. **Statistical oracles**: esreg consistency on DB 2019 DGP-(2) (`x ~ chi2(1)`,
   `y = -x + (1 + 0.5x) eps`, eps ~ N(0,1); true parameters at alpha = 0.025:
   `(-1.959964, -1.979982, -2.337803, -2.168901)`) at n = 20,000, within 4 estimated SEs.
   Intercept-only covariance vs DB 2019 Remark 2.10. EVT-POT on simulated t4 losses recovers
   the 1% VaR/ES within MC error.
3. **R reference fixtures** (webR; `scripts/r_fixtures/`). Python writes inputs to
   `tests/fixtures/r/inputs/*.csv`; an R script (written in M4) computes, and stores as JSON
   in `tests/fixtures/r/outputs/` (committed, small):
   (a) `quantreg::rq` coefficients at several tau;
   (b) `esreg` coefficients and loss for (g1, g2) in {(2,1), (1,2), (2,3)} on DGP-(2) with
       n = 1000;
   (c) `vcovA` for iid/ind, nid/scl_N and nid/scl_sp, each with misspec TRUE and FALSE, plus
       `density_quantile_function`, `conditional_truncated_variance` and `cdf_at_quantile`
       outputs;
   (d) `bw.SJ` on 3 vectors;
   (e) `esr_backtest` p-values for versions 1, 2, 3 on three forecast sets (correct GARCH-t,
       scale-0.85, normal-tails; n = 1500, alpha = 0.025 and 0.01);
   (f) `er_backtest` with B = 20,000.
   Also record R and package versions in `outputs/r_versions.json`.
   Tolerances: QR coefficients 1e-6. vcov at R's coefficients: 1e-6 relative for iid/ind and
   scl_N, 2e-2 for scl_sp (KDE binning differs), with misspec=TRUE expected to differ only
   through the G2'' term. esreg fit: our loss <= R loss + 1e-7 and coefficients within
   0.05 SE (the ILS is random in both). ESR p-values: absolute difference <= 0.01 for
   nid/scl_N and <= 0.03 for nid/scl_sp. er_backtest: <= 0.02.
   Command (from repo root; needs Node 20+):
   `npm ci --ignore-scripts --prefix scripts/r_fixtures` then
   `node scripts/r_fixtures/run_r.mjs scripts/r_fixtures/<script>.R tests/fixtures/r esreg esback quantreg`.
   Tests read only the committed JSON; they never need Node or R.
4. **Monte Carlo size/power** (section 6).
5. **Property tests**: invariances, including VaR tests that depend only on hits; Z2 scaling
   (r, q, e scaled by c > 0 leaves Z unchanged); esreg equivariance in scale for g2 = 1; FZ0
   rejects e > q. Input validation errors.
6. **Study smoke test**: a 2-asset, 2-model, 60-day OOS run on synthetic prices under
   `tmp_path`, including resume (kill after one unit, rerun, assert skip).

CI runs layers 1, 2 (fast parts), 3 (JSON comparisons), 5, 6 and the fast MC subset. Slow
tests are marked `slow` and run with `uv run pytest --runslow`.

## 8. Architecture

```
src/tailcheck/
  results.py         BacktestResult
  _validation.py     input checks
  var_tests.py       Kupiec, Christoffersen, DQ, traffic light            (M2)
  es_tests.py        Acerbi-Szekely Z1/Z2, McNeil-Frey                    (M3)
  esreg/             joint quantile/ES regression                          (M4)
    gfuns.py loss.py quantreg.py estimator.py covariance.py
  esr_backtest.py    Bayer-Dimitriadis ESR tests v1-v3                     (M4)
  scoring.py         FZ0, Diebold-Mariano, MCS (arch)                      (M5)
  forecasters/       hs, gjr-t, fhs, evt-pot, caviar-sav                   (M6)
  data.py            Binance + FRED download, cache, returns               (M7)
  checkpoint.py      resumable unit store, progress log                    (M1)
  montecarlo/        DGPs, runner, report                                  (M8)
  study/             runner, report                                        (M7, M9)
  cli.py             tailcheck backtest | mc | study | data                (M10)
configs/             study.toml, study_full.toml, mc_quick.toml, mc_full.toml
scripts/r_fixtures/  webR runner + R scripts for reference fixtures
tests/fixtures/r/    committed R outputs (JSON) and their inputs (CSV)
results/             small committed derived results; REPORT.md at repo root
```

Runtime dependencies: numpy, scipy, pandas and arch (arch brings statsmodels). The optional
`plot` extra adds matplotlib. Pinned in `uv.lock` (2026-10-23): numpy 2.5.4 (Python 3.12+) /
2.4.6 (Python 3.11), scipy 1.18.1 / 1.17.1, pandas 3.1.0, arch 8.0.0, statsmodels 0.15.0,
matplotlib 3.11.2. Licences: numpy BSD-3, scipy BSD-3, pandas BSD-3, arch NCSA,
statsmodels BSD-3, matplotlib PSF-style. All are compatible with MIT distribution. `esreg`
and `esback` (GPL-3) are not dependencies; they are used only as external oracles.

## 9. Milestones

Each milestone ends with `uv run ruff check .`, `uv run ruff format --check .` and
`uv run pytest` green, an updated AGENTS.md "Status and next steps", and one commit.

| # | Scope | Done-check |
|---|---|---|
| M1 | `checkpoint.UnitStore`, progress log, worker-pool helper with 4-process cap and thread env vars | tests: write/skip/resume, hash mismatch refusal, atomic write survives a simulated crash |
| M2 | `var_tests` | all section 4.1 reference values; DQ equals the OLS form; Basel table; edge cases (0 hits, all hits, single hit at the end) |
| M3 | `es_tests` | Z1/Z2 formulas; Z2 null 5% quantile -0.70 +/- 0.05 (normal, T=250, alpha=0.025); scale invariance; MF bootstrap size at R=200 in [0.02, 0.10] |
| M4a | `esreg` loss, G functions, LP quantile regression, estimator | FZ0 identity; finite-difference G-derivatives; DGP-(2) consistency; QR equals R `rq` |
| M4b | `vcov_asymptotic`, nuisance estimators, SJ bandwidth, covariance derivation note | Remark 2.10 oracle; R fixtures (c), (d); G2'' discrepancy checked as in 4.3.1 |
| M4c | `esr_backtest` v1-v3, asymptotic and bootstrap | R fixtures (e); bootstrap runs and is seeded; `nan` + reason on too few hits |
| M5 | `scoring` | FZ0 minimized at truth on simulated data; DM against a hand-computed HAC example; MCS wrapper keeps the known-best model on a synthetic panel |
| M6 | forecasters | standardized-t references; GJR recursion matches arch; EVT recovers t4 tail; CAViaR recovers simulated SAV parameters; no look-ahead test (forecast at t unchanged when data after t-1 is perturbed) |
| M7 | `data` + study forecast stage | downloads verified by checksum; manifest hashes; smoke study on synthetic data; real quick run of the forecast stage with timings logged |
| M8 | Monte Carlo runner and report | calibration recorded; quick profile completes within 45 CPU-minutes; `results/mc-quick/size_power.csv` committed; fast subset in pytest |
| M9 | study backtest + compare stages, report generator, `REPORT.md` | all numbers in REPORT.md come from committed results files; Holm/BH applied to the primary family; regime tables |
| M10 | CLI `backtest` on a user CSV, README usage, CHANGELOG, handoff | `tailcheck backtest example.csv` prints a table; CI green; AGENTS.md status current |

Ordering: M1 -> M2 -> M3 -> M4a-c -> M5 -> M6 -> M7 -> M8 and M9 (M8 can run while M9 is
written) -> M10. If time runs short, drop in this order: AS in the study, ESR bootstrap
p-values, `scl_sp` (fall back to the `scl_N` default and document the deviation from esback),
the regime tables.

## 10. Risks

- **Non-convex joint loss.** The ILS can stop in local minima. Mitigation: compare the loss
  value (not only the coefficients) with R; record `n_local_searches`; MC failure counts.
- **Covariance fidelity** (`scl_sp`, SJ bandwidth, misspec terms) drives the ESR p-values.
  Mitigation: R fixtures at R's coefficients isolate covariance from optimization.
- **Small exceedance counts.** At alpha = 1% equity OOS gives about 14 hits. ESR and AS are
  then weak and possibly oversized. Report the MC size at matching n and avoid strong claims.
- **Data drift**: FRED SP500 is a 10-year rolling window, and Binance changed its timestamp
  unit in 2025. Mitigation: manifest hashes and unit detection.
- **Licensing**: GPL-3 reference code. Mitigation: implement from papers; oracles only.
- **Compute**: the CAViaR multi-start dominates the study. If over budget, lower
  `caviar_random_starts` to 500 and record it.

## 11. References

- Acerbi, C. and Szekely, B. (2014). Backtesting expected shortfall. *Risk*, December 2014.
- Basel Committee on Banking Supervision (1996). Supervisory framework for the use of
  "backtesting" in conjunction with the internal models approach to market risk capital
  requirements.
- Bayer, S. and Dimitriadis, T. (2022). Regression-based expected shortfall backtesting.
  *Journal of Financial Econometrics* 20(3), 437-471. doi:10.1093/jjfinec/nbaa013
- Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*
  39(4), 841-862.
- Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business
  & Economic Statistics* 13(3), 253-263.
- Dimitriadis, T. and Bayer, S. (2019). A joint quantile and expected shortfall regression
  framework. *Electronic Journal of Statistics* 13(1), 1823-1871. doi:10.1214/19-EJS1560
- Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Engle, R. F. and Manganelli, S. (2004). CAViaR: conditional autoregressive value at risk by
  regression quantiles. *Journal of Business & Economic Statistics* 22(4), 367-381.
- Fissler, T. and Ziegel, J. F. (2016). Higher order elicitability and Osband's principle.
  *Annals of Statistics* 44(4), 1680-1707.
- Glosten, L. R., Jagannathan, R. and Runkle, D. E. (1993). On the relation between the
  expected value and the volatility of the nominal excess return on stocks. *Journal of
  Finance* 48(5), 1779-1801.
- Hall, P. and Sheather, S. J. (1988). On the distribution of a studentized quantile. *JRSS B*
  50(3), 381-391.
- Hansen, P. R., Lunde, A. and Nason, J. M. (2011). The model confidence set. *Econometrica*
  79(2), 453-497.
- Harvey, D., Leybourne, S. and Newbold, P. (1997). Testing the equality of prediction mean
  squared errors. *International Journal of Forecasting* 13(2), 281-291.
- Hendricks, W. and Koenker, R. (1992). Hierarchical spline models for conditional quantiles
  and the demand for electricity. *JASA* 87(417), 58-68.
- Hosking, J. R. M. and Wallis, J. R. (1987). Parameter and quantile estimation for the
  generalized Pareto distribution. *Technometrics* 29(3), 339-349.
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
- Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models.
  *Journal of Derivatives* 3(2), 73-84.
- McNeil, A. J. and Frey, R. (2000). Estimation of tail-related risk measures for
  heteroscedastic financial time series: an extreme value approach. *Journal of Empirical
  Finance* 7(3-4), 271-300. doi:10.1016/S0927-5398(00)00012-8
- Patton, A. J., Ziegel, J. F. and Chen, R. (2019). Dynamic semiparametric models for expected
  shortfall (and Value-at-Risk). *Journal of Econometrics* 211(2), 388-413.
- Sheather, S. J. and Jones, M. C. (1991). A reliable data-based bandwidth selection method
  for kernel density estimation. *JRSS B* 53(3), 683-690.
- Software used as oracles: R packages `esreg` 0.6.2 and `esback` 0.3.1 (GPL-3,
  github.com/BayerSe), run in webR 0.6.0 (R 4.6.0). Library wrapped: `arch` 8.0.0 (MCS, GARCH).
