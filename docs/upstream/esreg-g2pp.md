# Upstream issue draft: esreg `G2_prime_prime(z, 1)` returns `-1/z^3` instead of `-2/z^3`

Status: filed 2026-11-06 as <https://github.com/BayerSe/esreg/issues/17>; fixed in esreg
0.6.3 (2026-11-09). The draft as written follows. Target: <https://github.com/BayerSe/esreg/issues> (the CRAN page
lists the maintainer's e-mail address as undeliverable). Checked 2026-10-30: esreg 0.6.2
(published 2023-05-13) is still the current CRAN release, the GitHub `main` branch has the
same code, and no existing issue mentions `G2_prime_prime`.

---

**Title:** `G2_prime_prime(z, type = 1)` returns `-1/z^3`; the second derivative of
`G2(z) = -1/z` is `-2/z^3`

**Version:** esreg 0.6.2 (CRAN), R 4.6.0. Also present on GitHub `main`.

**Description.** For `type = 1`, `G2_fun(z) = -1/z` and `G2_prime_fun(z) = 1/z^2`, so the
second derivative is `-2/z^3`. The roxygen documentation of `G2_prime_prime` also states
`-2/z^3`, but `src/target_functions.cpp` computes `out = -1/pow(z, 3);`. Types 2-5 are
consistent with `G2_prime_fun` (checked by differentiation); only type 1, the default
`g2`, is affected.

**Minimal repro.**

```r
library(esreg)  # 0.6.2
z <- -2; h <- 1e-5
G2_prime_prime(z, 1)                                          # 0.125
(G2_prime_fun(z + h, 1) - G2_prime_fun(z - h, 1)) / (2 * h)   # 0.25
-2 / z^3                                                      # 0.25
```

**Expected:** `G2_prime_prime(-2, 1) == 0.25` (equal to `-2/z^3` and to the finite
difference of `G2_prime_fun`).

**Actual:** `0.125`, i.e. half the correct value for every `z < 0`.

**Impact.** `G2_prime_prime` enters only the `misspec = TRUE` term of `Lambda_22` in
`vcovA` (`G2_prime_prime_xe * xbq * (F - alpha) / alpha`, `src/sandwich_loops.cpp`).
`misspec = TRUE` is the default of `vcovA`, hence of `vcov(fit)`, and of
`esback::esr_backtest` (`cov_config`). So with the default `g2 = 1`:

- the misspecification-robust covariance of the ES coefficients is wrong whenever the fitted
  quantile model is not exactly calibrated (`F - alpha != 0`). On DGP (2) of Dimitriadis and
  Bayer (2019), n = 1000, alpha = 0.025, the `nid`/`scl_N` covariance entries differ between the
  two G2'' values by up to 3.2 in relative terms;
- `esback::esr_backtest` p-values change: strict ESR (versions 1 and 2) by up to 0.25 on
  three test forecast sets, and by up to 0.76 (median 0.03) across 50 forecaster/asset/level
  cells of a crypto and equity-index study; the one-sided intercept test (version 3) by at
  most 0.01 on the test sets;
- under the null (GARCH-t5 with true forecasts, n = 1000, alpha = 0.025, 250 samples) the
  strict ESR test rejects 8.0% at 5% with the current value (`esback`, and an independent
  implementation reproducing it) and 6.4% with the corrected one.

**Suggested fix.** `out = -2/pow(z, 3);` for `type == 1` in `G2_prime_prime`
(`src/target_functions.cpp`), plus a test comparing `G2_prime_prime` with a finite difference
of `G2_prime_fun` for all types.

---

Evidence in this repository: the repro script `scripts/r_fixtures/g2pp_repro.R` (run through
webR on 2026-10-23 and 2026-10-30, same output), the covariance comparison in
[`docs/esreg-covariance.md`](../esreg-covariance.md), the ESR fixture comparison
(`tests/test_esr_backtest.py`), the study numbers in [`REPORT.md`](../../REPORT.md)
section 2 and the null-sample comparison in `results/esback-null/`.
