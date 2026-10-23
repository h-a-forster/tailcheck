# Covariance of the joint quantile/ES estimator

This note derives the sandwich covariance used by `tailcheck.esreg.covariance`. It follows
Dimitriadis and Bayer (2019), Theorem 2.6, and adds the terms that vanish only under correct
specification. It also records how the implementation was checked against the R package
`esreg` 0.6.2, and one discrepancy found in that package.

## Setup

Per observation, write `xi = x_q' b_q` (quantile), `eta = x_e' b_e` (ES), `I = 1{y <= xi}`,
`F = P(y <= xi | x)`, `f` the conditional density at `xi`, and
`v = Var(y - xi | y <= xi, x)`. The loss is

```
rho = (I - a) G1(xi) - I G1(y) + G2(eta) (eta - xi + (xi - y) I / a) - curlyG2(eta)
```

Its gradient (the derivative of `I` vanishes almost surely) is

```
psi_q = x_q (1/a) (I - a) (a G1'(xi) + G2(eta))
psi_e = x_e G2'(eta) (eta - xi + (xi - y) I / a)
```

and `sqrt(n)(b_hat - b) -> N(0, Lambda^{-1} C Lambda^{-1})` with `Lambda = dE[psi]/db'` and
`C = E[psi psi']`.

## Lambda

`E[psi_q | x] = (1/a)(F - a)(a G1' + G2)`, so

```
Lambda_11 = E[x_q x_q' ( f (G1' + G2/a) + G1''(xi) (F - a) )]
Lambda_12 = E[x_q x_e' G2'(eta) (F - a) / a]
```

`E[psi_e | x] = G2'(eta) (eta - xi + (xi F - E[y I | x]) / a)`. Using
`d/dxi E[y I | x] = xi f`, the derivative with respect to `b_q` is
`x_e x_q' G2'(eta)(F - a)/a = Lambda_12'`. With respect to `b_e`:

```
Lambda_22 = E[x_e x_e' ( G2'(eta) + G2''(eta) (eta - xi + (xi F - E[y I | x]) / a) )]
```

Under correct specification `F = a` and `E[y I | x] = a eta`, so the bracket after `G2''`
vanishes. The misspecification-robust version keeps `F` but evaluates `E[y I | x]` at
`a eta` (the ES model), which gives the term `G2''(eta) xi (F - a) / a`.

## C

With `E[(I - a)^2 | x] = F(1 - 2a) + a^2 = a(1 - a)[1 + (1 - 2a)(F - a)/(a(1 - a))]`:

```
C_11 = E[x_q x_q' (a G1' + G2)^2 (1 - a)/a [1 + (1 - 2a)(F - a)/(a (1 - a))]]
```

For the cross term, `E[(I - a)(eta - xi + (xi - y) I / a) | x] = (F - a)(eta - xi) +
(1 - a)/a (xi F - E[y I | x])`. With `E[y I | x] = a eta`:

```
C_21 = E[x_e x_q' (a G1' + G2) G2' ( (1 - a)/a (xi - eta)
                                      + (1 - a)/a xi (F - a)/a - (F - a)/a (xi - eta) )]
```

For `C_22`, write `A = eta - xi` and `B = (xi - y) I / a`. Then `E[A^2] = A^2`,
`2 E[A B] = 2 A (xi F - a eta)/a = -2 (xi - eta)^2 + 2 (xi - eta) xi (a - F)/a`, and
`E[B^2] = (v + (xi - eta)^2)/a` at `F = a`:

```
C_22 = E[x_e x_e' G2'^2 ( v/a + (1 - a)/a (xi - eta)^2 + 2 (xi - eta) xi (a - F)/a )]
```

`E[B^2]` is kept at its correctly specified value; only the cross term carries `F - a`.
These are the formulas in DESIGN 4.3.3. With `F = a` they reduce to DB 2019 eqs. 2.6-2.10,
and with intercepts only to Remark 2.10 (`n Cov = Sigma` with `Sigma_11 = a(1 - a)/f^2`,
`Sigma_12 = (1 - a)(q - e)/f`, `Sigma_22 = v/a + (1 - a)/a (q - e)^2`). The test
`test_intercept_only_covariance_formula` checks that identity to 1e-8 for two (G1, G2) pairs.

## Checks against R (`tests/fixtures/r/`, webR 0.6.0, R 4.6.0)

All comparisons evaluate our covariance at R's coefficients, so optimisation noise does not
enter. DGP-(2) of DB 2019, n = 1000, alpha = 0.025, (G1, G2) = (2, 1).

| sparsity / sigma | misspec | max relative difference vs `esreg::vcovA` |
|---|---|---|
| nid / scl_N | FALSE | 4e-6 |
| nid / scl_N | TRUE | 3.2 with the correct G2''; 4e-6 with esreg's G2'' |
| nid / scl_sp | FALSE | 0.004 |
| nid / scl_sp | TRUE | 3.1 with the correct G2''; 0.003 with esreg's G2'' |
| iid / ind | either | 0.33 (see below) |

Sheather-Jones bandwidths agree with `bw.SJ` within 0.5% (R bins to 1000 points; we use exact
pair sums). The KDE truncated moments for `scl_sp` use the closed form of the Gaussian-kernel
integrals instead of a trapezoidal grid.

**iid sparsity.** Our `iid` density (Koenker 1994: median-regression slope of the ordered
smallest-|u| residuals against their ranks) is about 22% higher than `esreg`'s on the
fixture. In simulation (intercept only, normal data, 10 seeds each) our estimate is unbiased
(mean ratio to the true density 1.019 at n = 20,000 and 1.010 at n = 200,000, sd 0.10 and
0.05). We did not establish which estimator `esreg` uses. `iid` is not the default and is
only forced for intercept-only fits, which the backtests do not use.

## Discrepancy in esreg 0.6.2: G2'' for G2 type 1

For `G2(z) = -1/z`, `G2'(z) = 1/z^2` and `G2''(z) = -2/z^3`. `esreg` 0.6.2 returns `-1/z^3`.
Minimal repro (`scripts/r_fixtures/g2pp_repro.R`), using only esreg's own functions:

```
G2_prime_fun(-2, 1)                 = 0.25   (1/z^2)
G2_prime_prime(-2, 1)               = 0.125
finite difference of G2_prime_fun   = 0.25
-2/z^3                              = 0.25
```

How we ruled out an adapter mistake on our side:

1. Our G-function derivatives pass finite-difference checks for all five G2 types
   (`test_g2_derivatives_by_finite_differences`).
2. With `misspec = FALSE` (no G2'' term) our covariance matches `vcovA` to 4e-6 (scl_N).
3. With `misspec = TRUE` it differs, and switching only G2'' to `-1/z^3` (private flag
   `_esreg_compat_g2pp`) restores agreement to 4e-6. Every other misspecification term
   therefore matches.
4. The same switch makes all 36 `esback::esr_backtest` reference p-values agree within
   0.006 (`tests/test_esr_backtest.py`). With the correct G2'', strict ESR (versions 1 and
   2) p-values on these fixtures move by up to 0.25; version 3 moves by at most 0.01.

The term enters only `Lambda_22` under `misspec = TRUE` (the `esback` default), multiplied by
the estimated `F - a`, so it is noise around 0 when the model is right. tailcheck uses the
mathematically correct `-2/z^3` by default. The Monte Carlo (REPORT.md) reports the size of
strict ESR under both variants so the practical effect is visible.
