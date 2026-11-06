# Minimal repro: esback::esr_backtest depends on the unit of the returns.
# Run (repo root): node scripts/r_fixtures/run_r.mjs scripts/r_fixtures/scale_repro.R examples esreg esback
# Output on 2026-11-06 (webR 0.6.0, R 4.6.0, esback 0.3.1, esreg 0.6.2), strict ESR p-value of
# the same forecasts in different units; a Wald test of coefficients (0, 1) should not change:
#   c=1      0.1925
#   c=0.01   0.3792
#   c=0.002  0.9528
#   c=0.001  1.0000
#   c=0.0001 0.0000
# tailcheck gives 0.215 at all of them (test_esr_statistic_does_not_depend_on_scale).
library(esback)
d <- read.csv("/io/garch_t5.csv")
for (c in c(1, 0.01, 0.002, 0.001, 0.0001)) {
  set.seed(1)
  b <- esr_backtest(r = d$r * c, q = d$q * c, e = d$e * c, alpha = 0.025, version = 1)
  cat(sprintf("c=%g %.4f\n", c, b$pvalue_twosided_asymptotic))
}
