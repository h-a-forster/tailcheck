# Minimal repro: esreg 0.6.2 second derivative of G2 for type 1 (G2(z) = -1/z).
# Run: node scripts/r_fixtures/run_r.mjs scripts/r_fixtures/g2pp_repro.R tmp esreg
# Output on 2026-10-23 (webR 0.6.0, R 4.6.0): G2_prime_prime(-2, 1) = 0.125, while the
# finite difference of esreg's own G2_prime_fun and calculus both give 0.25.
library(esreg)
z <- -2; h <- 1e-5
cat("G2_prime_fun(z,1)         =", esreg:::G2_prime_fun(z, 1), " (1/z^2 =", 1/z^2, ")\n")
cat("G2_prime_prime(z,1)       =", esreg:::G2_prime_prime(z, 1), "\n")
cat("finite difference of G2'  =", (esreg:::G2_prime_fun(z + h, 1) - esreg:::G2_prime_fun(z - h, 1)) / (2*h), "\n")
cat("-2/z^3                    =", -2/z^3, "\n")
