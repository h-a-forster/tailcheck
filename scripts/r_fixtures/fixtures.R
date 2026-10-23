# Reference outputs from the R packages esreg / esback / quantreg (GPL-3), used only as
# black-box oracles. Run via run_r.mjs with tests/fixtures/r mounted at /io:
#   node scripts/r_fixtures/run_r.mjs scripts/r_fixtures/fixtures.R tests/fixtures/r \
#        esreg esback quantreg jsonlite
library(esreg); library(esback); library(quantreg); library(jsonlite)
out <- list()
d <- read.csv("/io/inputs/dgp2.csv"); x <- d$x; y <- d$y

# (a) quantile regression
out$rq <- lapply(c(0.01, 0.025, 0.1, 0.5), function(tau)
  list(tau = tau, coef = unname(coef(rq(y ~ x, tau = tau)))))

# (b) esreg fits for several (g1, g2)
out$esreg <- list()
for (g in list(c(2, 1), c(1, 2), c(2, 3))) {
  set.seed(1)
  fit <- esreg(y ~ x | x, alpha = 0.025, g1 = g[1], g2 = g[2])
  out$esreg[[length(out$esreg) + 1]] <- list(g1 = g[1], g2 = g[2],
    coef = unname(fit$coefficients), loss = fit$loss)
}

# (c) covariance at R's (2, 1) coefficients
set.seed(1)
fit <- esreg(y ~ x | x, alpha = 0.025, g1 = 2, g2 = 1)
out$vcov_coef <- unname(fit$coefficients)
out$vcov <- list()
for (cfg in list(c("iid", "ind"), c("nid", "scl_N"), c("nid", "scl_sp"))) {
  for (m in c(FALSE, TRUE)) {
    v <- vcovA(fit, sparsity = cfg[1], sigma_est = cfg[2], misspec = m)
    out$vcov[[length(out$vcov) + 1]] <- list(sparsity = cfg[1], sigma_est = cfg[2],
      misspec = m, cov = unname(as.vector(v)))
  }
}
fit1 <- esreg(y ~ 1, alpha = 0.025)
out$vcov_intercept <- list(coef = unname(fit1$coefficients),
  cov = unname(as.vector(vcovA(fit1))))

# (d) Sheather-Jones bandwidths
out$bw_sj <- sapply(0:2, function(i) bw.SJ(read.csv(sprintf("/io/inputs/bw_%d.csv", i))$x))

# (e) ESR backtests and (f) exceedance-residual backtest
f <- read.csv("/io/inputs/forecasts.csv")
out$esr <- list()
for (set in c("true", "scale", "normal")) for (tag in c("025", "01")) {
  a <- if (tag == "025") 0.025 else 0.01
  q <- f[[paste0("q_", set, "_", tag)]]; e <- f[[paste0("e_", set, "_", tag)]]
  for (ver in 1:3) for (se in c("scl_N", "scl_sp")) {
    set.seed(1)
    b <- tryCatch(esr_backtest(r = f$r, q = q, e = e, alpha = a, version = ver,
      cov_config = list(sparsity = "nid", sigma_est = se, misspec = TRUE)),
      error = function(z) list(error = conditionMessage(z)))
    out$esr[[length(out$esr) + 1]] <- c(list(set = set, alpha = a, version = ver,
      sigma_est = se), lapply(b, function(v) if (is.null(v) || is.na(v)) NULL else v))
  }
}
out$er <- list()
for (set in c("true", "scale", "normal")) {
  set.seed(1)
  b <- er_backtest(r = f$r, q = f[[paste0("q_", set, "_025")]],
    e = f[[paste0("e_", set, "_025")]], s = f$s, B = 20000)
  out$er[[length(out$er) + 1]] <- c(list(set = set), b)
}
out$versions <- list(R = R.version.string, esreg = as.character(packageVersion("esreg")),
  esback = as.character(packageVersion("esback")),
  quantreg = as.character(packageVersion("quantreg")))
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE),
  "/io/outputs/r_reference.json")
cat("done\n")
