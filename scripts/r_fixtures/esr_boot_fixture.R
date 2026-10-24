# Bootstrap p-values of esback::esr_backtest (GPL-3, used only as a black-box oracle) on
# the committed forecasts.csv, for a distributional check of tailcheck's ESR bootstrap.
# R's and Python's random streams differ, so only agreement within Monte Carlo error is
# expected. About 12 minutes in webR. Run via run_r.mjs with tests/fixtures/r at /io:
#   node scripts/r_fixtures/run_r.mjs scripts/r_fixtures/esr_boot_fixture.R tests/fixtures/r \
#        esreg esback jsonlite
library(esback); library(jsonlite)
B <- 499
f <- read.csv("/io/inputs/forecasts.csv")
cases <- list(c("scale", 3), c("scale", 1), c("true", 3))
out <- list(B = B, cases = list())
for (cs in cases) {
  set <- cs[1]; ver <- as.integer(cs[2])
  set.seed(20261024)
  b <- esr_backtest(r = f$r, q = f[[paste0("q_", set, "_025")]],
    e = f[[paste0("e_", set, "_025")]], alpha = 0.025, version = ver, B = B,
    cov_config = list(sparsity = "nid", sigma_est = "scl_sp", misspec = TRUE))
  out$cases[[length(out$cases) + 1]] <- c(list(set = set, alpha = 0.025, version = ver), b)
}
out$versions <- list(R = R.version.string, esreg = as.character(packageVersion("esreg")),
  esback = as.character(packageVersion("esback")))
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE),
  "/io/outputs/r_esr_bootstrap.json")
cat("done\n")
