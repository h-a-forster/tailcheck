# esreg fits behind esback's strict ESR (v1) for the replications listed in /io/trace.txt.
# Writes /io/trace_out.json (coefficients and esreg's loss). Black-box oracle use only.
#   node scripts/r_fixtures/run_r.mjs scripts/esback_null/trace_v1.R tmp/esback_null \
#     esreg jsonlite
library(esreg); library(jsonlite)
alpha <- as.numeric(readLines("/io/alpha.txt"))
reps <- readLines("/io/trace.txt")
out <- list()
for (f in reps) {
  d <- read.csv(file.path("/io/inputs", paste0(f, ".csv")))
  set.seed(1)
  fit <- esreg(r ~ e | e, data = d, alpha = alpha, g1 = 2, g2 = 1, early_stopping = 10)
  out[[f]] <- list(coef = unname(fit$coefficients), loss = fit$loss)
}
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA), "/io/trace_out.json")
cat("done\n")
