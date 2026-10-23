# Run esback::esr_backtest (asymptotic, package defaults) on simulated null samples.
# Black-box oracle use only (esback is GPL-3). Inputs: /io/inputs/rep_XXX.csv (r, q, e),
# written by scripts/esback_null/esback_null.py. One JSON per replication goes to
# /io/out/rep_XXX.json; finished replications are skipped, so a rerun resumes.
#
#   node scripts/r_fixtures/run_r.mjs scripts/esback_null/esback_null.R tmp/esback_null \
#     esreg esback jsonlite
library(esback); library(jsonlite)
alpha <- as.numeric(readLines("/io/alpha.txt"))
files <- sort(list.files("/io/inputs", pattern = "^rep_[0-9]+\\.csv$"))
dir.create("/io/out", showWarnings = FALSE)
log_line <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n",
  file = "/io/progress.log", append = TRUE)
for (f in files) {
  out_file <- file.path("/io/out", sub("\\.csv$", ".json", f))
  if (file.exists(out_file)) next
  d <- read.csv(file.path("/io/inputs", f))
  t0 <- proc.time()[["elapsed"]]
  res <- list(file = f)
  for (ver in c(1, 3)) {
    set.seed(1)
    b <- tryCatch(esr_backtest(r = d$r, q = d$q, e = d$e, alpha = alpha, version = ver),
      error = function(z) list(error = conditionMessage(z)))
    res[[paste0("v", ver)]] <- lapply(b, function(v) if (is.null(v) || length(v) == 0 ||
      all(is.na(v))) NULL else v)
  }
  res$seconds <- proc.time()[["elapsed"]] - t0
  part <- paste0(out_file, ".part")
  writeLines(toJSON(res, auto_unbox = TRUE, digits = NA), part)
  file.rename(part, out_file)
  log_line("done", f, sprintf("%.1fs", res$seconds))
}
writeLines(c(R.version.string, paste("esreg", packageVersion("esreg")),
  paste("esback", packageVersion("esback"))), "/io/r_versions.txt")
cat("finished\n")
