# Smoke script for run_r.mjs: prints the R and package versions used for fixtures.
cat(R.version.string, "\n")
for (p in c("esreg", "esback", "quantreg")) cat(p, as.character(packageVersion(p)), "\n")
writeLines(c(R.version.string, paste("esreg", packageVersion("esreg")),
             paste("esback", packageVersion("esback"))), "/io/r_versions.txt")
