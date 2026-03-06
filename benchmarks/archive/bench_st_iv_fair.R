# Fair benchmark for ST_IV models (subprocess + cooldown)
models <- c("PG_ST_IV", "Bin_ST_IV")
stan_ref <- c(PG_ST_IV = 82.5, Bin_ST_IV = 56.0)
seeds <- 1:3

for (model in models) {
  cat(sprintf("\n=== %s ===\n", model))
  times <- numeric(length(seeds))
  for (i in seq_along(seeds)) {
    Sys.sleep(10)  # longer cooldown for heavy models
    out <- system2(
      "C:/Program Files/R/R-4.5.2/bin/Rscript.exe",
      args = c("benchmarks/bench_single_model.R", model, as.character(seeds[i])),
      stdout = TRUE, stderr = TRUE
    )
    cat(paste(out, collapse = "\n"), "\n")
    m <- regmatches(out, regexpr("[0-9]+\\.[0-9]+s", out))
    times[i] <- as.numeric(sub("s", "", m[1]))
  }
  med <- median(times)
  stan <- stan_ref[model]
  ratio <- med / stan
  v <- if (ratio <= 1) "WIN" else if (ratio <= 1.2) "~" else "LOSS"
  cat(sprintf("  MEDIAN: %.1fs vs Stan %.1fs = %.2fx %s\n", med, stan, ratio, v))
}
