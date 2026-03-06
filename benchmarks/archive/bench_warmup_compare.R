# Compare warmup=250 vs 500 for parity models
models <- c("NB_HSGP", "NB_ICAR")
warmups <- c(250L, 500L)
seeds <- 1:3

for (model in models) {
  cat(sprintf("\n=== %s ===\n", model))
  for (w in warmups) {
    times <- numeric(length(seeds))
    for (i in seq_along(seeds)) {
      Sys.sleep(5)
      out <- system2("C:/Program Files/R/R-4.5.2/bin/Rscript.exe",
                      args = c("benchmarks/bench_warmup_test.R", model,
                               as.character(seeds[i]), as.character(w)),
                      stdout = TRUE, stderr = TRUE)
      cat(paste(out, collapse = "\n"), "\n")
      m <- regmatches(out, regexpr("[0-9]+\\.[0-9]+s", out))
      times[i] <- if (length(m) > 0) as.numeric(sub("s", "", m[1])) else NA
    }
    cat(sprintf("  warmup=%d MEDIAN: %.1fs\n", w, median(times, na.rm = TRUE)))
  }
}
