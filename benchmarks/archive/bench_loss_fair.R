# Fair benchmark: subprocess isolation + cooldown to avoid thermal throttling
models <- c("NB_ICAR", "NB_HSGP", "PG_GPt", "Bin_GPt")
stan_ref <- c(NB_ICAR = 3.5, NB_HSGP = 2.9, PG_GPt = 10.0, Bin_GPt = 3.3)
seeds <- 1:5

results <- list()
for (model in models) {
  cat(sprintf("\n=== %s ===\n", model))
  times <- numeric(length(seeds))
  for (i in seq_along(seeds)) {
    Sys.sleep(5)  # cooldown
    out <- system2(
      "C:/Program Files/R/R-4.5.2/bin/Rscript.exe",
      args = c("benchmarks/bench_single_model.R", model, as.character(seeds[i])),
      stdout = TRUE, stderr = TRUE
    )
    cat(paste(out, collapse = "\n"), "\n")
    m <- regmatches(out, regexpr("[0-9]+\\.[0-9]+s", out))
    times[i] <- as.numeric(sub("s", "", m[1]))
  }
  results[[model]] <- median(times)
  cat(sprintf("  MEDIAN: %.1fs\n", median(times)))
}

cat("\n\n========== RESULTS ==========\n")
cat(sprintf("%-15s %8s %8s %8s %6s\n", "Model", "Median", "Stan", "Ratio", ""))
cat(paste0(rep("-", 50), collapse = ""), "\n")
for (nm in names(results)) {
  stan <- stan_ref[nm]
  ratio <- results[[nm]] / stan
  v <- if (ratio <= 1) "WIN" else if (ratio <= 1.2) "~" else "LOSS"
  cat(sprintf("%-15s %7.1fs %7.1fs %7.2fx %s\n", nm, results[[nm]], stan, ratio, v))
}
