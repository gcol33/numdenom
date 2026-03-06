# Run each seed as a subprocess with 5s cooldown to avoid thermal throttling
times <- numeric(5)
for (i in 1:5) {
  Sys.sleep(5)  # cooldown between runs
  out <- system2(
    "C:/Program Files/R/R-4.5.2/bin/Rscript.exe",
    args = c("benchmarks/bench_nb_icar_single.R", as.character(i)),
    stdout = TRUE, stderr = TRUE
  )
  # Parse time from output like "seed=4: 2.8s"
  m <- regmatches(out, regexpr("[0-9]+\\.[0-9]+s", out))
  times[i] <- as.numeric(sub("s", "", m[1]))
  cat(out, sep = "\n")
}
cat(sprintf("\nMEDIAN: %.1fs  MEAN: %.1fs  [%s]\n",
            median(times), mean(times),
            paste(round(times, 1), collapse = ", ")))
cat(sprintf("Stan ref: 3.5s  Ratio: %.2fx\n", median(times) / 3.5))
