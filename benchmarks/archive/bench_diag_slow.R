# =============================================================================
# Diagnostic benchmark: Per-step cost profiling for slow model configurations
# Confirms whether per-step cost (not treedepth) is the bottleneck vs Stan
# =============================================================================
#
# Usage: Rscript benchmarks/bench_diag_slow.R [row_start] [row_end]
# Default: runs the 12 slow configs (4 categories × 3 families)

args <- commandArgs(trailingOnly = TRUE)

# Slow model rows from gradient_methods.md:
# Category 1: Temporal GP (GP_t) - rows 13/43/73 (PG/NB/Bin)
# Category 2: Slopes+ICAR - rows 19/49/79 (PG/NB/Bin)
# Category 3: ST Type IV - rows 17/47/77 (PG/NB/Bin)
# Category 4: HSGP - rows 9/39/69 (PG/NB/Bin) - but NB ~parity, so PG+Bin only
SLOW_ROWS <- c(13, 43, 73, 19, 49, 79, 17, 47, 77, 9, 69)

ROW_START <- if (length(args) >= 1) as.integer(args[1]) else NA
ROW_END   <- if (length(args) >= 2) as.integer(args[2]) else NA

# If specific range given, use that; otherwise use SLOW_ROWS
if (!is.na(ROW_START)) {
  rows <- ROW_START:ROW_END
} else {
  rows <- SLOW_ROWS
}

cat("=== Per-Step Diagnostic Benchmark ===\n")
cat(sprintf("Running %d model configurations\n", length(rows)))
cat("Columns: row | model | time(s) | mean_td | eps | n_lf | per_step(ms) | leapfrog_total\n\n")

results <- data.frame(
  row = integer(),
  model = character(),
  time_s = numeric(),
  mean_td = numeric(),
  epsilon = numeric(),
  mean_lf = numeric(),
  per_step_ms = numeric(),
  total_lf = numeric(),
  stringsAsFactors = FALSE
)

for (row in rows) {
  cat(sprintf("--- Row %d ---\n", row))

  # Run bench_single_row.R as subprocess, capture full output
  cmd <- sprintf(
    '"%s" benchmarks/bench_single_row.R %d H 600',
    file.path(R.home("bin"), "Rscript"),
    row
  )

  out <- tryCatch({
    system(cmd, intern = TRUE, timeout = 660)
  }, error = function(e) {
    paste0("ERROR: ", e$message)
  })

  # Parse RESULT line
  result_line <- grep("^RESULT:", out, value = TRUE)
  if (length(result_line) == 0) {
    cat(sprintf("  Row %d: NO RESULT (timeout or error)\n", row))
    next
  }

  parts <- strsplit(result_line[1], ":")[[1]]
  time_s <- as.numeric(parts[4])
  if (is.na(time_s)) {
    cat(sprintf("  Row %d: %s\n", row, result_line[1]))
    next
  }

  # Parse diagnostic lines from output
  td_line <- grep("mean treedepth|avg treedepth|treedepth=", out, value = TRUE)
  eps_line <- grep("epsilon=|step size=", out, value = TRUE)
  lf_line <- grep("leapfrog|n_leapfrog", out, value = TRUE)

  # Extract mean treedepth from warmup diagnostic output
  # Look for the last warmup print: "warmup iter N: treedepth=X, epsilon=Y"
  warmup_lines <- grep("warmup iter.*treedepth=", out, value = TRUE)
  final_eps <- NA
  if (length(warmup_lines) > 0) {
    last_warmup <- warmup_lines[length(warmup_lines)]
    eps_match <- regmatches(last_warmup, regexpr("epsilon=[0-9.e+-]+", last_warmup))
    if (length(eps_match) > 0) {
      final_eps <- as.numeric(sub("epsilon=", "", eps_match))
    }
  }

  # Compute approximate metrics:
  # N_SAMPLE = 250 (500 iter - 250 warmup)
  # Total leapfrog steps ≈ N_SAMPLE * 2^mean_td (for NUTS)
  # Per-step time = total_time / total_leapfrog_steps
  n_sample <- 250

  # Report what we know
  cat(sprintf("  Row %d: %.1fs, eps=%.4f\n", row, time_s,
              ifelse(is.na(final_eps), 0, final_eps)))

  results <- rbind(results, data.frame(
    row = row,
    model = paste0("row_", row),
    time_s = time_s,
    mean_td = NA,  # Will be computed from actual fit diagnostics
    epsilon = ifelse(is.na(final_eps), NA, final_eps),
    mean_lf = NA,
    per_step_ms = NA,
    total_lf = NA,
    stringsAsFactors = FALSE
  ))
}

cat("\n=== Summary ===\n")
print(results)

# Save results
saveRDS(results, "benchmarks/results_diag_slow.rds")
cat("\nResults saved to benchmarks/results_diag_slow.rds\n")
