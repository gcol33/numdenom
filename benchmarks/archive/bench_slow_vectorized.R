# =============================================================================
# Benchmark: Vectorized slow models vs previous results
# Uses CORRECT bench_single_row.R row numbers (not gradient_methods.md rows)
# =============================================================================

# Correct bench_single_row.R row numbers for slow models:
# Category 1: HSGP - rows 8/68 (PG/Bin)
# Category 2: Temporal GP (GP_t) - rows 14/44/74 (PG/NB/Bin)
# Category 3: Slopes+ICAR - rows 25/55/87 (PG/NB/Bin)
# Category 4: ST Type IV - rows 29/59/91 (PG/NB/Bin)

SLOW_ROWS <- c(8, 68, 14, 44, 74, 25, 55, 87, 29, 59, 91)

LABELS <- c(
  "8"  = "PG+HSGP",
  "68" = "Bin+HSGP",
  "14" = "PG+GP_t",
  "44" = "NB+GP_t",
  "74" = "Bin+GP_t",
  "25" = "PG+slopes+ICAR",
  "55" = "NB+slopes+ICAR",
  "87" = "Bin+slopes+ICAR",
  "29" = "PG+ST_IV",
  "59" = "NB+ST_IV",
  "91" = "Bin+ST_IV"
)

# Previous (pre-vectorization) times from MEMORY.md
BEFORE <- c(
  "8"  = 19.0,
  "68" = 6.7,
  "14" = 40.6,
  "44" = 70.0,
  "74" = 9.7,
  "25" = 50.8,
  "55" = 70.0,
  "87" = 25.3,
  "29" = 190.8,
  "59" = 287.9,
  "91" = 80.7
)

STAN <- c(
  "8"  = 10.9,
  "68" = 2.9,
  "14" = 10.0,
  "44" = 21.3,
  "74" = 3.3,
  "25" = 15.4,
  "55" = 46.6,
  "87" = 14.5,
  "29" = 82.5,
  "59" = 133.5,
  "91" = 56.0
)

cat("=== Slow Model Vectorization Benchmark ===\n")
cat(sprintf("%-20s %8s %8s %8s %8s %8s\n",
            "Model", "Before", "After", "Stan", "Speedup", "vs Stan"))
cat(paste0(rep("-", 72), collapse = ""), "\n")

for (row in SLOW_ROWS) {
  label <- LABELS[as.character(row)]

  # Run bench_single_row.R as subprocess
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
    # Print any warnings or errors from output
    warn_lines <- grep("WARNING|ERROR|mismatch|Falling back", out, value = TRUE)
    cat(sprintf("%-20s %8s %8s %8s %8s %8s  TIMEOUT/ERROR\n",
                label, sprintf("%.1f", BEFORE[as.character(row)]),
                "---", sprintf("%.1f", STAN[as.character(row)]), "---", "---"))
    if (length(warn_lines) > 0) {
      for (w in warn_lines) cat(sprintf("  >> %s\n", w))
    }
    next
  }

  parts <- strsplit(result_line[1], ":")[[1]]
  time_s <- as.numeric(parts[4])
  if (is.na(time_s)) {
    cat(sprintf("%-20s PARSE ERROR: %s\n", label, result_line[1]))
    next
  }

  before <- BEFORE[as.character(row)]
  stan <- STAN[as.character(row)]
  speedup <- before / time_s
  vs_stan <- time_s / stan

  # Print warnings if any
  warn_lines <- grep("WARNING|mismatch|Falling back", out, value = TRUE)
  warn_flag <- if (length(warn_lines) > 0) " *WARN*" else ""

  cat(sprintf("%-20s %8.1f %8.1f %8.1f %7.1fx %7.1fx%s\n",
              label, before, time_s, stan, speedup, vs_stan, warn_flag))
  if (length(warn_lines) > 0) {
    for (w in warn_lines) cat(sprintf("  >> %s\n", w))
  }
}

cat("\nDone.\n")
