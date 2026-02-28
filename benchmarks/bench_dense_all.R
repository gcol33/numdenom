# =============================================================================
# Dense mass matrix benchmark: all 107 rows
# Runs each row via subprocess using bench_dense_row.R
# Saves results to benchmarks/results_dense_all.rds
#
# Usage:
#   Rscript benchmarks/bench_dense_all.R
#   Rscript benchmarks/bench_dense_all.R --resume   # skip completed rows
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
resume <- "--resume" %in% args

results_file <- "benchmarks/results_dense_all.rds"
TIMEOUT <- 660  # 11 min subprocess timeout (600s model + 60s overhead)
ALL_ROWS <- 1:107

# Load existing results if resuming
if (resume && file.exists(results_file)) {
  results <- readRDS(results_file)
  cat(sprintf("Resuming: %d/%d rows already done\n",
              sum(!is.na(results$dense_s) | results$dense_status != "pending"),
              nrow(results)))
} else {
  results <- data.frame(
    row = ALL_ROWS,
    dense_s = NA_real_,
    dense_status = "pending",
    stringsAsFactors = FALSE
  )
}

# Existing diagonal timings for comparison at the end
diag_timings <- c(
  1.2, 2.3, 145.1, 2.9, 3.1, 158.6, 600,        # rows 1-7
  90.8, 1.7, 2.7, 3.5, 2.9, 6.4, 84.6,           # rows 8-14
  66.5, 1.8, 47.2, 6.3, 156.9, 97.0, 600,         # rows 15-21
  92.2, 298.4, 4.5, 158.4, 600, 68.0, 90.3,       # rows 22-28
  140.9, 10.4, 1.3, 2.2, 135.6, 2.4, 3.1,         # rows 29-35
  161.1, 600, 210.4, 600, 4.2, 9.0, 3.8,           # rows 36-42
  3.5, 170.2, 107.3, 3.4, 155.0, 9.1, 158.6,      # rows 43-49
  119.8, 600, 207.1, 600, 2.9, 160.2, 600,         # rows 50-56
  183.2, 132.7, 184.4, 18.4, 0.5, 3.2, 140.3,     # rows 57-63
  2.9, 3.8, 139.0, 600, 50.9, 116.1, 2.9,         # rows 64-70
  3.5, 3.3, 82.9, 24.0, 24.1, 2.2, 1.9,           # rows 71-77
  3.6, 2.5, 3.4, 138.4, 5.2, 600, 50.4,           # rows 78-84
  531.8, 3.0, 136.5, 600, 24.4, 44.1, 93.9,       # rows 85-91
  3.0, 1.1, 1.9, 2.9, 3.1, 6.1, 0.8,              # rows 92-98
  1.9, 2.1, 6.7, 15.0, 1.0, 3.1, 6.6,             # rows 99-105
  10.5, 9.1                                         # rows 106-107
)

cat("=== Dense Mass Matrix Benchmark: All 107 Rows ===\n")
cat(sprintf("Started: %s\n\n", Sys.time()))

n_done <- 0
n_total <- length(ALL_ROWS)

for (row in ALL_ROWS) {
  # Skip if already done
  if (resume && results$dense_status[row] %in% c("done", "timeout", "error")) {
    n_done <- n_done + 1
    next
  }

  n_done <- n_done + 1
  diag_t <- diag_timings[row]
  diag_str <- if (diag_t >= 600) ">600" else sprintf("%.1f", diag_t)

  cat(sprintf("[%3d/107] Row %3d (diag=%ss) ... ", n_done, row, diag_str))
  flush.console()

  cmd <- sprintf("Rscript benchmarks/bench_dense_row.R %d 2>&1", row)

  tryCatch({
    output <- system(cmd, intern = TRUE, timeout = TIMEOUT)
    result_line <- grep("^RESULT:", output, value = TRUE)

    if (length(result_line) > 0) {
      parts <- strsplit(result_line[1], ":")[[1]]
      if (length(parts) >= 3) {
        val <- parts[3]
        if (val == "TIMEOUT") {
          results$dense_s[row] <- NA
          results$dense_status[row] <- "timeout"
          cat("TIMEOUT\n")
        } else if (grepl("^ERROR", val)) {
          results$dense_s[row] <- NA
          results$dense_status[row] <- "error"
          err_msg <- paste(parts[3:length(parts)], collapse = ":")
          cat(sprintf("ERROR: %s\n", substr(err_msg, 1, 80)))
        } else {
          t <- as.numeric(val)
          results$dense_s[row] <- t
          results$dense_status[row] <- "done"
          # Compare to diagonal
          speedup <- diag_t / t
          if (diag_t >= 600 && t < 600) {
            cat(sprintf("%.1fs (was >600s diag -> NOW FITS!)\n", t))
          } else if (speedup > 1.5) {
            cat(sprintf("%.1fs (%.1fx faster than diag %.1fs)\n", t, speedup, diag_t))
          } else if (speedup < 0.67) {
            cat(sprintf("%.1fs (%.1fx slower than diag %.1fs)\n", t, 1/speedup, diag_t))
          } else {
            cat(sprintf("%.1fs (similar to diag %.1fs)\n", t, diag_t))
          }
        }
      }
    } else {
      # No RESULT line — check for crash
      results$dense_s[row] <- NA
      results$dense_status[row] <- "error"
      last_lines <- tail(output, 3)
      cat(sprintf("NO RESULT: %s\n", paste(last_lines, collapse = " | ")))
    }
  }, error = function(e) {
    results$dense_s[row] <<- NA
    results$dense_status[row] <<- "timeout"
    cat("SUBPROCESS TIMEOUT\n")
  })

  # Save after each row
  saveRDS(results, results_file)
}

# === Summary ===
cat("\n=== SUMMARY ===\n")
cat(sprintf("Completed: %s\n\n", Sys.time()))

done <- results$dense_status == "done"
timeout <- results$dense_status == "timeout"
errored <- results$dense_status == "error"

cat(sprintf("Done:    %d\n", sum(done)))
cat(sprintf("Timeout: %d\n", sum(timeout)))
cat(sprintf("Error:   %d\n", sum(errored)))

if (any(done)) {
  cat("\n--- Comparison: Dense vs Diagonal ---\n")
  cat(sprintf("%-5s %-8s %-8s %-10s\n", "Row", "Dense", "Diag", "Speedup"))
  cat(paste(rep("-", 35), collapse = ""), "\n")

  for (row in ALL_ROWS[done]) {
    dt <- results$dense_s[row]
    diag_t <- diag_timings[row]
    if (diag_t >= 600) {
      cat(sprintf("%-5d %-8.1f >600     FIXED!\n", row, dt))
    } else {
      sp <- diag_t / dt
      marker <- if (sp > 2) " ***" else if (sp > 1.5) " **" else if (sp < 0.5) " (slower)" else ""
      cat(sprintf("%-5d %-8.1f %-8.1f %-6.1fx%s\n", row, dt, diag_t, sp, marker))
    }
  }

  # Summary stats for rows that were slow (>50s) with diagonal
  slow_rows <- ALL_ROWS[done & diag_timings[ALL_ROWS] > 50]
  if (length(slow_rows) > 0) {
    cat("\n--- Slow rows (diag >50s) with dense mass matrix ---\n")
    dense_times <- results$dense_s[slow_rows]
    diag_times <- diag_timings[slow_rows]
    speedups <- diag_times / dense_times
    cat(sprintf("  Median speedup: %.1fx\n", median(speedups, na.rm = TRUE)))
    cat(sprintf("  Mean speedup:   %.1fx\n", mean(speedups, na.rm = TRUE)))
    cat(sprintf("  Max speedup:    %.1fx (row %d)\n",
                max(speedups, na.rm = TRUE), slow_rows[which.max(speedups)]))
    cat(sprintf("  Rows faster:    %d/%d\n",
                sum(speedups > 1, na.rm = TRUE), length(slow_rows)))
    cat(sprintf("  Rows slower:    %d/%d\n",
                sum(speedups < 1, na.rm = TRUE), length(slow_rows)))
  }

  # Timeout rows that previously timed out with diagonal
  was_timeout <- ALL_ROWS[done & diag_timings[ALL_ROWS] >= 600]
  if (length(was_timeout) > 0) {
    cat("\n--- Previously-timeout rows now completing ---\n")
    for (row in was_timeout) {
      cat(sprintf("  Row %d: %.1fs (was >600s)\n", row, results$dense_s[row]))
    }
  }
}

cat("\nResults saved to:", results_file, "\n")
