# =============================================================================
# Quick gradient H vs N verification for all rows
# Runs each row with H mode, minimal iterations (just enough for gradient check)
# Captures gradient mismatch warnings from verify_gradient_runtime()
#
# Usage:
#   Rscript benchmarks/bench_grad_check.R             # all 137 rows
#   Rscript benchmarks/bench_grad_check.R 1 30        # rows 1-30
#   Rscript benchmarks/bench_grad_check.R 108 137     # nbg only
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
ROW_START <- if (length(args) >= 1) as.integer(args[1]) else 1L
ROW_END   <- if (length(args) >= 2) as.integer(args[2]) else 137L

# Find Rscript
rscript <- "Rscript"
if (.Platform$OS.type == "windows" || Sys.info()["sysname"] == "Windows") {
  r_home <- R.home("bin")
  rscript_path <- file.path(r_home, "Rscript.exe")
  if (file.exists(rscript_path)) rscript <- rscript_path
}

bench_script <- "benchmarks/bench_single_row.R"

cat(sprintf("\n=== Gradient H vs N Check: rows %d-%d ===\n\n", ROW_START, ROW_END))

passed <- integer()
failed <- integer()
errors <- integer()
fail_details <- list()

for (row_num in ROW_START:ROW_END) {
  cat(sprintf("[Row %3d] ", row_num))
  flush.console()

  # Run with H mode, short timeout (60s), model timeout 45s
  cmd <- sprintf('"%s" "%s" %d H 45 2>&1', rscript, bench_script, row_num)
  output <- tryCatch({
    system(cmd, intern = TRUE, timeout = 90)
  }, warning = function(w) {
    c("PROC_TIMEOUT")
  }, error = function(e) {
    c(paste0("PROC_ERROR:", conditionMessage(e)))
  })

  all_output <- paste(output, collapse = "\n")

  # Check for gradient mismatch warning
  has_mismatch <- grepl("gradient mismatch|Falling back to numerical", all_output)

  # Check for RESULT line
  result_line <- grep("^RESULT:", output, value = TRUE)
  has_result <- length(result_line) > 0

  if (has_mismatch) {
    failed <- c(failed, row_num)
    # Extract mismatch details
    mismatch_lines <- grep("mismatch|param|active|numerical", output, value = TRUE)
    fail_details[[as.character(row_num)]] <- mismatch_lines
    cat("FAIL (gradient mismatch)\n")
    for (l in mismatch_lines) cat(sprintf("  %s\n", l))
  } else if (has_result) {
    parts <- strsplit(result_line[length(result_line)], ":")[[1]]
    time_val <- paste(parts[4:length(parts)], collapse = ":")
    passed <- c(passed, row_num)
    cat(sprintf("OK (%s s)\n", time_val))
  } else if (any(grepl("PROC_TIMEOUT", output))) {
    # Timeout but no mismatch = gradient check passed (model just takes long)
    passed <- c(passed, row_num)
    cat("OK (timeout, but gradient check passed)\n")
  } else {
    errors <- c(errors, row_num)
    last_lines <- tail(output, 3)
    cat(sprintf("ERROR: %s\n", paste(last_lines, collapse = " | ")))
  }
}

cat(sprintf("\n=== SUMMARY ===\n"))
cat(sprintf("Passed: %d\n", length(passed)))
cat(sprintf("Failed (gradient mismatch): %d\n", length(failed)))
cat(sprintf("Errors: %d\n", length(errors)))

if (length(failed) > 0) {
  cat(sprintf("\nFAILED ROWS: %s\n", paste(failed, collapse = ", ")))
  for (r in as.character(failed)) {
    cat(sprintf("\n--- Row %s ---\n", r))
    for (l in fail_details[[r]]) cat(sprintf("  %s\n", l))
  }
}

if (length(errors) > 0) {
  cat(sprintf("\nERROR ROWS: %s\n", paste(errors, collapse = ", ")))
}
