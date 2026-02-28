# =============================================================================
# NUTS Benchmark Coordinator - runs each row in a separate R subprocess
# =============================================================================
# Each row runs in isolation so crashes don't propagate.
# Usage:
#   Rscript benchmarks/bench_nuts_all.R              # all 107 rows
#   Rscript benchmarks/bench_nuts_all.R 1 30         # rows 1-30
#   Rscript benchmarks/bench_nuts_all.R --resume     # skip completed rows
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
ROW_START <- 1
ROW_END   <- 107
RESUME    <- FALSE
TIMEOUT   <- 660  # 11 min per subprocess (10 min model + 1 min overhead)

i <- 1
while (i <= length(args)) {
  if (args[i] == "--resume") {
    RESUME <- TRUE
    i <- i + 1
  } else if (grepl("^[0-9]+$", args[i])) {
    ROW_START <- as.integer(args[i])
    if (i + 1 <= length(args) && grepl("^[0-9]+$", args[i + 1])) {
      ROW_END <- as.integer(args[i + 1])
      i <- i + 2
    } else {
      ROW_END <- ROW_START
      i <- i + 1
    }
  } else {
    i <- i + 1
  }
}

# Row descriptions for display
row_descs <- c(
  "pg base", "pg+RE", "pg+slopes", "pg+crossed",
  "pg+ICAR", "pg+BYM2", "pg+GP", "pg+HSGP", "pg+MSGP", "pg+PCAR",
  "pg+RW1", "pg+RW2", "pg+AR1", "pg+GP_t", "pg+MS_t",
  "pg+ZI", "pg+hurdle",
  "pg+ICAR+RW1", "pg+BYM2+RW1", "pg+ICAR+AR1",
  "pg+GP+RW1", "pg+HSGP+RW1", "pg+MSGP+RW1",
  "pg+ICAR+ZI", "pg+slopes+ICAR",
  "pg+SVC", "pg+TVC",
  "pg+ST-I", "pg+ST-IV", "pg+latent",
  # nb (31-60)
  "nb base", "nb+RE", "nb+slopes", "nb+crossed",
  "nb+ICAR", "nb+BYM2", "nb+GP", "nb+HSGP", "nb+MSGP", "nb+PCAR",
  "nb+RW1", "nb+RW2", "nb+AR1", "nb+GP_t", "nb+MS_t",
  "nb+ZI", "nb+hurdle",
  "nb+ICAR+RW1", "nb+BYM2+RW1", "nb+ICAR+AR1",
  "nb+GP+RW1", "nb+HSGP+RW1", "nb+MSGP+RW1",
  "nb+ICAR+ZI", "nb+slopes+ICAR",
  "nb+SVC", "nb+TVC",
  "nb+ST-I", "nb+ST-IV", "nb+latent",
  # bin (61-92)
  "bin base", "bin+RE", "bin+slopes", "bin+crossed",
  "bin+ICAR", "bin+BYM2", "bin+GP", "bin+HSGP", "bin+MSGP", "bin+PCAR",
  "bin+RW1", "bin+RW2", "bin+AR1", "bin+GP_t", "bin+MS_t",
  "bin+ZI", "bin+hurdle", "bin+OI", "bin+ZOIB",
  "bin+ICAR+RW1", "bin+BYM2+RW1", "bin+ICAR+AR1",
  "bin+GP+RW1", "bin+HSGP+RW1", "bin+MSGP+RW1",
  "bin+ICAR+ZI", "bin+slopes+ICAR",
  "bin+SVC", "bin+TVC",
  "bin+ST-I", "bin+ST-IV", "bin+latent",
  # gg (93-97)
  "gg base", "gg+RE", "gg+ICAR", "gg+RW1", "gg+ICAR+RW1",
  # ln (98-102)
  "ln base", "ln+RE", "ln+ICAR", "ln+RW1", "ln+ICAR+RW1",
  # bb (103-107)
  "bb base", "bb+RE", "bb+ICAR", "bb+RW1", "bb+ICAR+RW1"
)

# Results file
results_file <- "benchmarks/results_nuts_all.rds"
if (RESUME && file.exists(results_file)) {
  results <- readRDS(results_file)
  cat(sprintf("Resuming: %d rows already completed\n", nrow(results)))
} else {
  results <- data.frame(
    row = integer(), desc = character(), time_s = character(),
    stringsAsFactors = FALSE
  )
}

completed_rows <- results$row

cat(sprintf("\n=== NUTS Benchmark: rows %d-%d (H mode, subprocess isolation) ===\n",
            ROW_START, ROW_END))
cat(sprintf("Standard: N=500, iter=500, warmup=250, chains=1\n"))
cat(sprintf("GP: N=80 | Latent: N=50 | Timeout: 600s per model\n\n"))

start_time <- Sys.time()
n_done <- 0
n_total <- ROW_END - ROW_START + 1

for (row_num in ROW_START:ROW_END) {
  n_done <- n_done + 1

  if (row_num %in% completed_rows) {
    existing <- results$time_s[results$row == row_num]
    cat(sprintf("[%3d/%3d] Row %3d %-25s : %s (cached)\n",
                n_done, n_total, row_num, row_descs[row_num], existing))
    next
  }

  desc <- row_descs[row_num]
  cat(sprintf("[%3d/%3d] Row %3d %-25s : ", n_done, n_total, row_num, desc))
  flush.console()

  # Run in subprocess
  cmd <- sprintf('Rscript benchmarks/bench_single_row.R %d 2>&1', row_num)
  output <- tryCatch(
    system(cmd, intern = TRUE, timeout = TIMEOUT),
    error = function(e) paste0("CRASH:", conditionMessage(e)),
    warning = function(w) {
      # system() returns warning on non-zero exit
      paste0("CRASH:", conditionMessage(w))
    }
  )

  # Parse result
  result_line <- grep("^RESULT:", output, value = TRUE)
  if (length(result_line) > 0) {
    parts <- strsplit(result_line[1], ":")[[1]]
    time_str <- parts[3]
    if (length(parts) > 3) {
      # ERROR with message
      time_str <- paste(parts[3:length(parts)], collapse = ":")
    }
  } else {
    # Process crashed without producing RESULT line
    crash_lines <- tail(output, 3)
    crash_msg <- paste(crash_lines, collapse = " | ")
    if (nchar(crash_msg) > 80) crash_msg <- substr(crash_msg, 1, 80)
    time_str <- paste0("CRASH:", crash_msg)
  }

  cat(sprintf("%s\n", time_str))

  # Save result
  new_row <- data.frame(
    row = row_num, desc = desc, time_s = time_str,
    stringsAsFactors = FALSE
  )
  results <- rbind(results, new_row)
  saveRDS(results, results_file)
}

# =============================================================================
# Summary
# =============================================================================

elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
cat(sprintf("\n%s\nDone in %.1f minutes\n", strrep("=", 70), elapsed))

# Count status
n_ok      <- sum(grepl("^[0-9]", results$time_s))
n_timeout <- sum(grepl("TIMEOUT", results$time_s))
n_error   <- sum(grepl("ERROR|CRASH", results$time_s))
cat(sprintf("OK: %d | TIMEOUT: %d | ERROR/CRASH: %d | Total: %d\n",
            n_ok, n_timeout, n_error, nrow(results)))

# Print numeric results sorted
ok_rows <- results[grepl("^[0-9]", results$time_s), ]
if (nrow(ok_rows) > 0) {
  ok_rows$time_num <- as.numeric(ok_rows$time_s)
  ok_rows <- ok_rows[order(ok_rows$row), ]
  cat(sprintf("\n%-5s %-25s %10s\n", "Row", "Model", "H(s)"))
  cat(strrep("-", 45), "\n")
  for (i in seq_len(nrow(ok_rows))) {
    cat(sprintf("%-5d %-25s %10.1f\n",
                ok_rows$row[i], ok_rows$desc[i], ok_rows$time_num[i]))
  }
}

cat(sprintf("\nResults saved to %s\n", results_file))
