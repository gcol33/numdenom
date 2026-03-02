# =============================================================================
# N vs H Gradient Benchmark for ALL 137 Model Configurations
# =============================================================================
# Runs each row with both N (numerical) and H (hand-coded) gradient modes
# using subprocess isolation via bench_single_row.R.
#
# Usage:
#   Rscript benchmarks/bench_nh_all.R                   # all 137 rows
#   Rscript benchmarks/bench_nh_all.R 1 30              # rows 1-30
#   Rscript benchmarks/bench_nh_all.R 108 137            # negbin_gamma only
#   Rscript benchmarks/bench_nh_all.R --resume           # skip completed rows
#   Rscript benchmarks/bench_nh_all.R --modes H          # H only
#   Rscript benchmarks/bench_nh_all.R --modes N          # N only
#
# Results saved incrementally to benchmarks/results_nh_all.rds
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)

ROW_START <- 1L
ROW_END <- 137L
RESUME <- FALSE
MODES <- c("H", "N")
H_TIMEOUT <- 660       # 11 min for H mode (10 min model + 1 min overhead)
N_TIMEOUT <- 200       # 3.3 min for N mode (3 min model + overhead)
N_MODEL_TIMEOUT <- 180 # 3 min internal timeout for N mode in bench_single_row.R

i <- 1
while (i <= length(args)) {
  if (args[i] == "--resume") {
    RESUME <- TRUE
    i <- i + 1
  } else if (args[i] == "--modes") {
    MODES <- strsplit(args[i + 1], ",")[[1]]
    i <- i + 2
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

# Row descriptions (for display)
ROW_DESCS <- c(
  # PG (1-30)
  "pg", "pg+RE", "pg+slopes", "pg+crossed",
  "pg+ICAR", "pg+BYM2", "pg+GP", "pg+HSGP", "pg+MSGP", "pg+pCAR",
  "pg+RW1", "pg+RW2", "pg+AR1", "pg+GP_t", "pg+MS_t",
  "pg+ZI", "pg+hurdle",
  "pg+ICAR+RW1", "pg+BYM2+RW1", "pg+ICAR+AR1",
  "pg+GP+RW1", "pg+HSGP+RW1", "pg+MSGP+RW1",
  "pg+ICAR+ZI", "pg+slopes+ICAR", "pg+SVC", "pg+TVC",
  "pg+ST-I", "pg+ST-IV", "pg+latent",
  # NB (31-60)
  "nb", "nb+RE", "nb+slopes", "nb+crossed",
  "nb+ICAR", "nb+BYM2", "nb+GP", "nb+HSGP", "nb+MSGP", "nb+pCAR",
  "nb+RW1", "nb+RW2", "nb+AR1", "nb+GP_t", "nb+MS_t",
  "nb+ZI", "nb+hurdle",
  "nb+ICAR+RW1", "nb+BYM2+RW1", "nb+ICAR+AR1",
  "nb+GP+RW1", "nb+HSGP+RW1", "nb+MSGP+RW1",
  "nb+ICAR+ZI", "nb+slopes+ICAR", "nb+SVC", "nb+TVC",
  "nb+ST-I", "nb+ST-IV", "nb+latent",
  # Bin (61-92)
  "bin", "bin+RE", "bin+slopes", "bin+crossed",
  "bin+ICAR", "bin+BYM2", "bin+GP", "bin+HSGP", "bin+MSGP", "bin+pCAR",
  "bin+RW1", "bin+RW2", "bin+AR1", "bin+GP_t", "bin+MS_t",
  "bin+ZI", "bin+hurdle", "bin+OI", "bin+ZOIB",
  "bin+ICAR+RW1", "bin+BYM2+RW1", "bin+ICAR+AR1",
  "bin+GP+RW1", "bin+HSGP+RW1", "bin+MSGP+RW1",
  "bin+ICAR+ZI", "bin+slopes+ICAR", "bin+SVC", "bin+TVC",
  "bin+ST-I", "bin+ST-IV", "bin+latent",
  # GG (93-97)
  "gg", "gg+RE", "gg+ICAR", "gg+RW1", "gg+ICAR+RW1",
  # LN (98-102)
  "ln", "ln+RE", "ln+ICAR", "ln+RW1", "ln+ICAR+RW1",
  # BB (103-107)
  "bb", "bb+RE", "bb+ICAR", "bb+RW1", "bb+ICAR+RW1",
  # NBG (108-137)
  "nbg", "nbg+RE", "nbg+slopes", "nbg+crossed",
  "nbg+ICAR", "nbg+BYM2", "nbg+GP", "nbg+HSGP", "nbg+MSGP", "nbg+pCAR",
  "nbg+RW1", "nbg+RW2", "nbg+AR1", "nbg+GP_t", "nbg+MS_t",
  "nbg+ZI", "nbg+hurdle",
  "nbg+ICAR+RW1", "nbg+BYM2+RW1", "nbg+ICAR+AR1",
  "nbg+GP+RW1", "nbg+HSGP+RW1", "nbg+MSGP+RW1",
  "nbg+ICAR+ZI", "nbg+slopes+ICAR", "nbg+SVC", "nbg+TVC",
  "nbg+ST-I", "nbg+ST-IV", "nbg+latent"
)

# Load existing results
results_file <- "benchmarks/results_nh_all.rds"
if (file.exists(results_file)) {
  all_results <- readRDS(results_file)
  cat(sprintf("Loaded %d existing results from %s\n", nrow(all_results), results_file))
} else {
  all_results <- data.frame(
    row    = integer(),
    mode   = character(),
    desc   = character(),
    time_s = character(),
    stringsAsFactors = FALSE
  )
}

# Find Rscript
rscript <- "Rscript"
if (.Platform$OS.type == "windows" || Sys.info()["sysname"] == "Windows") {
  r_home <- R.home("bin")
  rscript_path <- file.path(r_home, "Rscript.exe")
  if (file.exists(rscript_path)) rscript <- rscript_path
}

bench_script <- "benchmarks/bench_single_row.R"
if (!file.exists(bench_script)) {
  # Try from benchmarks/ directory
  bench_script <- "bench_single_row.R"
  if (!file.exists(bench_script)) {
    stop("Cannot find bench_single_row.R")
  }
}

# =============================================================================
# Main loop
# =============================================================================

rows_to_run <- ROW_START:ROW_END
total_jobs <- length(rows_to_run) * length(MODES)

cat(sprintf("\n=== N vs H Benchmark: rows %d-%d, modes: %s ===\n",
            ROW_START, ROW_END, paste(MODES, collapse = ", ")))
cat(sprintf("Timeouts: H=%ds, N=%ds (shorter to avoid wasting time on O(p*N) timeouts)\n\n",
            H_TIMEOUT, N_TIMEOUT))

completed <- 0
skipped <- 0
start_time <- Sys.time()

for (row_num in rows_to_run) {
  desc <- if (row_num <= length(ROW_DESCS)) ROW_DESCS[row_num] else paste0("row", row_num)

  for (mode in MODES) {
    completed <- completed + 1

    # Skip if already completed (resume mode)
    if (RESUME) {
      existing <- all_results[all_results$row == row_num & all_results$mode == mode, ]
      if (nrow(existing) > 0) {
        skipped <- skipped + 1
        next
      }
    }

    cat(sprintf("[%d/%d] Row %3d %-25s %s: ",
                completed, total_jobs, row_num, desc, mode))
    flush.console()

    # Run subprocess (shorter timeout for N mode)
    sub_timeout <- if (mode == "N") N_TIMEOUT else H_TIMEOUT
    model_timeout <- if (mode == "N") N_MODEL_TIMEOUT else 600
    cmd <- sprintf('"%s" "%s" %d %s %d', rscript, bench_script, row_num, mode, model_timeout)
    result_text <- tryCatch({
      output <- system(cmd, intern = TRUE, timeout = sub_timeout)
      # Find RESULT line
      result_lines <- grep("^RESULT:", output, value = TRUE)
      if (length(result_lines) > 0) {
        result_lines[length(result_lines)]  # last RESULT line
      } else {
        # No RESULT line = crash
        last_lines <- tail(output, 3)
        paste0("CRASH:", paste(last_lines, collapse = " | "))
      }
    }, error = function(e) {
      paste0("CRASH:", conditionMessage(e))
    }, warning = function(w) {
      # system() with timeout gives a warning on timeout
      "TIMEOUT"
    })

    # Parse result
    # Format: RESULT:<row>:<mode>:<value>
    if (grepl("^RESULT:", result_text)) {
      parts <- strsplit(result_text, ":")[[1]]
      # parts[1]="RESULT", parts[2]=row, parts[3]=mode, parts[4+]=value
      time_val <- paste(parts[4:length(parts)], collapse = ":")
    } else if (grepl("^TIMEOUT", result_text)) {
      time_val <- "TIMEOUT"
    } else {
      time_val <- result_text
    }

    cat(sprintf("%s\n", time_val))

    # Save result
    new_row <- data.frame(
      row    = row_num,
      mode   = mode,
      desc   = desc,
      time_s = time_val,
      stringsAsFactors = FALSE
    )
    all_results <- rbind(all_results, new_row)

    # Incremental save after each result
    saveRDS(all_results, results_file)
  }
}

# Final save
saveRDS(all_results, results_file)

# =============================================================================
# Summary table
# =============================================================================

cat("\n\n", strrep("=", 90), "\n")
cat("N vs H BENCHMARK SUMMARY\n")
cat(strrep("=", 90), "\n\n")

if (skipped > 0) cat(sprintf("Skipped %d cached results (--resume)\n\n", skipped))

# Filter to requested range
res <- all_results[all_results$row >= ROW_START & all_results$row <= ROW_END, ]

if (nrow(res) > 0) {
  benchmarked_rows <- sort(unique(res$row))

  cat(sprintf("%-5s %-25s %12s %12s %10s\n",
              "Row", "Model", "H(s)", "N(s)", "N/H ratio"))
  cat(paste(rep("-", 70), collapse = ""), "\n")

  for (r in benchmarked_rows) {
    desc <- res$desc[res$row == r][1]
    h_val <- res$time_s[res$row == r & res$mode == "H"]
    n_val <- res$time_s[res$row == r & res$mode == "N"]

    h_str <- if (length(h_val) > 0) h_val[1] else "-"
    n_str <- if (length(n_val) > 0) n_val[1] else "-"

    # Compute ratio if both are numeric
    ratio_str <- ""
    h_num <- suppressWarnings(as.numeric(h_str))
    n_num <- suppressWarnings(as.numeric(n_str))
    if (!is.na(h_num) && !is.na(n_num) && h_num > 0) {
      ratio_str <- sprintf("%.1fx", n_num / h_num)
    }

    cat(sprintf("%-5d %-25s %12s %12s %10s\n",
                r, substr(desc, 1, 25), h_str, n_str, ratio_str))
  }
}

elapsed_min <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
cat(sprintf("\nTotal time: %.1f minutes\n", elapsed_min))

# Count outcomes
ok_h <- sum(grepl("^[0-9]", res$time_s[res$mode == "H"]))
ok_n <- sum(grepl("^[0-9]", res$time_s[res$mode == "N"]))
to_h <- sum(res$time_s[res$mode == "H"] == "TIMEOUT")
to_n <- sum(res$time_s[res$mode == "N"] == "TIMEOUT")
err_h <- sum(grepl("^ERROR|^CRASH", res$time_s[res$mode == "H"]))
err_n <- sum(grepl("^ERROR|^CRASH", res$time_s[res$mode == "N"]))

cat(sprintf("\nH mode: %d OK, %d TIMEOUT, %d ERROR\n", ok_h, to_h, err_h))
cat(sprintf("N mode: %d OK, %d TIMEOUT, %d ERROR\n", ok_n, to_n, err_n))
cat(sprintf("Results saved to %s\n", results_file))

# =============================================================================
# Markdown for gradient_methods.md
# =============================================================================

cat("\n\n--- H(s) and N(s) values for gradient_methods.md ---\n\n")
if (nrow(res) > 0) {
  for (r in benchmarked_rows) {
    h_val <- res$time_s[res$row == r & res$mode == "H"]
    n_val <- res$time_s[res$row == r & res$mode == "N"]

    h_str <- if (length(h_val) > 0 && !grepl("ERROR|CRASH", h_val[1])) {
      if (h_val[1] == "TIMEOUT") ">600" else h_val[1]
    } else ""

    n_str <- if (length(n_val) > 0 && !grepl("ERROR|CRASH", n_val[1])) {
      if (n_val[1] == "TIMEOUT") ">600" else n_val[1]
    } else ""

    cat(sprintf("Row %3d: H=%s  N=%s\n", r, h_str, n_str))
  }
}
