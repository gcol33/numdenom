#!/usr/bin/env Rscript
# Rerun base model benchmarks (base, +RE, +ICAR, +AR1, +ICAR+AR1 × 3 families)
library(numdenom)

cat("=== Base Model Benchmark Rerun ===\n")
cat(sprintf("Date: %s\n\n", Sys.time()))

# Stan reference times from MEMORY.md (2026-03-03 benchmarks)
STAN_REF <- list(
  "PG+base"      = 1.2, "PG+RE"       = 2.5, "PG+ICAR"      = 3.0,
  "PG+AR1"       = NA,  "PG+ICAR+AR1" = NA,
  "NB+base"      = 1.5, "NB+RE"       = 2.9, "NB+ICAR"      = 3.5,
  "NB+AR1"       = NA,  "NB+ICAR+AR1" = NA,
  "Bin+base"     = 9.4, "Bin+RE"      = NA,  "Bin+ICAR"     = NA,
  "Bin+AR1"      = NA,  "Bin+ICAR+AR1"= NA
)

MODELS <- list(
  list(label="PG+base",      row=1),  list(label="PG+RE",       row=2),
  list(label="PG+ICAR",      row=5),  list(label="PG+AR1",      row=13),
  list(label="PG+ICAR+AR1",  row=20),
  list(label="NB+base",      row=31), list(label="NB+RE",       row=32),
  list(label="NB+ICAR",      row=35), list(label="NB+AR1",      row=43),
  list(label="NB+ICAR+AR1",  row=50),
  list(label="Bin+base",     row=61), list(label="Bin+RE",      row=62),
  list(label="Bin+ICAR",     row=65), list(label="Bin+AR1",     row=73),
  list(label="Bin+ICAR+AR1", row=80)
)

RSCRIPT <- file.path(R.home("bin"), "Rscript")
results <- list()

for (m in MODELS) {
  cat(sprintf("--- %s (row %d) ... ", m$label, m$row))
  cmd <- sprintf('"%s" benchmarks/bench_single_row.R %d H 600', RSCRIPT, m$row)
  out <- tryCatch(
    system(cmd, intern = TRUE, timeout = 660),
    error = function(e) paste0("ERROR: ", e$message)
  )
  result_line <- grep("^RESULT:", out, value = TRUE)
  t <- NA_real_
  if (length(result_line) > 0) {
    parts <- strsplit(result_line[1], ":")[[1]]
    t <- suppressWarnings(as.numeric(parts[4]))
  }
  stan <- STAN_REF[[m$label]]
  if (!is.na(t)) {
    if (!is.na(stan)) {
      ratio <- t / stan
      faster <- sprintf("%.1fx", 1/ratio)
      verdict <- if (ratio <= 1.0) "WIN" else if (ratio <= 1.2) "~" else "LOSS"
    } else {
      faster <- "---"; verdict <- "---"
    }
    cat(sprintf("%.1fs %s %s\n", t, faster, verdict))
  } else {
    faster <- "---"; verdict <- "ERROR"
    cat("TIMEOUT/ERROR\n")
  }
  results[[length(results) + 1]] <- data.frame(
    model=m$label, nd_s=t, stan_s=ifelse(is.null(stan), NA, stan),
    verdict=verdict, stringsAsFactors=FALSE)
}

cat("\n\n========== SUMMARY ==========\n")
df <- do.call(rbind, results)
cat(sprintf("%-20s %8s %8s %s\n", "Model", "numdenom", "Stan", ""))
cat(paste0(rep("-", 45), collapse=""), "\n")
for (i in 1:nrow(df)) {
  r <- df[i,]
  stan_str <- if (is.na(r$stan_s)) "---" else sprintf("%.1f", r$stan_s)
  cat(sprintf("%-20s %7.1fs %7s %s\n", r$model, r$nd_s, stan_str, r$verdict))
}
