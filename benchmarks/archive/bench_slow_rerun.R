#!/usr/bin/env Rscript
# Rerun all 18 slow model benchmarks (6 categories × 3 families)
# Compares against known Stan reference times from custom joint models
library(numdenom)

cat("=== Slow Model Benchmark Rerun ===\n")
cat(sprintf("numdenom version: %s\n", packageVersion("numdenom")))
cat(sprintf("Date: %s\n\n", Sys.time()))

# Stan reference times (from 2026-03-04 benchmarks, custom joint models)
STAN_REF <- list(
  "PG+BYM2"         = 15.5, "NB+BYM2"         = 46.8, "Bin+BYM2"        = 13.5,
  "PG+BYM2+RW1"     = 48.5, "NB+BYM2+RW1"     = 79.5, "Bin+BYM2+RW1"    = 38.4,
  "PG+HSGP"         = 10.9, "NB+HSGP"         = 2.9,  "Bin+HSGP"        = 2.9,
  "PG+slopes+ICAR"  = 15.4, "NB+slopes+ICAR"  = 46.6, "Bin+slopes+ICAR" = 14.5,
  "PG+GP_t"         = 10.0, "NB+GP_t"         = 21.3, "Bin+GP_t"        = 3.3,
  "PG+ST_IV"        = 82.5, "NB+ST_IV"        = 133.5,"Bin+ST_IV"       = 56.0
)

MODELS <- list(
  list(label="PG+BYM2",         row=6),
  list(label="NB+BYM2",         row=36),
  list(label="Bin+BYM2",        row=66),
  list(label="PG+BYM2+RW1",     row=19),
  list(label="NB+BYM2+RW1",     row=49),
  list(label="Bin+BYM2+RW1",    row=81),
  list(label="PG+HSGP",         row=8),
  list(label="NB+HSGP",         row=38),
  list(label="Bin+HSGP",        row=68),
  list(label="PG+slopes+ICAR",  row=25),
  list(label="NB+slopes+ICAR",  row=55),
  list(label="Bin+slopes+ICAR", row=87),
  list(label="PG+GP_t",         row=14),
  list(label="NB+GP_t",         row=44),
  list(label="Bin+GP_t",        row=74),
  list(label="PG+ST_IV",        row=29),
  list(label="NB+ST_IV",        row=59),
  list(label="Bin+ST_IV",       row=91)
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
    ratio <- t / stan
    verdict <- if (ratio <= 1.0) "WIN" else if (ratio <= 1.2) "~" else "LOSS"
    cat(sprintf("%.1fs (Stan %.1fs) %.1fx %s\n", t, stan, 1/ratio, verdict))
  } else {
    ratio <- NA; verdict <- "ERROR"
    cat("TIMEOUT/ERROR\n")
  }
  results[[length(results) + 1]] <- data.frame(
    model=m$label, nd_s=t, stan_s=stan, ratio=ratio, verdict=verdict,
    stringsAsFactors=FALSE)
}

cat("\n\n========== SUMMARY ==========\n")
df <- do.call(rbind, results)
cat(sprintf("%-20s %8s %8s %8s %s\n", "Model", "numdenom", "Stan", "Faster", ""))
cat(paste0(rep("-", 55), collapse=""), "\n")
for (i in 1:nrow(df)) {
  r <- df[i,]
  faster <- if (!is.na(r$ratio)) sprintf("%.1fx", 1/r$ratio) else "---"
  cat(sprintf("%-20s %7.1fs %7.1fs %7s %s\n", r$model, r$nd_s, r$stan_s, faster, r$verdict))
}
w <- sum(df$verdict=="WIN",na.rm=TRUE); p <- sum(df$verdict=="~",na.rm=TRUE)
l <- sum(df$verdict=="LOSS",na.rm=TRUE)
cat(sprintf("\n%d WIN, %d PARITY, %d LOSS out of %d\n", w, p, l, nrow(df)))
