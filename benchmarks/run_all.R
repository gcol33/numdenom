# Benchmark all model configurations vs Stan
# Validates speedups reported in gradient_methods.md
#
# Usage: Rscript benchmarks/run_all.R [--quick]
#   --quick: Run with fewer iterations for fast validation

library(numdenom)

args <- commandArgs(trailingOnly = TRUE)
quick_mode <- "--quick" %in% args

# Settings
if (quick_mode) {
  n_iter <- 100
  n_warmup <- 50
  cat("Running in QUICK mode (100 iterations)\n\n")
} else {
  n_iter <- 400
  n_warmup <- 200
  cat("Running in FULL mode (400 iterations)\n\n")
}

# Source helper functions
source("benchmarks/helpers.R")

# Results storage
results <- data.frame(

row = integer(),
  family = character(),
  re = character(),
  spatial = character(),
  temporal = character(),
  zi = character(),
  grad = character(),
  nd_time = numeric(),
  stan_time = numeric(),
  speedup = numeric(),
  stringsAsFactors = FALSE
)

cat("=== numdenom Benchmark Suite ===\n\n")

# ============================================================================
# POISSON_GAMMA family
# ============================================================================

cat("--- Poisson-Gamma Family ---\n\n")

# Row 1: poisson_gamma, no RE (H)
results <- rbind(results, benchmark_config(
  row = 1, family = "poisson_gamma", re = "none",
  spatial = "none", temporal = "none", zi = "none", grad = "H",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 2: poisson_gamma, RE (H)
results <- rbind(results, benchmark_config(
  row = 2, family = "poisson_gamma", re = "intercept",
  spatial = "none", temporal = "none", zi = "none", grad = "H",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 3: poisson_gamma, slopes (A)
results <- rbind(results, benchmark_config(
  row = 3, family = "poisson_gamma", re = "slopes",
  spatial = "none", temporal = "none", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 4: poisson_gamma, crossed (A)
results <- rbind(results, benchmark_config(
  row = 4, family = "poisson_gamma", re = "crossed",
  spatial = "none", temporal = "none", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 5: poisson_gamma, RE + ICAR (A)
results <- rbind(results, benchmark_config(
  row = 5, family = "poisson_gamma", re = "intercept",
  spatial = "ICAR", temporal = "none", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 6: poisson_gamma, RE + BYM2 (A)
results <- rbind(results, benchmark_config(
  row = 6, family = "poisson_gamma", re = "intercept",
  spatial = "BYM2", temporal = "none", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Rows 7-8: GP (skipped - known issues)
cat("Row 7-8: GP configs - SKIPPED (known heisenbug)\n\n")

# Row 9: poisson_gamma, RE + RW1 (A)
results <- rbind(results, benchmark_config(
  row = 9, family = "poisson_gamma", re = "intercept",
  spatial = "none", temporal = "RW1", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 10: poisson_gamma, RE + RW2 (A)
results <- rbind(results, benchmark_config(
  row = 10, family = "poisson_gamma", re = "intercept",
  spatial = "none", temporal = "RW2", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 11: poisson_gamma, RE + AR1 (A)
results <- rbind(results, benchmark_config(
  row = 11, family = "poisson_gamma", re = "intercept",
  spatial = "none", temporal = "AR1", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 13: poisson_gamma, RE + ZI (A)
results <- rbind(results, benchmark_config(
  row = 13, family = "poisson_gamma", re = "intercept",
  spatial = "none", temporal = "none", zi = "ZI", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 14: poisson_gamma, RE + Hurdle (A)
results <- rbind(results, benchmark_config(
  row = 14, family = "poisson_gamma", re = "intercept",
  spatial = "none", temporal = "none", zi = "Hurdle", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 15: poisson_gamma, RE + ICAR + RW1 (A)
results <- rbind(results, benchmark_config(
  row = 15, family = "poisson_gamma", re = "intercept",
  spatial = "ICAR", temporal = "RW1", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 16: poisson_gamma, RE + BYM2 + RW1 (A)
results <- rbind(results, benchmark_config(
  row = 16, family = "poisson_gamma", re = "intercept",
  spatial = "BYM2", temporal = "RW1", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 17: poisson_gamma, RE + ICAR + AR1 (A)
results <- rbind(results, benchmark_config(
  row = 17, family = "poisson_gamma", re = "intercept",
  spatial = "ICAR", temporal = "AR1", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 19: poisson_gamma, RE + ICAR + ZI (A)
results <- rbind(results, benchmark_config(
  row = 19, family = "poisson_gamma", re = "intercept",
  spatial = "ICAR", temporal = "none", zi = "ZI", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 20: poisson_gamma, slopes + ICAR (A)
results <- rbind(results, benchmark_config(
  row = 20, family = "poisson_gamma", re = "slopes",
  spatial = "ICAR", temporal = "none", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# ============================================================================
# NEGBIN_NEGBIN family
# ============================================================================

cat("\n--- NegBin-NegBin Family ---\n\n")

# Row 21: negbin_negbin, no RE (A)
results <- rbind(results, benchmark_config(
  row = 21, family = "negbin_negbin", re = "none",
  spatial = "none", temporal = "none", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 22: negbin_negbin, RE (A)
results <- rbind(results, benchmark_config(
  row = 22, family = "negbin_negbin", re = "intercept",
  spatial = "none", temporal = "none", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 23: negbin_negbin, slopes (A)
results <- rbind(results, benchmark_config(
  row = 23, family = "negbin_negbin", re = "slopes",
  spatial = "none", temporal = "none", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 24: negbin_negbin, crossed (A)
results <- rbind(results, benchmark_config(
  row = 24, family = "negbin_negbin", re = "crossed",
  spatial = "none", temporal = "none", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 25: negbin_negbin, RE + ICAR (A)
results <- rbind(results, benchmark_config(
  row = 25, family = "negbin_negbin", re = "intercept",
  spatial = "ICAR", temporal = "none", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 26: negbin_negbin, RE + BYM2 (A)
results <- rbind(results, benchmark_config(
  row = 26, family = "negbin_negbin", re = "intercept",
  spatial = "BYM2", temporal = "none", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 28: negbin_negbin, RE + RW1 (A)
results <- rbind(results, benchmark_config(
  row = 28, family = "negbin_negbin", re = "intercept",
  spatial = "none", temporal = "RW1", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 29: negbin_negbin, RE + AR1 (A)
results <- rbind(results, benchmark_config(
  row = 29, family = "negbin_negbin", re = "intercept",
  spatial = "none", temporal = "AR1", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 30: negbin_negbin, RE + ZI (A)
results <- rbind(results, benchmark_config(
  row = 30, family = "negbin_negbin", re = "intercept",
  spatial = "none", temporal = "none", zi = "ZI", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 31: negbin_negbin, RE + Hurdle (A)
results <- rbind(results, benchmark_config(
  row = 31, family = "negbin_negbin", re = "intercept",
  spatial = "none", temporal = "none", zi = "Hurdle", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 32: negbin_negbin, RE + ICAR + RW1 (A)
results <- rbind(results, benchmark_config(
  row = 32, family = "negbin_negbin", re = "intercept",
  spatial = "ICAR", temporal = "RW1", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# ============================================================================
# BINOMIAL family
# ============================================================================

cat("\n--- Binomial Family ---\n\n")

# Row 33: binomial, no RE (A)
results <- rbind(results, benchmark_config(
  row = 33, family = "binomial", re = "none",
  spatial = "none", temporal = "none", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 34: binomial, RE (A)
results <- rbind(results, benchmark_config(
  row = 34, family = "binomial", re = "intercept",
  spatial = "none", temporal = "none", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 35: binomial, slopes (A)
results <- rbind(results, benchmark_config(
  row = 35, family = "binomial", re = "slopes",
  spatial = "none", temporal = "none", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 36: binomial, RE + ICAR (A)
results <- rbind(results, benchmark_config(
  row = 36, family = "binomial", re = "intercept",
  spatial = "ICAR", temporal = "none", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 37: binomial, RE + BYM2 (A)
results <- rbind(results, benchmark_config(
  row = 37, family = "binomial", re = "intercept",
  spatial = "BYM2", temporal = "none", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 39: binomial, RE + RW1 (A)
results <- rbind(results, benchmark_config(
  row = 39, family = "binomial", re = "intercept",
  spatial = "none", temporal = "RW1", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 40: binomial, RE + AR1 (A)
results <- rbind(results, benchmark_config(
  row = 40, family = "binomial", re = "intercept",
  spatial = "none", temporal = "AR1", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# Row 41: binomial, RE + ICAR + RW1 (A)
results <- rbind(results, benchmark_config(
  row = 41, family = "binomial", re = "intercept",
  spatial = "ICAR", temporal = "RW1", zi = "none", grad = "A",
  n_iter = n_iter, n_warmup = n_warmup
))

# ============================================================================
# Summary
# ============================================================================

cat("\n", strrep("=", 60), "\n")
cat("BENCHMARK SUMMARY\n")
cat(strrep("=", 60), "\n\n")

# Print results table
print(results[, c("row", "family", "re", "spatial", "temporal", "zi", "speedup")])

cat("\n")
cat("Configs tested:", nrow(results), "\n")
cat("All faster than Stan:", all(results$speedup > 1, na.rm = TRUE), "\n")
cat("Mean speedup:", round(mean(results$speedup, na.rm = TRUE), 1), "x\n")
cat("Min speedup:", round(min(results$speedup, na.rm = TRUE), 1), "x\n")
cat("Max speedup:", round(max(results$speedup, na.rm = TRUE), 1), "x\n")

# Save results
saveRDS(results, "benchmarks/results.rds")
cat("\nResults saved to benchmarks/results.rds\n")
