# Detailed benchmark comparing all gradient modes for multiscale temporal
# Testing with larger data to see if differences are more pronounced

library(numdenom)

# Larger parameters to see gradient differences
N_OBS <- 500
N_ITER <- 500
N_CHAINS <- 1
N_TIMES <- 20

set.seed(42)

# Generate temporal data
time_idx <- rep(1:N_TIMES, length.out = N_OBS)
x <- rnorm(N_OBS)
eta <- 2 + 0.3 * x + cumsum(rnorm(N_TIMES, 0, 0.1))[time_idx]
y <- rpois(N_OBS, exp(eta))
effort <- rgamma(N_OBS, 10, 1)

df <- data.frame(
  y = y,
  effort = effort,
  x = x,
  time = factor(time_idx)
)

cat("\n", strrep("=", 60), "\n")
cat("Multiscale Temporal Gradient Benchmark\n")
cat("N =", N_OBS, " T =", N_TIMES, " Iter =", N_ITER, "\n")
cat(strrep("=", 60), "\n\n")

results <- list()

# Test all gradient modes
for (mode in c("N", "A_t", "A", "H")) {
  cat(sprintf("Testing gradient_mode = '%s'...\n", mode))

  time_mode <- system.time({
    tryCatch({
      fit <- tratio(
        y | effort ~ x,
        data = df,
        family = ratiod_poisson_gamma(),
        temporal = temporal_multiscale(time_var = "time"),
        control = list(iter = N_ITER, chains = N_CHAINS, gradient_mode = mode)
      )
      cat("  SUCCESS\n")
    }, error = function(e) {
      cat(sprintf("  ERROR: %s\n", conditionMessage(e)))
    })
  })["elapsed"]

  cat(sprintf("  Time: %.1f s\n\n", time_mode))
  results[[mode]] <- time_mode
}

# Summary
cat(strrep("=", 60), "\n")
cat("SUMMARY\n")
cat(strrep("=", 60), "\n\n")

for (mode in names(results)) {
  cat(sprintf("%s gradient: %.1f s\n", mode, results[[mode]]))
}

cat("\nSpeedups vs N (numerical):\n")
for (mode in c("A_t", "A", "H")) {
  if (!is.na(results[[mode]]) && !is.na(results[["N"]]) && results[[mode]] > 0) {
    speedup <- results[["N"]] / results[[mode]]
    cat(sprintf("  %s: %.1fx\n", mode, speedup))
  }
}

# If H isn't much faster than A, the gradient isn't the bottleneck
cat("\nSpeedups vs A:\n")
for (mode in c("H")) {
  if (!is.na(results[[mode]]) && !is.na(results[["A"]]) && results[[mode]] > 0) {
    speedup <- results[["A"]] / results[[mode]]
    cat(sprintf("  %s: %.1fx\n", mode, speedup))
    if (speedup < 2) {
      cat("\n⚠ H not significantly faster than A\n")
      cat("  This may indicate gradient is not the bottleneck\n")
      cat("  or the H implementation needs optimization\n")
    }
  }
}
