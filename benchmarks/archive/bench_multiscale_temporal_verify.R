# Quick verification that multiscale temporal (MS_t) uses H gradients
# Should see ~10s with H, ~700s with A

library(numdenom)

# Smaller parameters for quick verification
N_OBS <- 200
N_ITER <- 100
N_CHAINS <- 1
N_TIMES <- 10

set.seed(42)

# Generate simple temporal data
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
cat("Verifying multiscale temporal H gradient\n")
cat(strrep("=", 60), "\n\n")

# Test H gradient mode (should be fast ~10s)
cat("Testing gradient_mode = 'H'...\n")
time_h <- system.time({
  tryCatch({
    fit_h <- ratiod(
      y | effort ~ x,
      data = df,
      family = ratiod_poisson_gamma(),
      iter = N_ITER,
      chains = N_CHAINS,
      gradient_mode = "H",
      temporal = temporal_multiscale(time_var = "time"),
      refresh = 0
    )
    cat("  SUCCESS\n")
  }, error = function(e) {
    cat(sprintf("  ERROR: %s\n", conditionMessage(e)))
  })
})["elapsed"]
cat(sprintf("  H time: %.1f s\n\n", time_h))

# Test A gradient mode (should be slow ~100s for 100 iter)
cat("Testing gradient_mode = 'A' (expect slower)...\n")
time_a <- system.time({
  tryCatch({
    fit_a <- ratiod(
      y | effort ~ x,
      data = df,
      family = ratiod_poisson_gamma(),
      iter = N_ITER,
      chains = N_CHAINS,
      gradient_mode = "A",
      temporal = temporal_multiscale(time_var = "time"),
      refresh = 0
    )
    cat("  SUCCESS\n")
  }, error = function(e) {
    cat(sprintf("  ERROR: %s\n", conditionMessage(e)))
  })
})["elapsed"]
cat(sprintf("  A time: %.1f s\n\n", time_a))

# Summary
cat(strrep("=", 60), "\n")
cat("SUMMARY\n")
cat(strrep("=", 60), "\n\n")
cat(sprintf("H gradient: %.1f s\n", time_h))
cat(sprintf("A gradient: %.1f s\n", time_a))
if (!is.na(time_h) && !is.na(time_a) && time_a > 0) {
  speedup <- time_a / time_h
  cat(sprintf("Speedup (A/H): %.1fx\n", speedup))
  if (speedup > 3) {
    cat("\n✓ H gradients are working correctly!\n")
  } else {
    cat("\n⚠ H gradients may not be activating (speedup < 3x)\n")
  }
}
