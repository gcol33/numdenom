# Test that lognormal H-mode gradients match A-mode (autodiff)
# This script tests gradient correctness before enabling H-mode for LOGNORMAL

library(numdenom)
set.seed(42)

# Simple lognormal model without RE
N_OBS <- 50
x <- rnorm(N_OBS)
eta_num <- 2 + 0.3 * x
eta_denom <- 4 + 0.2 * x

df <- data.frame(
  y = rlnorm(N_OBS, eta_num, 0.5),
  denom = rlnorm(N_OBS, eta_denom, 0.5),
  x = x
)

cat("Testing lognormal gradient modes...\n")

# Currently A-mode (since LOGNORMAL is excluded from H)
cat("\n=== Test 1: Base lognormal (no RE) ===\n")
t_A <- system.time({
  fit_A <- tratio(
    y | denom ~ x, data = df,
    family = ratiod_lognormal(),
    control = list(iter = 200, warmup = 100, chains = 1, gradient_mode = "A", verbose = FALSE)
  )
})[[3]]
cat(sprintf("A-mode: %.2fs\n", t_A))

# Force H-mode to test gradient correctness
# This will currently error or use A fallback since LOGNORMAL is excluded
tryCatch({
  t_H <- system.time({
    fit_H <- tratio(
      y | denom ~ x, data = df,
      family = ratiod_lognormal(),
      control = list(iter = 200, warmup = 100, chains = 1, gradient_mode = "H", verbose = FALSE)
    )
  })[[3]]
  cat(sprintf("H-mode: %.2fs\n", t_H))

  # Compare posteriors
  draws_A <- as.matrix(fit_A$draws)
  draws_H <- as.matrix(fit_H$draws)

  params <- c("beta_num[1]", "beta_num[2]", "beta_denom[1]", "beta_denom[2]",
              "sigma_num", "sigma_denom")

  cat("\nPosterior comparison (should be nearly identical):\n")
  all_pass <- TRUE
  for (p in params) {
    mean_A <- mean(draws_A[, p])
    mean_H <- mean(draws_H[, p])
    sd_A <- sd(draws_A[, p])
    se <- sd_A / sqrt(nrow(draws_A))
    diff_se <- abs(mean_A - mean_H) / se
    pass <- diff_se < 3  # Allow 3 SE for randomness
    if (!pass) all_pass <- FALSE
    cat(sprintf("  %s: A=%.4f, H=%.4f (%.1f SE) %s\n",
                p, mean_A, mean_H, diff_se, ifelse(pass, "OK", "MISMATCH")))
  }

  if (all_pass) {
    cat("\n*** LOGNORMAL H-mode gradients are CORRECT ***\n")
    cat(sprintf("Speedup: %.1fx\n", t_A / t_H))
  } else {
    cat("\n*** LOGNORMAL H-mode gradients have MISMATCH - need debugging ***\n")
  }

}, error = function(e) {
  cat(sprintf("H-mode not available or errored: %s\n", e$message))
  cat("This is expected since LOGNORMAL is currently excluded from can_use_analytical_gradient()\n")
})

# Test with RE
cat("\n=== Test 2: Lognormal with RE ===\n")
N_SITES <- 10
site <- sample(1:N_SITES, N_OBS, replace = TRUE)
true_re <- rnorm(N_SITES, 0, 0.3)
eta_num_re <- eta_num + true_re[site]
eta_denom_re <- eta_denom + true_re[site]

df_re <- data.frame(
  y = rlnorm(N_OBS, eta_num_re, 0.5),
  denom = rlnorm(N_OBS, eta_denom_re, 0.5),
  x = x,
  site = factor(site)
)

t_A_re <- system.time({
  fit_A_re <- tratio(
    y | denom ~ x + (1 | site), data = df_re,
    family = ratiod_lognormal(),
    control = list(iter = 200, warmup = 100, chains = 1, gradient_mode = "A", verbose = FALSE)
  )
})[[3]]
cat(sprintf("A-mode with RE: %.2fs\n", t_A_re))

tryCatch({
  t_H_re <- system.time({
    fit_H_re <- tratio(
      y | denom ~ x + (1 | site), data = df_re,
      family = ratiod_lognormal(),
      control = list(iter = 200, warmup = 100, chains = 1, gradient_mode = "H", verbose = FALSE)
    )
  })[[3]]
  cat(sprintf("H-mode with RE: %.2fs\n", t_H_re))

  # Compare posteriors
  draws_A <- as.matrix(fit_A_re$draws)
  draws_H <- as.matrix(fit_H_re$draws)

  params <- c("beta_num[1]", "beta_num[2]", "beta_denom[1]", "beta_denom[2]",
              "sigma_num", "sigma_denom", "sigma_re")

  cat("\nPosterior comparison with RE:\n")
  all_pass <- TRUE
  for (p in params) {
    mean_A <- mean(draws_A[, p])
    mean_H <- mean(draws_H[, p])
    sd_A <- sd(draws_A[, p])
    se <- sd_A / sqrt(nrow(draws_A))
    diff_se <- abs(mean_A - mean_H) / se
    pass <- diff_se < 3
    if (!pass) all_pass <- FALSE
    cat(sprintf("  %s: A=%.4f, H=%.4f (%.1f SE) %s\n",
                p, mean_A, mean_H, diff_se, ifelse(pass, "OK", "MISMATCH")))
  }

  if (all_pass) {
    cat("\n*** LOGNORMAL+RE H-mode gradients are CORRECT ***\n")
    cat(sprintf("Speedup: %.1fx\n", t_A_re / t_H_re))
  } else {
    cat("\n*** LOGNORMAL+RE H-mode gradients have MISMATCH ***\n")
  }

}, error = function(e) {
  cat(sprintf("H-mode with RE not available: %s\n", e$message))
})

cat("\nDone.\n")
