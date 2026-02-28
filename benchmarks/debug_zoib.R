# Detailed ZOIB diagnostic
library(numdenom)
library(posterior)

set.seed(42)

N_OBS <- 200
N_SITES <- 20
N_ITER <- 2000  # More iterations for better diagnostics
N_WARMUP <- 1000
N_CHAINS <- 2

# Setup
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)
trials <- sample(20:50, N_OBS, replace = TRUE)

# True parameters
true_intercept <- 0.5
true_slope <- 0.3
true_sigma_re <- 0.3
site_effects <- rnorm(N_SITES, 0, true_sigma_re)

# ZOIB parameters
true_zi_prob <- 0.10
true_oi_prob <- 0.10

cat("=== ZOIB Diagnostic ===\n\n")
cat(sprintf("True parameters:\n"))
cat(sprintf("  intercept = %.2f\n", true_intercept))
cat(sprintf("  slope     = %.2f\n", true_slope))
cat(sprintf("  sigma_re  = %.2f\n", true_sigma_re))
cat(sprintf("  zi_prob   = %.2f\n", true_zi_prob))
cat(sprintf("  oi_prob   = %.2f\n", true_oi_prob))

# Generate data
eta <- true_intercept + true_slope * x + site_effects[as.integer(site)]
prob <- plogis(eta)

# Inflation mechanism
inflate_type <- sample(c("zero", "one", "neither"), N_OBS, replace = TRUE,
                       prob = c(true_zi_prob, true_oi_prob, 1 - true_zi_prob - true_oi_prob))

successes <- ifelse(inflate_type == "zero", 0,
              ifelse(inflate_type == "one", trials,
                     rbinom(N_OBS, trials, prob)))

df <- data.frame(successes = successes, trials = trials, x = x, site = site)

cat(sprintf("\nData summary:\n"))
cat(sprintf("  N = %d\n", N_OBS))
cat(sprintf("  Observed zeros: %d (%.1f%%)\n", sum(successes == 0), 100 * mean(successes == 0)))
cat(sprintf("  Observed ones:  %d (%.1f%%)\n", sum(successes == trials), 100 * mean(successes == trials)))
cat(sprintf("  Mean success rate: %.2f\n", mean(successes / trials)))

# Also check: how many zeros come from binomial vs inflation?
cat(sprintf("\n  Generated as zero-inflated: %d\n", sum(inflate_type == "zero")))
cat(sprintf("  Generated as one-inflated:  %d\n", sum(inflate_type == "one")))
cat(sprintf("  Generated as binomial:      %d\n", sum(inflate_type == "neither")))

# Among binomial-generated, how many happen to be 0 or trials?
binomial_zeros <- sum(successes[inflate_type == "neither"] == 0)
binomial_ones <- sum(successes[inflate_type == "neither"] == trials[inflate_type == "neither"])
cat(sprintf("  Binomial that happened to be 0: %d\n", binomial_zeros))
cat(sprintf("  Binomial that happened to be trials: %d\n", binomial_ones))

cat("\nFitting ZOIB model...\n")
fit <- tryCatch({
  ratiod(successes | trials ~ x + (1 | site), data = df,
         family = ratiod_zoibinomial(),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = TRUE)
}, error = function(e) {
  cat(sprintf("ERROR: %s\n", e$message))
  NULL
})

if (!is.null(fit)) {
  draws <- as.matrix(fit$draws)

  cat("\n=== Posterior Summary ===\n")
  cat(sprintf("Available parameters: %s\n", paste(head(colnames(draws), 20), collapse = ", ")))

  # Find slope parameter
  slope_cols <- grep("beta", colnames(draws), value = TRUE)
  cat(sprintf("\nBeta columns: %s\n", paste(slope_cols, collapse = ", ")))

  # Extract key parameters
  if (length(slope_cols) >= 2) {
    intercept_draws <- draws[, slope_cols[1]]
    slope_draws <- draws[, slope_cols[2]]

    cat(sprintf("\nIntercept: true=%.3f, post_mean=%.3f, post_sd=%.3f\n",
                true_intercept, mean(intercept_draws), sd(intercept_draws)))
    cat(sprintf("Slope:     true=%.3f, post_mean=%.3f, post_sd=%.3f\n",
                true_slope, mean(slope_draws), sd(slope_draws)))

    diff_sd <- abs(mean(slope_draws) - true_slope) / sd(slope_draws)
    cat(sprintf("\nSlope recovery: %.2f SD from true value\n", diff_sd))

    # Check 95% CI
    ci <- quantile(slope_draws, c(0.025, 0.975))
    cat(sprintf("95%% CI: [%.3f, %.3f]\n", ci[1], ci[2]))
    cat(sprintf("True value in CI: %s\n", if(true_slope >= ci[1] && true_slope <= ci[2]) "YES" else "NO"))
  }

  # Check ZI/OI parameters if present
  zi_cols <- grep("zi|oi|inflate", colnames(draws), value = TRUE, ignore.case = TRUE)
  if (length(zi_cols) > 0) {
    cat(sprintf("\nInflation parameters found: %s\n", paste(zi_cols, collapse = ", ")))
    for (col in zi_cols) {
      cat(sprintf("  %s: mean=%.3f, sd=%.3f\n", col, mean(draws[, col]), sd(draws[, col])))
    }
  }

  # Check RE sigma
  sigma_cols <- grep("sigma|sd_", colnames(draws), value = TRUE, ignore.case = TRUE)
  if (length(sigma_cols) > 0) {
    cat(sprintf("\nSigma parameters: %s\n", paste(sigma_cols, collapse = ", ")))
    for (col in sigma_cols) {
      cat(sprintf("  %s: mean=%.3f (true=%.3f)\n", col, mean(draws[, col]), true_sigma_re))
    }
  }

  # MCMC diagnostics
  cat("\n=== MCMC Diagnostics ===\n")
  if (!is.null(fit$diagnostics)) {
    cat(sprintf("Divergences: %d\n", sum(fit$diagnostics$divergent, na.rm = TRUE)))
    cat(sprintf("Max treedepth: %d\n", max(fit$diagnostics$treedepth, na.rm = TRUE)))
  }

  # Rhat and ESS for key parameters
  summ <- summarise_draws(fit$draws)
  key_params <- summ[grep("beta|sigma|zi|oi", summ$variable, ignore.case = TRUE), ]
  if (nrow(key_params) > 0) {
    cat("\nKey parameter diagnostics:\n")
    print(key_params[, c("variable", "mean", "sd", "rhat", "ess_bulk", "ess_tail")])
  }

} else {
  cat("Model fitting failed.\n")
}
