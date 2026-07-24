# ZOIB validation with NUMERICAL gradient mode
# Test if the issue is in the gradient computation
library(numdenom)
library(posterior)

set.seed(42)

N_OBS <- 400
N_SITES <- 20
N_ITER <- 1000  # shorter for numerical
N_WARMUP <- 500
N_CHAINS <- 1

# Setup
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)
trials <- sample(30:60, N_OBS, replace = TRUE)

# True parameters
true_intercept <- 0.5
true_slope <- 0.3
true_sigma_re <- 0.3
site_effects <- rnorm(N_SITES, 0, true_sigma_re)

# ZOIB parameters in model's CONDITIONAL parameterization
true_pi0 <- 0.15  # P(structural zero)
true_pi1 <- 0.12  # P(structural one | not zero)

cat("=== ZOIB Validation with NUMERICAL Gradient Mode ===\n\n")
cat(sprintf("True parameters:\n"))
cat(sprintf("  intercept   = %.2f\n", true_intercept))
cat(sprintf("  slope       = %.2f\n", true_slope))
cat(sprintf("  sigma_re    = %.2f\n", true_sigma_re))
cat(sprintf("  pi_0 (ZI)   = %.3f  [logit = %.3f]\n", true_pi0, qlogis(true_pi0)))
cat(sprintf("  pi_1 (OI|~Z)= %.3f  [logit = %.3f]\n", true_pi1, qlogis(true_pi1)))

# Generate data using MODEL's parameterization
eta <- true_intercept + true_slope * x + site_effects[as.integer(site)]
prob <- plogis(eta)

# Two-stage inflation (matches the model)
is_struct_zero <- rbinom(N_OBS, 1, true_pi0)
is_struct_one <- ifelse(is_struct_zero == 1, 0, rbinom(N_OBS, 1, true_pi1))

successes <- ifelse(is_struct_zero == 1, 0,
              ifelse(is_struct_one == 1, trials,
                     rbinom(N_OBS, trials, prob)))

df <- data.frame(successes = successes, trials = trials, x = x, site = site)

cat(sprintf("\nData summary:\n"))
cat(sprintf("  N = %d\n", N_OBS))
cat(sprintf("  Observed zeros: %d (%.1f%%) [expected: %.1f%%]\n",
            sum(successes == 0), 100 * mean(successes == 0), 100 * true_pi0))
cat(sprintf("  Observed ones:  %d (%.1f%%) [expected: %.1f%%]\n",
            sum(successes == trials), 100 * mean(successes == trials),
            100 * (1 - true_pi0) * true_pi1))

cat("\nFitting ZOIB model with NUMERICAL (N) gradient mode...\n")
fit <- tryCatch({
  tratio(successes | trials ~ x + (1 | site), data = df,
         family = ratiod_zoibinomial(),
         control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = TRUE, gradient_mode = "N"))
}, error = function(e) {
  cat(sprintf("ERROR: %s\n", e$message))
  NULL
})

if (!is.null(fit)) {
  draws <- as.matrix(fit$draws)

  cat("\n=== Parameter Recovery ===\n")

  # Slope
  slope_cols <- grep("beta_num", colnames(draws), value = TRUE)
  if (length(slope_cols) >= 2) {
    slope_draws <- draws[, slope_cols[2]]
    intercept_draws <- draws[, slope_cols[1]]

    cat(sprintf("\nIntercept: true=%.3f, post=%.3f (SD=%.3f)\n",
                true_intercept, mean(intercept_draws), sd(intercept_draws)))
    cat(sprintf("Slope:     true=%.3f, post=%.3f (SD=%.3f)\n",
                true_slope, mean(slope_draws), sd(slope_draws)))

    diff_sd <- abs(mean(slope_draws) - true_slope) / sd(slope_draws)
    ci <- quantile(slope_draws, c(0.025, 0.975))
    cat(sprintf("           %.2f SD from true, 95%% CI: [%.3f, %.3f]\n",
                diff_sd, ci[1], ci[2]))
    cat(sprintf("           True in CI: %s\n",
                if(true_slope >= ci[1] && true_slope <= ci[2]) "YES" else "NO"))

    slope_pass <- diff_sd < 2.0
    cat(sprintf("\nSlope recovery: %s\n", if(slope_pass) "PASS" else "FAIL"))
  }

  # ZI/OI parameters
  zi_cols <- grep("zi|oi", colnames(draws), value = TRUE, ignore.case = TRUE)
  if (length(zi_cols) > 0) {
    cat("\nInflation parameters:\n")
    for (col in zi_cols) {
      draws_col <- draws[, col]
      prob_draws <- plogis(draws_col)
      cat(sprintf("  %s (logit): mean=%.3f, sd=%.3f\n", col, mean(draws_col), sd(draws_col)))
      cat(sprintf("  %s (prob):  mean=%.3f\n", col, mean(prob_draws)))

      if (grepl("zi", col, ignore.case = TRUE)) {
        cat(sprintf("             true=%.3f (logit=%.3f)\n", true_pi0, qlogis(true_pi0)))
      } else if (grepl("oi", col, ignore.case = TRUE)) {
        cat(sprintf("             true=%.3f (logit=%.3f)\n", true_pi1, qlogis(true_pi1)))
      }
    }
  }

  # RE sigma
  sigma_cols <- grep("sigma", colnames(draws), value = TRUE, ignore.case = TRUE)
  if (length(sigma_cols) > 0) {
    cat("\nRandom effects:\n")
    for (col in sigma_cols) {
      cat(sprintf("  %s: mean=%.3f (true=%.3f)\n", col, mean(draws[, col]), true_sigma_re))
    }
  }

  # MCMC diagnostics
  cat("\n=== MCMC Diagnostics ===\n")
  summ <- summarise_draws(fit$draws)
  key_params <- summ[grep("beta|sigma|zi|oi", summ$variable, ignore.case = TRUE), ]
  if (nrow(key_params) > 0) {
    print(key_params[, c("variable", "mean", "sd", "rhat", "ess_bulk", "ess_tail")])
  }
}
