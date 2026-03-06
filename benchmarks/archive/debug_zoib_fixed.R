# ZOIB validation with CORRECT parameterization
# The numdenom ZOIB uses conditional parameterization:
#   pi_0 = P(structural zero)
#   pi_1 = P(structural one | NOT structural zero)
#
# P(Y=0) = pi_0
# P(Y=n) = (1 - pi_0) * pi_1
# P(Y=y) = (1 - pi_0) * (1 - pi_1) * Binomial(y; n, p)

library(numdenom)
library(posterior)

set.seed(42)

N_OBS <- 400       # More data for better identifiability
N_SITES <- 20
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2

# Setup
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)
trials <- sample(30:60, N_OBS, replace = TRUE)  # More trials

# True parameters
true_intercept <- 0.5
true_slope <- 0.3
true_sigma_re <- 0.3
site_effects <- rnorm(N_SITES, 0, true_sigma_re)

# ZOIB parameters in model's CONDITIONAL parameterization
true_pi0 <- 0.15  # P(structural zero)
true_pi1 <- 0.12  # P(structural one | not zero) -- CONDITIONAL

# Convert to unconditional for simulation
# P(zero) = pi_0 = 0.15
# P(one) = (1 - pi_0) * pi_1 = 0.85 * 0.12 = 0.102
# P(binom) = (1 - pi_0) * (1 - pi_1) = 0.85 * 0.88 = 0.748

cat("=== ZOIB Validation (Corrected Parameterization) ===\n\n")
cat("True parameters:\n")
cat(sprintf("  intercept   = %.2f\n", true_intercept))
cat(sprintf("  slope       = %.2f\n", true_slope))
cat(sprintf("  sigma_re    = %.2f\n", true_sigma_re))
cat(sprintf("  pi_0 (ZI)   = %.3f  [logit = %.3f]\n", true_pi0, qlogis(true_pi0)))
cat(sprintf("  pi_1 (OI|~Z)= %.3f  [logit = %.3f]\n", true_pi1, qlogis(true_pi1)))
cat(sprintf("\nImplied unconditional probabilities:\n"))
cat(sprintf("  P(zero)   = %.3f\n", true_pi0))
cat(sprintf("  P(one)    = %.3f\n", (1 - true_pi0) * true_pi1))
cat(sprintf("  P(binom)  = %.3f\n", (1 - true_pi0) * (1 - true_pi1)))

# Generate data using MODEL's parameterization
eta <- true_intercept + true_slope * x + site_effects[as.integer(site)]
prob <- plogis(eta)

# Two-stage inflation (matches the model)
# Stage 1: is it a structural zero?
is_struct_zero <- rbinom(N_OBS, 1, true_pi0)

# Stage 2: if not struct zero, is it a structural one?
is_struct_one <- ifelse(is_struct_zero == 1, 0, rbinom(N_OBS, 1, true_pi1))

# Generate outcomes
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
cat(sprintf("  Mean success rate: %.2f\n", mean(successes / trials)))

# How many zeros/ones come from binomial vs inflation?
n_struct_zeros <- sum(is_struct_zero == 1)
n_struct_ones <- sum(is_struct_one == 1)
n_binom <- sum(is_struct_zero == 0 & is_struct_one == 0)
binom_zeros <- sum(successes[is_struct_zero == 0 & is_struct_one == 0] == 0)
binom_ones <- sum(successes[is_struct_zero == 0 & is_struct_one == 0] ==
                   trials[is_struct_zero == 0 & is_struct_one == 0])

cat(sprintf("\n  Structural zeros: %d\n", n_struct_zeros))
cat(sprintf("  Structural ones:  %d\n", n_struct_ones))
cat(sprintf("  Binomial obs:     %d (of which %d=0, %d=n)\n",
            n_binom, binom_zeros, binom_ones))

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
      # Convert from logit scale
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

} else {
  cat("Model fitting failed.\n")
}
