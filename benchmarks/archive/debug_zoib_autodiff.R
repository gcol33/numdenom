# Test ZOIB with autodiff (A_t) mode to verify likelihood is correct
library(numdenom)
library(posterior)

set.seed(42)

N_OBS <- 400
N_SITES <- 20
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2

site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)
trials <- sample(30:60, N_OBS, replace = TRUE)

# True parameters
true_intercept <- 0.5
true_slope <- 0.3
true_sigma_re <- 0.3
site_effects <- rnorm(N_SITES, 0, true_sigma_re)

# ZOIB parameters - in the MIXTURE parameterization of numdenom
# zi = P(structural zero)
# oi = P(structural one | not structural zero, not binomial)
# NOTE: In numdenom's ZOIB, zeros can come from ZI OR from the binomial*!(ZI+OI)
true_zi <- 0.15
true_oi <- 0.12

cat("=== ZOIB Validation with Autodiff Mode ===\n\n")
cat("True parameters:\n")
cat(sprintf("  intercept = %.2f\n", true_intercept))
cat(sprintf("  slope     = %.2f\n", true_slope))
cat(sprintf("  sigma_re  = %.2f\n", true_sigma_re))
cat(sprintf("  zi        = %.3f  [logit = %.3f]\n", true_zi, qlogis(true_zi)))
cat(sprintf("  oi        = %.3f  [logit = %.3f]\n", true_oi, qlogis(true_oi)))

# Generate data using numdenom's MIXTURE ZOIB parameterization
# P(Y=0) = zi + (1-zi)*(1-oi)*Binom(0|n,p)
# P(Y=n) = (1-zi)*(oi + (1-oi)*Binom(n|n,p))
# P(0<Y<n) = (1-zi)*(1-oi)*Binom(y|n,p)
eta <- true_intercept + true_slope * x + site_effects[as.integer(site)]
prob <- plogis(eta)

successes <- numeric(N_OBS)
for (i in 1:N_OBS) {
  u <- runif(1)
  n <- trials[i]
  p <- prob[i]

  # Compute probabilities
  p_zi <- true_zi  # P(structural zero)
  p_binom_zero <- (1 - true_zi) * (1 - true_oi) * dbinom(0, n, p)
  p_oi <- (1 - true_zi) * true_oi  # P(structural one)
  p_binom_one <- (1 - true_zi) * (1 - true_oi) * dbinom(n, n, p)

  # Sample
  if (u < p_zi + p_binom_zero) {
    successes[i] <- 0
  } else if (u < p_zi + p_binom_zero + p_oi + p_binom_one) {
    # Could be structural one or binomial one
    if (runif(1) < p_oi / (p_oi + p_binom_one)) {
      successes[i] <- n  # structural one
    } else {
      successes[i] <- n  # binomial one
    }
  } else {
    # Binomial draw (truncated to exclude 0 and n)
    repeat {
      successes[i] <- rbinom(1, n, p)
      if (successes[i] > 0 && successes[i] < n) break
    }
  }
}

df <- data.frame(successes = successes, trials = trials, x = x, site = site)

cat(sprintf("\nData summary:\n"))
cat(sprintf("  N = %d\n", N_OBS))
cat(sprintf("  Observed zeros: %d (%.1f%%)\n", sum(successes == 0), 100 * mean(successes == 0)))
cat(sprintf("  Observed ones:  %d (%.1f%%)\n", sum(successes == trials), 100 * mean(successes == trials)))

cat("\nFitting ZOIB model with AUTODIFF (A_t) mode...\n")
fit <- tryCatch({
  tratio(successes | trials ~ x + (1 | site), data = df,
         family = ratiod_zoibinomial(),
         control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = TRUE, gradient_mode = "A_t"))
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
  }

  # ZI/OI parameters
  zi_cols <- grep("zi|oi", colnames(draws), value = TRUE, ignore.case = TRUE)
  if (length(zi_cols) > 0) {
    cat("\nInflation parameters:\n")
    for (col in zi_cols) {
      draws_col <- draws[, col]
      prob_draws <- plogis(draws_col)
      cat(sprintf("  %s (logit): mean=%.3f, sd=%.3f\n", col, mean(draws_col), sd(draws_col)))
      cat(sprintf("  %s (prob):  mean=%.3f, sd=%.3f\n", col, mean(prob_draws), sd(prob_draws)))

      if (grepl("zi", col, ignore.case = TRUE)) {
        cat(sprintf("             true=%.3f (logit=%.3f)\n", true_zi, qlogis(true_zi)))
      } else if (grepl("oi", col, ignore.case = TRUE)) {
        cat(sprintf("             true=%.3f (logit=%.3f)\n", true_oi, qlogis(true_oi)))
      }
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
