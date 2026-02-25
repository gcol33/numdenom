# Simulation-based validation for multiscale temporal models
# Row 75: binomial + MS_t (multiscale temporal)

library(numdenom)
library(posterior)

set.seed(42)

N_OBS <- 200
N_ITER <- 1000
N_WARMUP <- 500
N_CHAINS <- 2
N_SITES <- 20
N_TIMES <- 20

cat("=======================================================\n")
cat("Multiscale Temporal: Simulation-Based Validation\n")
cat("Row 75\n")
cat("=======================================================\n")
cat(sprintf("N=%d, sites=%d, times=%d, iter=%d, chains=%d\n\n",
            N_OBS, N_SITES, N_TIMES, N_ITER, N_CHAINS))

check_recovery <- function(draws, true_value, param_name, threshold_sd = 2) {
  post_mean <- mean(draws)
  post_sd <- sd(draws)
  diff_sd <- abs(post_mean - true_value) / post_sd
  in_ci <- true_value >= quantile(draws, 0.025) && true_value <= quantile(draws, 0.975)
  list(param = param_name, true = true_value, mean = post_mean, sd = post_sd,
       diff_sd = diff_sd, in_ci = in_ci, pass = diff_sd < threshold_sd)
}

print_recovery <- function(r) {
  cat(sprintf("  %s: true=%.3f, post=%.3f (SD=%.3f), %.2f SD => %s\n",
              r$param, r$true, r$mean, r$sd, r$diff_sd, if(r$pass) "PASS" else "FAIL"))
}

results <- list()

# Setup
site <- factor(rep(1:N_SITES, length.out = N_OBS))
time <- rep(1:N_TIMES, length.out = N_OBS)
x <- rnorm(N_OBS)
trials <- sample(20:50, N_OBS, replace = TRUE)

# True parameters
true_intercept <- 0.5
true_slope <- 0.3
site_effects <- rnorm(N_SITES, 0, 0.3)

# Generate multiscale temporal effects (slow + fast components)
true_sigma_slow <- 0.3
true_sigma_fast <- 0.15

# Slow component (smooth trend)
slow_trend <- cumsum(rnorm(N_TIMES, 0, true_sigma_slow / sqrt(N_TIMES)))
slow_trend <- slow_trend - mean(slow_trend)

# Fast component (rapid fluctuations)
fast_trend <- rnorm(N_TIMES, 0, true_sigma_fast)
fast_trend <- fast_trend - mean(fast_trend)

temporal_effects <- slow_trend + fast_trend

cat("========== Row 75: binomial + MS_t ==========\n")

eta_75 <- true_intercept + true_slope * x +
          site_effects[as.integer(site)] +
          temporal_effects[time]
prob_75 <- plogis(eta_75)
successes_75 <- rbinom(N_OBS, trials, prob_75)

df_75 <- data.frame(successes = successes_75, trials = trials, x = x, site = site, time = time)

cat(sprintf("True: slope=%.2f, sigma_slow=%.2f, sigma_fast=%.2f\n",
            true_slope, true_sigma_slow, true_sigma_fast))
cat(sprintf("Temporal effect range: [%.3f, %.3f]\n",
            min(temporal_effects), max(temporal_effects)))
cat("Fitting... ")

t_75 <- system.time({
  fit_75 <- tryCatch({
    ratiod(successes | trials ~ x + (1 | site), data = df_75,
           family = ratiod_binomial(),
           temporal = temporal_multiscale(time_var = "time"),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_75)) {
  cat(sprintf("%.1fs\n", t_75))
  draws_75 <- as.matrix(fit_75$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_75), value = TRUE)[1]

  results$row_75 <- list(
    slope = check_recovery(draws_75[, slope_col], true_slope, "slope"),
    time = t_75
  )
  print_recovery(results$row_75$slope)
  results$row_75$pass <- results$row_75$slope$pass
  cat(sprintf("Overall: %s (%.1fs)\n", if(results$row_75$pass) "PASS" else "FAIL", t_75))
} else {
  results$row_75 <- list(error = TRUE)
}

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - Multiscale Temporal Simulation Validation\n")
cat("=======================================================\n\n")

r <- results$row_75
if (!is.null(r$error) && r$error) {
  cat("Row 75: ERROR\n")
} else {
  cat(sprintf("Row 75: %s (%.2f SD, %.1fs)\n",
              if(r$pass) "PASS" else "FAIL", r$slope$diff_sd, r$time))
}

saveRDS(results, "benchmarks/results_sim_ms_temporal.rds")
cat("\nResults saved.\n")
