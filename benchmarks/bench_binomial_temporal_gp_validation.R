# Binomial Temporal GP Validation - Row 74
# Uses single-outcome Stan model (trials are fixed)

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

# Use smaller parameters for faster testing (row 74 already has H=128.6s benchmark)
N_OBS <- 150
N_ITER <- 1500
N_WARMUP <- 750
N_CHAINS <- 2
N_TIMES <- 8

cat("=======================================================\n")
cat("Binomial Temporal GP Validation: Row 74\n")
cat("=======================================================\n\n")

compare_posteriors <- function(nd_draws, stan_draws, param_name, threshold_se = 2) {
  nd_mean <- mean(nd_draws)
  nd_sd <- sd(nd_draws)
  stan_mean <- mean(stan_draws)
  stan_sd <- sd(stan_draws)

  se_combined <- sqrt(nd_sd^2 / length(nd_draws) + stan_sd^2 / length(stan_draws))
  diff <- abs(nd_mean - stan_mean)
  ratio <- diff / se_combined
  pass <- ratio < threshold_se

  list(
    param = param_name,
    nd_mean = nd_mean, nd_sd = nd_sd,
    stan_mean = stan_mean, stan_sd = stan_sd,
    diff = diff, ratio = ratio, pass = pass
  )
}

print_result <- function(result) {
  cat(sprintf("  %s: nd=%.4f (SD=%.4f), Stan=%.4f (SD=%.4f), diff=%.4f (%.2f SE) => %s\n",
              result$param, result$nd_mean, result$nd_sd,
              result$stan_mean, result$stan_sd,
              result$diff, result$ratio, if(result$pass) "PASS" else "FAIL"))
}

# Setup data
x <- rnorm(N_OBS)
time <- rep(1:N_TIMES, length.out = N_OBS)
trials <- sample(10:50, N_OBS, replace = TRUE)

# Scale time for Stan (mean 0, sd 1)
time_scaled <- (time - mean(time)) / sd(time)
time_unique_scaled <- (1:N_TIMES - mean(1:N_TIMES)) / sd(1:N_TIMES)

# True GP effect (for data generation)
true_sigma_gp <- 0.5
true_phi_gp <- 2.0
dist_mat <- as.matrix(dist(1:N_TIMES))
K <- true_sigma_gp^2 * exp(-dist_mat / true_phi_gp)
gp_effects_true <- MASS::mvrnorm(1, rep(0, N_TIMES), K + diag(1e-6, N_TIMES))
gp_by_obs <- gp_effects_true[time]

# Linear predictor (logit scale for binomial)
eta <- 0.5 + 0.3 * x + gp_by_obs
prob <- plogis(eta)

# Generate binomial data
y <- rbinom(N_OBS, size = trials, prob = prob)

df <- data.frame(
  y = y,
  trials = trials,
  x = x,
  time = time
)

cat("Data summary:\n")
cat(sprintf("  N=%d, T=%d, mean(trials)=%.1f\n", N_OBS, N_TIMES, mean(trials)))
cat(sprintf("  True: beta0=0.5, beta1=0.3, sigma_gp=%.1f, phi_gp=%.1f\n\n",
            true_sigma_gp, true_phi_gp))

# =============================================================================
# Row 74: binomial + temporal GP
# =============================================================================
cat("========== Row 74: binomial + temporal GP ==========\n")

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | trials ~ x, data = df,
    family = ratiod_binomial(),
    temporal = temporal_gp(time_var = "time", cov = "exponential"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/temporal_gp_binomial.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  T = N_TIMES,
  y = y,
  trials = trials,
  x = x,
  time_idx = time,
  time_values = time_unique_scaled,
  sigma2_prior_U = 1.0,
  sigma2_prior_alpha = 0.01,
  phi_prior_lower = 0.01,
  phi_prior_upper = 10.0
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model$sample(
    data = stan_data,
    iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
    chains = N_CHAINS, parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE, adapt_delta = 0.95
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_stan))

# Extract draws
draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

# Compare posteriors
# numdenom uses beta_num[1] for intercept, beta_num[2] for slope (binomial treats trials as denom)
# Stan uses beta_0 for intercept, beta_1 for slope
nd_intercept_col <- grep("^beta_num\\[1\\]$", colnames(draws_nd), value = TRUE)[1]
nd_slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_nd), value = TRUE)[1]

results <- list()

cat("\nComparing posteriors:\n")

results$intercept <- compare_posteriors(
  draws_nd[, nd_intercept_col], draws_stan$beta_0, "beta[1] (intercept)"
)
print_result(results$intercept)

results$slope <- compare_posteriors(
  draws_nd[, nd_slope_col], draws_stan$beta_1, "beta[2] (slope)"
)
print_result(results$slope)

# GP parameters - numdenom uses sigma2_temporal_gp (variance), Stan uses sigma_gp (SD)
nd_sigma2_col <- grep("sigma2_temporal_gp", colnames(draws_nd), value = TRUE)[1]
nd_phi_col <- grep("phi_temporal_gp", colnames(draws_nd), value = TRUE)[1]

if (!is.na(nd_sigma2_col)) {
  # numdenom stores variance, Stan stores SD - compare sqrt(sigma2) to sigma
  nd_sigma <- sqrt(draws_nd[, nd_sigma2_col])
  results$sigma_gp <- compare_posteriors(
    nd_sigma, draws_stan$sigma_gp, "sigma_gp"
  )
  print_result(results$sigma_gp)
}

if (!is.na(nd_phi_col)) {
  results$phi_gp <- compare_posteriors(
    draws_nd[, nd_phi_col], draws_stan$phi_gp, "phi_gp"
  )
  print_result(results$phi_gp)
}

cat(sprintf("\nTiming: numdenom=%.1fs, Stan=%.1fs, speedup=%.1fx\n",
            t_nd, t_stan, t_stan / t_nd))

# Check for divergences
diag <- fit_stan$diagnostic_summary()
div_pct <- 100 * sum(diag$num_divergent) / (N_CHAINS * (N_ITER - N_WARMUP))
cat(sprintf("Stan divergences: %.1f%%\n", div_pct))

# numdenom diagnostics
nd_diag <- fit_nd$sampler_diagnostics
if (!is.null(nd_diag)) {
  nd_div <- sum(nd_diag[, , "divergent__"])
  nd_div_pct <- 100 * nd_div / (N_CHAINS * (N_ITER - N_WARMUP))
  cat(sprintf("numdenom divergences: %.1f%%\n", nd_div_pct))
}

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - Binomial Temporal GP Validation (Row 74)\n")
cat("=======================================================\n\n")

all_pass <- results$intercept$pass && results$slope$pass
cat(sprintf("Overall: %s\n", if(all_pass) "PASS" else "FAIL"))
cat(sprintf("  Intercept: %.2f SE (%s)\n", results$intercept$ratio,
            if(results$intercept$pass) "PASS" else "FAIL"))
cat(sprintf("  Slope: %.2f SE (%s)\n", results$slope$ratio,
            if(results$slope$pass) "PASS" else "FAIL"))

if (!is.null(results$sigma_gp)) {
  cat(sprintf("  sigma_gp: %.2f SE (%s)\n", results$sigma_gp$ratio,
              if(results$sigma_gp$pass) "PASS" else "FAIL"))
}
if (!is.null(results$phi_gp)) {
  cat(sprintf("  phi_gp: %.2f SE (%s)\n", results$phi_gp$ratio,
              if(results$phi_gp$pass) "PASS" else "FAIL"))
}

cat(sprintf("\nTiming: numdenom=%.1fs, Stan=%.1fs\n", t_nd, t_stan))

saveRDS(list(results = results, time_nd = t_nd, time_stan = t_stan),
        "benchmarks/results_binomial_temporal_gp.rds")
cat("\nResults saved to benchmarks/results_binomial_temporal_gp.rds\n")

if (all_pass) {
  cat("\nUPDATE gradient_methods.md:\n")
  cat("  Row 74: Change to 'Stan (identifiable, <2SE)' in Notes column\n")
}
