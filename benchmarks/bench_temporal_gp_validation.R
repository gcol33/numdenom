# Temporal GP Validation - Rows 14, 44 (poisson_gamma and negbin_negbin with temporal GP)
# Uses joint Stan models for proper validation

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

N_OBS <- 200
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2
N_TIMES <- 10

cat("=======================================================\n")
cat("Temporal GP Validation: Rows 14, 44\n")
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

# Linear predictors
eta_num <- 2 + 0.3 * x + gp_by_obs
eta_denom <- 4 + 0.2 * x + gp_by_obs

results <- list()

# =============================================================================
# Row 44: negbin_negbin + temporal GP
# =============================================================================
cat("\n========== Row 44: negbin_negbin + temporal GP ==========\n")

df_nb_gp <- data.frame(
  y = rnbinom(N_OBS, mu = exp(eta_num), size = 5),
  denom = rnbinom(N_OBS, mu = exp(eta_denom), size = 10),
  x = x, time = time
)
df_nb_gp$denom[df_nb_gp$denom == 0] <- 1

cat("Fitting numdenom (temporal GP)... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x, data = df_nb_gp,
    family = ratiod_negbin_negbin(),
    temporal = temporal_gp(time_var = "time", cov = "exponential"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/temporal_gp_nb_joint.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  T = N_TIMES,
  y_num = df_nb_gp$y,
  y_denom = df_nb_gp$denom,
  x = df_nb_gp$x,
  time_idx = df_nb_gp$time,
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

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

nd_beta_col <- grep("^beta_num\\[2\\]$", colnames(draws_nd), value = TRUE)[1]
results$row_44 <- list(
  beta_num_1 = compare_posteriors(
    draws_nd[, nd_beta_col], draws_stan$beta_num_1, "beta_num[2] (slope)"
  ),
  time_nd = t_nd, time_stan = t_stan
)
print_result(results$row_44$beta_num_1)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan/t_nd))

# Check for divergences
diag <- fit_stan$diagnostic_summary()
div_pct <- 100 * sum(diag$num_divergent) / (N_CHAINS * (N_ITER - N_WARMUP))
cat(sprintf("  Stan divergences: %.1f%%\n", div_pct))

# =============================================================================
# Row 14: poisson_gamma + temporal GP
# =============================================================================
cat("\n========== Row 14: poisson_gamma + temporal GP ==========\n")

df_pg_gp <- data.frame(
  y = rpois(N_OBS, exp(eta_num)),
  denom = rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_denom)),
  x = x, time = time
)
df_pg_gp$denom[df_pg_gp$denom < 0.01] <- 0.01

cat("Fitting numdenom (temporal GP)... ")
t_nd_pg <- system.time({
  fit_nd_pg <- ratiod(
    y | denom ~ x, data = df_pg_gp,
    family = ratiod_poisson_gamma(),
    temporal = temporal_gp(time_var = "time", cov = "exponential"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_nd_pg))

cat("Compiling Stan model... ")
stan_model_pg <- cmdstan_model("benchmarks/stan/temporal_gp_pg_joint.stan")
cat("done\n")

stan_data_pg <- list(
  N = N_OBS,
  T = N_TIMES,
  y_num = df_pg_gp$y,
  y_denom = df_pg_gp$denom,
  x = df_pg_gp$x,
  time_idx = df_pg_gp$time,
  time_values = time_unique_scaled,
  sigma2_prior_U = 1.0,
  sigma2_prior_alpha = 0.01,
  phi_prior_lower = 0.01,
  phi_prior_upper = 10.0
)

cat("Fitting Stan model... ")
t_stan_pg <- system.time({
  fit_stan_pg <- stan_model_pg$sample(
    data = stan_data_pg,
    iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
    chains = N_CHAINS, parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE, adapt_delta = 0.95
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_stan_pg))

draws_nd_pg <- as.matrix(fit_nd_pg$draws)
draws_stan_pg <- fit_stan_pg$draws(format = "df")

nd_beta_col_pg <- grep("^beta_num\\[2\\]$", colnames(draws_nd_pg), value = TRUE)[1]
results$row_14 <- list(
  beta_num_1 = compare_posteriors(
    draws_nd_pg[, nd_beta_col_pg], draws_stan_pg$beta_num_1, "beta_num[2] (slope)"
  ),
  time_nd = t_nd_pg, time_stan = t_stan_pg
)
print_result(results$row_14$beta_num_1)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd_pg, t_stan_pg, t_stan_pg/t_nd_pg))

# Check for divergences
diag_pg <- fit_stan_pg$diagnostic_summary()
div_pct_pg <- 100 * sum(diag_pg$num_divergent) / (N_CHAINS * (N_ITER - N_WARMUP))
cat(sprintf("  Stan divergences: %.1f%%\n", div_pct_pg))

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - Temporal GP Validation (Rows 14, 44)\n")
cat("=======================================================\n\n")

cat(sprintf("%-8s %-30s %10s %10s %8s %s\n",
            "Row", "Model", "numdenom", "Stan", "Speedup", "Status"))
cat(paste(rep("-", 80), collapse = ""), "\n")

r44 <- results$row_44
cat(sprintf("%-8s %-30s %10.1fs %10.1fs %8.1fx %s\n",
            "44", "nb + temporal GP", r44$time_nd, r44$time_stan, r44$time_stan / r44$time_nd,
            if(r44$beta_num_1$pass) "PASS" else "FAIL"))

r14 <- results$row_14
cat(sprintf("%-8s %-30s %10.1fs %10.1fs %8.1fx %s\n",
            "14", "pg + temporal GP", r14$time_nd, r14$time_stan, r14$time_stan / r14$time_nd,
            if(r14$beta_num_1$pass) "PASS" else "FAIL"))

cat(paste(rep("-", 80), collapse = ""), "\n")

saveRDS(results, "benchmarks/results_temporal_gp_validation.rds")
cat("\nResults saved to benchmarks/results_temporal_gp_validation.rds\n")

cat("\n\nUPDATE gradient_methods.md:\n")
if (r44$beta_num_1$pass) cat("  Row 44: PASS - add '✓Stan (joint)' to Notes\n")
if (r14$beta_num_1$pass) cat("  Row 14: PASS - add '✓Stan (joint)' to Notes\n")
