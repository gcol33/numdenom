# Validation of numdenom two-process models against custom joint Stan models
# Batch 7: Random slopes (rows 3, 33)
#
# Rows validated:
#   - Row 3: poisson_gamma + random slopes
#   - Row 33: negbin_negbin + random slopes (TODO)

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

# Standard benchmark parameters
N_OBS <- 200
N_ITER <- 4000
N_WARMUP <- 1500
N_CHAINS <- 4
N_SITES <- 20

cat("=======================================================\n")
cat("Joint Model Validation Batch 7: Random Slopes\n")
cat("=======================================================\n")
cat(sprintf("N=%d, iter=%d, warmup=%d, chains=%d\n", N_OBS, N_ITER, N_WARMUP, N_CHAINS))
cat(sprintf("sites=%d\n\n", N_SITES))

# Helper function to compare posteriors
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
    nd_mean = nd_mean,
    nd_sd = nd_sd,
    stan_mean = stan_mean,
    stan_sd = stan_sd,
    diff = diff,
    ratio = ratio,
    pass = pass
  )
}

print_result <- function(result) {
  cat(sprintf("\n%s:\n", result$param))
  cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", result$nd_mean, result$nd_sd))
  cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", result$stan_mean, result$stan_sd))
  cat(sprintf("  Diff: %.4f (%.2f SE) => %s\n",
              result$diff, result$ratio, if(result$pass) "PASS" else "FAIL"))
}

results <- list()

# Generate shared data
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)

# =============================================================================
# Row 3: poisson_gamma + random slopes
# =============================================================================
cat("\n========== Row 3: poisson_gamma + random slopes ==========\n")

# Generate data WITH known random effects
true_sigma_int <- 0.4
true_sigma_slope <- 0.2
true_rho <- 0.3
true_beta_num <- c(3, 0.5)
true_beta_denom <- c(4, 0.2)

# Generate correlated RE
L <- matrix(c(1, 0, true_rho, sqrt(1-true_rho^2)), 2, 2)
true_re <- matrix(0, N_SITES, 2)
for (g in 1:N_SITES) {
  z <- rnorm(2)
  true_re[g, ] <- c(true_sigma_int, true_sigma_slope) * (L %*% z)
}

# Compute linear predictors
eta_num <- true_beta_num[1] + true_beta_num[2] * x
eta_denom <- true_beta_denom[1] + true_beta_denom[2] * x
for (i in 1:N_OBS) {
  g <- as.integer(site[i])
  re_effect <- true_re[g, 1] + true_re[g, 2] * x[i]
  eta_num[i] <- eta_num[i] + re_effect
  eta_denom[i] <- eta_denom[i] + re_effect  # SHARED
}

df_pg_slopes <- data.frame(
  y = rpois(N_OBS, exp(eta_num)),
  effort = rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_denom)),
  x = x,
  site = site
)
df_pg_slopes$effort[df_pg_slopes$effort < 0.01] <- 0.01

cat(sprintf("True params: beta_num=[%.2f, %.2f], beta_denom=[%.2f, %.2f]\n",
            true_beta_num[1], true_beta_num[2], true_beta_denom[1], true_beta_denom[2]))
cat(sprintf("True RE: sigma_int=%.2f, sigma_slope=%.2f, rho=%.2f\n",
            true_sigma_int, true_sigma_slope, true_rho))

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | effort ~ x + (1 + x|site),
    data = df_pg_slopes,
    family = ratiod_poisson_gamma(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model_pg_slopes <- cmdstan_model("stan/joint_pg_slopes.stan")
cat("done\n")

stan_data_pg_slopes <- list(
  N = N_OBS,
  y_num = df_pg_slopes$y,
  y_denom = df_pg_slopes$effort,
  p = 2,
  X = cbind(1, df_pg_slopes$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_pg_slopes$site),
  x_slope = df_pg_slopes$x
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_pg_slopes$sample(
    data = stan_data_pg_slopes,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE,
    adapt_delta = 0.95  # Increase for correlated RE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

# Compare multiple parameters
cat("\n--- Parameter Comparisons ---\n")

results$row_3_beta1 <- compare_posteriors(
  draws_nd[, "beta_num[1]"],
  draws_stan$`beta_num[1]`,
  "beta_num[1] (intercept)"
)
print_result(results$row_3_beta1)

results$row_3_beta2 <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)
print_result(results$row_3_beta2)

results$row_3_beta_denom1 <- compare_posteriors(
  draws_nd[, "beta_denom[1]"],
  draws_stan$`beta_denom[1]`,
  "beta_denom[1] (intercept)"
)
print_result(results$row_3_beta_denom1)

results$row_3_beta_denom2 <- compare_posteriors(
  draws_nd[, "beta_denom[2]"],
  draws_stan$`beta_denom[2]`,
  "beta_denom[2] (x slope)"
)
print_result(results$row_3_beta_denom2)

# Check if shape parameter exists
if ("log_shape" %in% colnames(draws_nd)) {
  results$row_3_shape <- compare_posteriors(
    exp(draws_nd[, "log_shape"]),
    draws_stan$shape,
    "shape (gamma)"
  )
  print_result(results$row_3_shape)
}

cat(sprintf("\n  numdenom time: %.1fs\n", t_nd))
cat(sprintf("  Stan time: %.1fs\n", t_stan))
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Summary
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - BATCH 7 (poisson_gamma slopes only)\n")
cat("=======================================================\n")

n_pass <- sum(sapply(results, function(r) r$pass))
n_total <- length(results)

for (name in names(results)) {
  cat(sprintf("%s: %s\n", name, if(results[[name]]$pass) "PASS" else "FAIL"))
}

cat(sprintf("\nOverall: %d/%d parameters PASS (within 2 SE)\n", n_pass, n_total))

if (n_pass == n_total) {
  cat("\n** All posteriors match custom joint Stan model! **\n")
  cat("Row 3 validated: poisson_gamma + random slopes\n")
} else {
  cat("\n!! Some posteriors differ - investigate !!\n")
}

# Save results
saveRDS(results, "results_joint_batch7.rds")
cat("\nResults saved to results_joint_batch7.rds\n")
