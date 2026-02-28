# Validation of numdenom two-process models against custom joint Stan models
# This script validates that numdenom posteriors match Stan posteriors
# for models where brms comparison is INVALID (poisson_gamma, negbin_negbin)
#
# Unlike brms with offset(), these Stan models correctly model BOTH num and denom
# as random variables with shared random effects.

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

# Standard benchmark parameters
N_OBS <- 200
N_ITER <- 1000
N_WARMUP <- 500
N_CHAINS <- 2
N_SITES <- 20

cat("=======================================================\n")
cat("Joint Model Validation: numdenom vs Custom Stan\n")
cat("=======================================================\n")
cat(sprintf("N=%d, iter=%d, warmup=%d, chains=%d\n\n", N_OBS, N_ITER, N_WARMUP, N_CHAINS))

# Helper function to compare posteriors
compare_posteriors <- function(nd_draws, stan_draws, param_name, threshold_se = 2) {
  nd_mean <- mean(nd_draws)
  nd_sd <- sd(nd_draws)
  stan_mean <- mean(stan_draws)
  stan_sd <- sd(stan_draws)

  # Combined SE for comparison
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

# =============================================================================
# Row 1: poisson_gamma (no RE)
# =============================================================================
cat("\n========== Row 1: poisson_gamma (no RE) ==========\n")

# Generate data - ensure effort > 0
df_pg <- data.frame(
  y = rpois(N_OBS, lambda = 30),
  effort = rgamma(N_OBS, shape = 5, rate = 0.1),
  x = rnorm(N_OBS)
)
# Ensure no zero effort
df_pg$effort[df_pg$effort < 0.01] <- 0.01

# Fit numdenom
cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | effort ~ x,
    data = df_pg,
    family = ratiod_poisson_gamma(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

# Compile and fit Stan model
cat("Compiling Stan model... ")
stan_model <- cmdstan_model("stan/joint_pg_base.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = df_pg$y,
  y_denom = df_pg$effort,
  p = 2,
  X = cbind(1, df_pg$x)
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model$sample(
    data = stan_data,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

# Extract and compare posteriors
draws_nd <- fit_nd$draws
draws_stan <- fit_stan$draws(format = "df")

# Compare beta_num[2] (slope for x)
result_1 <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)

cat(sprintf("\n%s:\n", result_1$param))
cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", result_1$nd_mean, result_1$nd_sd))
cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", result_1$stan_mean, result_1$stan_sd))
cat(sprintf("  Diff: %.4f (%.2f SE) => %s\n",
            result_1$diff, result_1$ratio, if(result_1$pass) "PASS" else "FAIL"))
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Row 2: poisson_gamma with RE
# =============================================================================
cat("\n========== Row 2: poisson_gamma + RE ==========\n")

# Generate data with site effects
site <- factor(rep(1:N_SITES, length.out = N_OBS))
site_effects <- rnorm(N_SITES, 0, 0.5)

df_pg_re <- data.frame(
  y = rpois(N_OBS, lambda = 30 * exp(site_effects[as.numeric(site)])),
  effort = rgamma(N_OBS, shape = 5, rate = 0.1) * exp(site_effects[as.numeric(site)]),
  x = rnorm(N_OBS),
  site = site
)
# Ensure no zero effort
df_pg_re$effort[df_pg_re$effort < 0.01] <- 0.01

# Fit numdenom
cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | effort ~ x + (1|site),
    data = df_pg_re,
    family = ratiod_poisson_gamma(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

# Compile and fit Stan model
cat("Compiling Stan model... ")
stan_model_re <- cmdstan_model("stan/joint_pg_re.stan")
cat("done\n")

stan_data_re <- list(
  N = N_OBS,
  y_num = df_pg_re$y,
  y_denom = df_pg_re$effort,
  p = 2,
  X = cbind(1, df_pg_re$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_pg_re$site)
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_re$sample(
    data = stan_data_re,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

# Compare posteriors
draws_nd <- fit_nd$draws
draws_stan <- fit_stan$draws(format = "df")

result_2_beta <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)

result_2_sigma <- compare_posteriors(
  draws_nd[, "sigma_re"],
  draws_stan$sigma_re,
  "sigma_re"
)

cat(sprintf("\n%s:\n", result_2_beta$param))
cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", result_2_beta$nd_mean, result_2_beta$nd_sd))
cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", result_2_beta$stan_mean, result_2_beta$stan_sd))
cat(sprintf("  Diff: %.4f (%.2f SE) => %s\n",
            result_2_beta$diff, result_2_beta$ratio, if(result_2_beta$pass) "PASS" else "FAIL"))

cat(sprintf("\n%s:\n", result_2_sigma$param))
cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", result_2_sigma$nd_mean, result_2_sigma$nd_sd))
cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", result_2_sigma$stan_mean, result_2_sigma$stan_sd))
cat(sprintf("  Diff: %.4f (%.2f SE) => %s\n",
            result_2_sigma$diff, result_2_sigma$ratio, if(result_2_sigma$pass) "PASS" else "FAIL"))
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Row 31: negbin_negbin (no RE)
# =============================================================================
cat("\n========== Row 31: negbin_negbin (no RE) ==========\n")

# Generate data
df_nb <- data.frame(
  y = rnbinom(N_OBS, size = 5, mu = 30),
  denom = rnbinom(N_OBS, size = 8, mu = 100),
  x = rnorm(N_OBS)
)

# Fit numdenom
cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x,
    data = df_nb,
    family = ratiod_negbin_negbin(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

# Compile and fit Stan model
cat("Compiling Stan model... ")
stan_model_nb <- cmdstan_model("stan/joint_nb_base.stan")
cat("done\n")

stan_data_nb <- list(
  N = N_OBS,
  y_num = df_nb$y,
  y_denom = df_nb$denom,
  p = 2,
  X = cbind(1, df_nb$x)
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_nb$sample(
    data = stan_data_nb,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

# Compare posteriors
draws_nd <- fit_nd$draws
draws_stan <- fit_stan$draws(format = "df")

result_31 <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)

cat(sprintf("\n%s:\n", result_31$param))
cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", result_31$nd_mean, result_31$nd_sd))
cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", result_31$stan_mean, result_31$stan_sd))
cat(sprintf("  Diff: %.4f (%.2f SE) => %s\n",
            result_31$diff, result_31$ratio, if(result_31$pass) "PASS" else "FAIL"))
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Row 32: negbin_negbin with RE
# =============================================================================
cat("\n========== Row 32: negbin_negbin + RE ==========\n")

# Generate data with site effects
site_effects_nb <- rnorm(N_SITES, 0, 0.5)

df_nb_re <- data.frame(
  y = rnbinom(N_OBS, size = 5, mu = 30 * exp(site_effects_nb[as.numeric(site)])),
  denom = rnbinom(N_OBS, size = 8, mu = 100 * exp(site_effects_nb[as.numeric(site)])),
  x = rnorm(N_OBS),
  site = site
)

# Fit numdenom
cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x + (1|site),
    data = df_nb_re,
    family = ratiod_negbin_negbin(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

# Compile and fit Stan model
cat("Compiling Stan model... ")
stan_model_nb_re <- cmdstan_model("stan/joint_nb_re.stan")
cat("done\n")

stan_data_nb_re <- list(
  N = N_OBS,
  y_num = df_nb_re$y,
  y_denom = df_nb_re$denom,
  p = 2,
  X = cbind(1, df_nb_re$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_nb_re$site)
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_nb_re$sample(
    data = stan_data_nb_re,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

# Compare posteriors
draws_nd <- fit_nd$draws
draws_stan <- fit_stan$draws(format = "df")

result_32_beta <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)

result_32_sigma <- compare_posteriors(
  draws_nd[, "sigma_re"],
  draws_stan$sigma_re,
  "sigma_re"
)

cat(sprintf("\n%s:\n", result_32_beta$param))
cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", result_32_beta$nd_mean, result_32_beta$nd_sd))
cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", result_32_beta$stan_mean, result_32_beta$stan_sd))
cat(sprintf("  Diff: %.4f (%.2f SE) => %s\n",
            result_32_beta$diff, result_32_beta$ratio, if(result_32_beta$pass) "PASS" else "FAIL"))

cat(sprintf("\n%s:\n", result_32_sigma$param))
cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", result_32_sigma$nd_mean, result_32_sigma$nd_sd))
cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", result_32_sigma$stan_mean, result_32_sigma$stan_sd))
cat(sprintf("  Diff: %.4f (%.2f SE) => %s\n",
            result_32_sigma$diff, result_32_sigma$ratio, if(result_32_sigma$pass) "PASS" else "FAIL"))
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Summary
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY\n")
cat("=======================================================\n")

results <- list(
  row_1 = result_1,
  row_2_beta = result_2_beta,
  row_2_sigma = result_2_sigma,
  row_31 = result_31,
  row_32_beta = result_32_beta,
  row_32_sigma = result_32_sigma
)

n_pass <- sum(sapply(results, function(r) r$pass))
n_total <- length(results)

cat(sprintf("\nRow  1 (pg_base):     %s\n", if(result_1$pass) "PASS" else "FAIL"))
cat(sprintf("Row  2 (pg_re):       %s (beta), %s (sigma)\n",
            if(result_2_beta$pass) "PASS" else "FAIL",
            if(result_2_sigma$pass) "PASS" else "FAIL"))
cat(sprintf("Row 31 (nb_base):     %s\n", if(result_31$pass) "PASS" else "FAIL"))
cat(sprintf("Row 32 (nb_re):       %s (beta), %s (sigma)\n",
            if(result_32_beta$pass) "PASS" else "FAIL",
            if(result_32_sigma$pass) "PASS" else "FAIL"))

cat(sprintf("\nOverall: %d/%d parameters PASS (within 2 SE)\n", n_pass, n_total))

if (n_pass == n_total) {
  cat("\n✓ All posteriors match custom joint Stan models!\n")
  cat("  This validates that numdenom correctly models two-process data.\n")
} else {
  cat("\n✗ Some posteriors differ - investigate!\n")
}
