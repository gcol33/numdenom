# Validation of numdenom two-process models against custom joint Stan models
# Batch 5: Zero-Inflation and Hurdle variants
#
# Rows validated:
#   - Row 16: poisson_gamma + RE + ZI
#   - Row 17: poisson_gamma + RE + Hurdle
#   - Row 46: negbin_negbin + RE + ZI
#   - Row 47: negbin_negbin + RE + Hurdle

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

# Standard benchmark parameters
N_OBS <- 300  # More obs for ZI models to get enough zeros
N_ITER <- 1000
N_WARMUP <- 500
N_CHAINS <- 2
N_SITES <- 20

cat("=======================================================\n")
cat("Joint Model Validation Batch 5: ZI + Hurdle\n")
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

# Generate shared effects for consistent data generation
site <- factor(rep(1:N_SITES, length.out = N_OBS))

# Simulate shared site RE
re_true <- rnorm(N_SITES, 0, 0.3)

# ZI probability (around 20% excess zeros)
zi_prob_true <- 0.2

# =============================================================================
# Row 16: poisson_gamma + RE + ZI
# =============================================================================
cat("\n========== Row 16: poisson_gamma + RE + ZI ==========\n")

# Generate ZI-Poisson data
mu_pg <- 25 * exp(re_true[as.numeric(site)])
y_zi_pg <- ifelse(
  runif(N_OBS) < zi_prob_true,
  0,  # Structural zero
  rpois(N_OBS, lambda = mu_pg)  # Count process
)

df_pg_zi <- data.frame(
  y = y_zi_pg,
  effort = rgamma(N_OBS, shape = 5, rate = 0.1) * exp(re_true[as.numeric(site)]),
  x = rnorm(N_OBS),
  site = site
)
df_pg_zi$effort[df_pg_zi$effort < 0.01] <- 0.01

cat(sprintf("Zero rate in data: %.1f%%\n", 100 * mean(df_pg_zi$y == 0)))

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | effort ~ x + (1|site),
    data = df_pg_zi,
    family = ratiod_poisson_gamma(),
    zi = zi_poisson(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model_zi_pg <- cmdstan_model("stan/joint_pg_zi.stan")
cat("done\n")

stan_data_zi_pg <- list(
  N = N_OBS,
  y_num = df_pg_zi$y,
  y_denom = df_pg_zi$effort,
  p = 2,
  X = cbind(1, df_pg_zi$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_pg_zi$site)
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_zi_pg$sample(
    data = stan_data_zi_pg,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

results$row_16_beta <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)
print_result(results$row_16_beta)
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Row 17: poisson_gamma + RE + Hurdle
# =============================================================================
cat("\n========== Row 17: poisson_gamma + RE + Hurdle ==========\n")

# Generate Hurdle-Poisson data
theta_true <- 0.75  # P(Y > 0)
y_hurdle_pg <- ifelse(
  runif(N_OBS) < theta_true,
  # Truncated Poisson: sample until y > 0
  sapply(mu_pg, function(m) {
    y <- rpois(1, m)
    while(y == 0) y <- rpois(1, m)
    y
  }),
  0  # Structural zero
)

df_pg_hurdle <- data.frame(
  y = y_hurdle_pg,
  effort = rgamma(N_OBS, shape = 5, rate = 0.1) * exp(re_true[as.numeric(site)]),
  x = rnorm(N_OBS),
  site = site
)
df_pg_hurdle$effort[df_pg_hurdle$effort < 0.01] <- 0.01

cat(sprintf("Zero rate in data: %.1f%%\n", 100 * mean(df_pg_hurdle$y == 0)))

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | effort ~ x + (1|site),
    data = df_pg_hurdle,
    family = ratiod_poisson_gamma(),
    zi = hurdle_poisson(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model_hurdle_pg <- cmdstan_model("stan/joint_pg_hurdle.stan")
cat("done\n")

stan_data_hurdle_pg <- list(
  N = N_OBS,
  y_num = df_pg_hurdle$y,
  y_denom = df_pg_hurdle$effort,
  p = 2,
  X = cbind(1, df_pg_hurdle$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_pg_hurdle$site)
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_hurdle_pg$sample(
    data = stan_data_hurdle_pg,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

results$row_17_beta <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)
print_result(results$row_17_beta)
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Row 46: negbin_negbin + RE + ZI
# =============================================================================
cat("\n========== Row 46: negbin_negbin + RE + ZI ==========\n")

# Generate ZI-NegBin data
mu_nb <- 25 * exp(re_true[as.numeric(site)])
phi_nb <- 5
y_zi_nb <- ifelse(
  runif(N_OBS) < zi_prob_true,
  0,
  rnbinom(N_OBS, size = phi_nb, mu = mu_nb)
)

df_nb_zi <- data.frame(
  y = y_zi_nb,
  denom = rnbinom(N_OBS, size = 8, mu = 100 * exp(re_true[as.numeric(site)])),
  x = rnorm(N_OBS),
  site = site
)

cat(sprintf("Zero rate in data: %.1f%%\n", 100 * mean(df_nb_zi$y == 0)))

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x + (1|site),
    data = df_nb_zi,
    family = ratiod_negbin_negbin(),
    zi = zi_negbin(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model_zi_nb <- cmdstan_model("stan/joint_nb_zi.stan")
cat("done\n")

stan_data_zi_nb <- list(
  N = N_OBS,
  y_num = df_nb_zi$y,
  y_denom = df_nb_zi$denom,
  p = 2,
  X = cbind(1, df_nb_zi$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_nb_zi$site)
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_zi_nb$sample(
    data = stan_data_zi_nb,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

results$row_46_beta <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)
print_result(results$row_46_beta)
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Row 47: negbin_negbin + RE + Hurdle
# =============================================================================
cat("\n========== Row 47: negbin_negbin + RE + Hurdle ==========\n")

# Generate Hurdle-NegBin data
y_hurdle_nb <- ifelse(
  runif(N_OBS) < theta_true,
  sapply(mu_nb, function(m) {
    y <- rnbinom(1, size = phi_nb, mu = m)
    while(y == 0) y <- rnbinom(1, size = phi_nb, mu = m)
    y
  }),
  0
)

df_nb_hurdle <- data.frame(
  y = y_hurdle_nb,
  denom = rnbinom(N_OBS, size = 8, mu = 100 * exp(re_true[as.numeric(site)])),
  x = rnorm(N_OBS),
  site = site
)

cat(sprintf("Zero rate in data: %.1f%%\n", 100 * mean(df_nb_hurdle$y == 0)))

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x + (1|site),
    data = df_nb_hurdle,
    family = ratiod_negbin_negbin(),
    zi = hurdle_negbin(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model_hurdle_nb <- cmdstan_model("stan/joint_nb_hurdle.stan")
cat("done\n")

stan_data_hurdle_nb <- list(
  N = N_OBS,
  y_num = df_nb_hurdle$y,
  y_denom = df_nb_hurdle$denom,
  p = 2,
  X = cbind(1, df_nb_hurdle$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_nb_hurdle$site)
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_hurdle_nb$sample(
    data = stan_data_hurdle_nb,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

results$row_47_beta <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)
print_result(results$row_47_beta)
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Summary
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - BATCH 5\n")
cat("=======================================================\n")

n_pass <- sum(sapply(results, function(r) r$pass))
n_total <- length(results)

cat(sprintf("\nRow 16 (pg_zi):       %s\n", if(results$row_16_beta$pass) "PASS" else "FAIL"))
cat(sprintf("Row 17 (pg_hurdle):   %s\n", if(results$row_17_beta$pass) "PASS" else "FAIL"))
cat(sprintf("Row 46 (nb_zi):       %s\n", if(results$row_46_beta$pass) "PASS" else "FAIL"))
cat(sprintf("Row 47 (nb_hurdle):   %s\n", if(results$row_47_beta$pass) "PASS" else "FAIL"))

cat(sprintf("\nOverall: %d/%d parameters PASS (within 2 SE)\n", n_pass, n_total))

if (n_pass == n_total) {
  cat("\n** All posteriors match custom joint Stan models! **\n")
} else {
  cat("\n!! Some posteriors differ - investigate !!\n")
}

# Save results
saveRDS(results, "results_joint_batch5.rds")
cat("\nResults saved to results_joint_batch5.rds\n")

# Print update instructions
cat("\n=======================================================\n")
cat("UPDATE gradient_methods.md:\n")
cat("=======================================================\n")
cat("Add '✓Stan (joint)' to Notes column for these rows:\n")
for (name in names(results)) {
  if (results[[name]]$pass) {
    row_num <- gsub("row_([0-9]+).*", "\\1", name)
    cat(sprintf("  Row %s: PASS\n", row_num))
  }
}
