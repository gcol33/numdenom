# Validation of negbin_gamma family against custom joint Stan models
# Rows 108 (base) and 109 (+RE) in gradient_methods.md

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

# Standard benchmark parameters
N_OBS <- 500
N_ITER <- 1000
N_WARMUP <- 500
N_CHAINS <- 2
N_SITES <- 50

cat("=======================================================\n")
cat("NegBin-Gamma Validation: numdenom vs Custom Stan\n")
cat("=======================================================\n")
cat(sprintf("N=%d, iter=%d, warmup=%d, chains=%d\n\n", N_OBS, N_ITER, N_WARMUP, N_CHAINS))

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

print_result <- function(r, t_nd, t_stan) {
  cat(sprintf("  %s:\n", r$param))
  cat(sprintf("    numdenom: %.4f (SD=%.4f)\n", r$nd_mean, r$nd_sd))
  cat(sprintf("    Stan:     %.4f (SD=%.4f)\n", r$stan_mean, r$stan_sd))
  cat(sprintf("    Diff: %.4f (%.2f SE) => %s\n", r$diff, r$ratio,
              if(r$pass) "PASS" else "FAIL"))
}

# =============================================================================
# Row 108: negbin_gamma (no RE)
# =============================================================================
cat("\n========== Row 108: negbin_gamma (no RE) ==========\n")

# Simulate data using numdenom
sim <- sim_ratiod(
  n = N_OBS,
  family = ratiod_negbin_gamma(),
  beta_num = c(2, 0.3),
  beta_denom = c(1, 0.2),
  phi_num = 5,
  phi_denom = 3,
  n_groups = 1,
  seed = 108
)

cat(sprintf("True params: beta_num=(%s), beta_denom=(%s), phi_num=5, phi_denom=3\n",
            paste(c(2, 0.3), collapse=", "), paste(c(1, 0.2), collapse=", ")))

# Fit numdenom
cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y_num | y_denom ~ x1,
    data = sim$data,
    family = ratiod_negbin_gamma(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

# Compile and fit Stan model
cat("Compiling Stan model... ")
stan_model <- cmdstan_model("stan/joint_nb_gamma_base.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = as.integer(sim$data$y_num),
  y_denom = as.numeric(sim$data$y_denom),
  p = 2,
  X = cbind(1, sim$data$x1)
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

# Extract draws
draws_nd <- fit_nd$draws
draws_stan <- fit_stan$draws(format = "df")

# Compare all key parameters
results_108 <- list(
  compare_posteriors(draws_nd[, "beta_num[1]"], draws_stan$`beta_num[1]`, "beta_num[1] (intercept)"),
  compare_posteriors(draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`, "beta_num[2] (x slope)"),
  compare_posteriors(draws_nd[, "beta_denom[1]"], draws_stan$`beta_denom[1]`, "beta_denom[1] (intercept)"),
  compare_posteriors(draws_nd[, "beta_denom[2]"], draws_stan$`beta_denom[2]`, "beta_denom[2] (x slope)"),
  compare_posteriors(draws_nd[, "phi_num"], draws_stan$phi_num, "phi_num (NB overdispersion)"),
  compare_posteriors(draws_nd[, "phi_denom"], draws_stan$phi_denom, "phi_denom (Gamma shape)")
)

cat("\nResults:\n")
for (r in results_108) print_result(r, t_nd, t_stan)
all_pass_108 <- all(sapply(results_108, function(r) r$pass))
cat(sprintf("\nRow 108 overall: %s\n", if(all_pass_108) "PASS" else "FAIL"))
cat(sprintf("Timing: numdenom=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan / t_nd))

# =============================================================================
# Row 109: negbin_gamma + RE
# =============================================================================
cat("\n\n========== Row 109: negbin_gamma + RE ==========\n")

# Simulate data with RE
sim_re <- sim_ratiod(
  n = N_OBS,
  family = ratiod_negbin_gamma(),
  beta_num = c(2, 0.3),
  beta_denom = c(1, 0.2),
  sigma_re = 0.5,
  phi_num = 5,
  phi_denom = 3,
  n_groups = N_SITES,
  seed = 109
)

cat(sprintf("True params: beta_num=(%s), beta_denom=(%s), sigma_re=0.5, phi_num=5, phi_denom=3\n",
            paste(c(2, 0.3), collapse=", "), paste(c(1, 0.2), collapse=", ")))

# Fit numdenom
cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd_re <- ratiod(
    y_num | y_denom ~ x1 + (1|group),
    data = sim_re$data,
    family = ratiod_negbin_gamma(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

# Compile and fit Stan model
cat("Compiling Stan model... ")
stan_model_re <- cmdstan_model("stan/joint_nb_gamma_re.stan")
cat("done\n")

stan_data_re <- list(
  N = N_OBS,
  y_num = as.integer(sim_re$data$y_num),
  y_denom = as.numeric(sim_re$data$y_denom),
  p = 2,
  X = cbind(1, sim_re$data$x1),
  n_groups = N_SITES,
  group_idx = as.integer(sim_re$data$group)
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan_re <- stan_model_re$sample(
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

# Extract draws
draws_nd_re <- fit_nd_re$draws
draws_stan_re <- fit_stan_re$draws(format = "df")

# Compare all key parameters
results_109 <- list(
  compare_posteriors(draws_nd_re[, "beta_num[1]"], draws_stan_re$`beta_num[1]`, "beta_num[1] (intercept)"),
  compare_posteriors(draws_nd_re[, "beta_num[2]"], draws_stan_re$`beta_num[2]`, "beta_num[2] (x slope)"),
  compare_posteriors(draws_nd_re[, "beta_denom[1]"], draws_stan_re$`beta_denom[1]`, "beta_denom[1] (intercept)"),
  compare_posteriors(draws_nd_re[, "beta_denom[2]"], draws_stan_re$`beta_denom[2]`, "beta_denom[2] (x slope)"),
  compare_posteriors(draws_nd_re[, "phi_num"], draws_stan_re$phi_num, "phi_num (NB overdispersion)"),
  compare_posteriors(draws_nd_re[, "phi_denom"], draws_stan_re$phi_denom, "phi_denom (Gamma shape)"),
  compare_posteriors(draws_nd_re[, "sigma_re"], draws_stan_re$sigma_re, "sigma_re")
)

cat("\nResults:\n")
for (r in results_109) print_result(r, t_nd, t_stan)
all_pass_109 <- all(sapply(results_109, function(r) r$pass))
cat(sprintf("\nRow 109 overall: %s\n", if(all_pass_109) "PASS" else "FAIL"))
cat(sprintf("Timing: numdenom=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan / t_nd))

# =============================================================================
# Summary
# =============================================================================
cat("\n\n=======================================================\n")
cat("SUMMARY\n")
cat("=======================================================\n")
cat(sprintf("Row 108 (negbin_gamma base):  %s\n", if(all_pass_108) "PASS" else "FAIL"))
cat(sprintf("Row 109 (negbin_gamma + RE):  %s\n", if(all_pass_109) "PASS" else "FAIL"))
cat("=======================================================\n")
