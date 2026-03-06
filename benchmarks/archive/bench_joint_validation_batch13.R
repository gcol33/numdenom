# Validation of numdenom models against custom joint Stan models
# Batch 13: Temporal GP and HSGP with matching parameterization
#
# Rows validated:
#   - Row 44: negbin_negbin + temporal_gp (updated Stan model with scaled coords)
#   - Row 14: poisson_gamma + temporal_gp
#   - Row 38: negbin_negbin + HSGP
#   - Row 8: poisson_gamma + HSGP

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

# Standard benchmark parameters
N_OBS <- 200
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2
N_SITES <- 20
N_TIMES <- 15

cat("=======================================================\n")
cat("Joint Model Validation Batch 13: Temporal GP & HSGP\n")
cat("=======================================================\n")
cat(sprintf("N=%d, iter=%d, warmup=%d, chains=%d\n", N_OBS, N_ITER, N_WARMUP, N_CHAINS))
cat(sprintf("sites=%d, times=%d\n\n", N_SITES, N_TIMES))

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
  cat(sprintf("  %s: nd=%.4f (SD=%.4f), Stan=%.4f (SD=%.4f), diff=%.4f (%.2f SE) => %s\n",
              result$param, result$nd_mean, result$nd_sd,
              result$stan_mean, result$stan_sd,
              result$diff, result$ratio, if(result$pass) "PASS" else "FAIL"))
}

# =============================================================================
# DATA SETUP
# =============================================================================

site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)
time <- rep(1:N_TIMES, length.out = N_OBS)

# Spatial coordinates
n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = seq(0, 1, length.out = n_side),
                    lat = seq(0, 1, length.out = n_side))[1:N_SITES, ]
site_int <- as.integer(site)
lon <- grid$lon[site_int]
lat <- grid$lat[site_int]

# Generate data
eta_num <- 2 + 0.3 * x
eta_denom <- 4 + 0.2 * x

results <- list()

# =============================================================================
# Row 44: negbin_negbin + temporal_gp (Updated Stan model)
# =============================================================================
cat("\n========== Row 44: negbin_negbin + temporal_gp ==========\n")

df_nb_gp <- data.frame(
  y = rnbinom(N_OBS, mu = exp(eta_num), size = 5),
  denom = rnbinom(N_OBS, mu = exp(eta_denom), size = 10),
  x = x,
  time = time
)
df_nb_gp$denom[df_nb_gp$denom == 0] <- 1

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x,
    data = df_nb_gp,
    family = ratiod_negbin_negbin(),
    temporal = temporal_gp("time", cov = "exponential"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_nd))

# Get scaled time values from numdenom
time_vals <- df_nb_gp$time
time_scaled <- as.vector(scale(time_vals))
unique_times_scaled <- sort(unique(time_scaled))
time_idx <- match(time_scaled, unique_times_scaled)

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/temporal_gp_nb_joint.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  T = N_TIMES,
  y_num = df_nb_gp$y,
  y_denom = df_nb_gp$denom,
  x = df_nb_gp$x,
  time_idx = time_idx,
  time_values = unique_times_scaled,
  sigma2_prior_U = 1.0,
  sigma2_prior_alpha = 0.01,
  phi_prior_lower = 0.01,
  phi_prior_upper = 10.0
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
    show_messages = FALSE,
    adapt_delta = 0.9
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_stan))

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

# Compare beta_num[2] (slope)
nd_beta_col <- grep("^beta_num\\[2\\]$", colnames(draws_nd), value = TRUE)[1]
results$row_44 <- list(
  beta_num_1 = compare_posteriors(
    draws_nd[, nd_beta_col],
    draws_stan$beta_num_1,
    "beta_num[2] (slope)"
  ),
  time_nd = t_nd,
  time_stan = t_stan
)
print_result(results$row_44$beta_num_1)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan/t_nd))

# =============================================================================
# Row 14: poisson_gamma + temporal_gp
# =============================================================================
cat("\n========== Row 14: poisson_gamma + temporal_gp ==========\n")

y_num_pg <- rpois(N_OBS, exp(eta_num))
y_denom_pg <- rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_denom))

df_pg_gp <- data.frame(
  y = y_num_pg,
  denom = y_denom_pg,
  x = x,
  time = time
)
df_pg_gp$denom[df_pg_gp$denom < 0.01] <- 0.01

cat("Fitting numdenom... ")
t_nd_pg <- system.time({
  fit_nd_pg <- ratiod(
    y | denom ~ x,
    data = df_pg_gp,
    family = ratiod_poisson_gamma(),
    temporal = temporal_gp("time", cov = "exponential"),
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
  time_idx = time_idx,
  time_values = unique_times_scaled,
  sigma2_prior_U = 1.0,
  sigma2_prior_alpha = 0.01,
  phi_prior_lower = 0.01,
  phi_prior_upper = 10.0
)

cat("Fitting Stan model... ")
t_stan_pg <- system.time({
  fit_stan_pg <- stan_model_pg$sample(
    data = stan_data_pg,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE,
    adapt_delta = 0.9
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_stan_pg))

draws_nd_pg <- as.matrix(fit_nd_pg$draws)
draws_stan_pg <- fit_stan_pg$draws(format = "df")

nd_beta_col_pg <- grep("^beta_num\\[2\\]$", colnames(draws_nd_pg), value = TRUE)[1]
results$row_14 <- list(
  beta_num_1 = compare_posteriors(
    draws_nd_pg[, nd_beta_col_pg],
    draws_stan_pg$beta_num_1,
    "beta_num[2] (slope)"
  ),
  time_nd = t_nd_pg,
  time_stan = t_stan_pg
)
print_result(results$row_14$beta_num_1)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd_pg, t_stan_pg, t_stan_pg/t_nd_pg))

# =============================================================================
# Row 38: negbin_negbin + HSGP
# =============================================================================
cat("\n========== Row 38: negbin_negbin + HSGP ==========\n")

df_nb_hsgp <- data.frame(
  y = rnbinom(N_OBS, mu = exp(eta_num), size = 5),
  denom = rnbinom(N_OBS, mu = exp(eta_denom), size = 10),
  x = x,
  lon = lon,
  lat = lat
)
df_nb_hsgp$denom[df_nb_hsgp$denom == 0] <- 1

cat("Fitting numdenom (HSGP)... ")
t_nd_hsgp <- system.time({
  fit_nd_hsgp <- ratiod(
    y | denom ~ x,
    data = df_nb_hsgp,
    family = ratiod_negbin_negbin(),
    spatial = spatial_hsgp(coords = c("lon", "lat"), m = 6),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_nd_hsgp))

cat("Compiling Stan model... ")
stan_model_hsgp <- cmdstan_model("benchmarks/stan/hsgp_nb_joint.stan")
cat("done\n")

stan_data_hsgp <- list(
  N = N_OBS,
  M = 6L,
  y_num = df_nb_hsgp$y,
  y_denom = df_nb_hsgp$denom,
  x = df_nb_hsgp$x,
  coords = cbind(df_nb_hsgp$lon, df_nb_hsgp$lat),
  c = 1.5,
  sigma2_prior_U = 1.0,
  sigma2_prior_alpha = 0.01,
  phi_prior_lower = 0.01,
  phi_prior_upper = 100.0
)

cat("Fitting Stan model... ")
t_stan_hsgp <- system.time({
  fit_stan_hsgp <- stan_model_hsgp$sample(
    data = stan_data_hsgp,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE,
    adapt_delta = 0.9
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_stan_hsgp))

draws_nd_hsgp <- as.matrix(fit_nd_hsgp$draws)
draws_stan_hsgp <- fit_stan_hsgp$draws(format = "df")

nd_beta_col_hsgp <- grep("^beta_num\\[2\\]$", colnames(draws_nd_hsgp), value = TRUE)[1]
results$row_38 <- list(
  beta_num_1 = compare_posteriors(
    draws_nd_hsgp[, nd_beta_col_hsgp],
    draws_stan_hsgp$beta_num_1,
    "beta_num[2] (slope)"
  ),
  time_nd = t_nd_hsgp,
  time_stan = t_stan_hsgp
)
print_result(results$row_38$beta_num_1)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd_hsgp, t_stan_hsgp, t_stan_hsgp/t_nd_hsgp))

# =============================================================================
# Row 8: poisson_gamma + HSGP
# =============================================================================
cat("\n========== Row 8: poisson_gamma + HSGP ==========\n")

df_pg_hsgp <- data.frame(
  y = rpois(N_OBS, exp(eta_num)),
  denom = rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_denom)),
  x = x,
  lon = lon,
  lat = lat
)
df_pg_hsgp$denom[df_pg_hsgp$denom < 0.01] <- 0.01

cat("Fitting numdenom (HSGP)... ")
t_nd_pg_hsgp <- system.time({
  fit_nd_pg_hsgp <- ratiod(
    y | denom ~ x,
    data = df_pg_hsgp,
    family = ratiod_poisson_gamma(),
    spatial = spatial_hsgp(coords = c("lon", "lat"), m = 6),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_nd_pg_hsgp))

cat("Compiling Stan model... ")
stan_model_pg_hsgp <- cmdstan_model("benchmarks/stan/hsgp_pg_joint.stan")
cat("done\n")

stan_data_pg_hsgp <- list(
  N = N_OBS,
  M = 6L,
  y_num = df_pg_hsgp$y,
  y_denom = df_pg_hsgp$denom,
  x = df_pg_hsgp$x,
  coords = cbind(df_pg_hsgp$lon, df_pg_hsgp$lat),
  c = 1.5,
  sigma2_prior_U = 1.0,
  sigma2_prior_alpha = 0.01,
  phi_prior_lower = 0.01,
  phi_prior_upper = 100.0
)

cat("Fitting Stan model... ")
t_stan_pg_hsgp <- system.time({
  fit_stan_pg_hsgp <- stan_model_pg_hsgp$sample(
    data = stan_data_pg_hsgp,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE,
    adapt_delta = 0.9
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_stan_pg_hsgp))

draws_nd_pg_hsgp <- as.matrix(fit_nd_pg_hsgp$draws)
draws_stan_pg_hsgp <- fit_stan_pg_hsgp$draws(format = "df")

nd_beta_col_pg_hsgp <- grep("^beta_num\\[2\\]$", colnames(draws_nd_pg_hsgp), value = TRUE)[1]
results$row_8 <- list(
  beta_num_1 = compare_posteriors(
    draws_nd_pg_hsgp[, nd_beta_col_pg_hsgp],
    draws_stan_pg_hsgp$beta_num_1,
    "beta_num[2] (slope)"
  ),
  time_nd = t_nd_pg_hsgp,
  time_stan = t_stan_pg_hsgp
)
print_result(results$row_8$beta_num_1)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd_pg_hsgp, t_stan_pg_hsgp, t_stan_pg_hsgp/t_nd_pg_hsgp))

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - BATCH 13 (Temporal GP & HSGP with joint Stan)\n")
cat("=======================================================\n\n")

cat(sprintf("%-8s %-30s %10s %10s %8s %s\n",
            "Row", "Model", "numdenom", "Stan", "Speedup", "Status"))
cat(paste(rep("-", 80), collapse = ""), "\n")

# Row 44
r44 <- results$row_44
cat(sprintf("%-8s %-30s %10.1fs %10.1fs %8.1fx %s\n",
            "44", "nb + temporal_gp",
            r44$time_nd, r44$time_stan, r44$time_stan / r44$time_nd,
            if(r44$beta_num_1$pass) "PASS" else "FAIL"))

# Row 14
r14 <- results$row_14
cat(sprintf("%-8s %-30s %10.1fs %10.1fs %8.1fx %s\n",
            "14", "pg + temporal_gp",
            r14$time_nd, r14$time_stan, r14$time_stan / r14$time_nd,
            if(r14$beta_num_1$pass) "PASS" else "FAIL"))

# Row 38
r38 <- results$row_38
cat(sprintf("%-8s %-30s %10.1fs %10.1fs %8.1fx %s\n",
            "38", "nb + HSGP",
            r38$time_nd, r38$time_stan, r38$time_stan / r38$time_nd,
            if(r38$beta_num_1$pass) "PASS" else "FAIL"))

# Row 8
r8 <- results$row_8
cat(sprintf("%-8s %-30s %10.1fs %10.1fs %8.1fx %s\n",
            "8", "pg + HSGP",
            r8$time_nd, r8$time_stan, r8$time_stan / r8$time_nd,
            if(r8$beta_num_1$pass) "PASS" else "FAIL"))

cat(paste(rep("-", 80), collapse = ""), "\n")

# Save results
saveRDS(results, "benchmarks/results_joint_batch13.rds")
cat("\nResults saved to benchmarks/results_joint_batch13.rds\n")

# Print rows to update in gradient_methods.md
cat("\n\nUPDATE gradient_methods.md:\n")
if (r44$beta_num_1$pass) cat("  Row 44: PASS - add '✓Stan (joint)' to Notes\n")
if (r14$beta_num_1$pass) cat("  Row 14: PASS - add '✓Stan (joint)' to Notes\n")
if (r38$beta_num_1$pass) cat("  Row 38: PASS - add '✓Stan (joint)' to Notes\n")
if (r8$beta_num_1$pass) cat("  Row 8: PASS - add '✓Stan (joint)' to Notes\n")
