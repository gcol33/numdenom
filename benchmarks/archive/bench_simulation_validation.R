# Simulation-based validation for models where Stan has convergence issues
# Validates against known true parameters from simulated data
#
# Rows needing simulation validation:
#   95: gamma_gamma + RE + ICAR (Stan ~5 SE)
#   96: gamma_gamma + RE + RW1 (Stan ~4 SE)
#   99: lognormal + RE (Stan convergence issues)
#  100: lognormal + RE + ICAR (Stan convergence issues)
#  101: lognormal + RE + RW1 (Stan convergence issues)
#  102: lognormal + RE + ICAR + RW1 (Stan 100% treedepth)

library(numdenom)
library(posterior)

set.seed(2026)

# Standard benchmark parameters
N_OBS <- 200
N_ITER <- 1000
N_WARMUP <- 500
N_CHAINS <- 2
N_SITES <- 20
N_TIMES <- 10

cat("=======================================================\n")
cat("Simulation-Based Validation (Stan Has Convergence Issues)\n")
cat("=======================================================\n")
cat(sprintf("N=%d, iter=%d, warmup=%d, chains=%d\n", N_OBS, N_ITER, N_WARMUP, N_CHAINS))
cat(sprintf("sites=%d, times=%d\n\n", N_SITES, N_TIMES))

# Helper function to validate against true parameters
validate_against_truth <- function(draws, param_name, true_value, threshold_sd = 2) {
  post_mean <- mean(draws)
  post_sd <- sd(draws)
  q025 <- quantile(draws, 0.025)
  q975 <- quantile(draws, 0.975)

  # Check if posterior mean is within threshold_sd of true value
  diff_sd <- abs(post_mean - true_value) / post_sd
  pass <- diff_sd < threshold_sd

  # Also check if 95% CI contains true value
  in_ci <- true_value >= q025 && true_value <= q975

  list(
    param = param_name,
    true_value = true_value,
    post_mean = post_mean,
    post_sd = post_sd,
    diff_sd = diff_sd,
    q025 = q025,
    q975 = q975,
    in_ci = in_ci,
    pass = pass
  )
}

print_result <- function(result) {
  ci_status <- if(result$in_ci) "in CI" else "NOT in CI"
  cat(sprintf("  %s: true=%.3f, post=%.3f (SD=%.3f), diff=%.2f SD, 95%% CI=[%.3f, %.3f] %s => %s\n",
              result$param, result$true_value, result$post_mean, result$post_sd,
              result$diff_sd, result$q025, result$q975, ci_status,
              if(result$pass) "PASS" else "FAIL"))
}

# =============================================================================
# DATA SETUP
# =============================================================================

# True parameters
TRUE_BETA_NUM <- c(2.0, 0.3)
TRUE_BETA_DENOM <- c(1.5, 0.2)
TRUE_SIGMA_RE <- 0.4
TRUE_SIGMA_SPATIAL <- 0.3
TRUE_SIGMA_TEMPORAL <- 0.25
TRUE_SHAPE_NUM <- 5
TRUE_SHAPE_DENOM <- 8
TRUE_SIGMA_OBS <- 0.5  # For lognormal

# Generate base data
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)
time <- rep(1:N_TIMES, length.out = N_OBS)
time_factor <- factor(time)

# Spatial grid and adjacency
n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_SITES, ]
adj_mat <- matrix(0, N_SITES, N_SITES)
for (i in 1:N_SITES) {
  for (j in 1:N_SITES) {
    if (i != j) {
      dist <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
      if (dist <= 1.5) adj_mat[i, j] <- 1
    }
  }
}
spatial_site <- factor(rep(1:N_SITES, length.out = N_OBS))

# Generate true random effects
true_re <- rnorm(N_SITES, 0, TRUE_SIGMA_RE)

# Generate true spatial effects (approximate ICAR by correlated normals)
# For simulation, use spatially correlated noise
true_spatial <- rnorm(N_SITES, 0, TRUE_SIGMA_SPATIAL)
# Apply some spatial smoothing
for (iter in 1:3) {
  for (i in 1:N_SITES) {
    neighbors <- which(adj_mat[i, ] == 1)
    if (length(neighbors) > 0) {
      true_spatial[i] <- 0.5 * true_spatial[i] + 0.5 * mean(true_spatial[neighbors])
    }
  }
}
true_spatial <- true_spatial - mean(true_spatial)  # Center
true_spatial <- true_spatial * (TRUE_SIGMA_SPATIAL / sd(true_spatial))  # Scale

# Generate true temporal effects (RW1)
true_temporal <- cumsum(rnorm(N_TIMES, 0, TRUE_SIGMA_TEMPORAL / sqrt(N_TIMES)))
true_temporal <- true_temporal - mean(true_temporal)  # Center

results <- list()

# =============================================================================
# Row 95: gamma_gamma + RE + ICAR
# =============================================================================
cat("\n========== Row 95: gamma_gamma + RE + ICAR ==========\n")

eta_num_95 <- TRUE_BETA_NUM[1] + TRUE_BETA_NUM[2] * x +
              true_re[as.integer(site)] + true_spatial[as.integer(spatial_site)]
eta_denom_95 <- TRUE_BETA_DENOM[1] + TRUE_BETA_DENOM[2] * x +
                true_re[as.integer(site)] + true_spatial[as.integer(spatial_site)]

df_95 <- data.frame(
  y = rgamma(N_OBS, shape = TRUE_SHAPE_NUM, rate = TRUE_SHAPE_NUM / exp(eta_num_95)),
  denom = rgamma(N_OBS, shape = TRUE_SHAPE_DENOM, rate = TRUE_SHAPE_DENOM / exp(eta_denom_95)),
  x = x,
  site = site,
  spatial_site = spatial_site
)
df_95$y[df_95$y <= 0] <- 0.01
df_95$denom[df_95$denom <= 0] <- 0.01

cat("Fitting numdenom... ")
t_95 <- system.time({
  fit_95 <- ratiod(
    y | denom ~ x + (1|site),
    data = df_95,
    family = ratiod_gamma_gamma(),
    spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_95))

draws_95 <- as.matrix(fit_95$draws)
results$row_95 <- list(
  beta_num_2 = validate_against_truth(draws_95[, "beta_num[2]"], "beta_num[2]", TRUE_BETA_NUM[2]),
  beta_denom_2 = validate_against_truth(draws_95[, "beta_denom[2]"], "beta_denom[2]", TRUE_BETA_DENOM[2]),
  time = t_95
)
print_result(results$row_95$beta_num_2)
print_result(results$row_95$beta_denom_2)

# =============================================================================
# Row 96: gamma_gamma + RE + RW1
# =============================================================================
cat("\n========== Row 96: gamma_gamma + RE + RW1 ==========\n")

eta_num_96 <- TRUE_BETA_NUM[1] + TRUE_BETA_NUM[2] * x +
              true_re[as.integer(site)] + true_temporal[time]
eta_denom_96 <- TRUE_BETA_DENOM[1] + TRUE_BETA_DENOM[2] * x +
                true_re[as.integer(site)] + true_temporal[time]

df_96 <- data.frame(
  y = rgamma(N_OBS, shape = TRUE_SHAPE_NUM, rate = TRUE_SHAPE_NUM / exp(eta_num_96)),
  denom = rgamma(N_OBS, shape = TRUE_SHAPE_DENOM, rate = TRUE_SHAPE_DENOM / exp(eta_denom_96)),
  x = x,
  site = site,
  time = time,
  time_factor = time_factor
)
df_96$y[df_96$y <= 0] <- 0.01
df_96$denom[df_96$denom <= 0] <- 0.01

cat("Fitting numdenom... ")
t_96 <- system.time({
  fit_96 <- ratiod(
    y | denom ~ x + (1|site),
    data = df_96,
    family = ratiod_gamma_gamma(),
    temporal = temporal_rw1("time_factor"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_96))

draws_96 <- as.matrix(fit_96$draws)
results$row_96 <- list(
  beta_num_2 = validate_against_truth(draws_96[, "beta_num[2]"], "beta_num[2]", TRUE_BETA_NUM[2]),
  beta_denom_2 = validate_against_truth(draws_96[, "beta_denom[2]"], "beta_denom[2]", TRUE_BETA_DENOM[2]),
  time = t_96
)
print_result(results$row_96$beta_num_2)
print_result(results$row_96$beta_denom_2)

# =============================================================================
# Row 99: lognormal + RE
# =============================================================================
cat("\n========== Row 99: lognormal + RE ==========\n")

eta_num_99 <- TRUE_BETA_NUM[1] + TRUE_BETA_NUM[2] * x + true_re[as.integer(site)]
eta_denom_99 <- TRUE_BETA_DENOM[1] + TRUE_BETA_DENOM[2] * x + true_re[as.integer(site)]

df_99 <- data.frame(
  y = exp(eta_num_99 + rnorm(N_OBS, 0, TRUE_SIGMA_OBS)),
  denom = exp(eta_denom_99 + rnorm(N_OBS, 0, TRUE_SIGMA_OBS)),
  x = x,
  site = site
)

cat("Fitting numdenom... ")
t_99 <- system.time({
  fit_99 <- ratiod(
    y | denom ~ x + (1|site),
    data = df_99,
    family = ratiod_lognormal(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_99))

draws_99 <- as.matrix(fit_99$draws)
results$row_99 <- list(
  beta_num_2 = validate_against_truth(draws_99[, "beta_num[2]"], "beta_num[2]", TRUE_BETA_NUM[2]),
  beta_denom_2 = validate_against_truth(draws_99[, "beta_denom[2]"], "beta_denom[2]", TRUE_BETA_DENOM[2]),
  time = t_99
)
print_result(results$row_99$beta_num_2)
print_result(results$row_99$beta_denom_2)

# =============================================================================
# Row 100: lognormal + RE + ICAR
# =============================================================================
cat("\n========== Row 100: lognormal + RE + ICAR ==========\n")

eta_num_100 <- TRUE_BETA_NUM[1] + TRUE_BETA_NUM[2] * x +
               true_re[as.integer(site)] + true_spatial[as.integer(spatial_site)]
eta_denom_100 <- TRUE_BETA_DENOM[1] + TRUE_BETA_DENOM[2] * x +
                 true_re[as.integer(site)] + true_spatial[as.integer(spatial_site)]

df_100 <- data.frame(
  y = exp(eta_num_100 + rnorm(N_OBS, 0, TRUE_SIGMA_OBS)),
  denom = exp(eta_denom_100 + rnorm(N_OBS, 0, TRUE_SIGMA_OBS)),
  x = x,
  site = site,
  spatial_site = spatial_site
)

cat("Fitting numdenom... ")
t_100 <- system.time({
  fit_100 <- ratiod(
    y | denom ~ x + (1|site),
    data = df_100,
    family = ratiod_lognormal(),
    spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_100))

draws_100 <- as.matrix(fit_100$draws)
results$row_100 <- list(
  beta_num_2 = validate_against_truth(draws_100[, "beta_num[2]"], "beta_num[2]", TRUE_BETA_NUM[2]),
  beta_denom_2 = validate_against_truth(draws_100[, "beta_denom[2]"], "beta_denom[2]", TRUE_BETA_DENOM[2]),
  time = t_100
)
print_result(results$row_100$beta_num_2)
print_result(results$row_100$beta_denom_2)

# =============================================================================
# Row 101: lognormal + RE + RW1
# =============================================================================
cat("\n========== Row 101: lognormal + RE + RW1 ==========\n")

eta_num_101 <- TRUE_BETA_NUM[1] + TRUE_BETA_NUM[2] * x +
               true_re[as.integer(site)] + true_temporal[time]
eta_denom_101 <- TRUE_BETA_DENOM[1] + TRUE_BETA_DENOM[2] * x +
                 true_re[as.integer(site)] + true_temporal[time]

df_101 <- data.frame(
  y = exp(eta_num_101 + rnorm(N_OBS, 0, TRUE_SIGMA_OBS)),
  denom = exp(eta_denom_101 + rnorm(N_OBS, 0, TRUE_SIGMA_OBS)),
  x = x,
  site = site,
  time = time,
  time_factor = time_factor
)

cat("Fitting numdenom... ")
t_101 <- system.time({
  fit_101 <- ratiod(
    y | denom ~ x + (1|site),
    data = df_101,
    family = ratiod_lognormal(),
    temporal = temporal_rw1("time_factor"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_101))

draws_101 <- as.matrix(fit_101$draws)
results$row_101 <- list(
  beta_num_2 = validate_against_truth(draws_101[, "beta_num[2]"], "beta_num[2]", TRUE_BETA_NUM[2]),
  beta_denom_2 = validate_against_truth(draws_101[, "beta_denom[2]"], "beta_denom[2]", TRUE_BETA_DENOM[2]),
  time = t_101
)
print_result(results$row_101$beta_num_2)
print_result(results$row_101$beta_denom_2)

# =============================================================================
# Row 102: lognormal + RE + ICAR + RW1
# =============================================================================
cat("\n========== Row 102: lognormal + RE + ICAR + RW1 ==========\n")

eta_num_102 <- TRUE_BETA_NUM[1] + TRUE_BETA_NUM[2] * x +
               true_re[as.integer(site)] + true_spatial[as.integer(spatial_site)] + true_temporal[time]
eta_denom_102 <- TRUE_BETA_DENOM[1] + TRUE_BETA_DENOM[2] * x +
                 true_re[as.integer(site)] + true_spatial[as.integer(spatial_site)] + true_temporal[time]

df_102 <- data.frame(
  y = exp(eta_num_102 + rnorm(N_OBS, 0, TRUE_SIGMA_OBS)),
  denom = exp(eta_denom_102 + rnorm(N_OBS, 0, TRUE_SIGMA_OBS)),
  x = x,
  site = site,
  spatial_site = spatial_site,
  time = time,
  time_factor = time_factor
)

cat("Fitting numdenom... ")
t_102 <- system.time({
  fit_102 <- ratiod(
    y | denom ~ x + (1|site),
    data = df_102,
    family = ratiod_lognormal(),
    spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
    temporal = temporal_rw1("time_factor"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_102))

draws_102 <- as.matrix(fit_102$draws)
results$row_102 <- list(
  beta_num_2 = validate_against_truth(draws_102[, "beta_num[2]"], "beta_num[2]", TRUE_BETA_NUM[2]),
  beta_denom_2 = validate_against_truth(draws_102[, "beta_denom[2]"], "beta_denom[2]", TRUE_BETA_DENOM[2]),
  time = t_102
)
print_result(results$row_102$beta_num_2)
print_result(results$row_102$beta_denom_2)

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - Simulation-Based Validation\n")
cat("=======================================================\n\n")

cat("True parameter values used:\n")
cat(sprintf("  beta_num = [%.1f, %.1f]\n", TRUE_BETA_NUM[1], TRUE_BETA_NUM[2]))
cat(sprintf("  beta_denom = [%.1f, %.1f]\n", TRUE_BETA_DENOM[1], TRUE_BETA_DENOM[2]))
cat(sprintf("  sigma_re = %.2f, sigma_spatial = %.2f, sigma_temporal = %.2f\n\n",
            TRUE_SIGMA_RE, TRUE_SIGMA_SPATIAL, TRUE_SIGMA_TEMPORAL))

cat(sprintf("%-8s %-35s %8s %s\n", "Row", "Model", "Time", "Status"))
cat(paste(rep("-", 70), collapse = ""), "\n")

all_pass <- TRUE
for (row in names(results)) {
  r <- results[[row]]
  pass <- r$beta_num_2$pass && r$beta_denom_2$pass
  if (!pass) all_pass <- FALSE

  model_name <- switch(row,
    row_95 = "gamma_gamma + RE + ICAR",
    row_96 = "gamma_gamma + RE + RW1",
    row_99 = "lognormal + RE",
    row_100 = "lognormal + RE + ICAR",
    row_101 = "lognormal + RE + RW1",
    row_102 = "lognormal + RE + ICAR + RW1"
  )

  cat(sprintf("%-8s %-35s %8.1fs %s\n",
              gsub("row_", "", row), model_name, r$time,
              if(pass) "PASS" else "FAIL"))
}

cat(paste(rep("-", 70), collapse = ""), "\n")
cat(sprintf("\nOverall: %s\n", if(all_pass) "ALL PASS" else "SOME FAIL"))

# Save results
saveRDS(results, "benchmarks/results_simulation_validation.rds")
cat("\nResults saved to benchmarks/results_simulation_validation.rds\n")

# Print rows to update in gradient_methods.md
cat("\n\nUPDATE gradient_methods.md:\n")
cat("Change 'needs Sim' to 'Sim' for these rows:\n")
for (row in names(results)) {
  r <- results[[row]]
  pass <- r$beta_num_2$pass && r$beta_denom_2$pass
  if (pass) {
    row_num <- gsub("row_", "", row)
    cat(sprintf("  Row %s - PASS (validated against simulation truth)\n", row_num))
  } else {
    row_num <- gsub("row_", "", row)
    cat(sprintf("  Row %s - FAIL (needs investigation)\n", row_num))
  }
}
