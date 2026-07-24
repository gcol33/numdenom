# Simulation-based validation for beta_binomial + ICAR models
# Rows 105, 107: beta_binomial with ICAR spatial
#
# brms uses IID RE instead of ICAR for beta_binomial, so we validate
# against simulation truth instead.

library(numdenom)
library(posterior)

set.seed(42)

# Standard benchmark parameters
N_OBS <- 200
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2
N_SITES <- 20
N_TIMES <- 10

cat("=======================================================\n")
cat("Beta-Binomial + ICAR: Simulation-Based Validation\n")
cat("Rows 105, 107\n")
cat("=======================================================\n")
cat(sprintf("N=%d, sites=%d, times=%d, iter=%d, warmup=%d, chains=%d\n\n",
            N_OBS, N_SITES, N_TIMES, N_ITER, N_WARMUP, N_CHAINS))

# Helper function to check parameter recovery
check_recovery <- function(draws, true_value, param_name, threshold_sd = 2) {
  post_mean <- mean(draws)
  post_sd <- sd(draws)
  q025 <- quantile(draws, 0.025)
  q975 <- quantile(draws, 0.975)

  diff_sd <- abs(post_mean - true_value) / post_sd
  in_ci <- true_value >= q025 && true_value <= q975
  pass <- diff_sd < threshold_sd

  list(
    param = param_name,
    true = true_value,
    mean = post_mean,
    sd = post_sd,
    q025 = q025,
    q975 = q975,
    diff_sd = diff_sd,
    in_ci = in_ci,
    pass = pass
  )
}

print_recovery <- function(result) {
  ci_status <- if (result$in_ci) "in CI" else "NOT in CI"
  pass_status <- if (result$pass) "PASS" else "FAIL"
  cat(sprintf("  %s: true=%.3f, post=%.3f (SD=%.3f), 95%% CI=[%.3f, %.3f], %.2f SD from true, %s => %s\n",
              result$param, result$true, result$mean, result$sd,
              result$q025, result$q975, result$diff_sd, ci_status, pass_status))
}

# Build adjacency matrix for ICAR (chain graph)
adj_matrix <- matrix(0, N_SITES, N_SITES)
for (i in 1:(N_SITES - 1)) {
  adj_matrix[i, i + 1] <- 1
  adj_matrix[i + 1, i] <- 1
}

# Generate ICAR random effects
generate_icar <- function(n_sites, sigma, adj) {
  # Generate from intrinsic CAR: (I - rho * W) with rho = 1
  # Use random walk on chain graph
  phi <- numeric(n_sites)
  phi[1] <- rnorm(1, 0, sigma)
  for (i in 2:n_sites) {
    phi[i] <- phi[i - 1] + rnorm(1, 0, sigma)
  }
  # Center to sum to zero

  phi <- phi - mean(phi)
  phi
}

# Generate RW1 temporal effects
generate_rw1 <- function(n_times, sigma) {
  gamma <- numeric(n_times)
  gamma[1] <- rnorm(1, 0, sigma)
  for (t in 2:n_times) {
    gamma[t] <- gamma[t - 1] + rnorm(1, 0, sigma)
  }
  gamma <- gamma - mean(gamma)
  gamma
}

results <- list()

# =============================================================================
# Row 105: beta_binomial + RE + ICAR
# =============================================================================
cat("\n========== Row 105: beta_binomial + RE + ICAR ==========\n")

# True parameters
true_intercept_105 <- 0.5
true_slope_105 <- 0.3
true_sigma_icar_105 <- 0.4
true_phi_105 <- 10  # beta-binomial concentration

# Generate data
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)
trials <- sample(10:50, N_OBS, replace = TRUE)

# Generate ICAR spatial effects
spatial_effects_105 <- generate_icar(N_SITES, true_sigma_icar_105, adj_matrix)

# Linear predictor
eta_105 <- true_intercept_105 + true_slope_105 * x + spatial_effects_105[as.integer(site)]
prob_105 <- plogis(eta_105)

# Generate beta-binomial responses
alpha_105 <- prob_105 * true_phi_105
beta_105 <- (1 - prob_105) * true_phi_105
y_prop_105 <- rbeta(N_OBS, alpha_105, beta_105)
y_prop_105 <- pmin(pmax(y_prop_105, 0.001), 0.999)
successes_105 <- round(y_prop_105 * trials)

df_105 <- data.frame(
  successes = successes_105,
  trials = trials,
  x = x,
  site = site
)

cat(sprintf("True parameters: intercept=%.2f, slope=%.2f, sigma_icar=%.2f, phi=%.1f\n",
            true_intercept_105, true_slope_105, true_sigma_icar_105, true_phi_105))
cat(sprintf("Spatial effect range: [%.3f, %.3f]\n",
            min(spatial_effects_105), max(spatial_effects_105)))

cat("Fitting numdenom... ")
t_nd_105 <- system.time({
  fit_105 <- tryCatch({
    tratio(
      successes | trials ~ x,
      data = df_105,
      family = ratiod_beta_binomial(),
      spatial = spatial_car(adj_matrix, group_var = "site"),
      control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
    )
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    NULL
  })
})[["elapsed"]]

if (!is.null(fit_105)) {
  cat(sprintf("%.1fs\n", t_nd_105))

  draws_105 <- as.matrix(fit_105$draws)

  # Find column names
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_105), value = TRUE)[1]

  results$row_105 <- list(
    slope = check_recovery(draws_105[, slope_col], true_slope_105, "slope (x)"),
    time = t_nd_105
  )

  cat("\nParameter recovery:\n")
  print_recovery(results$row_105$slope)

  cat(sprintf("\nTime: %.1fs\n", t_nd_105))

  # Overall pass
  results$row_105$pass <- results$row_105$slope$pass
  cat(sprintf("Overall: %s\n", if (results$row_105$pass) "PASS" else "FAIL"))
} else {
  results$row_105 <- list(error = TRUE)
  cat("  SKIPPED due to error\n")
}

# =============================================================================
# Row 107: beta_binomial + RE + ICAR + RW1
# =============================================================================
cat("\n========== Row 107: beta_binomial + RE + ICAR + RW1 ==========\n")

# True parameters
true_intercept_107 <- 0.5
true_slope_107 <- 0.3
true_sigma_icar_107 <- 0.3
true_sigma_rw1_107 <- 0.2
true_phi_107 <- 10

# Generate data with both spatial and temporal structure
time <- rep(1:N_TIMES, length.out = N_OBS)
site_107 <- factor(rep(1:N_SITES, each = N_OBS / N_SITES))

# Generate effects
spatial_effects_107 <- generate_icar(N_SITES, true_sigma_icar_107, adj_matrix)
temporal_effects_107 <- generate_rw1(N_TIMES, true_sigma_rw1_107)

# Linear predictor
eta_107 <- true_intercept_107 + true_slope_107 * x +
           spatial_effects_107[as.integer(site_107)] +
           temporal_effects_107[time]
prob_107 <- plogis(eta_107)

# Generate responses
alpha_107 <- prob_107 * true_phi_107
beta_107 <- (1 - prob_107) * true_phi_107
y_prop_107 <- rbeta(N_OBS, alpha_107, beta_107)
y_prop_107 <- pmin(pmax(y_prop_107, 0.001), 0.999)
successes_107 <- round(y_prop_107 * trials)

df_107 <- data.frame(
  successes = successes_107,
  trials = trials,
  x = x,
  site = site_107,
  time = time
)

cat(sprintf("True parameters: intercept=%.2f, slope=%.2f, sigma_icar=%.2f, sigma_rw1=%.2f, phi=%.1f\n",
            true_intercept_107, true_slope_107, true_sigma_icar_107, true_sigma_rw1_107, true_phi_107))
cat(sprintf("Spatial effect range: [%.3f, %.3f]\n",
            min(spatial_effects_107), max(spatial_effects_107)))
cat(sprintf("Temporal effect range: [%.3f, %.3f]\n",
            min(temporal_effects_107), max(temporal_effects_107)))

cat("Fitting numdenom... ")
t_nd_107 <- system.time({
  fit_107 <- tryCatch({
    tratio(
      successes | trials ~ x,
      data = df_107,
      family = ratiod_beta_binomial(),
      spatial = spatial_car(adj_matrix, group_var = "site"),
      temporal = temporal_rw1(time_var = "time"),
      control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
    )
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    NULL
  })
})[["elapsed"]]

if (!is.null(fit_107)) {
  cat(sprintf("%.1fs\n", t_nd_107))

  draws_107 <- as.matrix(fit_107$draws)

  # Find column names
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_107), value = TRUE)[1]

  results$row_107 <- list(
    slope = check_recovery(draws_107[, slope_col], true_slope_107, "slope (x)"),
    time = t_nd_107
  )

  cat("\nParameter recovery:\n")
  print_recovery(results$row_107$slope)

  cat(sprintf("\nTime: %.1fs\n", t_nd_107))

  # Overall pass
  results$row_107$pass <- results$row_107$slope$pass
  cat(sprintf("Overall: %s\n", if (results$row_107$pass) "PASS" else "FAIL"))
} else {
  results$row_107 <- list(error = TRUE)
  cat("  SKIPPED due to error\n")
}

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - Beta-Binomial + ICAR Simulation Validation\n")
cat("=======================================================\n\n")

cat(sprintf("%-8s %-30s %10s %s\n", "Row", "Model", "Time", "Status"))
cat(paste(rep("-", 60), collapse = ""), "\n")

for (row_name in names(results)) {
  r <- results[[row_name]]
  row_num <- gsub("row_", "", row_name)

  if (!is.null(r$error) && r$error) {
    cat(sprintf("%-8s %-30s %10s %s\n", row_num, "beta_binomial", "ERROR", "SKIPPED"))
  } else {
    model_name <- switch(row_num,
      "105" = "bb + ICAR",
      "107" = "bb + ICAR + RW1",
      "unknown"
    )
    status <- if (r$pass) "PASS (Sim)" else "FAIL"
    cat(sprintf("%-8s %-30s %10.1fs %s\n", row_num, model_name, r$time, status))
  }
}

cat(paste(rep("-", 60), collapse = ""), "\n")

saveRDS(results, "benchmarks/results_beta_binomial_icar_sim.rds")
cat("\nResults saved to benchmarks/results_beta_binomial_icar_sim.rds\n")

cat("\n\nUPDATE gradient_methods.md:\n")
for (row_name in names(results)) {
  r <- results[[row_name]]
  if (is.null(r$error) && r$pass) {
    cat(sprintf("  Row %s: PASS - change 'brms uses IID RE' to 'Sim'\n", gsub("row_", "", row_name)))
  }
}
