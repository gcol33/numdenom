# Simulation-based validation for GP spatial models
# Rows 7, 21, 37, 51, 67, 83: GP spatial (with and without RW1)
# NOTE: GP is O(N³) - using smaller N for speed
# NOTE: numdenom GP operates at observation level with unique coordinates.
#   Each obs gets unique coords; spatial effects generated from GP covariance.

library(numdenom)
library(posterior)

set.seed(42)

# Smaller parameters for GP (O(N³))
N_OBS <- 80
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1
N_TIMES <- 10

cat("=======================================================\n")
cat("GP Spatial Models: Simulation-Based Validation\n")
cat("Rows 7, 21, 37, 51, 67, 83\n")
cat("=======================================================\n")
cat(sprintf("N=%d, iter=%d, chains=%d\n\n", N_OBS, N_ITER, N_CHAINS))

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

# Generate UNIQUE spatial coordinates per observation
set.seed(123)
coord_x <- runif(N_OBS, 0, 10)
coord_y <- runif(N_OBS, 0, 10)
coords_mat <- cbind(coord_x, coord_y)

# Generate GP spatial effects at observation level using exponential covariance
generate_gp_effects <- function(coords, sigma, lengthscale) {
  n <- nrow(coords)
  dist_mat <- as.matrix(dist(coords))
  cov_mat <- sigma^2 * exp(-dist_mat / lengthscale)
  diag(cov_mat) <- diag(cov_mat) + 1e-6  # Jitter for stability
  L <- chol(cov_mat)
  z <- rnorm(n)
  effects <- as.vector(t(L) %*% z)
  effects - mean(effects)
}

generate_rw1 <- function(n, sigma) {
  gamma <- cumsum(rnorm(n, 0, sigma))
  gamma - mean(gamma)
}

# True parameters
true_intercept <- 1.0
true_slope <- 0.3
true_sigma_gp <- 0.3
true_lengthscale <- 2.0

# Generate GP spatial effects (one per observation)
spatial_effects <- generate_gp_effects(coords_mat, true_sigma_gp, true_lengthscale)

# Common covariate
x <- rnorm(N_OBS)

# =============================================================================
# Row 7: poisson_gamma + GP
# =============================================================================
cat("\n========== Row 7: poisson_gamma + GP ==========\n")

eta_7 <- true_intercept + true_slope * x + spatial_effects
effort_7 <- rgamma(N_OBS, 5, 1)
count_7 <- rpois(N_OBS, exp(eta_7) * effort_7)
df_7 <- data.frame(count = count_7, effort = effort_7, x = x,
                   coord_x = coord_x, coord_y = coord_y)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_7 <- system.time({
  fit_7 <- tryCatch({
    tratio(count | effort ~ x, data = df_7, family = ratiod_poisson_gamma(),
           spatial = spatial_gp(coords = c("coord_x", "coord_y"), cov = "exponential"),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_7)) {
  cat(sprintf("%.1fs\n", t_7))
  draws_7 <- as.matrix(fit_7$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_7), value = TRUE)[1]
  results$row_7 <- list(slope = check_recovery(draws_7[, slope_col], true_slope, "slope"), time = t_7)
  print_recovery(results$row_7$slope)
  results$row_7$pass <- results$row_7$slope$pass
} else { results$row_7 <- list(error = TRUE) }

# =============================================================================
# Row 37: negbin_negbin + GP
# =============================================================================
cat("\n========== Row 37: negbin_negbin + GP ==========\n")

eta_num_37 <- true_intercept + true_slope * x + spatial_effects
eta_denom_37 <- 0.5 + 0.2 * x + spatial_effects * 0.8

num_37 <- rnbinom(N_OBS, size = 5, mu = exp(eta_num_37))
denom_37 <- rnbinom(N_OBS, size = 5, mu = exp(eta_denom_37))
denom_37[denom_37 == 0] <- 1
df_37 <- data.frame(num = num_37, denom = denom_37, x = x,
                    coord_x = coord_x, coord_y = coord_y)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_37 <- system.time({
  fit_37 <- tryCatch({
    tratio(num | denom ~ x, data = df_37, family = ratiod_negbin_negbin(),
           spatial = spatial_gp(coords = c("coord_x", "coord_y"), cov = "exponential"),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_37)) {
  cat(sprintf("%.1fs\n", t_37))
  draws_37 <- as.matrix(fit_37$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_37), value = TRUE)[1]
  results$row_37 <- list(slope = check_recovery(draws_37[, slope_col], true_slope, "slope"), time = t_37)
  print_recovery(results$row_37$slope)
  results$row_37$pass <- results$row_37$slope$pass
} else { results$row_37 <- list(error = TRUE) }

# =============================================================================
# Row 67: binomial + GP
# =============================================================================
cat("\n========== Row 67: binomial + GP ==========\n")

eta_67 <- true_intercept + true_slope * x + spatial_effects
prob_67 <- plogis(eta_67)
trials_67 <- sample(20:50, N_OBS, replace = TRUE)
successes_67 <- rbinom(N_OBS, trials_67, prob_67)
df_67 <- data.frame(successes = successes_67, trials = trials_67, x = x,
                    coord_x = coord_x, coord_y = coord_y)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_67 <- system.time({
  fit_67 <- tryCatch({
    tratio(successes | trials ~ x, data = df_67, family = ratiod_binomial(),
           spatial = spatial_gp(coords = c("coord_x", "coord_y"), cov = "exponential"),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_67)) {
  cat(sprintf("%.1fs\n", t_67))
  draws_67 <- as.matrix(fit_67$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_67), value = TRUE)[1]
  results$row_67 <- list(slope = check_recovery(draws_67[, slope_col], true_slope, "slope"), time = t_67)
  print_recovery(results$row_67$slope)
  results$row_67$pass <- results$row_67$slope$pass
} else { results$row_67 <- list(error = TRUE) }

# =============================================================================
# Now with RW1 temporal: Rows 21, 51, 83
# =============================================================================

true_sigma_rw1 <- 0.15
temporal_effects <- generate_rw1(N_TIMES, true_sigma_rw1)
time <- rep(1:N_TIMES, length.out = N_OBS)

# =============================================================================
# Row 21: poisson_gamma + GP + RW1
# =============================================================================
cat("\n========== Row 21: poisson_gamma + GP + RW1 ==========\n")

eta_21 <- true_intercept + true_slope * x + spatial_effects + temporal_effects[time]
count_21 <- rpois(N_OBS, exp(eta_21) * effort_7)
df_21 <- data.frame(count = count_21, effort = effort_7, x = x, time = time,
                    coord_x = coord_x, coord_y = coord_y)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_21 <- system.time({
  fit_21 <- tryCatch({
    tratio(count | effort ~ x, data = df_21, family = ratiod_poisson_gamma(),
           spatial = spatial_gp(coords = c("coord_x", "coord_y"), cov = "exponential"),
           temporal = temporal_rw1(time_var = "time"),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_21)) {
  cat(sprintf("%.1fs\n", t_21))
  draws_21 <- as.matrix(fit_21$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_21), value = TRUE)[1]
  results$row_21 <- list(slope = check_recovery(draws_21[, slope_col], true_slope, "slope"), time = t_21)
  print_recovery(results$row_21$slope)
  results$row_21$pass <- results$row_21$slope$pass
} else { results$row_21 <- list(error = TRUE) }

# =============================================================================
# Row 51: negbin_negbin + GP + RW1
# =============================================================================
cat("\n========== Row 51: negbin_negbin + GP + RW1 ==========\n")

eta_num_51 <- true_intercept + true_slope * x + spatial_effects + temporal_effects[time]
num_51 <- rnbinom(N_OBS, size = 5, mu = exp(eta_num_51))
df_51 <- data.frame(num = num_51, denom = denom_37, x = x, time = time,
                    coord_x = coord_x, coord_y = coord_y)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_51 <- system.time({
  fit_51 <- tryCatch({
    tratio(num | denom ~ x, data = df_51, family = ratiod_negbin_negbin(),
           spatial = spatial_gp(coords = c("coord_x", "coord_y"), cov = "exponential"),
           temporal = temporal_rw1(time_var = "time"),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_51)) {
  cat(sprintf("%.1fs\n", t_51))
  draws_51 <- as.matrix(fit_51$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_51), value = TRUE)[1]
  results$row_51 <- list(slope = check_recovery(draws_51[, slope_col], true_slope, "slope"), time = t_51)
  print_recovery(results$row_51$slope)
  results$row_51$pass <- results$row_51$slope$pass
} else { results$row_51 <- list(error = TRUE) }

# =============================================================================
# Row 83: binomial + GP + RW1
# =============================================================================
cat("\n========== Row 83: binomial + GP + RW1 ==========\n")

eta_83 <- true_intercept + true_slope * x + spatial_effects + temporal_effects[time]
prob_83 <- plogis(eta_83)
successes_83 <- rbinom(N_OBS, trials_67, prob_83)
df_83 <- data.frame(successes = successes_83, trials = trials_67, x = x, time = time,
                    coord_x = coord_x, coord_y = coord_y)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_83 <- system.time({
  fit_83 <- tryCatch({
    tratio(successes | trials ~ x, data = df_83, family = ratiod_binomial(),
           spatial = spatial_gp(coords = c("coord_x", "coord_y"), cov = "exponential"),
           temporal = temporal_rw1(time_var = "time"),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_83)) {
  cat(sprintf("%.1fs\n", t_83))
  draws_83 <- as.matrix(fit_83$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_83), value = TRUE)[1]
  results$row_83 <- list(slope = check_recovery(draws_83[, slope_col], true_slope, "slope"), time = t_83)
  print_recovery(results$row_83$slope)
  results$row_83$pass <- results$row_83$slope$pass
} else { results$row_83 <- list(error = TRUE) }

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - GP Spatial Simulation Validation\n")
cat("=======================================================\n\n")

for (rn in c("row_7", "row_21", "row_37", "row_51", "row_67", "row_83")) {
  r <- results[[rn]]
  row_num <- gsub("row_", "", rn)
  if (!is.null(r$error) && r$error) {
    cat(sprintf("Row %s: ERROR\n", row_num))
  } else {
    cat(sprintf("Row %s: %s (%.2f SD, %.1fs)\n", row_num,
                if(r$pass) "PASS" else "FAIL", r$slope$diff_sd, r$time))
  }
}

saveRDS(results, "benchmarks/results_sim_gp.rds")
cat("\nResults saved.\n")
