# Simulation-based validation for MSGP and SVC models
# MSGP: Rows 9, 39, 69
# SVC: Rows 26, 56, 88
# NOTE: Both are SLOW (O(N²) or O(N³)) - using smaller N
# NOTE: Uses unique coordinates per observation (NNGP requirement)

library(numdenom)
library(posterior)

set.seed(42)

# Very small parameters for MSGP/SVC (extremely slow)
N_OBS <- 50
N_ITER <- 300
N_WARMUP <- 150
N_CHAINS <- 1

cat("=======================================================\n")
cat("MSGP and SVC Models: Simulation-Based Validation\n")
cat("MSGP: Rows 9, 39, 69\n")
cat("SVC: Rows 26, 56, 88\n")
cat("=======================================================\n")
cat(sprintf("N=%d, iter=%d, chains=%d\n\n", N_OBS, N_ITER, N_CHAINS))
cat("NOTE: MSGP+RW1 rows (23, 53, 85) skipped due to extreme runtime\n\n")

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

# Generate UNIQUE coordinates per observation
set.seed(123)
coord_x <- runif(N_OBS, 0, 10)
coord_y <- runif(N_OBS, 0, 10)
coords_mat <- cbind(coord_x, coord_y)

# Generate multiscale GP effects (fine + coarse) at observation level
generate_msgp_effects <- function(coords, sigma_fine, sigma_coarse, ls_fine, ls_coarse) {
  n <- nrow(coords)
  dist_mat <- as.matrix(dist(coords))

  # Fine scale
  cov_fine <- sigma_fine^2 * exp(-dist_mat / ls_fine)
  diag(cov_fine) <- diag(cov_fine) + 1e-6
  L_fine <- chol(cov_fine)
  fine <- as.vector(t(L_fine) %*% rnorm(n))

  # Coarse scale
  cov_coarse <- sigma_coarse^2 * exp(-dist_mat / ls_coarse)
  diag(cov_coarse) <- diag(cov_coarse) + 1e-6
  L_coarse <- chol(cov_coarse)
  coarse <- as.vector(t(L_coarse) %*% rnorm(n))

  effects <- fine + coarse
  effects - mean(effects)
}

# Generate SVC effects
generate_svc_effects <- function(coords, sigma, lengthscale) {
  n <- nrow(coords)
  dist_mat <- as.matrix(dist(coords))
  cov_mat <- sigma^2 * exp(-dist_mat / lengthscale)
  diag(cov_mat) <- diag(cov_mat) + 1e-6
  L <- chol(cov_mat)
  as.vector(t(L) %*% rnorm(n))
}

# True parameters
true_intercept <- 1.0
true_slope <- 0.3

# Generate effects at observation level
spatial_effects_msgp <- generate_msgp_effects(coords_mat, 0.2, 0.3, 1.0, 4.0)
svc_slope <- generate_svc_effects(coords_mat, 0.2, 3.0)

# Common covariate
x <- rnorm(N_OBS)

# =============================================================================
# MSGP MODELS (Rows 9, 39, 69)
# =============================================================================

# Row 9: poisson_gamma + MSGP
cat("\n========== Row 9: poisson_gamma + MSGP ==========\n")

eta_9 <- true_intercept + true_slope * x + spatial_effects_msgp
effort_9 <- rgamma(N_OBS, 5, 1)
count_9 <- rpois(N_OBS, exp(eta_9) * effort_9)
df_9 <- data.frame(count = count_9, effort = effort_9, x = x,
                   coord_x = coord_x, coord_y = coord_y)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_9 <- system.time({
  fit_9 <- tryCatch({
    ratiod(count | effort ~ x, data = df_9, family = ratiod_poisson_gamma(),
           spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_9)) {
  cat(sprintf("%.1fs\n", t_9))
  draws_9 <- as.matrix(fit_9$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_9), value = TRUE)[1]
  results$row_9 <- list(slope = check_recovery(draws_9[, slope_col], true_slope, "slope"), time = t_9)
  print_recovery(results$row_9$slope)
  results$row_9$pass <- results$row_9$slope$pass
} else { results$row_9 <- list(error = TRUE) }

# Row 39: negbin_negbin + MSGP
cat("\n========== Row 39: negbin_negbin + MSGP ==========\n")

eta_num_39 <- true_intercept + true_slope * x + spatial_effects_msgp
eta_denom_39 <- 0.5 + 0.2 * x + spatial_effects_msgp * 0.8

num_39 <- rnbinom(N_OBS, size = 5, mu = exp(eta_num_39))
denom_39 <- rnbinom(N_OBS, size = 5, mu = exp(eta_denom_39))
denom_39[denom_39 == 0] <- 1
df_39 <- data.frame(num = num_39, denom = denom_39, x = x,
                    coord_x = coord_x, coord_y = coord_y)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_39 <- system.time({
  fit_39 <- tryCatch({
    ratiod(num | denom ~ x, data = df_39, family = ratiod_negbin_negbin(),
           spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_39)) {
  cat(sprintf("%.1fs\n", t_39))
  draws_39 <- as.matrix(fit_39$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_39), value = TRUE)[1]
  results$row_39 <- list(slope = check_recovery(draws_39[, slope_col], true_slope, "slope"), time = t_39)
  print_recovery(results$row_39$slope)
  results$row_39$pass <- results$row_39$slope$pass
} else { results$row_39 <- list(error = TRUE) }

# Row 69: binomial + MSGP
cat("\n========== Row 69: binomial + MSGP ==========\n")

eta_69 <- true_intercept + true_slope * x + spatial_effects_msgp
prob_69 <- plogis(eta_69)
trials_69 <- sample(20:50, N_OBS, replace = TRUE)
successes_69 <- rbinom(N_OBS, trials_69, prob_69)
df_69 <- data.frame(successes = successes_69, trials = trials_69, x = x,
                    coord_x = coord_x, coord_y = coord_y)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_69 <- system.time({
  fit_69 <- tryCatch({
    ratiod(successes | trials ~ x, data = df_69, family = ratiod_binomial(),
           spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_69)) {
  cat(sprintf("%.1fs\n", t_69))
  draws_69 <- as.matrix(fit_69$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_69), value = TRUE)[1]
  results$row_69 <- list(slope = check_recovery(draws_69[, slope_col], true_slope, "slope"), time = t_69)
  print_recovery(results$row_69$slope)
  results$row_69$pass <- results$row_69$slope$pass
} else { results$row_69 <- list(error = TRUE) }

# =============================================================================
# SVC MODELS (Rows 26, 56, 88)
# =============================================================================

# Row 26: poisson_gamma + SVC
cat("\n========== Row 26: poisson_gamma + SVC ==========\n")

eta_26 <- true_intercept + (true_slope + svc_slope) * x
count_26 <- rpois(N_OBS, exp(eta_26) * effort_9)
df_26 <- data.frame(count = count_26, effort = effort_9, x = x,
                    coord_x = coord_x, coord_y = coord_y)

cat(sprintf("True: mean_slope=%.2f (varies by location)\n", true_slope))
cat("Fitting... ")
t_26 <- system.time({
  fit_26 <- tryCatch({
    ratiod(count | effort ~ x, data = df_26, family = ratiod_poisson_gamma(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), varying = "x"),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_26)) {
  cat(sprintf("%.1fs\n", t_26))
  draws_26 <- as.matrix(fit_26$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_26), value = TRUE)[1]
  results$row_26 <- list(slope = check_recovery(draws_26[, slope_col], true_slope, "mean_slope"), time = t_26)
  print_recovery(results$row_26$slope)
  results$row_26$pass <- results$row_26$slope$pass
} else { results$row_26 <- list(error = TRUE) }

# Row 56: negbin_negbin + SVC
cat("\n========== Row 56: negbin_negbin + SVC ==========\n")

eta_num_56 <- true_intercept + (true_slope + svc_slope) * x
num_56 <- rnbinom(N_OBS, size = 5, mu = exp(eta_num_56))
df_56 <- data.frame(num = num_56, denom = denom_39, x = x,
                    coord_x = coord_x, coord_y = coord_y)

cat(sprintf("True: mean_slope=%.2f (varies by location)\n", true_slope))
cat("Fitting... ")
t_56 <- system.time({
  fit_56 <- tryCatch({
    ratiod(num | denom ~ x, data = df_56, family = ratiod_negbin_negbin(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), varying = "x"),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_56)) {
  cat(sprintf("%.1fs\n", t_56))
  draws_56 <- as.matrix(fit_56$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_56), value = TRUE)[1]
  results$row_56 <- list(slope = check_recovery(draws_56[, slope_col], true_slope, "mean_slope"), time = t_56)
  print_recovery(results$row_56$slope)
  results$row_56$pass <- results$row_56$slope$pass
} else { results$row_56 <- list(error = TRUE) }

# Row 88: binomial + SVC
cat("\n========== Row 88: binomial + SVC ==========\n")

eta_88 <- true_intercept + (true_slope + svc_slope) * x
prob_88 <- plogis(eta_88)
successes_88 <- rbinom(N_OBS, trials_69, prob_88)
df_88 <- data.frame(successes = successes_88, trials = trials_69, x = x,
                    coord_x = coord_x, coord_y = coord_y)

cat(sprintf("True: mean_slope=%.2f (varies by location)\n", true_slope))
cat("Fitting... ")
t_88 <- system.time({
  fit_88 <- tryCatch({
    ratiod(successes | trials ~ x, data = df_88, family = ratiod_binomial(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), varying = "x"),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_88)) {
  cat(sprintf("%.1fs\n", t_88))
  draws_88 <- as.matrix(fit_88$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_88), value = TRUE)[1]
  results$row_88 <- list(slope = check_recovery(draws_88[, slope_col], true_slope, "mean_slope"), time = t_88)
  print_recovery(results$row_88$slope)
  results$row_88$pass <- results$row_88$slope$pass
} else { results$row_88 <- list(error = TRUE) }

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - MSGP and SVC Simulation Validation\n")
cat("=======================================================\n\n")

for (rn in c("row_9", "row_39", "row_69", "row_26", "row_56", "row_88")) {
  r <- results[[rn]]
  row_num <- gsub("row_", "", rn)
  if (!is.null(r$error) && r$error) {
    cat(sprintf("Row %s: ERROR\n", row_num))
  } else {
    cat(sprintf("Row %s: %s (%.2f SD, %.1fs)\n", row_num,
                if(r$pass) "PASS" else "FAIL", r$slope$diff_sd, r$time))
  }
}

cat("\nNote: MSGP+RW1 rows (23, 53, 85) marked as 'runs only' due to extreme runtime\n")

saveRDS(results, "benchmarks/results_sim_msgp_svc.rds")
cat("\nResults saved.\n")
