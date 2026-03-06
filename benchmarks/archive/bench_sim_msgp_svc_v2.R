# Simulation-based validation for MSGP and SVC models (v2)
# MSGP: Rows 9, 23, 39, 53, 69, 85
# SVC:  Rows 26, 56, 88
#
# FIXES from v1:
#  - MSGP: N=200 instead of N=50 (was overparameterized: 2*50 spatial > 50 obs)
#  - SVC:  Fixed API: `terms = 1` instead of `varying = "x"` (wrong parameter name)
#  - Added MSGP+RW1 rows (23, 53, 85)

library(numdenom)
library(posterior)

set.seed(42)

# Parameters — N=200 gives enough data for MSGP (2*200=400 GP effects,
# constrained by GP prior). SVC uses N=100 (O(N^2) NNGP, faster).
N_MSGP <- 200
N_SVC  <- 100
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1
N_TIMES <- 10  # for MSGP+RW1 rows

cat("=======================================================\n")
cat("MSGP and SVC Models: Simulation-Based Validation (v2)\n")
cat("MSGP: Rows 9, 23, 39, 53, 69, 85  (N=", N_MSGP, ")\n")
cat("SVC:  Rows 26, 56, 88             (N=", N_SVC, ")\n")
cat("=======================================================\n")
cat(sprintf("MSGP: N=%d, iter=%d, chains=%d\n", N_MSGP, N_ITER, N_CHAINS))
cat(sprintf("SVC:  N=%d, iter=%d, chains=%d\n", N_SVC, N_ITER, N_CHAINS))
cat(sprintf("MSGP+RW1: N_TIMES=%d\n\n", N_TIMES))

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

# True parameters
true_intercept <- 1.0
true_slope <- 0.3

# =============================================================================
# MSGP DATA GENERATION (N=200, unique coords per observation)
# =============================================================================
set.seed(123)
coord_x_msgp <- runif(N_MSGP, 0, 10)
coord_y_msgp <- runif(N_MSGP, 0, 10)
coords_msgp <- cbind(coord_x_msgp, coord_y_msgp)
x_msgp <- rnorm(N_MSGP)

# Generate multiscale GP effects (fine + coarse)
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

spatial_effects_msgp <- generate_msgp_effects(coords_msgp, 0.2, 0.3, 1.0, 4.0)

# Generate RW1 temporal effects for MSGP+RW1 rows
generate_rw1 <- function(n, sigma) {
  gamma <- cumsum(rnorm(n, 0, sigma))
  gamma - mean(gamma)
}

set.seed(456)
temporal_effects <- generate_rw1(N_TIMES, 0.15)
time_idx_msgp <- rep(1:N_TIMES, length.out = N_MSGP)

# =============================================================================
# SVC DATA GENERATION (N=100, unique coords per observation)
# =============================================================================
set.seed(789)
coord_x_svc <- runif(N_SVC, 0, 10)
coord_y_svc <- runif(N_SVC, 0, 10)
coords_svc <- cbind(coord_x_svc, coord_y_svc)
x_svc <- rnorm(N_SVC)

generate_gp_effects <- function(coords, sigma, lengthscale) {
  n <- nrow(coords)
  dist_mat <- as.matrix(dist(coords))
  cov_mat <- sigma^2 * exp(-dist_mat / lengthscale)
  diag(cov_mat) <- diag(cov_mat) + 1e-6
  L <- chol(cov_mat)
  as.vector(t(L) %*% rnorm(n))
}

svc_slope <- generate_gp_effects(coords_svc, 0.2, 3.0)

# =============================================================================
# MSGP MODELS (Rows 9, 39, 69) - No temporal
# =============================================================================

# --- Row 9: poisson_gamma + MSGP ---
cat("\n========== Row 9: poisson_gamma + MSGP ==========\n")

eta_9 <- true_intercept + true_slope * x_msgp + spatial_effects_msgp
effort_9 <- rgamma(N_MSGP, 5, 1)
count_9 <- rpois(N_MSGP, exp(eta_9) * effort_9)
df_9 <- data.frame(count = count_9, effort = effort_9, x = x_msgp,
                   coord_x = coord_x_msgp, coord_y = coord_y_msgp)

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

# --- Row 39: negbin_negbin + MSGP ---
cat("\n========== Row 39: negbin_negbin + MSGP ==========\n")

eta_num_39 <- true_intercept + true_slope * x_msgp + spatial_effects_msgp
eta_denom_39 <- 0.5 + 0.2 * x_msgp + spatial_effects_msgp * 0.8
num_39 <- rnbinom(N_MSGP, size = 5, mu = exp(eta_num_39))
denom_39 <- rnbinom(N_MSGP, size = 5, mu = exp(eta_denom_39))
denom_39[denom_39 == 0] <- 1
df_39 <- data.frame(num = num_39, denom = denom_39, x = x_msgp,
                    coord_x = coord_x_msgp, coord_y = coord_y_msgp)

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

# --- Row 69: binomial + MSGP ---
cat("\n========== Row 69: binomial + MSGP ==========\n")

eta_69 <- true_intercept + true_slope * x_msgp + spatial_effects_msgp
prob_69 <- plogis(eta_69)
trials_69 <- sample(20:50, N_MSGP, replace = TRUE)
successes_69 <- rbinom(N_MSGP, trials_69, prob_69)
df_69 <- data.frame(successes = successes_69, trials = trials_69, x = x_msgp,
                    coord_x = coord_x_msgp, coord_y = coord_y_msgp)

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
# MSGP + RW1 MODELS (Rows 23, 53, 85)
# =============================================================================

# --- Row 23: poisson_gamma + MSGP + RW1 ---
cat("\n========== Row 23: poisson_gamma + MSGP + RW1 ==========\n")

eta_23 <- true_intercept + true_slope * x_msgp + spatial_effects_msgp + temporal_effects[time_idx_msgp]
effort_23 <- rgamma(N_MSGP, 5, 1)
count_23 <- rpois(N_MSGP, exp(eta_23) * effort_23)
df_23 <- data.frame(count = count_23, effort = effort_23, x = x_msgp,
                    coord_x = coord_x_msgp, coord_y = coord_y_msgp,
                    time = time_idx_msgp)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_23 <- system.time({
  fit_23 <- tryCatch({
    ratiod(count | effort ~ x, data = df_23, family = ratiod_poisson_gamma(),
           spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
           temporal = temporal_rw1(time_var = "time"),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_23)) {
  cat(sprintf("%.1fs\n", t_23))
  draws_23 <- as.matrix(fit_23$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_23), value = TRUE)[1]
  results$row_23 <- list(slope = check_recovery(draws_23[, slope_col], true_slope, "slope"), time = t_23)
  print_recovery(results$row_23$slope)
  results$row_23$pass <- results$row_23$slope$pass
} else { results$row_23 <- list(error = TRUE) }

# --- Row 53: negbin_negbin + MSGP + RW1 ---
cat("\n========== Row 53: negbin_negbin + MSGP + RW1 ==========\n")

eta_num_53 <- true_intercept + true_slope * x_msgp + spatial_effects_msgp + temporal_effects[time_idx_msgp]
eta_denom_53 <- 0.5 + 0.2 * x_msgp + spatial_effects_msgp * 0.8 + temporal_effects[time_idx_msgp] * 0.6
num_53 <- rnbinom(N_MSGP, size = 5, mu = exp(eta_num_53))
denom_53 <- rnbinom(N_MSGP, size = 5, mu = exp(eta_denom_53))
denom_53[denom_53 == 0] <- 1
df_53 <- data.frame(num = num_53, denom = denom_53, x = x_msgp,
                    coord_x = coord_x_msgp, coord_y = coord_y_msgp,
                    time = time_idx_msgp)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_53 <- system.time({
  fit_53 <- tryCatch({
    ratiod(num | denom ~ x, data = df_53, family = ratiod_negbin_negbin(),
           spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
           temporal = temporal_rw1(time_var = "time"),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_53)) {
  cat(sprintf("%.1fs\n", t_53))
  draws_53 <- as.matrix(fit_53$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_53), value = TRUE)[1]
  results$row_53 <- list(slope = check_recovery(draws_53[, slope_col], true_slope, "slope"), time = t_53)
  print_recovery(results$row_53$slope)
  results$row_53$pass <- results$row_53$slope$pass
} else { results$row_53 <- list(error = TRUE) }

# --- Row 85: binomial + MSGP + RW1 ---
cat("\n========== Row 85: binomial + MSGP + RW1 ==========\n")

eta_85 <- true_intercept + true_slope * x_msgp + spatial_effects_msgp + temporal_effects[time_idx_msgp]
prob_85 <- plogis(eta_85)
trials_85 <- sample(20:50, N_MSGP, replace = TRUE)
successes_85 <- rbinom(N_MSGP, trials_85, prob_85)
df_85 <- data.frame(successes = successes_85, trials = trials_85, x = x_msgp,
                    coord_x = coord_x_msgp, coord_y = coord_y_msgp,
                    time = time_idx_msgp)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_85 <- system.time({
  fit_85 <- tryCatch({
    ratiod(successes | trials ~ x, data = df_85, family = ratiod_binomial(),
           spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
           temporal = temporal_rw1(time_var = "time"),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_85)) {
  cat(sprintf("%.1fs\n", t_85))
  draws_85 <- as.matrix(fit_85$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_85), value = TRUE)[1]
  results$row_85 <- list(slope = check_recovery(draws_85[, slope_col], true_slope, "slope"), time = t_85)
  print_recovery(results$row_85$slope)
  results$row_85$pass <- results$row_85$slope$pass
} else { results$row_85 <- list(error = TRUE) }

# =============================================================================
# SVC MODELS (Rows 26, 56, 88) - Fixed: terms instead of varying
# =============================================================================

# --- Row 26: poisson_gamma + SVC ---
cat("\n========== Row 26: poisson_gamma + SVC ==========\n")

eta_26 <- true_intercept + (true_slope + svc_slope) * x_svc
effort_26 <- rgamma(N_SVC, 5, 1)
count_26 <- rpois(N_SVC, exp(eta_26) * effort_26)
df_26 <- data.frame(count = count_26, effort = effort_26, x = x_svc,
                    coord_x = coord_x_svc, coord_y = coord_y_svc)

cat(sprintf("True: mean_slope=%.2f (varies spatially)\n", true_slope))
cat("Fitting... ")
t_26 <- system.time({
  fit_26 <- tryCatch({
    ratiod(count | effort ~ x, data = df_26, family = ratiod_poisson_gamma(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 1),
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

# --- Row 56: negbin_negbin + SVC ---
cat("\n========== Row 56: negbin_negbin + SVC ==========\n")

eta_num_56 <- true_intercept + (true_slope + svc_slope) * x_svc
eta_denom_56 <- 0.5 + 0.2 * x_svc
num_56 <- rnbinom(N_SVC, size = 5, mu = exp(eta_num_56))
denom_56 <- rnbinom(N_SVC, size = 5, mu = exp(eta_denom_56))
denom_56[denom_56 == 0] <- 1
df_56 <- data.frame(num = num_56, denom = denom_56, x = x_svc,
                    coord_x = coord_x_svc, coord_y = coord_y_svc)

cat(sprintf("True: mean_slope=%.2f (varies spatially)\n", true_slope))
cat("Fitting... ")
t_56 <- system.time({
  fit_56 <- tryCatch({
    ratiod(num | denom ~ x, data = df_56, family = ratiod_negbin_negbin(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 1),
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

# --- Row 88: binomial + SVC ---
cat("\n========== Row 88: binomial + SVC ==========\n")

eta_88 <- true_intercept + (true_slope + svc_slope) * x_svc
prob_88 <- plogis(eta_88)
trials_88 <- sample(20:50, N_SVC, replace = TRUE)
successes_88 <- rbinom(N_SVC, trials_88, prob_88)
df_88 <- data.frame(successes = successes_88, trials = trials_88, x = x_svc,
                    coord_x = coord_x_svc, coord_y = coord_y_svc)

cat(sprintf("True: mean_slope=%.2f (varies spatially)\n", true_slope))
cat("Fitting... ")
t_88 <- system.time({
  fit_88 <- tryCatch({
    ratiod(successes | trials ~ x, data = df_88, family = ratiod_binomial(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 1),
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
cat("SUMMARY - MSGP and SVC Simulation Validation (v2)\n")
cat("=======================================================\n\n")

all_rows <- c("row_9", "row_23", "row_39", "row_53", "row_69", "row_85",
              "row_26", "row_56", "row_88")

for (rn in all_rows) {
  r <- results[[rn]]
  row_num <- gsub("row_", "", rn)
  if (is.null(r)) {
    cat(sprintf("Row %s: SKIPPED\n", row_num))
  } else if (!is.null(r$error) && r$error) {
    cat(sprintf("Row %s: ERROR\n", row_num))
  } else {
    cat(sprintf("Row %s: %s (%.2f SD, %.1fs)\n", row_num,
                if(r$pass) "PASS" else "FAIL", r$slope$diff_sd, r$time))
  }
}

n_pass <- sum(sapply(all_rows, function(rn) {
  r <- results[[rn]]
  if (!is.null(r) && is.null(r$error) && r$pass) 1 else 0
}))
cat(sprintf("\n%d / %d PASSED\n", n_pass, length(all_rows)))

saveRDS(results, "benchmarks/results_sim_msgp_svc_v2.rds")
cat("\nResults saved to benchmarks/results_sim_msgp_svc_v2.rds\n")
