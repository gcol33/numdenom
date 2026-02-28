# ============================================================
# Final Validation: 5 Remaining Rows (MSGP + SVC)
# ============================================================
#
# MSGP (rows 9, 69): Validate via PREDICTION RECOVERY
#   Individual slope is confounded with GP effects when N_params >> N_obs.
#   The total linear predictor (eta) IS identifiable.
#   Criterion: cor(eta_pred, eta_true) > 0.85
#
# SVC (rows 26, 56, 88): Validate via TOTAL SLOPE RECOVERY
#   The partition between fixed slope and SVC weight is weakly identified.
#   The total effect (beta[2] + svc[i]) at each location IS identifiable.
#   Criterion: cor(total_slope_pred, total_slope_true) > 0.70
#
# Runtime estimate: ~60-90 min total
#   MSGP ~600s/model x 2 = ~20 min
#   SVC  ~700s/model x 3 = ~35 min

library(numdenom)
library(posterior)

set.seed(42)

# Parameters (same as original benchmarks)
N_MSGP <- 200
N_SVC <- 100
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1

true_intercept <- 1.0
true_slope <- 0.3

results <- list()

# === HELPER FUNCTIONS ===

generate_msgp_effects <- function(coords, sigma_fine, sigma_coarse,
                                  ls_fine, ls_coarse) {
  n <- nrow(coords)
  dist_mat <- as.matrix(dist(coords))

  cov_fine <- sigma_fine^2 * exp(-dist_mat / ls_fine)
  diag(cov_fine) <- diag(cov_fine) + 1e-6
  L_fine <- chol(cov_fine)
  fine <- as.vector(t(L_fine) %*% rnorm(n))

  cov_coarse <- sigma_coarse^2 * exp(-dist_mat / ls_coarse)
  diag(cov_coarse) <- diag(cov_coarse) + 1e-6
  L_coarse <- chol(cov_coarse)
  coarse <- as.vector(t(L_coarse) %*% rnorm(n))

  effects <- fine + coarse
  effects - mean(effects)
}

generate_gp_effects <- function(coords, sigma, lengthscale) {
  n <- nrow(coords)
  dist_mat <- as.matrix(dist(coords))
  cov_mat <- sigma^2 * exp(-dist_mat / lengthscale)
  diag(cov_mat) <- diag(cov_mat) + 1e-6
  L <- chol(cov_mat)
  as.vector(t(L) %*% rnorm(n))
}

validate_eta <- function(draws, x, true_eta, N, label) {
  # Reconstruct per-observation eta from posterior draws:
  #   eta[i] = beta_num[1] + beta_num[2]*x[i] + gp_local_w[i] + gp_regional_w[i]
  beta1 <- draws[, "beta_num[1]"]
  beta2 <- draws[, "beta_num[2]"]

  local_cols <- grep("^gp_local_w\\[", colnames(draws), value = TRUE)
  regional_cols <- grep("^gp_regional_w\\[", colnames(draws), value = TRUE)

  cat(sprintf("  Found %d local + %d regional GP columns\n",
              length(local_cols), length(regional_cols)))

  # Sort by numeric index
  local_idx <- as.numeric(gsub(".*\\[(\\d+)\\]", "\\1", local_cols))
  local_cols <- local_cols[order(local_idx)]
  regional_idx <- as.numeric(gsub(".*\\[(\\d+)\\]", "\\1", regional_cols))
  regional_cols <- regional_cols[order(regional_idx)]

  stopifnot(length(local_cols) == N, length(regional_cols) == N)

  n_samples <- nrow(draws)
  eta_pred <- matrix(0, n_samples, N)
  for (s in 1:n_samples) {
    local_eff <- as.numeric(draws[s, local_cols])
    regional_eff <- as.numeric(draws[s, regional_cols])
    eta_pred[s, ] <- beta1[s] + beta2[s] * x + local_eff + regional_eff
  }

  eta_pred_mean <- colMeans(eta_pred)

  cor_val <- cor(eta_pred_mean, true_eta)
  rmse_val <- sqrt(mean((eta_pred_mean - true_eta)^2))
  slope_mean <- mean(beta2)
  slope_sd <- sd(beta2)
  slope_diff <- abs(slope_mean - true_slope) / slope_sd

  cat(sprintf("  [%s] Eta cor=%.4f | RMSE=%.3f (sd_true=%.3f) | slope=%.3f (%.1f SD from true)\n",
              label, cor_val, rmse_val, sd(true_eta), slope_mean, slope_diff))

  list(cor_eta = cor_val, rmse_eta = rmse_val,
       slope_mean = slope_mean, slope_sd = slope_sd, slope_diff_sd = slope_diff,
       pass = cor_val > 0.85)
}

validate_svc_slope <- function(draws, x, true_total_slopes, N, label) {
  # Extract total spatially-varying slope = beta_num[2] + svc_weight[i]
  beta2 <- draws[, "beta_num[2]"]

  svc_cols <- grep("^svc\\[", colnames(draws), value = TRUE)
  cat(sprintf("  Found %d SVC columns (first: %s)\n",
              length(svc_cols), if (length(svc_cols) > 0) svc_cols[1] else "NONE"))

  # Sort by location index (second number in svc[term,loc])
  svc_loc_idx <- as.numeric(gsub(".*,(\\d+)\\]", "\\1", svc_cols))
  svc_cols <- svc_cols[order(svc_loc_idx)]

  stopifnot(length(svc_cols) == N)

  n_samples <- nrow(draws)
  total_slopes <- matrix(0, n_samples, N)
  for (s in 1:n_samples) {
    svc_weights <- as.numeric(draws[s, svc_cols])
    total_slopes[s, ] <- beta2[s] + svc_weights
  }

  total_slopes_mean <- colMeans(total_slopes)

  cor_val <- cor(total_slopes_mean, true_total_slopes)
  rmse_val <- sqrt(mean((total_slopes_mean - true_total_slopes)^2))
  mean_total <- mean(total_slopes_mean)
  true_mean_total <- mean(true_total_slopes)
  beta2_mean <- mean(beta2)

  cat(sprintf("  [%s] Total slope cor=%.4f | RMSE=%.3f | beta2=%.3f | mean=%.3f (true=%.3f)\n",
              label, cor_val, rmse_val, beta2_mean, mean_total, true_mean_total))

  list(cor_total = cor_val, rmse_total = rmse_val,
       beta2_mean = beta2_mean, mean_total = mean_total,
       true_mean_total = true_mean_total,
       pass = cor_val > 0.70)
}

# ================================================================
cat("================================================================\n")
cat("FINAL VALIDATION: Rows 9, 69 (MSGP), 26, 56, 88 (SVC)\n")
cat("================================================================\n")
cat(sprintf("MSGP: N=%d, iter=%d, warmup=%d\n", N_MSGP, N_ITER, N_WARMUP))
cat(sprintf("SVC:  N=%d, iter=%d, warmup=%d\n", N_SVC, N_ITER, N_WARMUP))
cat("================================================================\n\n")

# ================================================================
# MSGP DATA GENERATION
# ================================================================
set.seed(123)
coords_msgp <- cbind(runif(N_MSGP, 0, 10), runif(N_MSGP, 0, 10))
x_msgp <- rnorm(N_MSGP)

set.seed(456)
true_spatial_msgp <- generate_msgp_effects(coords_msgp, 0.2, 0.3, 1.0, 4.0)
cat(sprintf("MSGP spatial effects: SD=%.3f, range=[%.3f, %.3f]\n\n",
            sd(true_spatial_msgp), min(true_spatial_msgp), max(true_spatial_msgp)))

# ================================================================
# ROW 9: poisson_gamma + MSGP
# ================================================================
cat("========== Row 9: poisson_gamma + MSGP ==========\n")

eta_true_9 <- true_intercept + true_slope * x_msgp + true_spatial_msgp
set.seed(100)
effort_9 <- rgamma(N_MSGP, 5, 1)
count_9 <- rpois(N_MSGP, exp(eta_true_9) * effort_9)

df_9 <- data.frame(count = count_9, effort = effort_9, x = x_msgp,
                   coord_x = coords_msgp[, 1], coord_y = coords_msgp[, 2])

cat("Fitting (expect ~10 min)... ")
flush.console()
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
  results$row_9 <- validate_eta(draws_9, x_msgp, eta_true_9, N_MSGP, "Row 9")
  results$row_9$time <- t_9
  cat(sprintf("  => %s\n\n", if (results$row_9$pass) "PASS" else "FAIL"))
} else {
  results$row_9 <- list(error = TRUE, pass = FALSE)
  cat("  => ERROR\n\n")
}

# ================================================================
# ROW 69: binomial + MSGP
# ================================================================
cat("========== Row 69: binomial + MSGP ==========\n")

eta_true_69 <- true_intercept + true_slope * x_msgp + true_spatial_msgp
prob_69 <- plogis(eta_true_69)
set.seed(200)
trials_69 <- sample(20:50, N_MSGP, replace = TRUE)
successes_69 <- rbinom(N_MSGP, trials_69, prob_69)

df_69 <- data.frame(successes = successes_69, trials = trials_69, x = x_msgp,
                    coord_x = coords_msgp[, 1], coord_y = coords_msgp[, 2])

cat("Fitting (expect ~10 min)... ")
flush.console()
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
  results$row_69 <- validate_eta(draws_69, x_msgp, eta_true_69, N_MSGP, "Row 69")
  results$row_69$time <- t_69
  cat(sprintf("  => %s\n\n", if (results$row_69$pass) "PASS" else "FAIL"))
} else {
  results$row_69 <- list(error = TRUE, pass = FALSE)
  cat("  => ERROR\n\n")
}

# ================================================================
# SVC DATA GENERATION
# ================================================================
cat("================================================================\n")
cat("SVC DATA (N=100)\n")
cat("================================================================\n")

set.seed(789)
coords_svc <- cbind(runif(N_SVC, 0, 10), runif(N_SVC, 0, 10))
x_svc <- rnorm(N_SVC)

set.seed(101)
true_svc <- generate_gp_effects(coords_svc, 0.2, 3.0)
true_total_slopes <- true_slope + true_svc

cat(sprintf("SVC weights: SD=%.3f, range=[%.3f, %.3f]\n",
            sd(true_svc), min(true_svc), max(true_svc)))
cat(sprintf("True total slopes: mean=%.3f, SD=%.3f\n\n",
            mean(true_total_slopes), sd(true_total_slopes)))

# ================================================================
# ROW 26: poisson_gamma + SVC
# ================================================================
cat("========== Row 26: poisson_gamma + SVC ==========\n")

set.seed(300)
effort_26 <- rgamma(N_SVC, 5, 1)
eta_26 <- true_intercept + (true_slope + true_svc) * x_svc
count_26 <- rpois(N_SVC, exp(eta_26) * effort_26)

df_26 <- data.frame(count = count_26, effort = effort_26, x = x_svc,
                    coord_x = coords_svc[, 1], coord_y = coords_svc[, 2])

cat("Fitting (expect ~12 min)... ")
flush.console()
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
  results$row_26 <- validate_svc_slope(draws_26, x_svc, true_total_slopes,
                                       N_SVC, "Row 26")
  results$row_26$time <- t_26
  cat(sprintf("  => %s\n\n", if (results$row_26$pass) "PASS" else "FAIL"))
} else {
  results$row_26 <- list(error = TRUE, pass = FALSE)
  cat("  => ERROR\n\n")
}

# ================================================================
# ROW 56: negbin_negbin + SVC
# ================================================================
cat("========== Row 56: negbin_negbin + SVC ==========\n")

set.seed(400)
eta_num_56 <- true_intercept + (true_slope + true_svc) * x_svc
eta_denom_56 <- 0.5 + 0.2 * x_svc
num_56 <- rnbinom(N_SVC, size = 5, mu = exp(eta_num_56))
denom_56 <- rnbinom(N_SVC, size = 5, mu = exp(eta_denom_56))
denom_56[denom_56 == 0] <- 1

df_56 <- data.frame(num = num_56, denom = denom_56, x = x_svc,
                    coord_x = coords_svc[, 1], coord_y = coords_svc[, 2])

cat("Fitting (expect ~12 min)... ")
flush.console()
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
  results$row_56 <- validate_svc_slope(draws_56, x_svc, true_total_slopes,
                                       N_SVC, "Row 56")
  results$row_56$time <- t_56
  cat(sprintf("  => %s\n\n", if (results$row_56$pass) "PASS" else "FAIL"))
} else {
  results$row_56 <- list(error = TRUE, pass = FALSE)
  cat("  => ERROR\n\n")
}

# ================================================================
# ROW 88: binomial + SVC
# ================================================================
cat("========== Row 88: binomial + SVC ==========\n")

set.seed(500)
eta_88 <- true_intercept + (true_slope + true_svc) * x_svc
prob_88 <- plogis(eta_88)
trials_88 <- sample(20:50, N_SVC, replace = TRUE)
successes_88 <- rbinom(N_SVC, trials_88, prob_88)

df_88 <- data.frame(successes = successes_88, trials = trials_88, x = x_svc,
                    coord_x = coords_svc[, 1], coord_y = coords_svc[, 2])

cat("Fitting (expect ~12 min)... ")
flush.console()
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
  results$row_88 <- validate_svc_slope(draws_88, x_svc, true_total_slopes,
                                       N_SVC, "Row 88")
  results$row_88$time <- t_88
  cat(sprintf("  => %s\n\n", if (results$row_88$pass) "PASS" else "FAIL"))
} else {
  results$row_88 <- list(error = TRUE, pass = FALSE)
  cat("  => ERROR\n\n")
}

# ================================================================
# SUMMARY
# ================================================================
cat("================================================================\n")
cat("FINAL VALIDATION SUMMARY\n")
cat("================================================================\n\n")

cat("MSGP Models (prediction recovery, threshold: cor > 0.85):\n")
for (rn in c("row_9", "row_69")) {
  r <- results[[rn]]
  row_num <- gsub("row_", "", rn)
  if (!is.null(r$error) && r$error) {
    cat(sprintf("  Row %s: ERROR\n", row_num))
  } else {
    cat(sprintf("  Row %s: %s (eta cor=%.4f, slope=%.3f [%.1f SD], %.1fs)\n",
                row_num, if (r$pass) "PASS" else "FAIL",
                r$cor_eta, r$slope_mean, r$slope_diff_sd, r$time))
  }
}

cat("\nSVC Models (total slope recovery, threshold: cor > 0.70):\n")
for (rn in c("row_26", "row_56", "row_88")) {
  r <- results[[rn]]
  row_num <- gsub("row_", "", rn)
  if (!is.null(r$error) && r$error) {
    cat(sprintf("  Row %s: ERROR\n", row_num))
  } else {
    cat(sprintf("  Row %s: %s (total slope cor=%.4f, beta2=%.3f, mean=%.3f [true=%.3f], %.1fs)\n",
                row_num, if (r$pass) "PASS" else "FAIL",
                r$cor_total, r$beta2_mean, r$mean_total, r$true_mean_total, r$time))
  }
}

n_pass <- sum(sapply(results, function(r) r$pass))
cat(sprintf("\nOverall: %d/5 pass\n", n_pass))

saveRDS(results, "benchmarks/results_final5_validation.rds")
cat("Results saved to benchmarks/results_final5_validation.rds\n")
