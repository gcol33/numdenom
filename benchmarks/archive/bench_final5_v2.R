# ============================================================
# Final Validation v2: Rows 9, 69 (MSGP) + 26, 56, 88 (SVC)
# ============================================================
# Key fixes from v1:
# 1. SVC uses terms=1 with INTERCEPT-varying data (not slope)
#    Previous script had terms=1 but slope-varying data = mismatch
# 2. All validation uses PREDICTION RECOVERY (total eta correlation)
#    This is robust to identifiability issues
# 3. MSGP Row 9: uses grouped observations (20 sites x 10 obs/site)
#    Reduces GP params from 400 to 40
# ============================================================

library(numdenom)
library(posterior)

set.seed(42)

N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1

true_intercept <- 1.0
true_slope <- 0.3

generate_gp_effects <- function(coords, sigma, lengthscale) {
  n <- nrow(coords)
  dist_mat <- as.matrix(dist(coords))
  cov_mat <- sigma^2 * exp(-dist_mat / lengthscale)
  diag(cov_mat) <- diag(cov_mat) + 1e-6
  L <- chol(cov_mat)
  as.vector(t(L) %*% rnorm(n))
}

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

# Prediction recovery: compute total eta from draws and compare to true
validate_prediction <- function(draws, x, true_eta, N, label,
                                spatial_cols = NULL, site_map = NULL) {
  beta1 <- draws[, "beta_num[1]"]
  beta2 <- draws[, "beta_num[2]"]

  n_samples <- nrow(draws)
  eta_pred <- matrix(0, n_samples, N)

  for (s in 1:n_samples) {
    eta_pred[s, ] <- beta1[s] + beta2[s] * x
  }

  # Add spatial effects if present
  if (!is.null(spatial_cols)) {
    for (s in 1:n_samples) {
      spatial_eff <- as.numeric(draws[s, spatial_cols])
      if (!is.null(site_map)) {
        # Grouped: map site effects to observations
        eta_pred[s, ] <- eta_pred[s, ] + spatial_eff[site_map]
      } else {
        eta_pred[s, ] <- eta_pred[s, ] + spatial_eff
      }
    }
  }

  eta_pred_mean <- colMeans(eta_pred)
  cor_val <- cor(eta_pred_mean, true_eta)
  rmse_val <- sqrt(mean((eta_pred_mean - true_eta)^2))
  slope_mean <- mean(beta2)
  slope_sd <- sd(beta2)

  cat(sprintf("  [%s] Eta cor=%.4f | RMSE=%.3f (sd_true=%.3f) | slope=%.3f\n",
              label, cor_val, rmse_val, sd(true_eta), slope_mean))

  list(cor_eta = cor_val, rmse_eta = rmse_val,
       slope_mean = slope_mean, slope_sd = slope_sd,
       pass = cor_val > 0.80)  # Threshold 0.80 for prediction recovery
}

results <- list()

cat("================================================================\n")
cat("FINAL VALIDATION v2\n")
cat("================================================================\n")
cat("Threshold: cor(eta_pred, eta_true) > 0.80\n")
cat("================================================================\n\n")

# ================================================================
# ROW 9: poisson_gamma + MSGP (grouped observations)
# ================================================================
cat("========== Row 9: poisson_gamma + MSGP (grouped) ==========\n")

N_SITES_9 <- 30
N_PER_SITE_9 <- 7
N_TOTAL_9 <- N_SITES_9 * N_PER_SITE_9

set.seed(123)
site_coords_9 <- cbind(runif(N_SITES_9, 0, 10), runif(N_SITES_9, 0, 10))
site_id_9 <- rep(1:N_SITES_9, each = N_PER_SITE_9)
coords_9 <- site_coords_9[site_id_9, ]
x_9 <- rnorm(N_TOTAL_9)

set.seed(456)
true_spatial_9 <- generate_msgp_effects(site_coords_9, 0.3, 0.4, 1.5, 5.0)
spatial_per_obs_9 <- true_spatial_9[site_id_9]

cat(sprintf("  N=%d obs, %d sites, %d obs/site\n", N_TOTAL_9, N_SITES_9, N_PER_SITE_9))
cat(sprintf("  MSGP spatial: SD=%.3f, range=[%.3f, %.3f]\n",
            sd(true_spatial_9), min(true_spatial_9), max(true_spatial_9)))

eta_true_9 <- true_intercept + true_slope * x_9 + spatial_per_obs_9
set.seed(100)
effort_9 <- rgamma(N_TOTAL_9, 5, 1)
count_9 <- rpois(N_TOTAL_9, exp(eta_true_9) * effort_9)

df_9 <- data.frame(count = count_9, effort = effort_9, x = x_9,
                   coord_x = coords_9[, 1], coord_y = coords_9[, 2])

cat("Fitting... ")
flush.console()
t_9 <- system.time({
  fit_9 <- tryCatch({
    tratio(count | effort ~ x, data = df_9, family = ratiod_poisson_gamma(),
           spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_9)) {
  cat(sprintf("%.1fs\n", t_9))
  draws_9 <- as.matrix(fit_9$draws)

  local_cols <- grep("^gp_local_w\\[", colnames(draws_9), value = TRUE)
  regional_cols <- grep("^gp_regional_w\\[", colnames(draws_9), value = TRUE)
  n_gp <- length(local_cols)
  cat(sprintf("  GP columns: %d local + %d regional\n", n_gp, length(regional_cols)))

  # Sort by index
  local_idx <- as.numeric(gsub(".*\\[(\\d+)\\]", "\\1", local_cols))
  local_cols <- local_cols[order(local_idx)]
  regional_idx <- as.numeric(gsub(".*\\[(\\d+)\\]", "\\1", regional_cols))
  regional_cols <- regional_cols[order(regional_idx)]

  # Build combined spatial effect per sample
  n_samples <- nrow(draws_9)
  spatial_pred <- matrix(0, n_samples, n_gp)
  for (s in 1:n_samples) {
    spatial_pred[s, ] <- as.numeric(draws_9[s, local_cols]) +
                         as.numeric(draws_9[s, regional_cols])
  }
  spatial_cols_combined <- paste0("msgp_combined[", 1:n_gp, "]")
  colnames(spatial_pred) <- spatial_cols_combined
  draws_9_ext <- cbind(draws_9, spatial_pred)

  if (n_gp == N_SITES_9) {
    results$row_9 <- validate_prediction(draws_9_ext, x_9, eta_true_9, N_TOTAL_9,
                                          "Row 9", spatial_cols_combined, site_id_9)
  } else {
    results$row_9 <- validate_prediction(draws_9_ext, x_9, eta_true_9, N_TOTAL_9,
                                          "Row 9", spatial_cols_combined, NULL)
  }
  results$row_9$time <- t_9
  cat(sprintf("  => %s\n\n", if (results$row_9$pass) "PASS" else "FAIL"))
} else {
  results$row_9 <- list(error = TRUE, pass = FALSE)
  cat("  => ERROR\n\n")
}

# ================================================================
# ROW 69: binomial + MSGP (from v1, already passes)
# ================================================================
cat("========== Row 69: binomial + MSGP ==========\n")

N_MSGP <- 200
set.seed(123)
coords_69 <- cbind(runif(N_MSGP, 0, 10), runif(N_MSGP, 0, 10))
x_69 <- rnorm(N_MSGP)

set.seed(456)
true_spatial_69 <- generate_msgp_effects(coords_69, 0.2, 0.3, 1.0, 4.0)

eta_true_69 <- true_intercept + true_slope * x_69 + true_spatial_69
prob_69 <- plogis(eta_true_69)
set.seed(200)
trials_69 <- sample(20:50, N_MSGP, replace = TRUE)
successes_69 <- rbinom(N_MSGP, trials_69, prob_69)

df_69 <- data.frame(successes = successes_69, trials = trials_69, x = x_69,
                    coord_x = coords_69[, 1], coord_y = coords_69[, 2])

cat("Fitting... ")
flush.console()
t_69 <- system.time({
  fit_69 <- tryCatch({
    tratio(successes | trials ~ x, data = df_69, family = ratiod_binomial(),
           spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_69)) {
  cat(sprintf("%.1fs\n", t_69))
  draws_69 <- as.matrix(fit_69$draws)

  local_cols <- grep("^gp_local_w\\[", colnames(draws_69), value = TRUE)
  regional_cols <- grep("^gp_regional_w\\[", colnames(draws_69), value = TRUE)
  n_gp <- length(local_cols)

  local_idx <- as.numeric(gsub(".*\\[(\\d+)\\]", "\\1", local_cols))
  local_cols <- local_cols[order(local_idx)]
  regional_idx <- as.numeric(gsub(".*\\[(\\d+)\\]", "\\1", regional_cols))
  regional_cols <- regional_cols[order(regional_idx)]

  n_samples <- nrow(draws_69)
  spatial_pred <- matrix(0, n_samples, n_gp)
  for (s in 1:n_samples) {
    spatial_pred[s, ] <- as.numeric(draws_69[s, local_cols]) +
                         as.numeric(draws_69[s, regional_cols])
  }
  spatial_cols_combined <- paste0("msgp_combined[", 1:n_gp, "]")
  colnames(spatial_pred) <- spatial_cols_combined
  draws_69_ext <- cbind(draws_69, spatial_pred)

  results$row_69 <- validate_prediction(draws_69_ext, x_69, eta_true_69, N_MSGP,
                                          "Row 69", spatial_cols_combined, NULL)
  results$row_69$time <- t_69
  cat(sprintf("  => %s\n\n", if (results$row_69$pass) "PASS" else "FAIL"))
} else {
  results$row_69 <- list(error = TRUE, pass = FALSE)
  cat("  => ERROR\n\n")
}

# ================================================================
# SVC DATA GENERATION (INTERCEPT SVC — correct specification)
# ================================================================
cat("================================================================\n")
cat("SVC VALIDATION: Intercept SVC (terms=1, intercept-varying data)\n")
cat("================================================================\n\n")

N_SVC <- 100
set.seed(789)
coords_svc <- cbind(runif(N_SVC, 0, 10), runif(N_SVC, 0, 10))
x_svc <- rnorm(N_SVC)

set.seed(202)
true_svc_intercept <- generate_gp_effects(coords_svc, 0.5, 3.0)

cat(sprintf("SVC intercept weights: SD=%.3f, range=[%.3f, %.3f]\n",
            sd(true_svc_intercept), min(true_svc_intercept), max(true_svc_intercept)))

# eta[i] = (intercept + svc[i]) + slope * x[i]
# SVC on intercept: terms=1 selects (Intercept) column
true_eta_svc <- (true_intercept + true_svc_intercept) + true_slope * x_svc

# ================================================================
# ROW 26: poisson_gamma + SVC (intercept)
# ================================================================
cat("\n========== Row 26: poisson_gamma + SVC(intercept) ==========\n")

set.seed(300)
effort_26 <- rgamma(N_SVC, 5, 1)
count_26 <- rpois(N_SVC, exp(true_eta_svc) * effort_26)

df_26 <- data.frame(count = count_26, effort = effort_26, x = x_svc,
                    coord_x = coords_svc[, 1], coord_y = coords_svc[, 2])

cat("Fitting... ")
flush.console()
t_26 <- system.time({
  fit_26 <- tryCatch({
    tratio(count | effort ~ x, data = df_26, family = ratiod_poisson_gamma(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 1),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_26)) {
  cat(sprintf("%.1fs\n", t_26))
  draws_26 <- as.matrix(fit_26$draws)

  svc_cols <- grep("^svc\\[", colnames(draws_26), value = TRUE)
  svc_loc_idx <- as.numeric(gsub(".*,(\\d+)\\]", "\\1", svc_cols))
  svc_cols <- svc_cols[order(svc_loc_idx)]

  # SVC on intercept: eta[i] = beta1 + svc[i] + beta2*x[i]
  # svc_eta[i] = 1 * w[i] = w[i]
  # Total spatial cols = svc cols (they are the intercept deviations)
  results$row_26 <- validate_prediction(draws_26, x_svc, true_eta_svc, N_SVC,
                                         "Row 26", svc_cols, NULL)
  results$row_26$time <- t_26
  cat(sprintf("  => %s\n\n", if (results$row_26$pass) "PASS" else "FAIL"))
} else {
  results$row_26 <- list(error = TRUE, pass = FALSE)
  cat("  => ERROR\n\n")
}

# ================================================================
# ROW 56: negbin_negbin + SVC (intercept)
# ================================================================
cat("========== Row 56: negbin_negbin + SVC(intercept) ==========\n")

set.seed(400)
eta_num_56 <- true_eta_svc
eta_denom_56 <- 0.5 + 0.2 * x_svc
num_56 <- rnbinom(N_SVC, size = 5, mu = exp(eta_num_56))
denom_56 <- rnbinom(N_SVC, size = 5, mu = exp(eta_denom_56))
denom_56[denom_56 == 0] <- 1

df_56 <- data.frame(num = num_56, denom = denom_56, x = x_svc,
                    coord_x = coords_svc[, 1], coord_y = coords_svc[, 2])

cat("Fitting... ")
flush.console()
t_56 <- system.time({
  fit_56 <- tryCatch({
    tratio(num | denom ~ x, data = df_56, family = ratiod_negbin_negbin(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 1),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_56)) {
  cat(sprintf("%.1fs\n", t_56))
  draws_56 <- as.matrix(fit_56$draws)

  svc_cols <- grep("^svc\\[", colnames(draws_56), value = TRUE)
  svc_loc_idx <- as.numeric(gsub(".*,(\\d+)\\]", "\\1", svc_cols))
  svc_cols <- svc_cols[order(svc_loc_idx)]

  results$row_56 <- validate_prediction(draws_56, x_svc, true_eta_svc, N_SVC,
                                         "Row 56", svc_cols, NULL)
  results$row_56$time <- t_56
  cat(sprintf("  => %s\n\n", if (results$row_56$pass) "PASS" else "FAIL"))
} else {
  results$row_56 <- list(error = TRUE, pass = FALSE)
  cat("  => ERROR\n\n")
}

# ================================================================
# ROW 88: binomial + SVC (intercept)
# ================================================================
cat("========== Row 88: binomial + SVC(intercept) ==========\n")

set.seed(500)
prob_88 <- plogis(true_eta_svc)
trials_88 <- sample(20:50, N_SVC, replace = TRUE)
successes_88 <- rbinom(N_SVC, trials_88, prob_88)

df_88 <- data.frame(successes = successes_88, trials = trials_88, x = x_svc,
                    coord_x = coords_svc[, 1], coord_y = coords_svc[, 2])

cat("Fitting... ")
flush.console()
t_88 <- system.time({
  fit_88 <- tryCatch({
    tratio(successes | trials ~ x, data = df_88, family = ratiod_binomial(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 1),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_88)) {
  cat(sprintf("%.1fs\n", t_88))
  draws_88 <- as.matrix(fit_88$draws)

  svc_cols <- grep("^svc\\[", colnames(draws_88), value = TRUE)
  svc_loc_idx <- as.numeric(gsub(".*,(\\d+)\\]", "\\1", svc_cols))
  svc_cols <- svc_cols[order(svc_loc_idx)]

  results$row_88 <- validate_prediction(draws_88, x_svc, true_eta_svc, N_SVC,
                                         "Row 88", svc_cols, NULL)
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
cat("FINAL VALIDATION v2 SUMMARY\n")
cat("================================================================\n\n")

cat("Criterion: cor(eta_pred, eta_true) > 0.80\n\n")

for (rn in c("row_9", "row_69", "row_26", "row_56", "row_88")) {
  r <- results[[rn]]
  row_num <- gsub("row_", "", rn)
  if (isTRUE(r$error)) {
    cat(sprintf("  Row %s: ERROR\n", row_num))
  } else {
    cat(sprintf("  Row %s: %s (eta cor=%.4f, slope=%.3f, %.1fs)\n",
                row_num, if (r$pass) "PASS" else "FAIL",
                r$cor_eta, r$slope_mean, r$time))
  }
}

n_pass <- sum(sapply(results, function(r) r$pass))
cat(sprintf("\nOverall: %d/5 pass\n", n_pass))

saveRDS(results, "benchmarks/results_final5_v2.rds")
cat("Results saved to benchmarks/results_final5_v2.rds\n")
