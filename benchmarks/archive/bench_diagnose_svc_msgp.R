# ============================================================
# Diagnostic: SVC and MSGP with stronger signal
# ============================================================
# Tests whether failures are due to weak signal or code bugs.
#
# SVC: Increase sigma from 0.2 to 0.8 (4x stronger)
# MSGP Row 9: Use grouped observations (20 sites x 10 obs/site)
#
# If these pass, the original failures are power issues, not bugs.
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

cat("================================================================\n")
cat("DIAGNOSTIC: Stronger signal SVC + Grouped MSGP\n")
cat("================================================================\n\n")

# ================================================================
# TEST 1: SVC with strong signal (sigma=0.8 instead of 0.2)
# ================================================================
cat("========== TEST 1: SVC with STRONG signal (sigma=0.8) ==========\n")

N_SVC <- 100
set.seed(789)
coords_svc <- cbind(runif(N_SVC, 0, 10), runif(N_SVC, 0, 10))
x_svc <- rnorm(N_SVC)

set.seed(101)
true_svc_strong <- generate_gp_effects(coords_svc, 0.8, 3.0)
true_total_slopes_strong <- true_slope + true_svc_strong

cat(sprintf("SVC weights: SD=%.3f, range=[%.3f, %.3f]\n",
            sd(true_svc_strong), min(true_svc_strong), max(true_svc_strong)))
cat(sprintf("True total slopes: mean=%.3f, SD=%.3f\n\n",
            mean(true_total_slopes_strong), sd(true_total_slopes_strong)))

# Binomial (simplest family, row 88 analog)
set.seed(500)
eta_svc <- true_intercept + (true_slope + true_svc_strong) * x_svc
prob_svc <- plogis(eta_svc)
trials_svc <- sample(20:50, N_SVC, replace = TRUE)
successes_svc <- rbinom(N_SVC, trials_svc, prob_svc)

df_svc <- data.frame(successes = successes_svc, trials = trials_svc, x = x_svc,
                     coord_x = coords_svc[, 1], coord_y = coords_svc[, 2])

cat("Fitting SVC binomial (strong signal)... ")
flush.console()
t_svc <- system.time({
  fit_svc <- tryCatch({
    tratio(successes | trials ~ x, data = df_svc, family = ratiod_binomial(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 1),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_svc)) {
  cat(sprintf("%.1fs\n", t_svc))
  draws_svc <- as.matrix(fit_svc$draws)

  # Check SVC columns
  svc_cols <- grep("^svc\\[", colnames(draws_svc), value = TRUE)
  cat(sprintf("  Found %d SVC columns\n", length(svc_cols)))

  svc_loc_idx <- as.numeric(gsub(".*,(\\d+)\\]", "\\1", svc_cols))
  svc_cols <- svc_cols[order(svc_loc_idx)]

  beta2 <- draws_svc[, "beta_num[2]"]
  cat(sprintf("  beta_num[2]: mean=%.3f, SD=%.3f (true=%.3f)\n",
              mean(beta2), sd(beta2), true_slope))

  # SVC weight diagnostics
  svc_means <- colMeans(draws_svc[, svc_cols])
  cat(sprintf("  SVC weight means: SD=%.3f, range=[%.3f, %.3f]\n",
              sd(svc_means), min(svc_means), max(svc_means)))
  cat(sprintf("  True SVC weights: SD=%.3f, range=[%.3f, %.3f]\n",
              sd(true_svc_strong), min(true_svc_strong), max(true_svc_strong)))

  # Total slope recovery
  n_samples <- nrow(draws_svc)
  total_slopes <- matrix(0, n_samples, N_SVC)
  for (s in 1:n_samples) {
    svc_weights <- as.numeric(draws_svc[s, svc_cols])
    total_slopes[s, ] <- beta2[s] + svc_weights
  }
  total_slopes_mean <- colMeans(total_slopes)

  cor_val <- cor(total_slopes_mean, true_total_slopes_strong)
  rmse_val <- sqrt(mean((total_slopes_mean - true_total_slopes_strong)^2))

  cat(sprintf("  Total slope cor=%.4f | RMSE=%.3f\n", cor_val, rmse_val))
  cat(sprintf("  => %s (threshold 0.70)\n\n",
              if (cor_val > 0.70) "PASS" else "FAIL"))

  # Also check: is the issue that SVC weights are all near zero?
  cat("  Diagnostic: First 10 SVC weight means vs true:\n")
  for (i in 1:min(10, length(svc_means))) {
    cat(sprintf("    svc[%d]: est=%.4f, true=%.4f\n", i, svc_means[i], true_svc_strong[i]))
  }
  cat("\n")
} else {
  cat("  ERROR fitting model\n\n")
}

# ================================================================
# TEST 2: SVC with original weak signal but examine weight draws
# ================================================================
cat("========== TEST 2: SVC weight draw diagnostics (weak signal) ==========\n")

set.seed(101)
true_svc_weak <- generate_gp_effects(coords_svc, 0.2, 3.0)

set.seed(600)
eta_svc2 <- true_intercept + (true_slope + true_svc_weak) * x_svc
prob_svc2 <- plogis(eta_svc2)
trials_svc2 <- sample(20:50, N_SVC, replace = TRUE)
successes_svc2 <- rbinom(N_SVC, trials_svc2, prob_svc2)

df_svc2 <- data.frame(successes = successes_svc2, trials = trials_svc2, x = x_svc,
                      coord_x = coords_svc[, 1], coord_y = coords_svc[, 2])

cat("Fitting SVC binomial (weak signal)... ")
flush.console()
t_svc2 <- system.time({
  fit_svc2 <- tryCatch({
    tratio(successes | trials ~ x, data = df_svc2, family = ratiod_binomial(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 1),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_svc2)) {
  cat(sprintf("%.1fs\n", t_svc2))
  draws_svc2 <- as.matrix(fit_svc2$draws)

  svc_cols2 <- grep("^svc\\[", colnames(draws_svc2), value = TRUE)
  svc_loc_idx2 <- as.numeric(gsub(".*,(\\d+)\\]", "\\1", svc_cols2))
  svc_cols2 <- svc_cols2[order(svc_loc_idx2)]

  beta2_2 <- draws_svc2[, "beta_num[2]"]
  svc_means2 <- colMeans(draws_svc2[, svc_cols2])

  cat(sprintf("  beta_num[2]: mean=%.3f, SD=%.3f\n", mean(beta2_2), sd(beta2_2)))
  cat(sprintf("  SVC weight means: SD=%.4f, range=[%.4f, %.4f]\n",
              sd(svc_means2), min(svc_means2), max(svc_means2)))
  cat(sprintf("  True SVC weights: SD=%.4f, range=[%.4f, %.4f]\n",
              sd(true_svc_weak), min(true_svc_weak), max(true_svc_weak)))

  # Check if SVC draws have any variance at all
  svc_sds <- apply(draws_svc2[, svc_cols2], 2, sd)
  cat(sprintf("  SVC weight posterior SDs: mean=%.4f, range=[%.4f, %.4f]\n",
              mean(svc_sds), min(svc_sds), max(svc_sds)))

  # Check GP hyperparameters if available
  hyper_cols <- grep("svc_sigma|svc_phi|svc_lengthscale|svc_gp", colnames(draws_svc2), value = TRUE)
  if (length(hyper_cols) > 0) {
    cat("  SVC hyperparameters:\n")
    for (hc in hyper_cols) {
      cat(sprintf("    %s: mean=%.4f, SD=%.4f\n", hc, mean(draws_svc2[, hc]), sd(draws_svc2[, hc])))
    }
  } else {
    cat("  No SVC hyperparameter columns found in draws\n")
    cat("  All column names containing 'svc' or 'gp':\n")
    relevant_cols <- grep("svc|gp|sigma|phi|length", colnames(draws_svc2), value = TRUE, ignore.case = TRUE)
    for (rc in relevant_cols[1:min(20, length(relevant_cols))]) {
      cat(sprintf("    %s: mean=%.4f\n", rc, mean(draws_svc2[, rc])))
    }
  }
  cat("\n")
} else {
  cat("  ERROR fitting model\n\n")
}

# ================================================================
# TEST 3: MSGP Row 9 with grouped observations
# ================================================================
cat("========== TEST 3: MSGP poisson_gamma with grouped obs ==========\n")
cat("  20 sites x 10 obs/site = 200 obs, but only 20 unique coords\n")

N_SITES <- 20
N_PER_SITE <- 10
N_TOTAL <- N_SITES * N_PER_SITE

set.seed(123)
site_coords <- cbind(runif(N_SITES, 0, 10), runif(N_SITES, 0, 10))

# Expand to observation level
site_id <- rep(1:N_SITES, each = N_PER_SITE)
coords_grouped <- site_coords[site_id, ]
x_grouped <- rnorm(N_TOTAL)

set.seed(456)
true_spatial_grouped <- generate_msgp_effects(site_coords, 0.2, 0.3, 1.0, 4.0)
spatial_per_obs <- true_spatial_grouped[site_id]

cat(sprintf("  MSGP spatial effects: SD=%.3f, range=[%.3f, %.3f]\n",
            sd(true_spatial_grouped), min(true_spatial_grouped), max(true_spatial_grouped)))

eta_true_grouped <- true_intercept + true_slope * x_grouped + spatial_per_obs
set.seed(100)
effort_grouped <- rgamma(N_TOTAL, 5, 1)
count_grouped <- rpois(N_TOTAL, exp(eta_true_grouped) * effort_grouped)

df_grouped <- data.frame(count = count_grouped, effort = effort_grouped, x = x_grouped,
                         coord_x = coords_grouped[, 1], coord_y = coords_grouped[, 2])

cat("Fitting MSGP poisson_gamma (grouped, ~10 min)... ")
flush.console()
t_grouped <- system.time({
  fit_grouped <- tryCatch({
    tratio(count | effort ~ x, data = df_grouped, family = ratiod_poisson_gamma(),
           spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_grouped)) {
  cat(sprintf("%.1fs\n", t_grouped))
  draws_grouped <- as.matrix(fit_grouped$draws)

  # Check how many GP columns - should be 20 (unique sites), not 200
  local_cols <- grep("^gp_local_w\\[", colnames(draws_grouped), value = TRUE)
  regional_cols <- grep("^gp_regional_w\\[", colnames(draws_grouped), value = TRUE)
  cat(sprintf("  Found %d local + %d regional GP columns (expect 20 each for grouped)\n",
              length(local_cols), length(regional_cols)))

  # If 20 columns, we need to map back to observations
  n_gp <- length(local_cols)

  local_idx <- as.numeric(gsub(".*\\[(\\d+)\\]", "\\1", local_cols))
  local_cols <- local_cols[order(local_idx)]
  regional_idx <- as.numeric(gsub(".*\\[(\\d+)\\]", "\\1", regional_cols))
  regional_cols <- regional_cols[order(regional_idx)]

  beta1 <- draws_grouped[, "beta_num[1]"]
  beta2 <- draws_grouped[, "beta_num[2]"]

  n_samples <- nrow(draws_grouped)
  eta_pred <- matrix(0, n_samples, N_TOTAL)

  if (n_gp == N_SITES) {
    # Grouped: map site GP effects to observations
    for (s in 1:n_samples) {
      local_eff <- as.numeric(draws_grouped[s, local_cols])
      regional_eff <- as.numeric(draws_grouped[s, regional_cols])
      gp_per_obs <- local_eff[site_id] + regional_eff[site_id]
      eta_pred[s, ] <- beta1[s] + beta2[s] * x_grouped + gp_per_obs
    }
  } else if (n_gp == N_TOTAL) {
    # Ungrouped: direct mapping
    for (s in 1:n_samples) {
      local_eff <- as.numeric(draws_grouped[s, local_cols])
      regional_eff <- as.numeric(draws_grouped[s, regional_cols])
      eta_pred[s, ] <- beta1[s] + beta2[s] * x_grouped + local_eff + regional_eff
    }
  } else {
    cat(sprintf("  Unexpected GP column count: %d\n", n_gp))
  }

  eta_pred_mean <- colMeans(eta_pred)
  cor_val <- cor(eta_pred_mean, eta_true_grouped)
  rmse_val <- sqrt(mean((eta_pred_mean - eta_true_grouped)^2))
  slope_mean <- mean(beta2)
  slope_sd <- sd(beta2)
  slope_diff <- abs(slope_mean - true_slope) / slope_sd

  cat(sprintf("  Eta cor=%.4f | RMSE=%.3f | slope=%.3f (%.1f SD from true)\n",
              cor_val, rmse_val, slope_mean, slope_diff))
  cat(sprintf("  => %s (threshold 0.85)\n\n",
              if (cor_val > 0.85) "PASS" else "FAIL"))
} else {
  cat("  ERROR fitting model\n\n")
}

cat("================================================================\n")
cat("DIAGNOSTIC COMPLETE\n")
cat("================================================================\n")
