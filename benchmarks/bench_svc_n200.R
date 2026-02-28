# ============================================================
# SVC Validation: N=200 for noisier families
# ============================================================
# Rows 26 (poisson_gamma) and 56 (negbin_negbin) failed with N=100.
# Both had cor > 0.70, suggesting power issue, not bug.
# Row 88 (binomial) passed with N=100 (cor=0.83).
# Re-run with N=200 to confirm.
# ============================================================

library(numdenom)
library(posterior)

set.seed(42)

N_SVC <- 200
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

cat("================================================================\n")
cat("SVC VALIDATION: N=200 for noisier families\n")
cat("================================================================\n\n")

set.seed(789)
coords_svc <- cbind(runif(N_SVC, 0, 10), runif(N_SVC, 0, 10))
x_svc <- rnorm(N_SVC)

set.seed(202)
true_svc <- generate_gp_effects(coords_svc, 0.5, 3.0)

cat(sprintf("SVC intercept weights: SD=%.3f, range=[%.3f, %.3f]\n",
            sd(true_svc), min(true_svc), max(true_svc)))

true_eta <- (true_intercept + true_svc) + true_slope * x_svc

results <- list()

# ================================================================
# ROW 26: poisson_gamma + SVC (N=200)
# ================================================================
cat("\n========== Row 26: poisson_gamma + SVC(intercept), N=200 ==========\n")

set.seed(300)
effort_26 <- rgamma(N_SVC, 5, 1)
count_26 <- rpois(N_SVC, exp(true_eta) * effort_26)

df_26 <- data.frame(count = count_26, effort = effort_26, x = x_svc,
                    coord_x = coords_svc[, 1], coord_y = coords_svc[, 2])

cat("Fitting... ")
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

  svc_cols <- grep("^svc\\[", colnames(draws_26), value = TRUE)
  svc_loc_idx <- as.numeric(gsub(".*,(\\d+)\\]", "\\1", svc_cols))
  svc_cols <- svc_cols[order(svc_loc_idx)]

  beta1 <- draws_26[, "beta_num[1]"]
  beta2 <- draws_26[, "beta_num[2]"]

  n_samples <- nrow(draws_26)
  eta_pred <- matrix(0, n_samples, N_SVC)
  for (s in 1:n_samples) {
    svc_eff <- as.numeric(draws_26[s, svc_cols])
    eta_pred[s, ] <- beta1[s] + beta2[s] * x_svc + svc_eff
  }
  eta_pred_mean <- colMeans(eta_pred)

  cor_val <- cor(eta_pred_mean, true_eta)
  rmse_val <- sqrt(mean((eta_pred_mean - true_eta)^2))

  cat(sprintf("  Eta cor=%.4f | RMSE=%.3f | slope=%.3f\n", cor_val, rmse_val, mean(beta2)))
  cat(sprintf("  => %s (threshold 0.80)\n\n", if (cor_val > 0.80) "PASS" else "FAIL"))
  results$row_26 <- list(cor = cor_val, rmse = rmse_val, time = t_26, pass = cor_val > 0.80)
} else {
  cat("  ERROR\n\n")
  results$row_26 <- list(error = TRUE, pass = FALSE)
}

# ================================================================
# ROW 56: negbin_negbin + SVC (N=200)
# ================================================================
cat("========== Row 56: negbin_negbin + SVC(intercept), N=200 ==========\n")

set.seed(400)
eta_num_56 <- true_eta
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
    ratiod(num | denom ~ x, data = df_56, family = ratiod_negbin_negbin(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 1),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_56)) {
  cat(sprintf("%.1fs\n", t_56))
  draws_56 <- as.matrix(fit_56$draws)

  svc_cols <- grep("^svc\\[", colnames(draws_56), value = TRUE)
  svc_loc_idx <- as.numeric(gsub(".*,(\\d+)\\]", "\\1", svc_cols))
  svc_cols <- svc_cols[order(svc_loc_idx)]

  beta1 <- draws_56[, "beta_num[1]"]
  beta2 <- draws_56[, "beta_num[2]"]

  n_samples <- nrow(draws_56)
  eta_pred <- matrix(0, n_samples, N_SVC)
  for (s in 1:n_samples) {
    svc_eff <- as.numeric(draws_56[s, svc_cols])
    eta_pred[s, ] <- beta1[s] + beta2[s] * x_svc + svc_eff
  }
  eta_pred_mean <- colMeans(eta_pred)

  cor_val <- cor(eta_pred_mean, true_eta)
  rmse_val <- sqrt(mean((eta_pred_mean - true_eta)^2))

  cat(sprintf("  Eta cor=%.4f | RMSE=%.3f | slope=%.3f\n", cor_val, rmse_val, mean(beta2)))
  cat(sprintf("  => %s (threshold 0.80)\n\n", if (cor_val > 0.80) "PASS" else "FAIL"))
  results$row_56 <- list(cor = cor_val, rmse = rmse_val, time = t_56, pass = cor_val > 0.80)
} else {
  cat("  ERROR\n\n")
  results$row_56 <- list(error = TRUE, pass = FALSE)
}

# ================================================================
# SUMMARY
# ================================================================
cat("================================================================\n")
cat("SUMMARY\n")
cat("================================================================\n\n")

for (name in names(results)) {
  r <- results[[name]]
  if (isTRUE(r$error)) {
    cat(sprintf("  %s: ERROR\n", name))
  } else {
    cat(sprintf("  %s: %s (cor=%.4f, %.1fs)\n", name,
                if (r$pass) "PASS" else "FAIL", r$cor, r$time))
  }
}

saveRDS(results, "benchmarks/results_svc_n200.rds")
cat("\nResults saved to benchmarks/results_svc_n200.rds\n")
