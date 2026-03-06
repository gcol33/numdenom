# ============================================================
# SVC Validation - FIXED terms specification
# ============================================================
# Previous script used terms=1 which selects (Intercept) column.
# Data was generated with SVC on the slope.
# Fix: use terms=2 to select the x column.
#
# Also test: SVC on intercept (terms=1) with data generated correctly.
# ============================================================

library(numdenom)
library(posterior)

set.seed(42)

N_SVC <- 100
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

validate_svc_slope <- function(draws, true_total_slopes, N, label) {
  beta2 <- draws[, "beta_num[2]"]

  svc_cols <- grep("^svc\\[", colnames(draws), value = TRUE)
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
  beta2_mean <- mean(beta2)

  cat(sprintf("  [%s] Total slope cor=%.4f | RMSE=%.3f | beta2=%.3f | mean_total=%.3f (true=%.3f)\n",
              label, cor_val, rmse_val, beta2_mean,
              mean(total_slopes_mean), mean(true_total_slopes)))

  list(cor_total = cor_val, rmse_total = rmse_val, beta2_mean = beta2_mean,
       pass = cor_val > 0.70)
}

validate_svc_intercept <- function(draws, true_total_intercepts, N, label) {
  beta1 <- draws[, "beta_num[1]"]

  svc_cols <- grep("^svc\\[", colnames(draws), value = TRUE)
  svc_loc_idx <- as.numeric(gsub(".*,(\\d+)\\]", "\\1", svc_cols))
  svc_cols <- svc_cols[order(svc_loc_idx)]
  stopifnot(length(svc_cols) == N)

  n_samples <- nrow(draws)
  total_intercepts <- matrix(0, n_samples, N)
  for (s in 1:n_samples) {
    svc_weights <- as.numeric(draws[s, svc_cols])
    total_intercepts[s, ] <- beta1[s] + svc_weights
  }
  total_intercepts_mean <- colMeans(total_intercepts)

  cor_val <- cor(total_intercepts_mean, true_total_intercepts)
  rmse_val <- sqrt(mean((total_intercepts_mean - true_total_intercepts)^2))
  beta1_mean <- mean(beta1)

  cat(sprintf("  [%s] Total intercept cor=%.4f | RMSE=%.3f | beta1=%.3f | mean_total=%.3f (true=%.3f)\n",
              label, cor_val, rmse_val, beta1_mean,
              mean(total_intercepts_mean), mean(true_total_intercepts)))

  list(cor_total = cor_val, rmse_total = rmse_val, beta1_mean = beta1_mean,
       pass = cor_val > 0.70)
}

cat("================================================================\n")
cat("SVC VALIDATION (FIXED terms specification)\n")
cat("================================================================\n\n")

# ================================================================
# Shared setup
# ================================================================
set.seed(789)
coords_svc <- cbind(runif(N_SVC, 0, 10), runif(N_SVC, 0, 10))
x_svc <- rnorm(N_SVC)

# ================================================================
# TEST A: SVC on SLOPE (terms=2), data has SVC on slope
# ================================================================
cat("========== TEST A: SVC on slope (terms=2) ==========\n")
cat("  Model: eta[i] = beta1 + (beta2 + w[i]) * x[i]\n")
cat("  Data:  eta[i] = 1.0 + (0.3 + svc[i]) * x[i]\n\n")

set.seed(101)
true_svc_slope <- generate_gp_effects(coords_svc, 0.5, 3.0)  # Stronger signal
true_total_slopes <- true_slope + true_svc_slope

cat(sprintf("SVC slope weights: SD=%.3f, range=[%.3f, %.3f]\n",
            sd(true_svc_slope), min(true_svc_slope), max(true_svc_slope)))
cat(sprintf("True total slopes: mean=%.3f, SD=%.3f\n\n",
            mean(true_total_slopes), sd(true_total_slopes)))

results <- list()

# Row 88 analog: binomial + SVC on slope
set.seed(500)
eta_A <- true_intercept + (true_slope + true_svc_slope) * x_svc
prob_A <- plogis(eta_A)
trials_A <- sample(20:50, N_SVC, replace = TRUE)
successes_A <- rbinom(N_SVC, trials_A, prob_A)

df_A <- data.frame(successes = successes_A, trials = trials_A, x = x_svc,
                   coord_x = coords_svc[, 1], coord_y = coords_svc[, 2])

cat("Fitting binomial + SVC(terms=2)... ")
flush.console()
t_A <- system.time({
  fit_A <- tryCatch({
    ratiod(successes | trials ~ x, data = df_A, family = ratiod_binomial(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 2),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_A)) {
  cat(sprintf("%.1fs\n", t_A))
  draws_A <- as.matrix(fit_A$draws)
  results$test_A <- validate_svc_slope(draws_A, true_total_slopes, N_SVC, "Test A")
  results$test_A$time <- t_A
  cat(sprintf("  => %s\n\n", if (results$test_A$pass) "PASS" else "FAIL"))
} else {
  results$test_A <- list(error = TRUE, pass = FALSE)
  cat("  => ERROR\n\n")
}

# ================================================================
# TEST B: SVC on INTERCEPT (terms=1), data has SVC on intercept
# ================================================================
cat("========== TEST B: SVC on intercept (terms=1) ==========\n")
cat("  Model: eta[i] = (beta1 + w[i]) + beta2 * x[i]\n")
cat("  Data:  eta[i] = (1.0 + svc[i]) + 0.3 * x[i]\n\n")

set.seed(202)
true_svc_intcpt <- generate_gp_effects(coords_svc, 0.5, 3.0)
true_total_intercepts <- true_intercept + true_svc_intcpt

cat(sprintf("SVC intercept weights: SD=%.3f, range=[%.3f, %.3f]\n",
            sd(true_svc_intcpt), min(true_svc_intcpt), max(true_svc_intcpt)))

set.seed(600)
eta_B <- (true_intercept + true_svc_intcpt) + true_slope * x_svc
prob_B <- plogis(eta_B)
trials_B <- sample(20:50, N_SVC, replace = TRUE)
successes_B <- rbinom(N_SVC, trials_B, prob_B)

df_B <- data.frame(successes = successes_B, trials = trials_B, x = x_svc,
                   coord_x = coords_svc[, 1], coord_y = coords_svc[, 2])

cat("Fitting binomial + SVC(terms=1)... ")
flush.console()
t_B <- system.time({
  fit_B <- tryCatch({
    ratiod(successes | trials ~ x, data = df_B, family = ratiod_binomial(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 1),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_B)) {
  cat(sprintf("%.1fs\n", t_B))
  draws_B <- as.matrix(fit_B$draws)
  results$test_B <- validate_svc_intercept(draws_B, true_total_intercepts, N_SVC, "Test B")
  results$test_B$time <- t_B
  cat(sprintf("  => %s\n\n", if (results$test_B$pass) "PASS" else "FAIL"))
} else {
  results$test_B <- list(error = TRUE, pass = FALSE)
  cat("  => ERROR\n\n")
}

# ================================================================
# TEST C: All 3 families with SVC on slope (terms=2)
# ================================================================
cat("========== TEST C: All families with SVC on slope (terms=2) ==========\n\n")

# Use same SVC slope data from Test A

# C1: poisson_gamma (Row 26 analog)
cat("--- C1: poisson_gamma + SVC(terms=2) ---\n")
set.seed(300)
effort_C1 <- rgamma(N_SVC, 5, 1)
eta_C1 <- true_intercept + (true_slope + true_svc_slope) * x_svc
count_C1 <- rpois(N_SVC, exp(eta_C1) * effort_C1)

df_C1 <- data.frame(count = count_C1, effort = effort_C1, x = x_svc,
                    coord_x = coords_svc[, 1], coord_y = coords_svc[, 2])

cat("Fitting... ")
flush.console()
t_C1 <- system.time({
  fit_C1 <- tryCatch({
    ratiod(count | effort ~ x, data = df_C1, family = ratiod_poisson_gamma(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 2),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_C1)) {
  cat(sprintf("%.1fs\n", t_C1))
  draws_C1 <- as.matrix(fit_C1$draws)
  results$test_C1 <- validate_svc_slope(draws_C1, true_total_slopes, N_SVC, "C1 pg")
  results$test_C1$time <- t_C1
  cat(sprintf("  => %s\n\n", if (results$test_C1$pass) "PASS" else "FAIL"))
} else {
  results$test_C1 <- list(error = TRUE, pass = FALSE)
  cat("  => ERROR\n\n")
}

# C2: negbin_negbin (Row 56 analog)
cat("--- C2: negbin_negbin + SVC(terms=2) ---\n")
set.seed(400)
eta_num_C2 <- true_intercept + (true_slope + true_svc_slope) * x_svc
eta_denom_C2 <- 0.5 + 0.2 * x_svc
num_C2 <- rnbinom(N_SVC, size = 5, mu = exp(eta_num_C2))
denom_C2 <- rnbinom(N_SVC, size = 5, mu = exp(eta_denom_C2))
denom_C2[denom_C2 == 0] <- 1

df_C2 <- data.frame(num = num_C2, denom = denom_C2, x = x_svc,
                    coord_x = coords_svc[, 1], coord_y = coords_svc[, 2])

cat("Fitting... ")
flush.console()
t_C2 <- system.time({
  fit_C2 <- tryCatch({
    ratiod(num | denom ~ x, data = df_C2, family = ratiod_negbin_negbin(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 2),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_C2)) {
  cat(sprintf("%.1fs\n", t_C2))
  draws_C2 <- as.matrix(fit_C2$draws)
  results$test_C2 <- validate_svc_slope(draws_C2, true_total_slopes, N_SVC, "C2 nb")
  results$test_C2$time <- t_C2
  cat(sprintf("  => %s\n\n", if (results$test_C2$pass) "PASS" else "FAIL"))
} else {
  results$test_C2 <- list(error = TRUE, pass = FALSE)
  cat("  => ERROR\n\n")
}

# ================================================================
# SUMMARY
# ================================================================
cat("================================================================\n")
cat("SUMMARY\n")
cat("================================================================\n\n")

for (name in names(results)) {
  r <- results[[name]]
  status <- if (isTRUE(r$error)) "ERROR" else if (r$pass) "PASS" else "FAIL"
  cor_val <- if (!is.null(r$cor_total)) sprintf("%.4f", r$cor_total) else "N/A"
  time_val <- if (!is.null(r$time)) sprintf("%.1fs", r$time) else "N/A"
  cat(sprintf("  %s: %s (cor=%s, %s)\n", name, status, cor_val, time_val))
}

saveRDS(results, "benchmarks/results_svc_fixed.rds")
cat("\nResults saved to benchmarks/results_svc_fixed.rds\n")
