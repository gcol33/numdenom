# Final validation for remaining unvalidated rows
# Row 84: binomial + HSGP + RW1
# Rows 26, 56, 88: SVC models
# Uses simulation-based validation

library(numdenom)
library(posterior)

set.seed(42)

# Parameters - smaller for speed since these are slow models
N_OBS <- 100
N_SITES <- 20
N_TIMES <- 10
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1

cat("=======================================================\n")
cat("Final Validation: Rows 84, 26, 56, 88\n")
cat("=======================================================\n")
cat(sprintf("N=%d, sites=%d, times=%d, iter=%d\n\n", N_OBS, N_SITES, N_TIMES, N_ITER))

check_recovery <- function(draws, true_value, param_name, threshold_sd = 2) {
  post_mean <- mean(draws)
  post_sd <- sd(draws)
  diff_sd <- abs(post_mean - true_value) / post_sd
  in_ci <- true_value >= quantile(draws, 0.025) && true_value <= quantile(draws, 0.975)
  list(param = param_name, true = true_value, mean = post_mean, sd = post_sd,
       diff_sd = diff_sd, in_ci = in_ci, pass = diff_sd < threshold_sd)
}

print_recovery <- function(r) {
  status <- if(r$pass) "PASS" else "FAIL"
  cat(sprintf("  %s: true=%.3f, post=%.3f (SD=%.3f), %.2f SD => %s\n",
              r$param, r$true, r$mean, r$sd, r$diff_sd, status))
}

results <- list()

# Generate spatial coordinates on a grid
coords <- expand.grid(
  x = seq(0, 1, length.out = ceiling(sqrt(N_SITES))),
  y = seq(0, 1, length.out = ceiling(sqrt(N_SITES)))
)[1:N_SITES, ]

# True parameters
true_intercept <- 1.0
true_slope <- 0.3

# =============================================================================
# Row 84: binomial + HSGP + RW1
# =============================================================================
cat("\n========== Row 84: binomial + HSGP + RW1 ==========\n")

# Generate data with temporal structure
site_84 <- factor(rep(1:N_SITES, length.out = N_OBS))
time_84 <- factor(rep(1:N_TIMES, each = N_OBS / N_TIMES))
x_84 <- rnorm(N_OBS)

# Generate RW1 temporal effects
set.seed(123)
rw1_effects <- cumsum(rnorm(N_TIMES, 0, 0.2))
rw1_effects <- rw1_effects - mean(rw1_effects)

eta_84 <- true_intercept + true_slope * x_84 + rw1_effects[as.integer(time_84)]
prob_84 <- plogis(eta_84)
trials_84 <- sample(20:50, N_OBS, replace = TRUE)
successes_84 <- rbinom(N_OBS, trials_84, prob_84)

df_84 <- data.frame(
  successes = successes_84, trials = trials_84, x = x_84,
  site = site_84, time = time_84,
  coord_x = coords$x[as.integer(site_84)],
  coord_y = coords$y[as.integer(site_84)]
)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_84 <- system.time({
  fit_84 <- tryCatch({
    ratiod(successes | trials ~ x, data = df_84, family = ratiod_binomial(),
           spatial = spatial_hsgp(coords = c("coord_x", "coord_y"), m = 5),
           temporal = temporal_rw1(time_var = "time"),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_84)) {
  cat(sprintf("%.1fs\n", t_84))
  draws_84 <- as.matrix(fit_84$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_84), value = TRUE)[1]
  if (!is.null(slope_col) && length(slope_col) > 0) {
    results$row_84 <- list(
      slope = check_recovery(draws_84[, slope_col], true_slope, "slope"),
      time = t_84
    )
    print_recovery(results$row_84$slope)
    results$row_84$pass <- results$row_84$slope$pass
  } else {
    cat("ERROR: slope column not found\n")
    results$row_84 <- list(error = TRUE)
  }
} else {
  results$row_84 <- list(error = TRUE)
}

# =============================================================================
# SVC Models (Rows 26, 56, 88)
# =============================================================================

# Generate SVC effects - spatially varying coefficient for x
generate_svc_effects <- function(coords, sigma, lengthscale) {
  n <- nrow(coords)
  dist_mat <- as.matrix(dist(coords))
  cov_mat <- sigma^2 * exp(-dist_mat / lengthscale)
  diag(cov_mat) <- diag(cov_mat) + 1e-6
  L <- chol(cov_mat)
  as.vector(t(L) %*% rnorm(n))
}

set.seed(456)
svc_slope <- generate_svc_effects(coords, 0.3, 0.3)  # Varying slope by site

# Common data for SVC models
site_svc <- factor(rep(1:N_SITES, length.out = N_OBS))
x_svc <- rnorm(N_OBS)

# Row 26: poisson_gamma + SVC
cat("\n========== Row 26: poisson_gamma + SVC ==========\n")

eta_26 <- true_intercept + (true_slope + svc_slope[as.integer(site_svc)]) * x_svc
effort_26 <- rgamma(N_OBS, 5, 1)
count_26 <- rpois(N_OBS, exp(eta_26) * effort_26)

df_26 <- data.frame(
  count = count_26, effort = effort_26, x = x_svc, site = site_svc,
  coord_x = coords$x[as.integer(site_svc)],
  coord_y = coords$y[as.integer(site_svc)]
)

cat(sprintf("True: mean_slope=%.2f (varies by site)\n", true_slope))
cat("Fitting... (SVC is slow ~700s) ")
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
  if (!is.null(slope_col) && length(slope_col) > 0) {
    results$row_26 <- list(
      slope = check_recovery(draws_26[, slope_col], true_slope, "mean_slope"),
      time = t_26
    )
    print_recovery(results$row_26$slope)
    results$row_26$pass <- results$row_26$slope$pass
  } else {
    cat("ERROR: slope column not found\n")
    results$row_26 <- list(error = TRUE)
  }
} else {
  results$row_26 <- list(error = TRUE)
}

# Row 56: negbin_negbin + SVC
cat("\n========== Row 56: negbin_negbin + SVC ==========\n")

eta_num_56 <- true_intercept + (true_slope + svc_slope[as.integer(site_svc)]) * x_svc
eta_denom_56 <- 0.5 + 0.2 * x_svc
num_56 <- rnbinom(N_OBS, size = 5, mu = exp(eta_num_56))
denom_56 <- rnbinom(N_OBS, size = 5, mu = exp(eta_denom_56))
denom_56[denom_56 == 0] <- 1

df_56 <- data.frame(
  num = num_56, denom = denom_56, x = x_svc, site = site_svc,
  coord_x = coords$x[as.integer(site_svc)],
  coord_y = coords$y[as.integer(site_svc)]
)

cat(sprintf("True: mean_slope=%.2f (varies by site)\n", true_slope))
cat("Fitting... (SVC is slow ~700s) ")
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
  if (!is.null(slope_col) && length(slope_col) > 0) {
    results$row_56 <- list(
      slope = check_recovery(draws_56[, slope_col], true_slope, "mean_slope"),
      time = t_56
    )
    print_recovery(results$row_56$slope)
    results$row_56$pass <- results$row_56$slope$pass
  } else {
    cat("ERROR: slope column not found\n")
    results$row_56 <- list(error = TRUE)
  }
} else {
  results$row_56 <- list(error = TRUE)
}

# Row 88: binomial + SVC
cat("\n========== Row 88: binomial + SVC ==========\n")

eta_88 <- true_intercept + (true_slope + svc_slope[as.integer(site_svc)]) * x_svc
prob_88 <- plogis(eta_88)
trials_88 <- sample(20:50, N_OBS, replace = TRUE)
successes_88 <- rbinom(N_OBS, trials_88, prob_88)

df_88 <- data.frame(
  successes = successes_88, trials = trials_88, x = x_svc, site = site_svc,
  coord_x = coords$x[as.integer(site_svc)],
  coord_y = coords$y[as.integer(site_svc)]
)

cat(sprintf("True: mean_slope=%.2f (varies by site)\n", true_slope))
cat("Fitting... (SVC is slow ~700s) ")
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
  if (!is.null(slope_col) && length(slope_col) > 0) {
    results$row_88 <- list(
      slope = check_recovery(draws_88[, slope_col], true_slope, "mean_slope"),
      time = t_88
    )
    print_recovery(results$row_88$slope)
    results$row_88$pass <- results$row_88$slope$pass
  } else {
    cat("ERROR: slope column not found\n")
    results$row_88 <- list(error = TRUE)
  }
} else {
  results$row_88 <- list(error = TRUE)
}

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - Final Validation\n")
cat("=======================================================\n\n")

summary_rows <- c("row_84", "row_26", "row_56", "row_88")
for (rn in summary_rows) {
  r <- results[[rn]]
  row_num <- gsub("row_", "", rn)
  if (is.null(r)) {
    cat(sprintf("Row %s: NOT RUN\n", row_num))
  } else if (!is.null(r$error) && r$error) {
    cat(sprintf("Row %s: ERROR\n", row_num))
  } else {
    status <- if(r$pass) "PASS" else "FAIL"
    cat(sprintf("Row %s: %s (%.2f SD, %.1fs)\n", row_num, status, r$slope$diff_sd, r$time))
  }
}

cat("\n")
n_pass <- sum(sapply(results, function(r) !is.null(r$pass) && r$pass))
n_total <- length(summary_rows)
cat(sprintf("Total: %d/%d passed\n", n_pass, n_total))

saveRDS(results, "benchmarks/results_final_validation.rds")
cat("\nResults saved to benchmarks/results_final_validation.rds\n")
