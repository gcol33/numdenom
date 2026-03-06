# =============================================================================
# Re-run GP+RW1 models with longer chains
# Rows 51 (nb+GP+RW1) and 83 (bin+GP+RW1)
# =============================================================================
# These rows are currently marginal (Sim*) due to chain-length issues:
#   Row 51: slope=0.528 vs true=0.30. High-dim (92 params for 80 obs).
#   Row 83: slope=0.293 (correct) but SD=0.001 (chain barely explored).
#
# Fix: longer chains (2000 iter, 1000 warmup, 2 chains) with unique
# coordinates per observation (required for GP/NNGP).
#
# Usage: Rscript benchmarks/bench_rerun_gp_rw1.R
# =============================================================================

library(numdenom)
library(posterior)

set.seed(42)

# Longer chains for proper exploration
N_OBS    <- 80L    # GP is O(N^3), keep small
N_ITER   <- 2000L
N_WARMUP <- 1000L
N_CHAINS <- 2L
N_TIMES  <- 10L

cat("=======================================================\n")
cat("GP+RW1 Re-run: Longer Chains for Marginal Rows\n")
cat("Rows 51, 83\n")
cat("=======================================================\n")
cat(sprintf("N=%d, iter=%d (warmup=%d), chains=%d\n\n",
            N_OBS, N_ITER, N_WARMUP, N_CHAINS))

# =============================================================================
# Helpers
# =============================================================================

check_recovery <- function(draws, true_value, param_name, threshold_sd = 2) {
  post_mean <- mean(draws)
  post_sd   <- sd(draws)
  diff_sd   <- abs(post_mean - true_value) / post_sd
  q025      <- quantile(draws, 0.025)
  q975      <- quantile(draws, 0.975)
  in_ci     <- true_value >= q025 && true_value <= q975
  list(param = param_name, true = true_value, mean = post_mean, sd = post_sd,
       diff_sd = diff_sd, in_ci = in_ci, pass = diff_sd < threshold_sd,
       q025 = q025, q975 = q975)
}

print_recovery <- function(r) {
  ci_str  <- if (r$in_ci) "in CI" else "NOT in CI"
  status  <- if (r$pass) "PASS" else "FAIL"
  cat(sprintf("  %s: true=%.3f, post=%.3f (SD=%.3f), %.2f SD, 95%%CI=[%.3f,%.3f], %s => %s\n",
              r$param, r$true, r$mean, r$sd, r$diff_sd,
              r$q025, r$q975, ci_str, status))
}

generate_gp_effects <- function(coords, sigma, lengthscale) {
  n <- nrow(coords)
  dist_mat <- as.matrix(dist(coords))
  cov_mat  <- sigma^2 * exp(-dist_mat / lengthscale)
  diag(cov_mat) <- diag(cov_mat) + 1e-6
  L <- chol(cov_mat)
  effects <- as.vector(t(L) %*% rnorm(n))
  effects - mean(effects)
}

generate_rw1 <- function(n, sigma) {
  gamma <- cumsum(rnorm(n, 0, sigma))
  gamma - mean(gamma)
}

# =============================================================================
# Common data setup
# =============================================================================

# Unique spatial coordinates per observation (required for GP)
coord_x <- runif(N_OBS, 0, 10)
coord_y <- runif(N_OBS, 0, 10)
coords_mat <- cbind(coord_x, coord_y)

# True parameters
true_intercept    <- 1.0
true_slope        <- 0.3
true_sigma_gp     <- 0.3
true_lengthscale  <- 2.0
true_sigma_rw1    <- 0.2
true_sigma_re     <- 0.3

# Generate shared effects
spatial_effects  <- generate_gp_effects(coords_mat, true_sigma_gp, true_lengthscale)
temporal_effects <- generate_rw1(N_TIMES, true_sigma_rw1)
n_sites_eff      <- 15L
site_effects     <- rnorm(n_sites_eff, 0, true_sigma_re)

# Observation structure
site <- factor(rep(1:n_sites_eff, length.out = N_OBS))
time <- rep(1:N_TIMES, length.out = N_OBS)
x    <- rnorm(N_OBS)

# Combined linear predictor
eta <- true_intercept + true_slope * x +
       site_effects[as.integer(site)] +
       spatial_effects +
       temporal_effects[time]

results <- list()

# =============================================================================
# Row 51: negbin_negbin + GP + RW1
# =============================================================================

cat("\n========== Row 51: negbin_negbin + GP + RW1 ==========\n")
cat(sprintf("True: intercept=%.2f, slope=%.2f\n", true_intercept, true_slope))

# Generate NegBin-NegBin data
eta_num_51   <- eta
eta_denom_51 <- 0.5 + 0.1 * x + site_effects[as.integer(site)] * 0.5

y_num_51   <- rnbinom(N_OBS, mu = exp(eta_num_51), size = 5)
y_denom_51 <- rnbinom(N_OBS, mu = exp(eta_denom_51), size = 8)
y_denom_51[y_denom_51 == 0] <- 1L

df_51 <- data.frame(
  y = y_num_51, denom = y_denom_51, x = x,
  site = site, time = factor(time),
  lon = coord_x, lat = coord_y
)

cat("Fitting (this will take a while, GP is O(N^3))... ")
flush.console()

t_51 <- system.time({
  fit_51 <- tryCatch({
    ratiod(y | denom ~ x + (1 | site), data = df_51,
           family = ratiod_negbin_negbin(),
           spatial = spatial_gp(coords = ~ lon + lat),
           temporal = temporal_rw1("time"),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
           gradient_mode = "H", verbose = FALSE)
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", conditionMessage(e)))
    NULL
  })
})[["elapsed"]]

if (!is.null(fit_51)) {
  cat(sprintf("%.1fs\n", t_51))

  draws_51 <- as.matrix(fit_51$draws)
  cat("\nParameter recovery:\n")

  # Check slope (most robust parameter)
  slope_cols <- grep("beta_num\\[2\\]|beta_num\\.2", colnames(draws_51), value = TRUE)
  if (length(slope_cols) > 0) {
    r_slope <- check_recovery(draws_51[, slope_cols[1]], true_slope, "slope_num")
    print_recovery(r_slope)
    results$row51_slope <- r_slope
  }

  # Chain diagnostics
  cat("\nChain diagnostics:\n")
  cat(sprintf("  Total samples: %d (after warmup)\n", nrow(draws_51)))
  cat(sprintf("  Columns: %d parameters\n", ncol(draws_51)))

  # Check ESS for slope
  if (length(slope_cols) > 0) {
    ess_bulk <- posterior::ess_bulk(draws_51[, slope_cols[1]])
    ess_tail <- posterior::ess_tail(draws_51[, slope_cols[1]])
    rhat     <- posterior::rhat(draws_51[, slope_cols[1]])
    cat(sprintf("  Slope ESS (bulk/tail): %.0f / %.0f\n", ess_bulk, ess_tail))
    cat(sprintf("  Slope R-hat: %.4f\n", rhat))
  }

  # Divergences
  if (!is.null(fit_51$diagnostics)) {
    n_div <- sum(fit_51$diagnostics$divergent, na.rm = TRUE)
    cat(sprintf("  Divergences: %d\n", n_div))
  }
} else {
  cat("FAILED\n")
}

# =============================================================================
# Row 83: binomial + GP + RW1
# =============================================================================

cat("\n========== Row 83: binomial + GP + RW1 ==========\n")
cat(sprintf("True: intercept=%.2f, slope=%.2f\n", 0.5, true_slope))

# Generate Binomial data with GP + RW1
eta_83  <- 0.5 + true_slope * x +
           site_effects[as.integer(site)] +
           spatial_effects * 0.5 +    # scale down spatial for binomial
           temporal_effects[time] * 0.5

trials_83 <- sample(20:50, N_OBS, replace = TRUE)
p_83      <- plogis(eta_83)
y_83      <- rbinom(N_OBS, trials_83, p_83)

df_83 <- data.frame(
  y = y_83, trials = trials_83, x = x,
  site = site, time = factor(time),
  lon = coord_x, lat = coord_y
)

cat("Fitting (this will take a while, GP is O(N^3))... ")
flush.console()

t_83 <- system.time({
  fit_83 <- tryCatch({
    ratiod(y | trials ~ x + (1 | site), data = df_83,
           family = ratiod_binomial(),
           spatial = spatial_gp(coords = ~ lon + lat),
           temporal = temporal_rw1("time"),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
           gradient_mode = "H", verbose = FALSE)
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", conditionMessage(e)))
    NULL
  })
})[["elapsed"]]

if (!is.null(fit_83)) {
  cat(sprintf("%.1fs\n", t_83))

  draws_83 <- as.matrix(fit_83$draws)
  cat("\nParameter recovery:\n")

  # Check slope
  slope_cols <- grep("beta_num\\[2\\]|beta_num\\.2|beta\\[2\\]|beta\\.2",
                     colnames(draws_83), value = TRUE)
  if (length(slope_cols) > 0) {
    r_slope <- check_recovery(draws_83[, slope_cols[1]], true_slope, "slope")
    print_recovery(r_slope)
    results$row83_slope <- r_slope
  }

  # Chain diagnostics
  cat("\nChain diagnostics:\n")
  cat(sprintf("  Total samples: %d (after warmup)\n", nrow(draws_83)))
  cat(sprintf("  Columns: %d parameters\n", ncol(draws_83)))

  if (length(slope_cols) > 0) {
    ess_bulk <- posterior::ess_bulk(draws_83[, slope_cols[1]])
    ess_tail <- posterior::ess_tail(draws_83[, slope_cols[1]])
    rhat     <- posterior::rhat(draws_83[, slope_cols[1]])
    cat(sprintf("  Slope ESS (bulk/tail): %.0f / %.0f\n", ess_bulk, ess_tail))
    cat(sprintf("  Slope R-hat: %.4f\n", rhat))
  }

  if (!is.null(fit_83$diagnostics)) {
    n_div <- sum(fit_83$diagnostics$divergent, na.rm = TRUE)
    cat(sprintf("  Divergences: %d\n", n_div))
  }
} else {
  cat("FAILED\n")
}

# =============================================================================
# Summary
# =============================================================================

cat("\n\n", strrep("=", 60), "\n")
cat("SUMMARY\n")
cat(strrep("=", 60), "\n\n")

cat(sprintf("%-10s %-10s %-10s %-10s %-10s\n",
            "Row", "Time(s)", "Slope", "Diff SD", "Status"))
cat(paste(rep("-", 55), collapse = ""), "\n")

if (!is.null(results$row51_slope)) {
  r <- results$row51_slope
  cat(sprintf("%-10d %-10.1f %-10.3f %-10.2f %-10s\n",
              51, t_51, r$mean, r$diff_sd, if(r$pass) "PASS" else "FAIL"))
}

if (!is.null(results$row83_slope)) {
  r <- results$row83_slope
  cat(sprintf("%-10d %-10.1f %-10.3f %-10.2f %-10s\n",
              83, t_83, r$mean, r$diff_sd, if(r$pass) "PASS" else "FAIL"))
}

cat("\nPrevious results (short chains):\n")
cat("  Row 51: slope=0.528 vs true=0.30, 2.79 SD => Sim* marginal\n")
cat("  Row 83: slope=0.293 vs true=0.30, SD=0.001 => Sim* marginal\n")

# Save results
saveRDS(results, "benchmarks/results_rerun_gp_rw1.rds")
cat("\nResults saved to benchmarks/results_rerun_gp_rw1.rds\n")
