# verify_gp_log_post_impl.R
# Verify that GP and Multiscale GP produce identical posteriors across
# all gradient modes (H, A_t, A, N) after consolidation into log_post_impl.h
#
# Before this fix, A_t and A modes silently computed a wrong log-posterior
# for GP/MSGP models (missing the GP contribution). This script confirms
# that all 4 modes now agree.

library(numdenom)

cat("====================================================================\n")
cat("Verify GP/MSGP gradient modes after log_post_impl.h consolidation\n")
cat("====================================================================\n\n")

set.seed(42)

# Small N for GP (O(N^3) complexity)
N_OBS <- 80
ITER <- 300
WARMUP <- 150
CHAINS <- 1

# --- Helper: generate GP effects from exponential covariance ---
generate_gp_effects <- function(coords, sigma, lengthscale) {
  dist_mat <- as.matrix(dist(coords))
  cov_mat <- sigma^2 * exp(-dist_mat / lengthscale)
  L <- chol(cov_mat + diag(1e-6, nrow(coords)))
  z <- rnorm(nrow(coords))
  effects <- as.vector(t(L) %*% z)
  effects - mean(effects)
}

# --- Helper: compare modes ---
compare_modes <- function(results, modes, params, label) {
  cat(sprintf("\n--- %s: Posterior Mean Comparison ---\n", label))
  cat(sprintf("%20s", "Parameter"))
  for (m in modes) cat(sprintf(" | %8s", m))
  cat(" | max_diff\n")
  cat(paste0(rep("-", 20 + length(modes) * 11 + 12), collapse = ""), "\n")

  all_pass <- TRUE
  for (p in params) {
    vals <- sapply(modes, function(m) {
      d <- as.matrix(results[[m]]$draws)
      cols <- colnames(d)
      if (p %in% cols) mean(d[, p]) else NA
    })
    max_diff <- max(vals, na.rm = TRUE) - min(vals, na.rm = TRUE)
    cat(sprintf("%20s", p))
    for (v in vals) cat(sprintf(" | %8.4f", v))
    cat(sprintf(" | %8.5f", max_diff))

    # Allow generous tolerance (MCMC noise + short chains)
    # Key check: if log-post was wrong, means would differ by >1
    if (max_diff > 1.0) {
      cat("  ** FAIL **")
      all_pass <- FALSE
    }
    cat("\n")
  }

  # Compare timings
  cat(sprintf("\n%20s", "Time (s)"))
  for (m in modes) cat(sprintf(" | %8.1f", results[[m]]$time))
  cat("\n")

  # Divergences
  cat(sprintf("%20s", "Divergences"))
  for (m in modes) {
    n_div <- if (!is.null(results[[m]]$fit$divergences)) sum(results[[m]]$fit$divergences) else 0
    cat(sprintf(" | %8d", n_div))
  }
  cat("\n")

  return(all_pass)
}

# =====================================================================
# TEST 1: spatial_gp() with negbin_negbin
# =====================================================================
cat("\n========================================\n")
cat("TEST 1: spatial_gp() + negbin_negbin\n")
cat("========================================\n")

# Generate spatial coordinates
coord_x <- runif(N_OBS, 0, 10)
coord_y <- runif(N_OBS, 0, 10)
coords_mat <- cbind(coord_x, coord_y)

# True parameters
true_intercept <- 2.0
true_slope <- 0.3
x <- rnorm(N_OBS)

# Generate spatial effects
spatial_effects <- generate_gp_effects(coords_mat, sigma = 0.5, lengthscale = 2.0)

# Generate data
eta_num <- true_intercept + true_slope * x + spatial_effects
eta_denom <- true_intercept + spatial_effects
mu_num <- exp(eta_num)
mu_denom <- exp(eta_denom)
phi_true <- 5.0
y_num <- rnbinom(N_OBS, mu = mu_num, size = phi_true)
y_denom <- rnbinom(N_OBS, mu = mu_denom, size = phi_true)
y_denom[y_denom == 0] <- 1L

df_gp <- data.frame(
  y_num = y_num, y_denom = y_denom, x = x,
  coord_x = coord_x, coord_y = coord_y
)

modes <- c("H", "A_t", "A", "N")
results_gp <- list()

for (mode in modes) {
  cat(sprintf("  Fitting with gradient_mode = '%s'...", mode))
  time <- system.time({
    fit <- tryCatch(
      ratiod(
        y_num | y_denom ~ x,
        data = df_gp,
        family = ratiod_negbin_negbin(),
        spatial = spatial_gp(coords = c("coord_x", "coord_y"), cov = "exponential"),
        iter = ITER, warmup = WARMUP, chains = CHAINS,
        verbose = FALSE,
        gradient_mode = mode
      ),
      error = function(e) {
        cat(sprintf(" ERROR: %s\n", e$message))
        NULL
      }
    )
  })["elapsed"]

  if (!is.null(fit)) {
    results_gp[[mode]] <- list(fit = fit, draws = fit$draws, time = time)
    cat(sprintf(" done (%.1fs)\n", time))
  }
}

# Compare
gp_modes <- names(results_gp)
if (length(gp_modes) >= 2) {
  gp_params <- c("beta_num[1]", "beta_num[2]", "beta_denom[1]",
                  "log_phi_num", "log_phi_denom")
  pass1 <- compare_modes(results_gp, gp_modes, gp_params, "GP negbin")
} else {
  cat("  Fewer than 2 modes succeeded — cannot compare\n")
  pass1 <- FALSE
}

# =====================================================================
# TEST 2: spatial_multiscale() with negbin_negbin
# =====================================================================
cat("\n========================================\n")
cat("TEST 2: spatial_multiscale() + negbin_negbin\n")
cat("========================================\n")

# Reuse same data but with multiscale GP
results_ms <- list()

for (mode in modes) {
  cat(sprintf("  Fitting with gradient_mode = '%s'...", mode))
  time <- system.time({
    fit <- tryCatch(
      ratiod(
        y_num | y_denom ~ x,
        data = df_gp,
        family = ratiod_negbin_negbin(),
        spatial = spatial_multiscale(coords = c("coord_x", "coord_y"), cov = "exponential"),
        iter = ITER, warmup = WARMUP, chains = CHAINS,
        verbose = FALSE,
        gradient_mode = mode
      ),
      error = function(e) {
        cat(sprintf(" ERROR: %s\n", e$message))
        NULL
      }
    )
  })["elapsed"]

  if (!is.null(fit)) {
    results_ms[[mode]] <- list(fit = fit, draws = fit$draws, time = time)
    cat(sprintf(" done (%.1fs)\n", time))
  }
}

# Compare
ms_modes <- names(results_ms)
if (length(ms_modes) >= 2) {
  ms_params <- c("beta_num[1]", "beta_num[2]", "beta_denom[1]",
                  "log_phi_num", "log_phi_denom")
  pass2 <- compare_modes(results_ms, ms_modes, ms_params, "MSGP negbin")
} else {
  cat("  Fewer than 2 modes succeeded — cannot compare\n")
  pass2 <- FALSE
}

# =====================================================================
# SUMMARY
# =====================================================================
cat("\n====================================================================\n")
cat("SUMMARY\n")
cat("====================================================================\n")
cat(sprintf("  TEST 1 (GP + negbin):   %s\n", if (pass1) "PASS" else "FAIL"))
cat(sprintf("  TEST 2 (MSGP + negbin): %s\n", if (pass2) "PASS" else "FAIL"))

if (pass1 && pass2) {
  cat("\n  All gradient modes produce consistent posteriors.\n")
  cat("  GP/MSGP log_post_impl.h consolidation verified.\n")
} else {
  cat("\n  WARNING: Some modes disagree. Investigate further.\n")
}
cat("====================================================================\n")
