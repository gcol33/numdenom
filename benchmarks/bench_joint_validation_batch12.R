# Validation of numdenom HSGP models
# Batch 12: HSGP spatial (rows 8, 38, 68)
#
# HSGP (Hilbert Space GP) is an efficient approximation to full GP
# These should work correctly since they use the same machinery as regular GP

library(numdenom)
library(brms)
library(cmdstanr)
library(posterior)

set.seed(42)

# Standard benchmark parameters
N_OBS <- 200
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2
N_SITES <- 20

cat("=======================================================\n")
cat("Joint Model Validation Batch 12: HSGP Spatial\n")
cat("=======================================================\n")
cat(sprintf("N=%d, iter=%d, warmup=%d, chains=%d\n", N_OBS, N_ITER, N_WARMUP, N_CHAINS))
cat(sprintf("sites=%d\n\n", N_SITES))

# Helper function to compare posteriors
compare_posteriors <- function(nd_draws, brms_draws, param_name, threshold_se = 2) {
  nd_mean <- mean(nd_draws)
  nd_sd <- sd(nd_draws)
  brms_mean <- mean(brms_draws)
  brms_sd <- sd(brms_draws)

  se_combined <- sqrt(nd_sd^2 / length(nd_draws) + brms_sd^2 / length(brms_draws))
  diff <- abs(nd_mean - brms_mean)
  ratio <- diff / se_combined

  pass <- ratio < threshold_se

  list(
    param = param_name,
    nd_mean = nd_mean,
    nd_sd = nd_sd,
    brms_mean = brms_mean,
    brms_sd = brms_sd,
    diff = diff,
    ratio = ratio,
    pass = pass
  )
}

print_result <- function(result) {
  cat(sprintf("  %s: nd=%.4f (SD=%.4f), brms=%.4f (SD=%.4f), diff=%.4f (%.2f SE) => %s\n",
              result$param, result$nd_mean, result$nd_sd,
              result$brms_mean, result$brms_sd,
              result$diff, result$ratio, if(result$pass) "PASS" else "FAIL"))
}

# =============================================================================
# DATA SETUP
# =============================================================================

site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)

# Spatial coordinates (random within unit square)
n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = seq(0, 1, length.out = n_side),
                    lat = seq(0, 1, length.out = n_side))[1:N_SITES, ]
site_int <- as.integer(site)
lon <- grid$lon[site_int]
lat <- grid$lat[site_int]

# Generate base linear predictors
eta_num <- 2 + 0.3 * x
eta_denom <- 4 + 0.2 * x

results <- list()

# =============================================================================
# Row 68: binomial + HSGP
# =============================================================================
cat("\n========== Row 68: binomial + HSGP ==========\n")

# Binomial data
trials <- sample(10:50, N_OBS, replace = TRUE)
y_bin <- rbinom(N_OBS, trials, plogis(0.5 + 0.3*x))

df_bin_hsgp <- data.frame(
  y = y_bin,
  trials = trials,
  x = x,
  site = site,
  lon = lon,
  lat = lat
)

cat("Fitting numdenom (HSGP)... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | trials ~ x + (1|site),
    data = df_bin_hsgp,
    family = ratiod_binomial(),
    spatial = spatial_hsgp(coords = c("lon", "lat"), m = 6),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

# brms with GP (no HSGP equivalent in brms, so we use regular GP approx)
cat("Fitting brms (gp())... ")
t_brms <- system.time({
  fit_brms <- brm(
    y | trials(trials) ~ x + (1|site) + gp(lon, lat),
    data = df_bin_hsgp,
    family = binomial(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    backend = "cmdstanr",
    silent = 2, refresh = 0
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_brms))

# Extract draws
nd_draws <- as.matrix(fit_nd$draws)
brms_draws <- as_draws_matrix(fit_brms)

# Compare x coefficient
nd_beta <- nd_draws[, "beta_num[2]"]
brms_beta <- brms_draws[, "b_x"]

results$row_68 <- list(
  beta = compare_posteriors(nd_beta, brms_beta, "beta[x]"),
  time_nd = t_nd,
  time_brms = t_brms
)
print_result(results$row_68$beta)
cat(sprintf("  Times: nd=%.1fs, brms=%.1fs, speedup=%.1fx\n", t_nd, t_brms, t_brms/t_nd))

# =============================================================================
# Row 8: poisson_gamma + HSGP (skip - needs joint Stan model)
# =============================================================================
cat("\n========== Row 8: poisson_gamma + HSGP ==========\n")
cat("SKIP: Requires custom joint Stan model for two-process validation\n")

# =============================================================================
# Row 38: negbin_negbin + HSGP (skip - needs joint Stan model)
# =============================================================================
cat("\n========== Row 38: negbin_negbin + HSGP ==========\n")
cat("SKIP: Requires custom joint Stan model for two-process validation\n")

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - BATCH 12 (HSGP Spatial)\n")
cat("=======================================================\n\n")

cat(sprintf("%-8s %-30s %10s %10s %8s %s\n",
            "Row", "Model", "numdenom", "brms", "Speedup", "Status"))
cat(paste(rep("-", 80), collapse = ""), "\n")

# Row 68
r68 <- results$row_68
cat(sprintf("%-8s %-30s %10.1fs %10.1fs %8.1fx %s\n",
            "68", "binomial + HSGP",
            r68$time_nd, r68$time_brms, r68$time_brms / r68$time_nd,
            if(r68$beta$pass) "PASS" else "FAIL"))

cat(sprintf("%-8s %-30s %10s %10s %8s %s\n",
            "8", "poisson_gamma + HSGP", "SKIP", "-", "-", "needs joint Stan"))
cat(sprintf("%-8s %-30s %10s %10s %8s %s\n",
            "38", "negbin_negbin + HSGP", "SKIP", "-", "-", "needs joint Stan"))

cat(paste(rep("-", 80), collapse = ""), "\n")

# Save results
saveRDS(results, "benchmarks/results_joint_batch12.rds")
cat("\nResults saved to benchmarks/results_joint_batch12.rds\n")

# Print rows to update in gradient_methods.md
cat("\n\nUPDATE gradient_methods.md:\n")
if (r68$beta$pass) {
  cat("  Row 68: PASS - add '✓Stan' to Notes\n")
}
