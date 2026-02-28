# Benchmark latent factor models with reduced N
# Latent factors add N×K parameters, so we use N=50 to keep total params manageable
# Standard benchmark uses N=500, but that yields 1000+ params which is too slow for HMC

library(numdenom)

set.seed(42)

# Reduced benchmark parameters for latent factors
N_OBS <- 50           # Reduced from 500 (keeps latent params at 100 instead of 1000)
N_ITER <- 500         # Standard
N_WARMUP <- 250       # Standard
N_CHAINS <- 1         # Single chain for timing
N_SITES <- 10         # Reduced from 50
N_FACTORS <- 2        # Standard

cat("=======================================================\n")
cat("Latent Factor Benchmark (Small Scale)\n")
cat("N =", N_OBS, ", iter =", N_ITER, ", K =", N_FACTORS, "\n")
cat("Total latent params:", N_OBS * N_FACTORS, "+ ~20 base =", N_OBS * N_FACTORS + 20, "\n")
cat("=======================================================\n\n")

# Generate test data
sites <- rep(1:N_SITES, length.out = N_OBS)
x <- rnorm(N_OBS)

# True latent factors (shared confounder)
true_sigma <- c(0.5, 0.3)
true_factors <- matrix(rnorm(N_OBS * N_FACTORS), N_OBS, N_FACTORS)
for (k in 1:N_FACTORS) {
  true_factors[, k] <- true_factors[, k] - mean(true_factors[, k])
}
latent_effect <- true_factors %*% true_sigma

# True parameters
beta_num <- c(0.5, 0.3)
beta_denom <- c(1.0, -0.2)
sigma_site <- 0.3
site_effects <- rnorm(N_SITES, 0, sigma_site)

# Linear predictors
eta_num <- beta_num[1] + beta_num[2] * x + site_effects[sites] + latent_effect
eta_denom <- beta_denom[1] + beta_denom[2] * x + site_effects[sites] + latent_effect

# Results storage
results <- list()

# =============================================================================
# Row 30: poisson_gamma + latent
# =============================================================================
cat("\n--- Row 30: poisson_gamma + latent (N=50) ---\n")

mu_num <- exp(eta_num)
mu_denom <- exp(eta_denom)
phi_denom <- 5

y_num <- rpois(N_OBS, mu_num)
y_denom <- rgamma(N_OBS, shape = phi_denom, rate = phi_denom / mu_denom)

df_pg <- data.frame(
  count = y_num,
  effort = y_denom,
  x = x,
  site = factor(sites)
)

cat("  Mode H: ")
flush.console()
tryCatch({
  time <- system.time({
    fit <- ratiod(
      count | effort ~ x + (1 | site),
      data = df_pg,
      family = ratiod_poisson_gamma(),
      latent = latent_factor(n_factors = N_FACTORS, shared = TRUE),
      iter = N_ITER,
      warmup = N_WARMUP,
      chains = N_CHAINS,
      gradient_mode = "H",
      refresh = 0
    )
  })["elapsed"]
  cat(round(time, 1), "s\n")
  results$pg_H <- time
}, error = function(e) {
  cat("ERROR:", conditionMessage(e), "\n")
  results$pg_H <- NA
})

# =============================================================================
# Row 60: negbin_negbin + latent
# =============================================================================
cat("\n--- Row 60: negbin_negbin + latent (N=50) ---\n")

phi_num <- 3
phi_denom_nb <- 4
y_num_nb <- rnbinom(N_OBS, size = phi_num, mu = mu_num)
y_denom_nb <- rnbinom(N_OBS, size = phi_denom_nb, mu = mu_denom)

df_nb <- data.frame(
  count_num = y_num_nb,
  count_denom = y_denom_nb,
  x = x,
  site = factor(sites)
)

cat("  Mode H: ")
flush.console()
tryCatch({
  time <- system.time({
    fit <- ratiod(
      count_num | count_denom ~ x + (1 | site),
      data = df_nb,
      family = ratiod_negbin_negbin(),
      latent = latent_factor(n_factors = N_FACTORS, shared = TRUE),
      iter = N_ITER,
      warmup = N_WARMUP,
      chains = N_CHAINS,
      gradient_mode = "H",
      refresh = 0
    )
  })["elapsed"]
  cat(round(time, 1), "s\n")
  results$nb_H <- time
}, error = function(e) {
  cat("ERROR:", conditionMessage(e), "\n")
  results$nb_H <- NA
})

# =============================================================================
# Row 92: binomial + latent
# =============================================================================
cat("\n--- Row 92: binomial + latent (N=50) ---\n")

n_trials <- sample(10:50, N_OBS, replace = TRUE)
p <- plogis(eta_num)
y_binom <- rbinom(N_OBS, size = n_trials, prob = p)

df_bin <- data.frame(
  successes = y_binom,
  trials = n_trials,
  x = x,
  site = factor(sites)
)

cat("  Mode H: ")
flush.console()
tryCatch({
  time <- system.time({
    fit <- ratiod(
      successes | trials ~ x + (1 | site),
      data = df_bin,
      family = ratiod_binomial(),
      latent = latent_factor(n_factors = N_FACTORS, shared = TRUE),
      iter = N_ITER,
      warmup = N_WARMUP,
      chains = N_CHAINS,
      gradient_mode = "H",
      refresh = 0
    )
  })["elapsed"]
  cat(round(time, 1), "s\n")
  results$bin_H <- time
}, error = function(e) {
  cat("ERROR:", conditionMessage(e), "\n")
  results$bin_H <- NA
})

# =============================================================================
# Summary
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY: Latent Factor Benchmark (N=50, K=2)\n")
cat("=======================================================\n\n")

cat("Timings (seconds):\n")
cat("-------------------------------------------------------\n")
cat(sprintf("%-25s %8.1f s\n", "poisson_gamma (row 30)", results$pg_H))
cat(sprintf("%-25s %8.1f s\n", "negbin_negbin (row 60)", results$nb_H))
cat(sprintf("%-25s %8.1f s\n", "binomial (row 92)", results$bin_H))
cat("-------------------------------------------------------\n")

cat("\nNote: Standard benchmark uses N=500, but latent factors add N×K params.\n")
cat("With N=500, K=2: 1000+ params → 45-70 min per model (exceeds timeout).\n")
cat("With N=50, K=2: ~120 params → manageable for HMC.\n")
cat("\nFor production latent factor models, consider mode='vi' for larger N.\n")

# Save results
saveRDS(results, "benchmarks/results_latent_small.rds")
cat("\nResults saved to benchmarks/results_latent_small.rds\n")
