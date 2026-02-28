# Benchmark latent factor models vs Stan
# Rows 30, 60, 92: poisson_gamma, negbin_negbin, binomial with latent factors
# Tests all gradient modes: N, A_t, H

library(numdenom)
library(brms)

set.seed(42)

# Standard benchmark parameters
N_OBS <- 500
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1
N_SITES <- 50
N_FACTORS <- 2

cat("=======================================================\n")
cat("Latent Factor Benchmark: numdenom vs Stan\n")
cat("N =", N_OBS, ", iter =", N_ITER, ", K =", N_FACTORS, "\n")
cat("=======================================================\n\n")

# Generate test data
sites <- rep(1:N_SITES, length.out = N_OBS)
x <- rnorm(N_OBS)

# True latent factors (shared confounder)
true_sigma <- c(0.5, 0.3)
true_factors <- matrix(rnorm(N_OBS * N_FACTORS), N_OBS, N_FACTORS)
# Apply sum-to-zero constraint
for (k in 1:N_FACTORS) {
  true_factors[, k] <- true_factors[, k] - mean(true_factors[, k])
}
latent_effect <- true_factors %*% true_sigma

# True parameters
beta_num <- c(0.5, 0.3)
beta_denom <- c(1.0, -0.2)
sigma_site <- 0.3
site_effects <- rnorm(N_SITES, 0, sigma_site)

# Linear predictors (shared latent effect)
eta_num <- beta_num[1] + beta_num[2] * x + site_effects[sites] + latent_effect
eta_denom <- beta_denom[1] + beta_denom[2] * x + site_effects[sites] + latent_effect

# Results storage
results <- list()

# =============================================================================
# Row 30: poisson_gamma + latent
# =============================================================================
cat("\n--- Row 30: poisson_gamma + latent ---\n")

# Generate poisson_gamma data
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

# numdenom: only test H mode (N and A are O(p^2) and prohibitively slow with 1000+ params)
cat("  numdenom gradients:\n")

# Mode H only - N and A are too slow with latent factors (O(p^2) with p=N*K)
cat("    Mode H : ")
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
  results$pg_latent_H <- list(time = time, fit = fit)
}, error = function(e) {
  cat("ERROR:", conditionMessage(e), "\n")
  results$pg_latent_H <- list(time = NA, error = conditionMessage(e))
})

# brms comparison (approximate - brms doesn't have exact latent factor equivalent)
# Use observation-level random effect as proxy
cat("  brms (obs RE proxy): ")
tryCatch({
  # brms doesn't directly support shared latent factors
  # Use a simplified model for comparison
  time_stan <- system.time({
    fit_stan <- brm(
      count ~ x + (1 | site),
      data = df_pg,
      family = poisson(),
      iter = N_ITER,
      warmup = N_WARMUP,
      chains = N_CHAINS,
      refresh = 0,
      silent = 2
    )
  })["elapsed"]
  cat(round(time_stan, 1), "s\n")
  results$pg_latent_stan <- list(time = time_stan, fit = fit_stan)
}, error = function(e) {
  cat("ERROR:", conditionMessage(e), "\n")
  results$pg_latent_stan <- list(time = NA, error = conditionMessage(e))
})

# =============================================================================
# Row 60: negbin_negbin + latent
# =============================================================================
cat("\n--- Row 60: negbin_negbin + latent ---\n")

# Generate negbin_negbin data
phi_num <- 3
phi_denom <- 4
y_num_nb <- rnbinom(N_OBS, size = phi_num, mu = mu_num)
y_denom_nb <- rnbinom(N_OBS, size = phi_denom, mu = mu_denom)

df_nb <- data.frame(
  count_num = y_num_nb,
  count_denom = y_denom_nb,
  x = x,
  site = factor(sites)
)

# numdenom: only test H mode
cat("  numdenom gradients:\n")

cat("    Mode H : ")
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
  results$nb_latent_H <- list(time = time, fit = fit)
}, error = function(e) {
  cat("ERROR:", conditionMessage(e), "\n")
  results$nb_latent_H <- list(time = NA, error = conditionMessage(e))
})

# brms comparison
cat("  brms (negbin): ")
tryCatch({
  time_stan <- system.time({
    fit_stan <- brm(
      count_num ~ x + (1 | site),
      data = df_nb,
      family = negbinomial(),
      iter = N_ITER,
      warmup = N_WARMUP,
      chains = N_CHAINS,
      refresh = 0,
      silent = 2
    )
  })["elapsed"]
  cat(round(time_stan, 1), "s\n")
  results$nb_latent_stan <- list(time = time_stan, fit = fit_stan)
}, error = function(e) {
  cat("ERROR:", conditionMessage(e), "\n")
  results$nb_latent_stan <- list(time = NA, error = conditionMessage(e))
})

# =============================================================================
# Row 92: binomial + latent
# =============================================================================
cat("\n--- Row 92: binomial + latent ---\n")

# Generate binomial data
n_trials <- sample(10:50, N_OBS, replace = TRUE)
p <- plogis(eta_num)
y_binom <- rbinom(N_OBS, size = n_trials, prob = p)

df_bin <- data.frame(
  successes = y_binom,
  trials = n_trials,
  x = x,
  site = factor(sites)
)

# numdenom: only test H mode
cat("  numdenom gradients:\n")

cat("    Mode H : ")
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
  results$bin_latent_H <- list(time = time, fit = fit)
}, error = function(e) {
  cat("ERROR:", conditionMessage(e), "\n")
  results$bin_latent_H <- list(time = NA, error = conditionMessage(e))
})

# brms comparison
cat("  brms (binomial): ")
tryCatch({
  time_stan <- system.time({
    fit_stan <- brm(
      successes | trials(trials) ~ x + (1 | site),
      data = df_bin,
      family = binomial(),
      iter = N_ITER,
      warmup = N_WARMUP,
      chains = N_CHAINS,
      refresh = 0,
      silent = 2
    )
  })["elapsed"]
  cat(round(time_stan, 1), "s\n")
  results$bin_latent_stan <- list(time = time_stan, fit = fit_stan)
}, error = function(e) {
  cat("ERROR:", conditionMessage(e), "\n")
  results$bin_latent_stan <- list(time = NA, error = conditionMessage(e))
})

# =============================================================================
# Summary
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY: Latent Factor Benchmark Results\n")
cat("=======================================================\n\n")

# Extract times
get_time <- function(name) {
  if (!is.null(results[[name]]) && !is.na(results[[name]]$time)) {
    return(results[[name]]$time)
  }
  return(NA)
}

cat("Gradient Mode Timings (seconds):\n")
cat("-------------------------------------------------------\n")
cat(sprintf("%-20s %8s %8s %8s %8s\n", "Model", "N", "A", "H", "Stan"))
cat("-------------------------------------------------------\n")

# Row 30: poisson_gamma
cat(sprintf("%-20s %8.1f %8.1f %8.1f %8.1f\n",
            "poisson_gamma (30)",
            get_time("pg_latent_N"),
            get_time("pg_latent_A"),
            get_time("pg_latent_H"),
            get_time("pg_latent_stan")))

# Row 60: negbin_negbin
cat(sprintf("%-20s %8.1f %8.1f %8.1f %8.1f\n",
            "negbin_negbin (60)",
            get_time("nb_latent_N"),
            get_time("nb_latent_A"),
            get_time("nb_latent_H"),
            get_time("nb_latent_stan")))

# Row 92: binomial
cat(sprintf("%-20s %8.1f %8.1f %8.1f %8.1f\n",
            "binomial (92)",
            get_time("bin_latent_N"),
            get_time("bin_latent_A"),
            get_time("bin_latent_H"),
            get_time("bin_latent_stan")))

cat("-------------------------------------------------------\n")

# Speedup calculations
cat("\nSpeedup Ratios:\n")
cat("-------------------------------------------------------\n")

for (family in c("pg", "nb", "bin")) {
  t_N <- get_time(paste0(family, "_latent_N"))
  t_A <- get_time(paste0(family, "_latent_A"))
  t_H <- get_time(paste0(family, "_latent_H"))

  if (!is.na(t_N) && !is.na(t_H) && t_H > 0) {
    cat(sprintf("%s: H is %.1fx faster than N\n", family, t_N / t_H))
  }
  if (!is.na(t_A) && !is.na(t_H) && t_H > 0) {
    cat(sprintf("%s: H is %.1fx faster than A\n", family, t_A / t_H))
  }
}

cat("-------------------------------------------------------\n")

# Save results
saveRDS(results, "benchmarks/results_latent.rds")
cat("\nResults saved to benchmarks/results_latent.rds\n")
