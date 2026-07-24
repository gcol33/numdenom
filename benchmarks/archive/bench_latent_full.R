# Full benchmark for latent factor models: N, A, A_t, H vs Stan
# Uses N=50 to keep runtime manageable

library(numdenom)

set.seed(42)

# Reduced benchmark parameters for latent factors
N_OBS <- 50
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1
N_SITES <- 10
N_FACTORS <- 2
TIMEOUT <- 600  # 10 min max per model

cat("=======================================================\n")
cat("Latent Factor Full Benchmark: N/A/A_t/H vs Stan\n")
cat("N =", N_OBS, ", iter =", N_ITER, ", K =", N_FACTORS, "\n")
cat("Total latent params:", N_OBS * N_FACTORS, "+ ~20 base\n")
cat("=======================================================\n\n")

# Generate test data
sites <- rep(1:N_SITES, length.out = N_OBS)
x <- rnorm(N_OBS)

# True latent factors
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

# Helper to run with timeout
run_with_timeout <- function(expr, timeout_sec = TIMEOUT) {
  tryCatch({
    setTimeLimit(cpu = timeout_sec, elapsed = timeout_sec, transient = TRUE)
    on.exit(setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE))
    time <- system.time(eval(expr))["elapsed"]
    return(time)
  }, error = function(e) {
    if (grepl("time limit|timeout", conditionMessage(e), ignore.case = TRUE)) {
      return(NA)  # Timeout
    }
    stop(e)
  })
}

# =============================================================================
# Row 30: poisson_gamma + latent
# =============================================================================
cat("\n--- Row 30: poisson_gamma + latent ---\n")

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

for (mode in c("N", "A", "A_t", "H")) {
  cat("  Mode", mode, ": ")
  flush.console()
  tryCatch({
    time <- system.time({
      fit <- tratio(
        count | effort ~ x + (1 | site),
        data = df_pg,
        family = ratiod_poisson_gamma(),
        latent = latent_factor(n_factors = N_FACTORS, shared = TRUE),
        control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = mode)
      )
    })["elapsed"]
    cat(round(time, 1), "s\n")
    results[[paste0("pg_", mode)]] <- time
  }, error = function(e) {
    cat("ERROR:", conditionMessage(e), "\n")
    results[[paste0("pg_", mode)]] <- NA
  })
}

# Stan comparison (brms) - use observation-level RE as proxy
cat("  Stan (obs RE): ")
flush.console()
tryCatch({
  if (requireNamespace("brms", quietly = TRUE)) {
    time <- system.time({
      fit_stan <- brms::brm(
        count ~ x + (1 | site) + (1 | obs),
        data = transform(df_pg, obs = 1:nrow(df_pg)),
        family = brms::poisson(),
        iter = N_ITER,
        warmup = N_WARMUP,
        chains = N_CHAINS,
        refresh = 0,
        silent = 2
      )
    })["elapsed"]
    cat(round(time, 1), "s\n")
    results$pg_stan <- time
  } else {
    cat("brms not available\n")
    results$pg_stan <- NA
  }
}, error = function(e) {
  cat("ERROR:", conditionMessage(e), "\n")
  results$pg_stan <- NA
})

# =============================================================================
# Row 60: negbin_negbin + latent
# =============================================================================
cat("\n--- Row 60: negbin_negbin + latent ---\n")

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

for (mode in c("N", "A", "A_t", "H")) {
  cat("  Mode", mode, ": ")
  flush.console()
  tryCatch({
    time <- system.time({
      fit <- tratio(
        count_num | count_denom ~ x + (1 | site),
        data = df_nb,
        family = ratiod_negbin_negbin(),
        latent = latent_factor(n_factors = N_FACTORS, shared = TRUE),
        control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = mode)
      )
    })["elapsed"]
    cat(round(time, 1), "s\n")
    results[[paste0("nb_", mode)]] <- time
  }, error = function(e) {
    cat("ERROR:", conditionMessage(e), "\n")
    results[[paste0("nb_", mode)]] <- NA
  })
}

# Stan comparison
cat("  Stan (obs RE): ")
flush.console()
tryCatch({
  if (requireNamespace("brms", quietly = TRUE)) {
    time <- system.time({
      fit_stan <- brms::brm(
        count_num ~ x + (1 | site) + (1 | obs),
        data = transform(df_nb, obs = 1:nrow(df_nb)),
        family = brms::negbinomial(),
        iter = N_ITER,
        warmup = N_WARMUP,
        chains = N_CHAINS,
        refresh = 0,
        silent = 2
      )
    })["elapsed"]
    cat(round(time, 1), "s\n")
    results$nb_stan <- time
  } else {
    cat("brms not available\n")
    results$nb_stan <- NA
  }
}, error = function(e) {
  cat("ERROR:", conditionMessage(e), "\n")
  results$nb_stan <- NA
})

# =============================================================================
# Row 92: binomial + latent
# =============================================================================
cat("\n--- Row 92: binomial + latent ---\n")

n_trials <- sample(10:50, N_OBS, replace = TRUE)
p <- plogis(eta_num)
y_binom <- rbinom(N_OBS, size = n_trials, prob = p)

df_bin <- data.frame(
  successes = y_binom,
  trials = n_trials,
  x = x,
  site = factor(sites)
)

for (mode in c("N", "A", "A_t", "H")) {
  cat("  Mode", mode, ": ")
  flush.console()
  tryCatch({
    time <- system.time({
      fit <- tratio(
        successes | trials ~ x + (1 | site),
        data = df_bin,
        family = ratiod_binomial(),
        latent = latent_factor(n_factors = N_FACTORS, shared = TRUE),
        control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = mode)
      )
    })["elapsed"]
    cat(round(time, 1), "s\n")
    results[[paste0("bin_", mode)]] <- time
  }, error = function(e) {
    cat("ERROR:", conditionMessage(e), "\n")
    results[[paste0("bin_", mode)]] <- NA
  })
}

# Stan comparison
cat("  Stan (obs RE): ")
flush.console()
tryCatch({
  if (requireNamespace("brms", quietly = TRUE)) {
    time <- system.time({
      fit_stan <- brms::brm(
        successes | trials(trials) ~ x + (1 | site) + (1 | obs),
        data = transform(df_bin, obs = 1:nrow(df_bin)),
        family = brms::binomial(),
        iter = N_ITER,
        warmup = N_WARMUP,
        chains = N_CHAINS,
        refresh = 0,
        silent = 2
      )
    })["elapsed"]
    cat(round(time, 1), "s\n")
    results$bin_stan <- time
  } else {
    cat("brms not available\n")
    results$bin_stan <- NA
  }
}, error = function(e) {
  cat("ERROR:", conditionMessage(e), "\n")
  results$bin_stan <- NA
})

# =============================================================================
# Summary
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY: Latent Factor Full Benchmark (N=50, K=2)\n")
cat("=======================================================\n\n")

get_time <- function(name) {
  if (!is.null(results[[name]]) && !is.na(results[[name]])) {
    return(results[[name]])
  }
  return(NA)
}

cat("Timings (seconds):\n")
cat("-----------------------------------------------------------------------\n")
cat(sprintf("%-20s %8s %8s %8s %8s %8s\n", "Model", "N", "A", "A_t", "H", "Stan"))
cat("-----------------------------------------------------------------------\n")

# Row 30
cat(sprintf("%-20s %8.1f %8.1f %8.1f %8.1f %8.1f\n",
            "poisson_gamma (30)",
            get_time("pg_N"), get_time("pg_A"), get_time("pg_A_t"),
            get_time("pg_H"), get_time("pg_stan")))

# Row 60
cat(sprintf("%-20s %8.1f %8.1f %8.1f %8.1f %8.1f\n",
            "negbin_negbin (60)",
            get_time("nb_N"), get_time("nb_A"), get_time("nb_A_t"),
            get_time("nb_H"), get_time("nb_stan")))

# Row 92
cat(sprintf("%-20s %8.1f %8.1f %8.1f %8.1f %8.1f\n",
            "binomial (92)",
            get_time("bin_N"), get_time("bin_A"), get_time("bin_A_t"),
            get_time("bin_H"), get_time("bin_stan")))

cat("-----------------------------------------------------------------------\n")

# Speedup calculations
cat("\nSpeedup Ratios (vs N):\n")
cat("-----------------------------------------------------------------------\n")

for (family in c("pg", "nb", "bin")) {
  t_N <- get_time(paste0(family, "_N"))
  t_A <- get_time(paste0(family, "_A"))
  t_At <- get_time(paste0(family, "_A_t"))
  t_H <- get_time(paste0(family, "_H"))
  t_stan <- get_time(paste0(family, "_stan"))

  if (!is.na(t_N) && !is.na(t_H) && t_H > 0) {
    cat(sprintf("%s: N/H = %.1fx", family, t_N / t_H))
    if (!is.na(t_A)) cat(sprintf(", N/A = %.1fx", t_N / t_A))
    if (!is.na(t_At)) cat(sprintf(", N/A_t = %.1fx", t_N / t_At))
    if (!is.na(t_stan) && t_stan > 0) cat(sprintf(", Stan/H = %.1fx", t_stan / t_H))
    cat("\n")
  }
}

cat("-----------------------------------------------------------------------\n")
cat("\nNote: Stan uses (1|obs) as proxy for latent factors.\n")
cat("      numdenom latent factors are shared between num/denom.\n")

# Save results
saveRDS(results, "benchmarks/results_latent_full.rds")
cat("\nResults saved to benchmarks/results_latent_full.rds\n")
