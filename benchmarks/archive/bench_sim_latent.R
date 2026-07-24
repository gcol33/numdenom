# Simulation-based validation for latent factor models
# Rows 30, 60, 92: latent factors with pg, nb, binomial families

library(numdenom)
library(posterior)

set.seed(42)

# Smaller N for latent (high dimensionality)
N_OBS <- 50
N_ITER <- 1000
N_WARMUP <- 500
N_CHAINS <- 2
N_SITES <- 10
K_FACTORS <- 2

cat("=======================================================\n")
cat("Latent Factor Models: Simulation-Based Validation\n")
cat("Rows 30, 60, 92\n")
cat("=======================================================\n")
cat(sprintf("N=%d, sites=%d, K=%d, iter=%d, chains=%d\n\n",
            N_OBS, N_SITES, K_FACTORS, N_ITER, N_CHAINS))

# Helper function to check parameter recovery
check_recovery <- function(draws, true_value, param_name, threshold_sd = 2) {
  post_mean <- mean(draws)
  post_sd <- sd(draws)
  q025 <- quantile(draws, 0.025)
  q975 <- quantile(draws, 0.975)

  diff_sd <- abs(post_mean - true_value) / post_sd
  in_ci <- true_value >= q025 && true_value <= q975
  pass <- diff_sd < threshold_sd

  list(param = param_name, true = true_value, mean = post_mean, sd = post_sd,
       diff_sd = diff_sd, in_ci = in_ci, pass = pass)
}

print_recovery <- function(result) {
  status <- if (result$pass) "PASS" else "FAIL"
  ci <- if (result$in_ci) "in CI" else "NOT in CI"
  cat(sprintf("  %s: true=%.3f, post=%.3f (SD=%.3f), %.2f SD, %s => %s\n",
              result$param, result$true, result$mean, result$sd, result$diff_sd, ci, status))
}

results <- list()

# Common setup
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)

# True parameters
true_intercept <- 1.0
true_slope <- 0.3

# Generate latent factors (shared confounders)
true_lambda <- matrix(rnorm(N_SITES * K_FACTORS, 0, 0.5), N_SITES, K_FACTORS)
latent_effect <- rowSums(true_lambda)[as.integer(site)]

# =============================================================================
# Row 30: poisson_gamma + latent
# =============================================================================
cat("\n========== Row 30: poisson_gamma + latent ==========\n")

eta_num_30 <- true_intercept + true_slope * x + latent_effect
eta_denom_30 <- 0.5 + 0.2 * x + latent_effect * 0.8  # Shared but scaled

# Generate Poisson-Gamma data
effort_30 <- rgamma(N_OBS, shape = 5, rate = 1)
count_30 <- rpois(N_OBS, lambda = exp(eta_num_30) * effort_30)

df_30 <- data.frame(count = count_30, effort = effort_30, x = x, site = site)

cat(sprintf("True: intercept=%.2f, slope=%.2f\n", true_intercept, true_slope))
cat("Fitting numdenom... ")

t_30 <- system.time({
  fit_30 <- tryCatch({
    tratio(count | effort ~ x + (1 | site), data = df_30,
           family = ratiod_poisson_gamma(),
           latent = latent_factor(K_FACTORS),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_30)) {
  cat(sprintf("%.1fs\n", t_30))
  draws_30 <- as.matrix(fit_30$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_30), value = TRUE)[1]

  results$row_30 <- list(
    slope = check_recovery(draws_30[, slope_col], true_slope, "slope"),
    time = t_30
  )
  print_recovery(results$row_30$slope)
  results$row_30$pass <- results$row_30$slope$pass
  cat(sprintf("Overall: %s (%.1fs)\n", if(results$row_30$pass) "PASS" else "FAIL", t_30))
} else {
  results$row_30 <- list(error = TRUE)
}

# =============================================================================
# Row 60: negbin_negbin + latent
# =============================================================================
cat("\n========== Row 60: negbin_negbin + latent ==========\n")

eta_num_60 <- true_intercept + true_slope * x + latent_effect
eta_denom_60 <- 0.5 + 0.2 * x + latent_effect * 0.8

# Generate NegBin-NegBin data
size_nb <- 5
num_60 <- rnbinom(N_OBS, size = size_nb, mu = exp(eta_num_60))
denom_60 <- rnbinom(N_OBS, size = size_nb, mu = exp(eta_denom_60))
denom_60[denom_60 == 0] <- 1  # Avoid zero denominators

df_60 <- data.frame(num = num_60, denom = denom_60, x = x, site = site)

cat(sprintf("True: intercept=%.2f, slope=%.2f\n", true_intercept, true_slope))
cat("Fitting numdenom... ")

t_60 <- system.time({
  fit_60 <- tryCatch({
    tratio(num | denom ~ x + (1 | site), data = df_60,
           family = ratiod_negbin_negbin(),
           latent = latent_factor(K_FACTORS),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_60)) {
  cat(sprintf("%.1fs\n", t_60))
  draws_60 <- as.matrix(fit_60$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_60), value = TRUE)[1]

  results$row_60 <- list(
    slope = check_recovery(draws_60[, slope_col], true_slope, "slope"),
    time = t_60
  )
  print_recovery(results$row_60$slope)
  results$row_60$pass <- results$row_60$slope$pass
  cat(sprintf("Overall: %s (%.1fs)\n", if(results$row_60$pass) "PASS" else "FAIL", t_60))
} else {
  results$row_60 <- list(error = TRUE)
}

# =============================================================================
# Row 92: binomial + latent
# =============================================================================
cat("\n========== Row 92: binomial + latent ==========\n")

eta_92 <- true_intercept + true_slope * x + latent_effect
prob_92 <- plogis(eta_92)
trials_92 <- sample(20:50, N_OBS, replace = TRUE)
successes_92 <- rbinom(N_OBS, size = trials_92, prob = prob_92)

df_92 <- data.frame(successes = successes_92, trials = trials_92, x = x, site = site)

cat(sprintf("True: intercept=%.2f, slope=%.2f\n", true_intercept, true_slope))
cat("Fitting numdenom... ")

t_92 <- system.time({
  fit_92 <- tryCatch({
    tratio(successes | trials ~ x + (1 | site), data = df_92,
           family = ratiod_binomial(),
           latent = latent_factor(K_FACTORS),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_92)) {
  cat(sprintf("%.1fs\n", t_92))
  draws_92 <- as.matrix(fit_92$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_92), value = TRUE)[1]

  results$row_92 <- list(
    slope = check_recovery(draws_92[, slope_col], true_slope, "slope"),
    time = t_92
  )
  print_recovery(results$row_92$slope)
  results$row_92$pass <- results$row_92$slope$pass
  cat(sprintf("Overall: %s (%.1fs)\n", if(results$row_92$pass) "PASS" else "FAIL", t_92))
} else {
  results$row_92 <- list(error = TRUE)
}

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - Latent Factor Simulation Validation\n")
cat("=======================================================\n\n")

for (rn in c("row_30", "row_60", "row_92")) {
  r <- results[[rn]]
  row_num <- gsub("row_", "", rn)
  if (!is.null(r$error) && r$error) {
    cat(sprintf("Row %s: ERROR\n", row_num))
  } else {
    cat(sprintf("Row %s: %s (%.2f SD, %.1fs)\n", row_num,
                if(r$pass) "PASS" else "FAIL", r$slope$diff_sd, r$time))
  }
}

saveRDS(results, "benchmarks/results_sim_latent.rds")
cat("\nResults saved.\n")
