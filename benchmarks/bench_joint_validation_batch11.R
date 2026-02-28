# Validation of numdenom temporal GP models against custom joint Stan models
# Batch 11: Temporal GP (rows 14, 44)
#
# Rows validated:
#   - Row 14: poisson_gamma + temporal_gp
#   - Row 44: negbin_negbin + temporal_gp
#
# Note: The existing temporal_gp_joint.stan is for negbin_negbin.
#       We need a poisson_gamma version as well.

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

# Standard benchmark parameters
N_OBS <- 200
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2
N_SITES <- 20
N_TIMES <- 15

cat("=======================================================\n")
cat("Joint Model Validation Batch 11: Temporal GP\n")
cat("=======================================================\n")
cat(sprintf("N=%d, iter=%d, warmup=%d, chains=%d\n", N_OBS, N_ITER, N_WARMUP, N_CHAINS))
cat(sprintf("sites=%d, times=%d\n\n", N_SITES, N_TIMES))

# Helper function to compare posteriors
compare_posteriors <- function(nd_draws, stan_draws, param_name, threshold_se = 2) {
  nd_mean <- mean(nd_draws)
  nd_sd <- sd(nd_draws)
  stan_mean <- mean(stan_draws)
  stan_sd <- sd(stan_draws)

  se_combined <- sqrt(nd_sd^2 / length(nd_draws) + stan_sd^2 / length(stan_draws))
  diff <- abs(nd_mean - stan_mean)
  ratio <- diff / se_combined

  pass <- ratio < threshold_se

  list(
    param = param_name,
    nd_mean = nd_mean,
    nd_sd = nd_sd,
    stan_mean = stan_mean,
    stan_sd = stan_sd,
    diff = diff,
    ratio = ratio,
    pass = pass
  )
}

print_result <- function(result) {
  cat(sprintf("  %s: nd=%.4f (SD=%.4f), Stan=%.4f (SD=%.4f), diff=%.4f (%.2f SE) => %s\n",
              result$param, result$nd_mean, result$nd_sd,
              result$stan_mean, result$stan_sd,
              result$diff, result$ratio, if(result$pass) "PASS" else "FAIL"))
}

# =============================================================================
# DATA SETUP
# =============================================================================

site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)
time <- rep(1:N_TIMES, length.out = N_OBS)
time_factor <- factor(time)
time_values <- 1:N_TIMES  # Unique time values for GP

# Generate base linear predictors
eta_num <- 2 + 0.3 * x
eta_denom <- 4 + 0.2 * x

results <- list()

# =============================================================================
# Row 44: negbin_negbin + temporal_gp
# =============================================================================
cat("\n========== Row 44: negbin_negbin + temporal_gp ==========\n")

df_nb_gp <- data.frame(
  y = rnbinom(N_OBS, mu = exp(eta_num), size = 5),
  denom = rnbinom(N_OBS, mu = exp(eta_denom), size = 10),
  x = x,
  site = site,
  time = time,
  time_factor = time_factor
)
df_nb_gp$denom[df_nb_gp$denom == 0] <- 1

cat("Fitting numdenom (no site RE, to match Stan model)... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x,  # No site RE - Stan model doesn't have it
    data = df_nb_gp,
    family = ratiod_negbin_negbin(),
    temporal = temporal_gp("time", cov = "exponential"),  # Use numeric time, not factor
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/temporal_gp_joint.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  T = N_TIMES,
  y_num = df_nb_gp$y,
  y_denom = df_nb_gp$denom,
  x = df_nb_gp$x,
  time_idx = df_nb_gp$time,
  time_values = as.numeric(time_values)
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model$sample(
    data = stan_data,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE,
    adapt_delta = 0.9
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

# Check what columns exist
cat("numdenom columns:", paste(head(colnames(draws_nd), 10), collapse = ", "), "...\n")
cat("Stan columns:", paste(head(names(draws_stan), 10), collapse = ", "), "...\n")

# Find beta_num[2] (slope) column
nd_beta_col <- grep("^beta_num\\[2\\]$|^beta\\[2\\]$", colnames(draws_nd), value = TRUE)[1]
if (is.na(nd_beta_col)) {
  nd_beta_col <- grep("beta_num", colnames(draws_nd), value = TRUE)[2]  # Second beta_num
}

results$row_44 <- list(
  beta_num_1 = compare_posteriors(
    draws_nd[, nd_beta_col],
    draws_stan$beta_num_1,
    "beta_num[2] (slope)"
  ),
  time_nd = t_nd,
  time_stan = t_stan
)
print_result(results$row_44$beta_num_1)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan/t_nd))

# =============================================================================
# Row 14: poisson_gamma + temporal_gp
# =============================================================================
cat("\n========== Row 14: poisson_gamma + temporal_gp ==========\n")

# Generate Poisson-Gamma data
y_num_pg <- rpois(N_OBS, exp(eta_num))
y_denom_pg <- rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_denom))

df_pg_gp <- data.frame(
  y = y_num_pg,
  denom = y_denom_pg,
  x = x,
  site = site,
  time = time,
  time_factor = time_factor
)
df_pg_gp$denom[df_pg_gp$denom < 0.01] <- 0.01

cat("Fitting numdenom... ")
t_nd_pg <- system.time({
  fit_nd_pg <- ratiod(
    y | denom ~ x + (1|site),
    data = df_pg_gp,
    family = ratiod_poisson_gamma(),
    temporal = temporal_gp("time", cov = "exponential"),  # Use numeric time, not factor
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd_pg))

# For row 14, we need a Poisson-Gamma Stan model
# The existing model is NegBin-NegBin, so we note this as "needs joint Stan"
cat("Note: No joint_pg_temporal_gp.stan exists yet - skipping Stan comparison\n")

results$row_14 <- list(
  time_nd = t_nd_pg,
  note = "needs joint Stan model (Poisson-Gamma with temporal GP)"
)

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - BATCH 11 (Temporal GP)\n")
cat("=======================================================\n\n")

cat(sprintf("%-8s %-30s %10s %10s %8s %s\n",
            "Row", "Model", "numdenom", "Stan", "Speedup", "Status"))
cat(paste(rep("-", 80), collapse = ""), "\n")

# Row 44
r44 <- results$row_44
pass44 <- r44$beta_num_1$pass
cat(sprintf("%-8s %-30s %10.1fs %10.1fs %8.1fx %s\n",
            "44", "nb + temporal_gp",
            r44$time_nd, r44$time_stan, r44$time_stan / r44$time_nd,
            if(pass44) "PASS" else "FAIL"))

# Row 14
r14 <- results$row_14
cat(sprintf("%-8s %-30s %10.1fs %10s %8s %s\n",
            "14", "pg + temporal_gp",
            r14$time_nd, "N/A", "N/A",
            "SKIP (needs joint Stan)"))

cat(paste(rep("-", 80), collapse = ""), "\n")

# Save results
saveRDS(results, "benchmarks/results_joint_batch11.rds")
cat("\nResults saved to benchmarks/results_joint_batch11.rds\n")

# Print rows to update in gradient_methods.md
cat("\n\nUPDATE gradient_methods.md:\n")
if (pass44) {
  cat("  Row 44: PASS - add '✓Stan (joint)' to Notes\n")
}
cat("  Row 14: needs joint_pg_temporal_gp.stan\n")
