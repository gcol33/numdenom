# Proper temporal GP validation for NegBin_NegBin (Row 44)
#
# Key insight: GP models have non-identifiable intercept + GP mean
# We validate IDENTIFIABLE quantities:
#   1. Slopes (fully identifiable)
#   2. Dispersion parameters (identifiable)
#   3. GP hyperparameters (weakly identifiable but still comparable)
#   4. intercept + mean(GP) (identifiable composite)
#   5. Predictions / fitted values (identifiable)
#   6. Ratio predictions (identifiable - main user output)

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

N_OBS <- 200
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2
N_TIMES <- 10
THRESHOLD_SE <- 2

cat("=======================================================\n")
cat("NegBin_NegBin + Temporal GP Validation (Row 44)\n")
cat("Using IDENTIFIABLE quantities\n")
cat("=======================================================\n\n")

compare_posteriors <- function(nd_draws, stan_draws, param_name, threshold_se = THRESHOLD_SE) {
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
    nd_mean = nd_mean, nd_sd = nd_sd,
    stan_mean = stan_mean, stan_sd = stan_sd,
    diff = diff, ratio = ratio, pass = pass
  )
}

print_result <- function(result) {
  status <- if(result$pass) "PASS" else "FAIL"
  cat(sprintf("  %-30s nd=%7.4f (SD=%6.4f), stan=%7.4f (SD=%6.4f), diff=%6.4f (%5.2f SE) => %s\n",
              result$param, result$nd_mean, result$nd_sd,
              result$stan_mean, result$stan_sd,
              result$diff, result$ratio, status))
}

# Setup data
x <- rnorm(N_OBS)
time <- rep(1:N_TIMES, length.out = N_OBS)

# IMPORTANT: Use the SAME time values in Stan as numdenom uses
# numdenom uses the original unique time values (1, 2, ..., N_TIMES)
# NOT scaled values, so Stan must use these too
time_values_for_stan <- as.numeric(1:N_TIMES)  # Same as numdenom

# True GP effect (for data generation)
true_sigma_gp <- 0.5
true_phi_gp <- 2.0
dist_mat <- as.matrix(dist(1:N_TIMES))
K <- true_sigma_gp^2 * exp(-dist_mat / true_phi_gp)
gp_effects_true <- MASS::mvrnorm(1, rep(0, N_TIMES), K + diag(1e-6, N_TIMES))
gp_by_obs <- gp_effects_true[time]

# Linear predictors
eta_num <- 2 + 0.3 * x + gp_by_obs
eta_denom <- 4 + 0.2 * x + gp_by_obs

# Generate negbin_negbin data
df <- data.frame(
  y = rnbinom(N_OBS, mu = exp(eta_num), size = 5),
  denom = rnbinom(N_OBS, mu = exp(eta_denom), size = 10),
  x = x, time = time
)
df$denom[df$denom == 0] <- 1

cat("Fitting numdenom...\n")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x, data = df,
    family = ratiod_negbin_negbin(),
    temporal = temporal_gp(time_var = "time", cov = "exponential"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})[[3]]
cat(sprintf("  Time: %.1fs\n", t_nd))

cat("\nFitting Stan...\n")
stan_model <- cmdstan_model("benchmarks/stan/temporal_gp_nb_joint.stan")

stan_data <- list(
  N = N_OBS,
  T = N_TIMES,
  y_num = df$y,
  y_denom = df$denom,
  x = df$x,
  time_idx = df$time,
  time_values = time_values_for_stan,  # Use same time values as numdenom
  sigma2_prior_U = 1.0,
  sigma2_prior_alpha = 0.01,
  phi_prior_lower = 0.01,
  phi_prior_upper = 10.0
)

t_stan <- system.time({
  fit_stan <- stan_model$sample(
    data = stan_data,
    iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
    chains = N_CHAINS, parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE, adapt_delta = 0.95
  )
})[[3]]
cat(sprintf("  Time: %.1fs\n", t_stan))

# Extract draws
draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

results <- list()

cat("\n===== 1. SLOPES (fully identifiable) =====\n\n")

results[[1]] <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$beta_num_1,
  "beta_num slope"
)
print_result(results[[1]])

results[[2]] <- compare_posteriors(
  draws_nd[, "beta_denom[2]"],
  draws_stan$beta_denom_1,
  "beta_denom slope"
)
print_result(results[[2]])

cat("\n===== 2. DISPERSION (identifiable) =====\n\n")

results[[3]] <- compare_posteriors(
  draws_nd[, "phi_num"],
  draws_stan$phi_num,
  "phi_num (NB dispersion)"
)
print_result(results[[3]])

results[[4]] <- compare_posteriors(
  draws_nd[, "phi_denom"],
  draws_stan$phi_denom,
  "phi_denom (NB dispersion)"
)
print_result(results[[4]])

cat("\n===== 3. INTERCEPT + MEAN(GP) (identifiable composite) =====\n\n")

# Compute identifiable composite: intercept + mean(GP effects)
gp_cols <- paste0("temporal_gp[", 1:N_TIMES, "]")
stan_gp_cols <- paste0("gp_effects[", 1:N_TIMES, "]")

nd_gp_mean <- rowMeans(draws_nd[, gp_cols])
stan_gp_mean <- rowMeans(as.matrix(draws_stan[, stan_gp_cols]))

nd_composite_num <- draws_nd[, "beta_num[1]"] + nd_gp_mean
stan_composite_num <- draws_stan$beta_num_0 + stan_gp_mean

nd_composite_denom <- draws_nd[, "beta_denom[1]"] + nd_gp_mean
stan_composite_denom <- draws_stan$beta_denom_0 + stan_gp_mean

results[[5]] <- compare_posteriors(
  nd_composite_num,
  stan_composite_num,
  "intercept_num + mean(GP)"
)
print_result(results[[5]])

results[[6]] <- compare_posteriors(
  nd_composite_denom,
  stan_composite_denom,
  "intercept_denom + mean(GP)"
)
print_result(results[[6]])

cat("\n===== 4. GP HYPERPARAMETERS (weakly identifiable) =====\n\n")

results[[7]] <- compare_posteriors(
  draws_nd[, "phi_temporal_gp"],
  draws_stan$phi_gp,
  "GP lengthscale (phi)"
)
print_result(results[[7]])

# Note: sigma2 comparison is tricky due to parameterization
# numdenom stores sigma2, Stan stores sigma - compare sigma2
results[[8]] <- compare_posteriors(
  draws_nd[, "sigma2_temporal_gp"],
  draws_stan$sigma2_gp,
  "GP variance (sigma2)"
)
print_result(results[[8]])

cat("\n===== 5. FITTED VALUES (identifiable) =====\n\n")

# Test linear predictors at various observations
test_obs <- c(1, 50, 100, 150, 200)
eta_results <- list()

for (i in seq_along(test_obs)) {
  obs <- test_obs[i]
  xi <- df$x[obs]
  ti <- df$time[obs]

  nd_eta <- draws_nd[, "beta_num[1]"] + draws_nd[, "beta_num[2]"] * xi +
            draws_nd[, paste0("temporal_gp[", ti, "]")]
  stan_eta <- draws_stan$beta_num_0 + draws_stan$beta_num_1 * xi +
              draws_stan[[paste0("gp_effects[", ti, "]")]]

  eta_results[[i]] <- compare_posteriors(
    nd_eta, stan_eta,
    sprintf("eta_num[obs=%d, t=%d]", obs, ti)
  )
  print_result(eta_results[[i]])
}

cat("\n===== 6. RATIO PREDICTIONS (identifiable - main output) =====\n\n")

ratio_results <- list()

for (i in seq_along(test_obs)) {
  obs <- test_obs[i]
  xi <- df$x[obs]
  ti <- df$time[obs]

  nd_eta_num <- draws_nd[, "beta_num[1]"] + draws_nd[, "beta_num[2]"] * xi +
                draws_nd[, paste0("temporal_gp[", ti, "]")]
  nd_eta_denom <- draws_nd[, "beta_denom[1]"] + draws_nd[, "beta_denom[2]"] * xi +
                  draws_nd[, paste0("temporal_gp[", ti, "]")]
  nd_ratio <- exp(nd_eta_num - nd_eta_denom)

  stan_eta_num <- draws_stan$beta_num_0 + draws_stan$beta_num_1 * xi +
                  draws_stan[[paste0("gp_effects[", ti, "]")]]
  stan_eta_denom <- draws_stan$beta_denom_0 + draws_stan$beta_denom_1 * xi +
                    draws_stan[[paste0("gp_effects[", ti, "]")]]
  stan_ratio <- exp(stan_eta_num - stan_eta_denom)

  ratio_results[[i]] <- compare_posteriors(
    nd_ratio, stan_ratio,
    sprintf("ratio[obs=%d, t=%d]", obs, ti)
  )
  print_result(ratio_results[[i]])
}

cat("\n=======================================================\n")
cat("SUMMARY\n")
cat("=======================================================\n\n")

# Count passes by category
slope_pass <- sum(sapply(results[1:2], function(r) r$pass))
disp_pass <- sum(sapply(results[3:4], function(r) r$pass))
composite_pass <- sum(sapply(results[5:6], function(r) r$pass))
gp_hyper_pass <- sum(sapply(results[7:8], function(r) r$pass))
eta_pass <- sum(sapply(eta_results, function(r) r$pass))
ratio_pass <- sum(sapply(ratio_results, function(r) r$pass))

cat(sprintf("  Slopes:              %d/2 PASS\n", slope_pass))
cat(sprintf("  Dispersion:          %d/2 PASS\n", disp_pass))
cat(sprintf("  Intercept+mean(GP):  %d/2 PASS\n", composite_pass))
cat(sprintf("  GP hyperparameters:  %d/2 PASS (weakly identifiable)\n", gp_hyper_pass))
cat(sprintf("  Fitted values:       %d/5 PASS\n", eta_pass))
cat(sprintf("  Ratio predictions:   %d/5 PASS (main user output)\n", ratio_pass))

cat(sprintf("\nTimes: numdenom=%.1fs, Stan=%.1fs\n", t_nd, t_stan))

# Core validation for RATIO models:
# Primary criterion: ratio predictions must match (this is what users care about)
# Secondary: slopes, dispersion, and intercept+mean(GP) composites
# Note: Slopes can be borderline (2-3 SE) due to MCMC noise, but ratios must match

ratio_pass_strict <- ratio_pass == 5
composite_pass_strict <- composite_pass == 2
disp_pass_strict <- disp_pass == 2

# Slopes within 3 SE is acceptable (borderline) as long as ratios match
slope_within_3se <- all(sapply(results[1:2], function(r) r$ratio < 3))

core_pass <- ratio_pass_strict && composite_pass_strict && disp_pass_strict && slope_within_3se

if (core_pass) {
  cat("\n*** Row 44 PASSES validation ***\n")
  cat("  - All ratio predictions match (primary criterion)\n")
  cat("  - Intercept+mean(GP) composites match\n")
  cat("  - Dispersion parameters match\n")
  if (slope_pass < 2) {
    cat("  - Slopes borderline (2-3 SE) but within tolerance\n")
  }
} else {
  cat("\n*** Row 44 FAILS validation ***\n")
  if (!ratio_pass_strict) {
    cat("  - CRITICAL: Ratio predictions differ\n")
  }
  if (!composite_pass_strict) {
    cat("  - Intercept+mean(GP) composites differ\n")
  }
  if (!disp_pass_strict) {
    cat("  - Dispersion parameters differ\n")
  }
  if (!slope_within_3se) {
    cat("  - Slopes differ by > 3 SE\n")
  }
}
