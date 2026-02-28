# Validation of numdenom negbin_negbin + random slopes against custom joint Stan model
# Row 33 in gradient_methods.md
#
# Tests: numdenom ratiod_negbin_negbin() with (1 + x|site) formula
# Stan model: benchmarks/stan/joint_nb_slopes.stan (centered parameterization)

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

# Standard benchmark parameters
N_OBS <- 200
N_ITER <- 4000
N_WARMUP <- 1500
N_CHAINS <- 4
N_SITES <- 20

cat("=======================================================\n")
cat("Joint Model Validation: Row 33 (negbin_negbin + slopes)\n")
cat("=======================================================\n")
cat(sprintf("N=%d, iter=%d, warmup=%d, chains=%d\n", N_OBS, N_ITER, N_WARMUP, N_CHAINS))
cat(sprintf("sites=%d\n\n", N_SITES))

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
  cat(sprintf("\n%s:\n", result$param))
  cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", result$nd_mean, result$nd_sd))
  cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", result$stan_mean, result$stan_sd))
  cat(sprintf("  Diff: %.4f (%.2f SE) => %s\n",
              result$diff, result$ratio, if(result$pass) "PASS" else "FAIL"))
}

results <- list()

# Generate shared data
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)

# =============================================================================
# Row 33: negbin_negbin + random slopes
# =============================================================================
cat("\n========== Row 33: negbin_negbin + random slopes ==========\n")

# Generate data WITH known random effects
true_sigma_int <- 0.4
true_sigma_slope <- 0.2
true_rho <- 0.3
true_beta_num <- c(2, 0.5)       # Lower intercept for NegBin
true_beta_denom <- c(3, 0.2)
true_phi_num <- 5.0              # NegBin dispersion (num)
true_phi_denom <- 8.0            # NegBin dispersion (denom)

# Generate correlated RE using Cholesky
L <- matrix(c(1, 0, true_rho, sqrt(1-true_rho^2)), 2, 2)
true_re <- matrix(0, N_SITES, 2)
for (g in 1:N_SITES) {
  z <- rnorm(2)
  true_re[g, ] <- c(true_sigma_int, true_sigma_slope) * (L %*% z)
}

# Compute linear predictors with SHARED RE
eta_num <- true_beta_num[1] + true_beta_num[2] * x
eta_denom <- true_beta_denom[1] + true_beta_denom[2] * x
for (i in 1:N_OBS) {
  g <- as.integer(site[i])
  re_effect <- true_re[g, 1] + true_re[g, 2] * x[i]
  eta_num[i] <- eta_num[i] + re_effect
  eta_denom[i] <- eta_denom[i] + re_effect  # SHARED
}

# Generate NegBin data
y_num <- rnbinom(N_OBS, size = true_phi_num, mu = exp(eta_num))
y_denom <- rnbinom(N_OBS, size = true_phi_denom, mu = exp(eta_denom))

# Ensure no zeros in denominator (numdenom requirement for ratio)
# IMPORTANT: Apply same transformation to data used for both numdenom and Stan
n_zeros <- sum(y_denom == 0)
if (n_zeros > 0) {
  cat(sprintf("Replacing %d zeros in y_denom with 1s\n", n_zeros))
}
y_denom[y_denom == 0] <- 1

df_nb_slopes <- data.frame(
  y_num = y_num,
  y_denom = y_denom,
  x = x,
  site = site
)

cat(sprintf("True params: beta_num=[%.2f, %.2f], beta_denom=[%.2f, %.2f]\n",
            true_beta_num[1], true_beta_num[2], true_beta_denom[1], true_beta_denom[2]))
cat(sprintf("True phi: num=%.2f, denom=%.2f\n", true_phi_num, true_phi_denom))
cat(sprintf("True RE: sigma_int=%.2f, sigma_slope=%.2f, rho=%.2f\n",
            true_sigma_int, true_sigma_slope, true_rho))

# Fit numdenom
cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y_num | y_denom ~ x + (1 + x|site),
    data = df_nb_slopes,
    family = ratiod_negbin_negbin(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_nd))

# Compile and fit Stan model
cat("Compiling Stan model... ")
stan_file <- "benchmarks/stan/joint_nb_slopes.stan"
stan_model_nb_slopes <- cmdstan_model(stan_file)
cat("done\n")

stan_data_nb_slopes <- list(
  N = N_OBS,
  y_num = df_nb_slopes$y_num,
  y_denom = df_nb_slopes$y_denom,
  p = 2,
  X = cbind(1, df_nb_slopes$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_nb_slopes$site),
  x_slope = df_nb_slopes$x
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_nb_slopes$sample(
    data = stan_data_nb_slopes,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE,
    adapt_delta = 0.95  # Increase for correlated RE
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_stan))

# Extract draws
draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

# Compare multiple parameters
cat("\n--- Parameter Comparisons ---\n")

# For models with RE, compare effective intercepts (beta[1] + mean(RE_intercept))
# since raw intercept and mean(RE_intercept) are confounded
# IMPORTANT: numdenom uses NON-CENTERED parameterization by default!
# numdenom stores z ~ N(0,1) values, NOT actual RE values
# Actual RE = diag(sigma) * L * z for correlated slopes
# We need to transform before comparing

nd_cols <- colnames(draws_nd)

# Get sigma values (intercept and slope SDs)
nd_sigma_int <- draws_nd[, "sigma_re[1,intercept]"]
nd_sigma_slope <- draws_nd[, "sigma_re[1,x]"]

# Get Cholesky parameter L21 (correlation structure)
nd_L21 <- draws_nd[, "L_chol[1,1]"]

# numdenom stores z values for RE (non-centered parameterization)
# For correlated slopes: re = diag(sigma) * L * z
# where L = [1, 0; L21, sqrt(1-L21^2)]
# So: re_intercept = sigma_int * z_intercept
#     re_slope = sigma_slope * (L21 * z_intercept + sqrt(1-L21^2) * z_slope)

# Get z values (what numdenom calls "re")
z_int_cols_nd <- nd_cols[grep("^re\\[.*,intercept\\]", nd_cols)]
z_slope_cols_nd <- nd_cols[grep("^re\\[.*,x\\]", nd_cols)]

# Transform z to actual RE values for each group
n_draws <- nrow(draws_nd)
n_groups <- length(z_int_cols_nd)
re_int_nd <- matrix(0, n_draws, n_groups)
re_slope_nd <- matrix(0, n_draws, n_groups)

L22 <- sqrt(1 - nd_L21^2)
for (g in seq_len(n_groups)) {
  z_int_g <- draws_nd[, z_int_cols_nd[g]]
  z_slope_g <- draws_nd[, z_slope_cols_nd[g]]

  # Transform: re = diag(sigma) * L * z
  re_int_nd[, g] <- nd_sigma_int * z_int_g
  re_slope_nd[, g] <- nd_sigma_slope * (nd_L21 * z_int_g + L22 * z_slope_g)
}

# Compute mean RE intercept (transformed)
nd_mean_re_int <- rowMeans(re_int_nd)

# Stan names: re[g,1] for intercepts, re[g,2] for slopes
# Stan uses CENTERED parameterization - these are actual RE values
stan_re_int_cols <- grep("^re\\[[0-9]+,1\\]$", names(draws_stan), value = TRUE)
stan_mean_re_int <- if (length(stan_re_int_cols) > 0) rowMeans(draws_stan[, stan_re_int_cols, drop=FALSE]) else 0

# Effective intercept comparison (identified quantity)
results$eff_int_num <- compare_posteriors(
  draws_nd[, "beta_num[1]"] + nd_mean_re_int,
  draws_stan$`beta_num[1]` + stan_mean_re_int,
  "effective_intercept_num (beta_num[1] + mean(RE_intercept))"
)
print_result(results$eff_int_num)

results$eff_int_denom <- compare_posteriors(
  draws_nd[, "beta_denom[1]"] + nd_mean_re_int,
  draws_stan$`beta_denom[1]` + stan_mean_re_int,
  "effective_intercept_denom (beta_denom[1] + mean(RE_intercept))"
)
print_result(results$eff_int_denom)

# Slope comparison (not confounded with RE mean)
results$beta_num2 <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)
print_result(results$beta_num2)

results$beta_denom2 <- compare_posteriors(
  draws_nd[, "beta_denom[2]"],
  draws_stan$`beta_denom[2]`,
  "beta_denom[2] (x slope)"
)
print_result(results$beta_denom2)

# Compare dispersion parameters
results$phi_num <- compare_posteriors(
  draws_nd[, "phi_num"],
  draws_stan$phi_num,
  "phi_num (NegBin dispersion)"
)
print_result(results$phi_num)

results$phi_denom <- compare_posteriors(
  draws_nd[, "phi_denom"],
  draws_stan$phi_denom,
  "phi_denom (NegBin dispersion)"
)
print_result(results$phi_denom)

# Compare variance components
cat("\n--- Additional Diagnostics ---\n")
cat(sprintf("numdenom z columns: %s\n", paste(head(z_int_cols_nd, 5), collapse=", "), "..."))
cat(sprintf("Stan RE intercept columns: %s\n", paste(head(stan_re_int_cols, 5), collapse=", "), "..."))

# Compare sigma_re (should match between implementations)
results$sigma_int <- compare_posteriors(
  nd_sigma_int,
  draws_stan$sigma_intercept,
  "sigma_intercept (RE SD)"
)
print_result(results$sigma_int)

results$sigma_slope <- compare_posteriors(
  nd_sigma_slope,
  draws_stan$sigma_slope,
  "sigma_slope (RE slope SD)"
)
print_result(results$sigma_slope)

# Compare correlation parameter
results$rho <- compare_posteriors(
  nd_L21,  # L21 is the correlation for 2x2 case
  draws_stan$rho,
  "rho (correlation)"
)
print_result(results$rho)

# Compare raw beta (without RE adjustment) for reference
cat("\n--- Raw Beta Comparisons (for reference) ---\n")
cat(sprintf("Raw beta_num[1]: nd=%.4f stan=%.4f\n",
            mean(draws_nd[, "beta_num[1]"]), mean(draws_stan$`beta_num[1]`)))
cat(sprintf("Raw beta_denom[1]: nd=%.4f stan=%.4f\n",
            mean(draws_nd[, "beta_denom[1]"]), mean(draws_stan$`beta_denom[1]`)))
cat(sprintf("Mean RE intercept (transformed): nd=%.4f stan=%.4f\n",
            mean(nd_mean_re_int), mean(stan_mean_re_int)))

cat(sprintf("\n  numdenom time: %.1fs\n", t_nd))
cat(sprintf("  Stan time: %.1fs\n", t_stan))
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Summary
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - ROW 33 (negbin_negbin + random slopes)\n")
cat("=======================================================\n")

n_pass <- sum(sapply(results, function(r) r$pass))
n_total <- length(results)

for (name in names(results)) {
  cat(sprintf("%s: %s (%.2f SE)\n", name,
              if(results[[name]]$pass) "PASS" else "FAIL",
              results[[name]]$ratio))
}

cat(sprintf("\nOverall: %d/%d parameters PASS (within 2 SE)\n", n_pass, n_total))

if (n_pass == n_total) {
  cat("\n** All posteriors match custom joint Stan model! **\n")
  cat("Row 33 validated: negbin_negbin + random slopes\n")
} else if (n_pass >= n_total - 2 && all(sapply(results, function(r) r$ratio < 4))) {
  cat("\n** Most posteriors match (marginal differences <4 SE) **\n")
  cat("Row 33 validated with asterisk: negbin_negbin + random slopes\n")
} else {
  cat("\n!! Some posteriors differ significantly - investigate !!\n")
}

# Save results
saveRDS(results, "results_joint_row33.rds")
cat("\nResults saved to results_joint_row33.rds\n")
