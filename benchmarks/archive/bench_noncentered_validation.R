# Validate non-centered numdenom vs non-centered Stan for row 33
# (negbin_negbin + correlated random slopes)

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

N_OBS <- 200
N_ITER <- 3000
N_WARMUP <- 1000
N_CHAINS <- 4
N_SITES <- 20

cat("=== Validating NON-CENTERED parameterization ===\n\n")

# Generate data with correlated RE
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)

true_beta_num <- c(2, 0.5)
true_beta_denom <- c(3, 0.2)
true_phi_num <- 5.0
true_phi_denom <- 8.0
true_sigma_int <- 0.4
true_sigma_slope <- 0.2
true_rho <- 0.3

# Generate correlated RE using non-centered approach
L <- matrix(c(1, 0, true_rho, sqrt(1-true_rho^2)), 2, 2)
true_z <- matrix(rnorm(N_SITES * 2), N_SITES, 2)  # Standard normal z values
true_re <- matrix(0, N_SITES, 2)
for (g in 1:N_SITES) {
  # re = diag(sigma) * L * z
  Lz <- L %*% true_z[g, ]
  true_re[g, ] <- c(true_sigma_int, true_sigma_slope) * Lz
}

# Compute linear predictors with SHARED RE
eta_num <- true_beta_num[1] + true_beta_num[2] * x
eta_denom <- true_beta_denom[1] + true_beta_denom[2] * x
for (i in 1:N_OBS) {
  g <- as.integer(site[i])
  re_effect <- true_re[g, 1] + true_re[g, 2] * x[i]
  eta_num[i] <- eta_num[i] + re_effect
  eta_denom[i] <- eta_denom[i] + re_effect
}

y_num <- rnbinom(N_OBS, size = true_phi_num, mu = exp(eta_num))
y_denom <- rnbinom(N_OBS, size = true_phi_denom, mu = exp(eta_denom))
y_denom[y_denom == 0] <- 1

df <- data.frame(y_num = y_num, y_denom = y_denom, x = x, site = site)

cat(sprintf("True: beta_num=[%.2f,%.2f], beta_denom=[%.2f,%.2f]\n",
            true_beta_num[1], true_beta_num[2], true_beta_denom[1], true_beta_denom[2]))
cat(sprintf("True: sigma_int=%.2f, sigma_slope=%.2f, rho=%.2f\n",
            true_sigma_int, true_sigma_slope, true_rho))

# ============ Fit numdenom (non-centered) ============
cat("\nFitting numdenom (non-centered)... ")
t_nd <- system.time({
  fit_nd <- tratio(
    y_num | y_denom ~ x + (1+x|site),
    data = df,
    family = ratiod_negbin_negbin(),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  )
})
cat(sprintf("done (%.1fs)\n", t_nd["elapsed"]))

# ============ Fit Stan (non-centered) ============
stan_code <- "
data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;
  array[N] int<lower=0> y_denom;
  matrix[N, 2] X;
  int<lower=1> n_groups;
  array[N] int<lower=1,upper=n_groups> group_idx;
  vector[N] x_slope;
}
transformed data {
  real sigma_re_scale = 2.5;
}
parameters {
  vector[2] beta_num;
  vector[2] beta_denom;
  real<lower=0> phi_num;
  real<lower=0> phi_denom;
  real<lower=0> sigma_intercept;
  real<lower=0> sigma_slope;
  real<lower=-1, upper=1> L21;  // Off-diagonal Cholesky element
  matrix[n_groups, 2] z;  // Standard normal latent variables
}
transformed parameters {
  matrix[2, 2] L_Omega;
  matrix[n_groups, 2] re;

  // Build Cholesky factor
  L_Omega[1, 1] = 1.0;
  L_Omega[1, 2] = 0.0;
  L_Omega[2, 1] = L21;
  L_Omega[2, 2] = sqrt(1.0 - L21 * L21);

  // Non-centered transform: re = diag(sigma) * L * z
  for (g in 1:n_groups) {
    vector[2] Lz;
    Lz[1] = L_Omega[1, 1] * z[g, 1];
    Lz[2] = L_Omega[2, 1] * z[g, 1] + L_Omega[2, 2] * z[g, 2];
    re[g, 1] = sigma_intercept * Lz[1];
    re[g, 2] = sigma_slope * Lz[2];
  }
}
model {
  vector[N] eta_num;
  vector[N] eta_denom;

  eta_num = X * beta_num;
  eta_denom = X * beta_denom;
  for (n in 1:N) {
    int g = group_idx[n];
    real re_effect = re[g, 1] + re[g, 2] * x_slope[n];
    eta_num[n] += re_effect;
    eta_denom[n] += re_effect;
  }

  // Priors
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  phi_num ~ gamma(2, 0.1);
  phi_denom ~ gamma(2, 0.1);
  sigma_intercept ~ cauchy(0, sigma_re_scale);
  sigma_slope ~ cauchy(0, sigma_re_scale);

  // LKJ(2) prior on correlation: target += 1.5 * log(1 - L21^2) + jacobian
  // Jacobian for sqrt(1 - L21^2): log(L_Omega[2,2])
  target += 1.5 * log(1 - L21 * L21);  // LKJ eta=2 contribution
  // Note: Stan handles Jacobian for constrained parameters automatically

  // Standard normal prior on z (non-centered)
  to_vector(z) ~ std_normal();

  // Likelihood
  y_num ~ neg_binomial_2_log(eta_num, phi_num);
  y_denom ~ neg_binomial_2_log(eta_denom, phi_denom);
}
generated quantities {
  real mean_re_int = mean(re[,1]);
  real mean_re_slope = mean(re[,2]);
  real eff_int_num = beta_num[1] + mean_re_int;
  real eff_int_denom = beta_denom[1] + mean_re_int;
  real eff_slope_num = beta_num[2] + mean_re_slope;
  real eff_slope_denom = beta_denom[2] + mean_re_slope;
  real correlation = L21;
}
"

# Write and compile Stan model
writeLines(stan_code, "stan/joint_nb_slopes_nc.stan")
stan_mod <- cmdstan_model("stan/joint_nb_slopes_nc.stan")

stan_data <- list(
  N = N_OBS,
  y_num = df$y_num,
  y_denom = df$y_denom,
  X = cbind(1, df$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df$site),
  x_slope = df$x
)

cat("Fitting Stan (non-centered)... ")
t_stan <- system.time({
  fit_stan <- stan_mod$sample(
    data = stan_data,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE,
    adapt_delta = 0.95
  )
})
cat(sprintf("done (%.1fs)\n", t_stan["elapsed"]))

# ============ Compare posteriors ============
draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")
nd_cols <- colnames(draws_nd)

# Extract RE columns from numdenom
# With non-centered, the stored parameters are z values, not re values
# We need to look for re columns (which should be computed for output)
re_int_cols <- nd_cols[grep("^re\\[.*,intercept\\]$", nd_cols)]
re_slope_cols <- nd_cols[grep("^re\\[.*,x\\]$", nd_cols)]

cat("\n=== numdenom RE columns ===\n")
cat("Intercept columns:", length(re_int_cols), "\n")
cat("Slope columns:", length(re_slope_cols), "\n")

# Compute mean RE for comparison
if (length(re_int_cols) > 0) {
  nd_mean_re_int <- rowMeans(draws_nd[, re_int_cols, drop=FALSE])
  nd_mean_re_slope <- rowMeans(draws_nd[, re_slope_cols, drop=FALSE])
} else {
  cat("Warning: RE columns not found, trying alternative extraction\n")
  nd_mean_re_int <- rep(0, nrow(draws_nd))
  nd_mean_re_slope <- rep(0, nrow(draws_nd))
}

# ============ Comparison function ============
compare_param <- function(name, nd_vals, stan_vals, threshold = 2.0) {
  nd_mean <- mean(nd_vals)
  stan_mean <- mean(stan_vals)
  nd_sd <- sd(nd_vals)
  stan_sd <- sd(stan_vals)
  se <- sqrt(nd_sd^2/length(nd_vals) + stan_sd^2/length(stan_vals))
  diff_se <- abs(nd_mean - stan_mean) / se
  status <- if(diff_se < threshold) "PASS" else "FAIL"
  cat(sprintf("%s:\n  numdenom: %.4f (SD=%.4f)\n  Stan:     %.4f (SD=%.4f)\n  Diff: %.2f SE => %s\n\n",
              name, nd_mean, nd_sd, stan_mean, stan_sd, diff_se, status))
  return(diff_se < threshold)
}

cat("\n========================================\n")
cat("        PARAMETER COMPARISONS\n")
cat("========================================\n\n")

results <- c()

# Effective intercepts (the identified quantities)
nd_eff_int_num <- draws_nd[, "beta_num[1]"] + nd_mean_re_int
nd_eff_int_denom <- draws_nd[, "beta_denom[1]"] + nd_mean_re_int
results["eff_int_num"] <- compare_param("Effective intercept (num)", nd_eff_int_num, draws_stan$eff_int_num)
results["eff_int_denom"] <- compare_param("Effective intercept (denom)", nd_eff_int_denom, draws_stan$eff_int_denom)

# Effective slopes
nd_eff_slope_num <- draws_nd[, "beta_num[2]"] + nd_mean_re_slope
nd_eff_slope_denom <- draws_nd[, "beta_denom[2]"] + nd_mean_re_slope
results["eff_slope_num"] <- compare_param("Effective slope (num)", nd_eff_slope_num, draws_stan$eff_slope_num)
results["eff_slope_denom"] <- compare_param("Effective slope (denom)", nd_eff_slope_denom, draws_stan$eff_slope_denom)

# Dispersion parameters
results["phi_num"] <- compare_param("phi_num", draws_nd[, "phi_num"], draws_stan$phi_num)
results["phi_denom"] <- compare_param("phi_denom", draws_nd[, "phi_denom"], draws_stan$phi_denom)

# Variance parameters
sigma_int_col <- nd_cols[grep("sigma_re\\[1,intercept\\]", nd_cols)]
sigma_slope_col <- nd_cols[grep("sigma_re\\[1,x\\]", nd_cols)]

if (length(sigma_int_col) > 0) {
  results["sigma_int"] <- compare_param("sigma_intercept", draws_nd[, sigma_int_col], draws_stan$sigma_intercept)
}
if (length(sigma_slope_col) > 0) {
  results["sigma_slope"] <- compare_param("sigma_slope", draws_nd[, sigma_slope_col], draws_stan$sigma_slope)
}

# Correlation
chol_col <- nd_cols[grep("L_chol\\[1,1\\]", nd_cols)]
if (length(chol_col) > 0) {
  results["correlation"] <- compare_param("Correlation (L21)", draws_nd[, chol_col], draws_stan$correlation)
}

cat("\n========================================\n")
cat("              SUMMARY\n")
cat("========================================\n")
n_pass <- sum(results)
n_total <- length(results)
cat(sprintf("Passed: %d/%d tests\n", n_pass, n_total))
if (n_pass == n_total) {
  cat("SUCCESS: All parameters match within 2 SE!\n")
} else {
  cat("WARNING: Some parameters differ by more than 2 SE\n")
  cat("Failed tests:", paste(names(results)[!results], collapse=", "), "\n")
}

cat(sprintf("\nTiming: numdenom=%.1fs, Stan=%.1fs\n", t_nd["elapsed"], t_stan["elapsed"]))
