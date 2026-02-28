# Final validation: non-centered numdenom vs Stan
# Using longer chains and visual comparison

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

N_OBS <- 200
N_ITER <- 4000
N_WARMUP <- 1000
N_CHAINS <- 4
N_SITES <- 20

cat("=== Final validation: NON-CENTERED parameterization ===\n\n")

# Generate data
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)

true_beta_num <- c(2, 0.5)
true_beta_denom <- c(3, 0.2)
true_phi_num <- 5.0
true_phi_denom <- 8.0
true_sigma_int <- 0.4
true_sigma_slope <- 0.2
true_rho <- 0.3

# Generate correlated RE
L <- matrix(c(1, 0, true_rho, sqrt(1-true_rho^2)), 2, 2)
true_z <- matrix(rnorm(N_SITES * 2), N_SITES, 2)
true_re <- matrix(0, N_SITES, 2)
for (g in 1:N_SITES) {
  Lz <- L %*% true_z[g, ]
  true_re[g, ] <- c(true_sigma_int, true_sigma_slope) * Lz
}

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

# ============ Fit numdenom ============
cat("\nFitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y_num | y_denom ~ x + (1+x|site),
    data = df,
    family = ratiod_negbin_negbin(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})
cat(sprintf("done (%.1fs)\n", t_nd["elapsed"]))

# ============ Fit Stan ============
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
  real<lower=-1, upper=1> L21;
  matrix[n_groups, 2] z;
}
transformed parameters {
  matrix[2, 2] L_Omega;
  matrix[n_groups, 2] re;
  L_Omega[1, 1] = 1.0;
  L_Omega[1, 2] = 0.0;
  L_Omega[2, 1] = L21;
  L_Omega[2, 2] = sqrt(1.0 - L21 * L21);
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
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  phi_num ~ gamma(2, 0.1);
  phi_denom ~ gamma(2, 0.1);
  sigma_intercept ~ cauchy(0, sigma_re_scale);
  sigma_slope ~ cauchy(0, sigma_re_scale);
  target += 1.5 * log(1 - L21 * L21);
  to_vector(z) ~ std_normal();
  y_num ~ neg_binomial_2_log(eta_num, phi_num);
  y_denom ~ neg_binomial_2_log(eta_denom, phi_denom);
}
generated quantities {
  real mean_re_int = mean(re[,1]);
  real mean_re_slope = mean(re[,2]);
}
"

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

cat("Fitting Stan... ")
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
cat(sprintf("done (%.1fs)\n\n", t_stan["elapsed"]))

# ============ Extract draws ============
draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")
nd_cols <- colnames(draws_nd)

# Compute mean RE for numdenom
re_int_cols <- nd_cols[grepl('intercept]$', nd_cols) & startsWith(nd_cols, 're[')]
re_slope_cols <- nd_cols[grepl('x]$', nd_cols) & startsWith(nd_cols, 're[')]

nd_mean_re_int <- if(length(re_int_cols) > 0) {
  rowMeans(draws_nd[, re_int_cols, drop=FALSE])
} else {
  rep(0, nrow(draws_nd))
}
nd_mean_re_slope <- if(length(re_slope_cols) > 0) {
  rowMeans(draws_nd[, re_slope_cols, drop=FALSE])
} else {
  rep(0, nrow(draws_nd))
}

# ============ Summary comparison ============
cat("=== POSTERIOR SUMMARY COMPARISON ===\n\n")

# Function to summarize and compare
compare_summary <- function(name, nd_vals, stan_vals, true_val = NA) {
  nd_q <- quantile(nd_vals, c(0.025, 0.5, 0.975))
  stan_q <- quantile(stan_vals, c(0.025, 0.5, 0.975))

  cat(sprintf("%s:\n", name))
  cat(sprintf("  numdenom:  %.4f [%.4f, %.4f]\n", nd_q[2], nd_q[1], nd_q[3]))
  cat(sprintf("  Stan:      %.4f [%.4f, %.4f]\n", stan_q[2], stan_q[1], stan_q[3]))
  if (!is.na(true_val)) {
    cat(sprintf("  True:      %.4f\n", true_val))
  }

  # Check overlap of 95% CIs
  overlap <- (nd_q[1] <= stan_q[3]) && (stan_q[1] <= nd_q[3])
  cat(sprintf("  95%% CI overlap: %s\n\n", if(overlap) "YES" else "NO"))

  return(overlap)
}

results <- c()

# Effective intercepts
nd_eff_int_num <- draws_nd[, "beta_num[1]"] + nd_mean_re_int
stan_eff_int_num <- draws_stan$`beta_num[1]` + draws_stan$mean_re_int
results["eff_int_num"] <- compare_summary("Effective intercept (num)",
                                          nd_eff_int_num, stan_eff_int_num, true_beta_num[1])

nd_eff_int_denom <- draws_nd[, "beta_denom[1]"] + nd_mean_re_int
stan_eff_int_denom <- draws_stan$`beta_denom[1]` + draws_stan$mean_re_int
results["eff_int_denom"] <- compare_summary("Effective intercept (denom)",
                                            nd_eff_int_denom, stan_eff_int_denom, true_beta_denom[1])

# Effective slopes
nd_eff_slope_num <- draws_nd[, "beta_num[2]"] + nd_mean_re_slope
stan_eff_slope_num <- draws_stan$`beta_num[2]` + draws_stan$mean_re_slope
results["eff_slope_num"] <- compare_summary("Effective slope (num)",
                                            nd_eff_slope_num, stan_eff_slope_num, true_beta_num[2])

nd_eff_slope_denom <- draws_nd[, "beta_denom[2]"] + nd_mean_re_slope
stan_eff_slope_denom <- draws_stan$`beta_denom[2]` + draws_stan$mean_re_slope
results["eff_slope_denom"] <- compare_summary("Effective slope (denom)",
                                              nd_eff_slope_denom, stan_eff_slope_denom, true_beta_denom[2])

# Dispersion
results["phi_num"] <- compare_summary("phi_num", draws_nd[, "phi_num"], draws_stan$phi_num, true_phi_num)
results["phi_denom"] <- compare_summary("phi_denom", draws_nd[, "phi_denom"], draws_stan$phi_denom, true_phi_denom)

# Variance parameters
sigma_int_col <- nd_cols[grepl("sigma_re.*intercept", nd_cols)]
if (length(sigma_int_col) > 0) {
  results["sigma_int"] <- compare_summary("sigma_intercept",
                                          draws_nd[, sigma_int_col], draws_stan$sigma_intercept, true_sigma_int)
}

sigma_slope_col <- nd_cols[grepl("sigma_re.*x]$", nd_cols)]
if (length(sigma_slope_col) > 0) {
  results["sigma_slope"] <- compare_summary("sigma_slope",
                                            draws_nd[, sigma_slope_col], draws_stan$sigma_slope, true_sigma_slope)
}

# Correlation
chol_col <- nd_cols[grepl("L_chol", nd_cols)]
if (length(chol_col) > 0) {
  results["correlation"] <- compare_summary("L21 (correlation)",
                                            draws_nd[, chol_col], draws_stan$L21, true_rho)
}

# ============ Summary ============
cat("========================================\n")
cat("              SUMMARY\n")
cat("========================================\n")
n_overlap <- sum(results)
n_total <- length(results)
cat(sprintf("Parameters with overlapping 95%% CIs: %d/%d\n", n_overlap, n_total))
cat(sprintf("Timing: numdenom=%.1fs, Stan=%.1fs\n", t_nd["elapsed"], t_stan["elapsed"]))

if (n_overlap == n_total) {
  cat("\nSUCCESS: All posterior credible intervals overlap!\n")
  cat("The non-centered implementation is validated.\n")
} else {
  cat("\nWARNING: Some credible intervals do not overlap.\n")
  cat("Failed parameters:", paste(names(results)[!results], collapse=", "), "\n")
}
