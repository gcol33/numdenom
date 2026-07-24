# Debug: Compare numdenom vs Stan for negbin_negbin with random SLOPES
# With proper handling of RE matrix structure

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

N_OBS <- 200
N_ITER <- 4000
N_WARMUP <- 1500
N_CHAINS <- 4
N_SITES <- 20

cat("=== DEBUG: NegBin-NegBin WITH RE slopes ===\n\n")

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

# Fit numdenom
cat("Fitting numdenom... ")
fit_nd <- tratio(
  y_num | y_denom ~ x + (1+x|site),
  data = df,
  family = ratiod_negbin_negbin(),
  control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
)
cat("done\n")

# Print parameter names in numdenom
nd_cols <- colnames(as.matrix(fit_nd$draws))
cat("\nnumdenom parameter names:\n")
print(head(nd_cols, 30))

# Fit Stan model
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
  matrix[n_groups, 2] re;
}
transformed parameters {
  matrix[2, 2] L_Omega;
  L_Omega[1, 1] = 1.0;
  L_Omega[1, 2] = 0.0;
  L_Omega[2, 1] = L21;
  L_Omega[2, 2] = sqrt(1.0 - L21 * L21);
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

  // MVN prior on RE (centered)
  for (g in 1:n_groups) {
    real y1 = re[g, 1] / sigma_intercept;
    real y2 = re[g, 2] / sigma_slope;
    real L22 = L_Omega[2, 2];
    real z1 = y1;
    real z2 = (y2 - L21 * z1) / L22;
    target += -0.5 * (z1 * z1 + z2 * z2);
  }
  target += -n_groups * (log(sigma_intercept) + log(sigma_slope) + log(L_Omega[2, 2]));

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
}
"

writeLines(stan_code, "stan/joint_nb_slopes_gq.stan")
stan_mod <- cmdstan_model("stan/joint_nb_slopes_gq.stan")

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
cat("done\n\n")

# Print Stan parameter names
stan_vars <- fit_stan$metadata()$stan_variables
cat("Stan parameter names:\n")
print(stan_vars)

# Extract draws
draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

# Find RE columns in numdenom
nd_cols <- colnames(draws_nd)
re_int_cols <- nd_cols[grep("^re\\[.*,1\\]$|^re_int\\[", nd_cols)]
re_slope_cols <- nd_cols[grep("^re\\[.*,2\\]$|^re_slope\\[", nd_cols)]

cat("\nnumdenom RE intercept columns:", paste(re_int_cols[1:min(3, length(re_int_cols))], collapse=", "), "...\n")
cat("numdenom RE slope columns:", paste(re_slope_cols[1:min(3, length(re_slope_cols))], collapse=", "), "...\n")

# Compute means
if (length(re_int_cols) > 0) {
  nd_mean_re_int <- rowMeans(draws_nd[, re_int_cols, drop=FALSE])
} else {
  # Try alternate naming
  cat("Trying alternate RE naming...\n")
  re_all_cols <- nd_cols[grep("^re\\[", nd_cols)]
  cat("All RE columns:", paste(re_all_cols[1:min(10, length(re_all_cols))], collapse=", "), "...\n")
  # If RE are stored as re[1]...re[40] for 20 groups with 2 coefficients
  # Then re[1], re[3], re[5],... are intercepts; re[2], re[4], re[6],... are slopes
  n_coefs <- 2
  re_int_idx <- seq(1, length(re_all_cols), by = n_coefs)
  re_slope_idx <- seq(2, length(re_all_cols), by = n_coefs)
  re_int_cols <- re_all_cols[re_int_idx]
  re_slope_cols <- re_all_cols[re_slope_idx]
  nd_mean_re_int <- rowMeans(draws_nd[, re_int_cols, drop=FALSE])
  nd_mean_re_slope <- rowMeans(draws_nd[, re_slope_cols, drop=FALSE])
  cat("RE intercept mean: mean=", mean(nd_mean_re_int), "\n")
  cat("RE slope mean: mean=", mean(nd_mean_re_slope), "\n")
}

cat("\n=== Effective intercept comparison ===\n")
cat(sprintf("effective_intercept_num:\n"))
cat(sprintf("  numdenom: %.4f (SD=%.4f)\n",
            mean(draws_nd[, "beta_num[1]"] + nd_mean_re_int),
            sd(draws_nd[, "beta_num[1]"] + nd_mean_re_int)))
cat(sprintf("  Stan:     %.4f (SD=%.4f)\n",
            mean(draws_stan$eff_int_num), sd(draws_stan$eff_int_num)))

se_eff <- sqrt(sd(draws_nd[, "beta_num[1]"] + nd_mean_re_int)^2/nrow(draws_nd) +
               sd(draws_stan$eff_int_num)^2/nrow(draws_stan))
diff_se <- abs(mean(draws_nd[, "beta_num[1]"] + nd_mean_re_int) - mean(draws_stan$eff_int_num)) / se_eff
cat(sprintf("  Diff: %.2f SE => %s\n", diff_se, if(diff_se < 2) "PASS" else "FAIL"))

cat(sprintf("\neffective_slope_num:\n"))
cat(sprintf("  numdenom: %.4f (SD=%.4f)\n",
            mean(draws_nd[, "beta_num[2]"] + nd_mean_re_slope),
            sd(draws_nd[, "beta_num[2]"] + nd_mean_re_slope)))
cat(sprintf("  Stan:     %.4f (SD=%.4f)\n",
            mean(draws_stan$eff_slope_num), sd(draws_stan$eff_slope_num)))

se_eff_slope <- sqrt(sd(draws_nd[, "beta_num[2]"] + nd_mean_re_slope)^2/nrow(draws_nd) +
                     sd(draws_stan$eff_slope_num)^2/nrow(draws_stan))
diff_se_slope <- abs(mean(draws_nd[, "beta_num[2]"] + nd_mean_re_slope) - mean(draws_stan$eff_slope_num)) / se_eff_slope
cat(sprintf("  Diff: %.2f SE => %s\n", diff_se_slope, if(diff_se_slope < 2) "PASS" else "FAIL"))

cat("\n=== Raw comparisons ===\n")
cat(sprintf("beta_num[1]: nd=%.4f, stan=%.4f\n", mean(draws_nd[, "beta_num[1]"]), mean(draws_stan$`beta_num[1]`)))
cat(sprintf("beta_num[2]: nd=%.4f, stan=%.4f\n", mean(draws_nd[, "beta_num[2]"]), mean(draws_stan$`beta_num[2]`)))
cat(sprintf("phi_num: nd=%.4f, stan=%.4f\n", mean(draws_nd[, "phi_num"]), mean(draws_stan$phi_num)))
cat(sprintf("phi_denom: nd=%.4f, stan=%.4f\n", mean(draws_nd[, "phi_denom"]), mean(draws_stan$phi_denom)))
