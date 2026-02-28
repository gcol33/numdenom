# Debug: Compare numdenom vs Stan for negbin_negbin WITH random INTERCEPTS only
# This isolates whether random intercepts work correctly

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

N_OBS <- 200
N_ITER <- 4000
N_WARMUP <- 1500
N_CHAINS <- 4
N_SITES <- 20

cat("=== DEBUG: NegBin-NegBin WITH RE (intercepts only) ===\n\n")

# Generate data with RE
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)

true_beta_num <- c(2, 0.5)
true_beta_denom <- c(3, 0.2)
true_phi_num <- 5.0
true_phi_denom <- 8.0
true_sigma_re <- 0.4

# Generate RE (SHARED between num and denom)
true_re <- rnorm(N_SITES, 0, true_sigma_re)

eta_num <- true_beta_num[1] + true_beta_num[2] * x
eta_denom <- true_beta_denom[1] + true_beta_denom[2] * x
for (i in 1:N_OBS) {
  g <- as.integer(site[i])
  eta_num[i] <- eta_num[i] + true_re[g]
  eta_denom[i] <- eta_denom[i] + true_re[g]  # SHARED
}

y_num <- rnbinom(N_OBS, size = true_phi_num, mu = exp(eta_num))
y_denom <- rnbinom(N_OBS, size = true_phi_denom, mu = exp(eta_denom))
y_denom[y_denom == 0] <- 1

df <- data.frame(y_num = y_num, y_denom = y_denom, x = x, site = site)

cat(sprintf("True: beta_num=[%.2f,%.2f], beta_denom=[%.2f,%.2f], sigma_re=%.2f\n",
            true_beta_num[1], true_beta_num[2], true_beta_denom[1], true_beta_denom[2], true_sigma_re))

# Fit numdenom
cat("Fitting numdenom... ")
fit_nd <- ratiod(
  y_num | y_denom ~ x + (1|site),
  data = df,
  family = ratiod_negbin_negbin(),
  iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
  verbose = FALSE
)
cat("done\n")

# Stan model with SHARED RE (centered)
stan_code <- "
data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;
  array[N] int<lower=0> y_denom;
  matrix[N, 2] X;
  int<lower=1> n_groups;
  array[N] int<lower=1,upper=n_groups> group_idx;
}
parameters {
  vector[2] beta_num;
  vector[2] beta_denom;
  real<lower=0> phi_num;
  real<lower=0> phi_denom;
  real<lower=0> sigma_re;
  vector[n_groups] re;  // SHARED RE (centered parameterization)
}
model {
  vector[N] eta_num;
  vector[N] eta_denom;

  // Priors
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  phi_num ~ gamma(2, 0.1);
  phi_denom ~ gamma(2, 0.1);
  sigma_re ~ cauchy(0, 2.5);

  // RE prior (centered)
  re ~ normal(0, sigma_re);

  // Linear predictors with SHARED RE
  eta_num = X * beta_num;
  eta_denom = X * beta_denom;
  for (n in 1:N) {
    eta_num[n] += re[group_idx[n]];
    eta_denom[n] += re[group_idx[n]];
  }

  // Likelihood
  y_num ~ neg_binomial_2_log(eta_num, phi_num);
  y_denom ~ neg_binomial_2_log(eta_denom, phi_denom);
}
"

writeLines(stan_code, "stan/joint_nb_intercept.stan")
stan_mod <- cmdstan_model("stan/joint_nb_intercept.stan")

stan_data <- list(
  N = N_OBS,
  y_num = df$y_num,
  y_denom = df$y_denom,
  X = cbind(1, df$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df$site)
)

cat("Fitting Stan... ")
fit_stan <- stan_mod$sample(
  data = stan_data,
  iter_sampling = N_ITER - N_WARMUP,
  iter_warmup = N_WARMUP,
  chains = N_CHAINS,
  parallel_chains = N_CHAINS,
  refresh = 0,
  show_messages = FALSE
)
cat("done\n\n")

# Compare
draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

cat("=== Comparison ===\n")

for (p in c("beta_num[1]", "beta_num[2]", "beta_denom[1]", "beta_denom[2]")) {
  nd_mean <- mean(draws_nd[, p])
  nd_sd <- sd(draws_nd[, p])
  stan_mean <- mean(draws_stan[[p]])
  stan_sd <- sd(draws_stan[[p]])

  se <- sqrt(nd_sd^2/nrow(draws_nd) + stan_sd^2/nrow(draws_stan))
  diff_se <- abs(nd_mean - stan_mean) / se

  cat(sprintf("%s:\n", p))
  cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", nd_mean, nd_sd))
  cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", stan_mean, stan_sd))
  cat(sprintf("  Diff: %.2f SE => %s\n\n", diff_se, if(diff_se < 2) "PASS" else "FAIL"))
}

# phi
for (p in c("phi_num", "phi_denom")) {
  nd_mean <- mean(draws_nd[, p])
  nd_sd <- sd(draws_nd[, p])
  stan_mean <- mean(draws_stan[[p]])
  stan_sd <- sd(draws_stan[[p]])

  se <- sqrt(nd_sd^2/nrow(draws_nd) + stan_sd^2/nrow(draws_stan))
  diff_se <- abs(nd_mean - stan_mean) / se

  cat(sprintf("%s:\n", p))
  cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", nd_mean, nd_sd))
  cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", stan_mean, stan_sd))
  cat(sprintf("  Diff: %.2f SE => %s\n\n", diff_se, if(diff_se < 2) "PASS" else "FAIL"))
}

# sigma_re - need to find the right column name in numdenom
nd_cols <- colnames(draws_nd)
sigma_cols <- nd_cols[grep("sigma|sd_", nd_cols, ignore.case=TRUE)]
cat("numdenom sigma columns:", paste(sigma_cols, collapse=", "), "\n")

if (length(sigma_cols) > 0) {
  nd_mean <- mean(draws_nd[, sigma_cols[1]])
  nd_sd <- sd(draws_nd[, sigma_cols[1]])
  stan_mean <- mean(draws_stan$sigma_re)
  stan_sd <- sd(draws_stan$sigma_re)

  se <- sqrt(nd_sd^2/nrow(draws_nd) + stan_sd^2/nrow(draws_stan))
  diff_se <- abs(nd_mean - stan_mean) / se

  cat(sprintf("\nsigma_re:\n"))
  cat(sprintf("  numdenom (%s): %.4f (SD=%.4f)\n", sigma_cols[1], nd_mean, nd_sd))
  cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", stan_mean, stan_sd))
  cat(sprintf("  Diff: %.2f SE => %s\n", diff_se, if(diff_se < 2) "PASS" else "FAIL"))
}
