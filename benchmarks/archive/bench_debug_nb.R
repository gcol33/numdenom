# Debug: Compare numdenom vs Stan for negbin_negbin WITHOUT random effects
# This isolates whether the base NB-NB model matches

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

N_OBS <- 200
N_ITER <- 4000
N_WARMUP <- 1500
N_CHAINS <- 4

cat("=== DEBUG: NegBin-NegBin WITHOUT RE ===\n\n")

# Generate simple data
x <- rnorm(N_OBS)
true_beta_num <- c(2, 0.5)
true_beta_denom <- c(3, 0.2)
true_phi_num <- 5.0
true_phi_denom <- 8.0

eta_num <- true_beta_num[1] + true_beta_num[2] * x
eta_denom <- true_beta_denom[1] + true_beta_denom[2] * x

y_num <- rnbinom(N_OBS, size = true_phi_num, mu = exp(eta_num))
y_denom <- rnbinom(N_OBS, size = true_phi_denom, mu = exp(eta_denom))
y_denom[y_denom == 0] <- 1

df <- data.frame(y_num = y_num, y_denom = y_denom, x = x)

# Fit numdenom
cat("Fitting numdenom... ")
fit_nd <- tratio(
  y_num | y_denom ~ x,
  data = df,
  family = ratiod_negbin_negbin(),
  control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
)
cat("done\n")

# Simple Stan model
stan_code <- "
data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;
  array[N] int<lower=0> y_denom;
  matrix[N, 2] X;
}
parameters {
  vector[2] beta_num;
  vector[2] beta_denom;
  real<lower=0> phi_num;
  real<lower=0> phi_denom;
}
model {
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  phi_num ~ gamma(2, 0.1);
  phi_denom ~ gamma(2, 0.1);

  y_num ~ neg_binomial_2_log(X * beta_num, phi_num);
  y_denom ~ neg_binomial_2_log(X * beta_denom, phi_denom);
}
"

writeLines(stan_code, "stan/simple_nb.stan")
stan_mod <- cmdstan_model("stan/simple_nb.stan")

stan_data <- list(
  N = N_OBS,
  y_num = df$y_num,
  y_denom = df$y_denom,
  X = cbind(1, df$x)
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
params <- c("beta_num[1]", "beta_num[2]", "beta_denom[1]", "beta_denom[2]")

for (p in params) {
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

# Also compare phi
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
