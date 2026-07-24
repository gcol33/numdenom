# Diagnose gamma_gamma sampler issues

library(numdenom)

set.seed(123)

# Simple intercept-only test
N <- 200

# True parameters
beta_num <- 2.0
beta_denom <- 3.0
shape_num <- 4
shape_denom <- 6

# Generate data
mu_num <- exp(beta_num)
mu_denom <- exp(beta_denom)

y_num <- rgamma(N, shape = shape_num, rate = shape_num / mu_num)
y_denom <- rgamma(N, shape = shape_denom, rate = shape_denom / mu_denom)

df <- data.frame(y = y_num, denom = y_denom)

cat("=== True parameters ===\n")
cat("beta_num =", beta_num, "\n")
cat("beta_denom =", beta_denom, "\n")
cat("shape_num =", shape_num, "\n")
cat("shape_denom =", shape_denom, "\n")

# Try with Stan for comparison
cat("\n=== Fitting Stan model for comparison ===\n")
library(cmdstanr)

stan_code <- "
data {
  int<lower=1> N;
  vector<lower=0>[N] y_num;
  vector<lower=0>[N] y_denom;
}

parameters {
  real beta_num;
  real beta_denom;
  real<lower=0> shape_num;
  real<lower=0> shape_denom;
}

model {
  vector[N] mu_num = exp(rep_vector(beta_num, N));
  vector[N] mu_denom = exp(rep_vector(beta_denom, N));

  // Priors
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  shape_num ~ gamma(2, 0.5);
  shape_denom ~ gamma(2, 0.5);

  // Likelihoods
  y_num ~ gamma(shape_num, shape_num ./ mu_num);
  y_denom ~ gamma(shape_denom, shape_denom ./ mu_denom);
}
"

writeLines(stan_code, "benchmarks/debug_gg_intercept.stan")
stan_model <- cmdstan_model("benchmarks/debug_gg_intercept.stan")

stan_data <- list(
  N = N,
  y_num = y_num,
  y_denom = y_denom
)

fit_stan <- stan_model$sample(
  data = stan_data,
  iter_sampling = 1000,
  iter_warmup = 1000,
  chains = 2,
  parallel_chains = 2,
  refresh = 200
)

draws_stan <- fit_stan$draws(format = "df")

cat("\n=== Stan estimates ===\n")
cat("beta_num:    mean =", mean(draws_stan$beta_num), ", true =", beta_num, "\n")
cat("beta_denom:  mean =", mean(draws_stan$beta_denom), ", true =", beta_denom, "\n")
cat("shape_num:   mean =", mean(draws_stan$shape_num), ", true =", shape_num, "\n")
cat("shape_denom: mean =", mean(draws_stan$shape_denom), ", true =", shape_denom, "\n")

# Now try numdenom
cat("\n=== Fitting numdenom ===\n")
fit <- tratio(
  y | denom ~ 1,
  data = df,
  family = ratiod_gamma_gamma(),
  control = list(iter = 2000, warmup = 1000, chains = 2, gradient_mode = "H", seed = 42)
)

draws <- as.matrix(fit$draws)

cat("\n=== Numdenom estimates ===\n")
cat("beta_num:    mean =", mean(draws[,"beta_num[1]"]), ", true =", beta_num, "\n")
cat("beta_denom:  mean =", mean(draws[,"beta_denom[1]"]), ", true =", beta_denom, "\n")
cat("shape_num:   mean =", mean(draws[,"shape_num"]), ", true =", shape_num, "\n")
cat("shape_denom: mean =", mean(draws[,"shape_denom"]), ", true =", shape_denom, "\n")

# Compare standard deviations
cat("\n=== Standard deviations ===\n")
cat("Stan:\n")
cat("  beta_num sd:", sd(draws_stan$beta_num), "\n")
cat("  beta_denom sd:", sd(draws_stan$beta_denom), "\n")
cat("  shape_num sd:", sd(draws_stan$shape_num), "\n")
cat("  shape_denom sd:", sd(draws_stan$shape_denom), "\n")
cat("Numdenom:\n")
cat("  beta_num sd:", sd(draws[,"beta_num[1]"]), "\n")
cat("  beta_denom sd:", sd(draws[,"beta_denom[1]"]), "\n")
cat("  shape_num sd:", sd(draws[,"shape_num"]), "\n")
cat("  shape_denom sd:", sd(draws[,"shape_denom"]), "\n")
