# Debug script for lognormal - compare to gamma_gamma
# Lognormal should be simpler (normal on log scale)

library(numdenom)
library(cmdstanr)

set.seed(42)

# Generate minimal data
N <- 100
x <- rnorm(N)

# True parameters for lognormal
mu_num <- c(2.0, 0.3)  # mean on log scale (intercept, slope)
mu_denom <- c(4.0, 0.2)
sigma_num <- 0.5
sigma_denom <- 0.3

# Generate responses - lognormal means log(Y) ~ Normal(mu, sigma^2)
eta_num <- mu_num[1] + mu_num[2] * x
eta_denom <- mu_denom[1] + mu_denom[2] * x

y_num <- exp(rnorm(N, eta_num, sigma_num))
y_denom <- exp(rnorm(N, eta_denom, sigma_denom))

dat <- data.frame(y_num = y_num, y_denom = y_denom, x = x)

cat("True values:\n")
cat("  beta_num: ", mu_num, "\n")
cat("  beta_denom: ", mu_denom, "\n")
cat("  sigma_num: ", sigma_num, "\n")
cat("  sigma_denom: ", sigma_denom, "\n\n")

# ============================================
# Fit with numdenom
# ============================================
cat("=== Fitting numdenom (lognormal) ===\n")
nd_fit <- ratiod(
  y_num | y_denom ~ x,
  data = dat,
  family = ratiod_lognormal(),
  iter = 2000,
  warmup = 1000,
  chains = 2,
  seed = 123
)

print(summary(nd_fit))

# ============================================
# Fit with Stan
# ============================================
cat("\n=== Fitting Stan ===\n")

stan_code <- "
data {
  int<lower=1> N;
  vector<lower=0>[N] y_num;
  vector<lower=0>[N] y_denom;
  int<lower=1> p;
  matrix[N, p] X;
}

parameters {
  vector[p] beta_num;
  vector[p] beta_denom;
  real<lower=0> sigma_num;
  real<lower=0> sigma_denom;
}

model {
  // Priors (matching numdenom defaults)
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  sigma_num ~ gamma(2, 0.1);
  sigma_denom ~ gamma(2, 0.1);

  // Likelihoods - log(y) ~ Normal(mu, sigma)
  for (n in 1:N) {
    if (y_num[n] > 0) {
      target += lognormal_lpdf(y_num[n] | X[n] * beta_num, sigma_num);
    }
    if (y_denom[n] > 0) {
      target += lognormal_lpdf(y_denom[n] | X[n] * beta_denom, sigma_denom);
    }
  }
}
"

stan_file <- tempfile(fileext = ".stan")
writeLines(stan_code, stan_file)
mod <- cmdstan_model(stan_file)

X <- model.matrix(~ x, data = dat)

stan_data <- list(
  N = N,
  y_num = dat$y_num,
  y_denom = dat$y_denom,
  p = ncol(X),
  X = X
)

stan_fit <- mod$sample(
  data = stan_data,
  iter_sampling = 500,
  iter_warmup = 500,
  chains = 2,
  seed = 123,
  show_messages = FALSE,
  refresh = 0
)

cat("Stan results:\n")
print(stan_fit$summary())

# ============================================
# Compare
# ============================================
cat("\n=== COMPARISON ===\n")
nd_draws <- as.matrix(nd_fit$draws)
stan_draws <- stan_fit$draws(format = "matrix")

cat("numdenom:\n")
cat("  beta_num[1]:", mean(nd_draws[,"beta_num[1]"]), "\n")
cat("  beta_num[2]:", mean(nd_draws[,"beta_num[2]"]), "\n")

cat("Stan:\n")
cat("  beta_num[1]:", mean(stan_draws[,"beta_num[1]"]), "\n")
cat("  beta_num[2]:", mean(stan_draws[,"beta_num[2]"]), "\n")

cat("\nTrue: beta_num[1] =", mu_num[1], ", beta_num[2] =", mu_num[2], "\n")
