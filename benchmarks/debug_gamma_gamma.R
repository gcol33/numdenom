# Debug script for gamma_gamma validation
# Minimal test case to isolate the bug

library(numdenom)
library(cmdstanr)

set.seed(42)

# Generate minimal data
N <- 100
x <- rnorm(N)

# True parameters
beta_num <- c(2.0, 0.3)  # intercept, slope
beta_denom <- c(4.0, 0.2)  # intercept, slope
shape_num <- 2.0
shape_denom <- 3.0

# Generate responses
eta_num <- beta_num[1] + beta_num[2] * x
eta_denom <- beta_denom[1] + beta_denom[2] * x
mu_num <- exp(eta_num)
mu_denom <- exp(eta_denom)

y_num <- rgamma(N, shape = shape_num, rate = shape_num / mu_num)
y_denom <- rgamma(N, shape = shape_denom, rate = shape_denom / mu_denom)

dat <- data.frame(
  y_num = y_num,
  y_denom = y_denom,
  x = x
)

cat("Data summary:\n")
cat("y_num range:", range(y_num), "\n")
cat("y_denom range:", range(y_denom), "\n")
cat("Any zeros? y_num:", any(y_num == 0), "y_denom:", any(y_denom == 0), "\n")

# ============================================
# Fit with numdenom
# ============================================
cat("\n=== Fitting numdenom ===\n")
nd_fit <- ratiod(
  y_num | y_denom ~ x,
  data = dat,
  family = ratiod_gamma_gamma(),
  iter = 1000,
  warmup = 500,
  chains = 2,
  seed = 123
)

nd_summary <- summary(nd_fit)
print(nd_summary)

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
  real<lower=0> shape_num;
  real<lower=0> shape_denom;
}

model {
  vector[N] mu_num = exp(X * beta_num);
  vector[N] mu_denom = exp(X * beta_denom);

  // Priors (matching numdenom defaults)
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  shape_num ~ gamma(2, 0.1);
  shape_denom ~ gamma(2, 0.1);

  // Likelihoods
  // Gamma: shape-rate parameterization where rate = shape/mu
  for (n in 1:N) {
    if (y_num[n] > 0) {
      target += gamma_lpdf(y_num[n] | shape_num, shape_num / mu_num[n]);
    }
    if (y_denom[n] > 0) {
      target += gamma_lpdf(y_denom[n] | shape_denom, shape_denom / mu_denom[n]);
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

stan_summary <- stan_fit$summary()
print(stan_summary)

# ============================================
# Compare
# ============================================
cat("\n=== COMPARISON ===\n")
cat("True values:\n")
cat("  beta_num[1] (intercept):", beta_num[1], "\n")
cat("  beta_num[2] (slope):", beta_num[2], "\n")
cat("  beta_denom[1] (intercept):", beta_denom[1], "\n")
cat("  beta_denom[2] (slope):", beta_denom[2], "\n")
cat("  shape_num:", shape_num, "\n")
cat("  shape_denom:", shape_denom, "\n")

cat("\nnumdenom estimates:\n")
nd_draws <- as.matrix(nd_fit$draws)
cat("  beta_num[1]:", mean(nd_draws[,"beta_num[1]"]), "\n")
cat("  beta_num[2]:", mean(nd_draws[,"beta_num[2]"]), "\n")
cat("  beta_denom[1]:", mean(nd_draws[,"beta_denom[1]"]), "\n")
cat("  beta_denom[2]:", mean(nd_draws[,"beta_denom[2]"]), "\n")

cat("\nStan estimates:\n")
stan_draws <- stan_fit$draws(format = "matrix")
cat("  beta_num[1]:", mean(stan_draws[,"beta_num[1]"]), "\n")
cat("  beta_num[2]:", mean(stan_draws[,"beta_num[2]"]), "\n")
cat("  beta_denom[1]:", mean(stan_draws[,"beta_denom[1]"]), "\n")
cat("  beta_denom[2]:", mean(stan_draws[,"beta_denom[2]"]), "\n")
