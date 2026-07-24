# Precise gradient comparison for lognormal
# Use numerical differentiation as ground truth

library(numdenom)
set.seed(42)

# Very simple data - single observation to trace exactly
N_OBS <- 5
x <- rnorm(N_OBS)
y_num <- rlnorm(N_OBS, 2 + 0.3 * x, 0.5)
y_denom <- rlnorm(N_OBS, 4 + 0.2 * x, 0.5)

df <- data.frame(y = y_num, denom = y_denom, x = x)

cat("Data:\n")
print(df)

# Compute log-posterior manually
log_posterior <- function(params, y_num, y_denom, X) {
  # params = [beta_num_0, beta_num_1, beta_denom_0, beta_denom_1, log_sigma_num, log_sigma_denom]
  beta_num <- params[1:2]
  beta_denom <- params[3:4]
  log_sigma_num <- params[5]
  log_sigma_denom <- params[6]

  sigma_num <- exp(log_sigma_num)
  sigma_denom <- exp(log_sigma_denom)

  N <- length(y_num)

  # Linear predictors
  eta_num <- X %*% beta_num
  eta_denom <- X %*% beta_denom

  # Likelihood
  log_lik <- 0
  for (i in 1:N) {
    # Lognormal: log(y) ~ N(mu, sigma^2)
    log_y_num <- log(y_num[i])
    log_y_denom <- log(y_denom[i])

    z_num <- (log_y_num - eta_num[i]) / sigma_num
    z_denom <- (log_y_denom - eta_denom[i]) / sigma_denom

    log_lik <- log_lik - log(y_num[i]) - log(sigma_num) - 0.5 * z_num^2
    log_lik <- log_lik - log(y_denom[i]) - log(sigma_denom) - 0.5 * z_denom^2
  }

  # Priors
  # beta ~ N(0, 10^2)
  tau_beta <- 1 / (10^2)
  log_prior_beta <- -0.5 * tau_beta * sum(beta_num^2) - 0.5 * tau_beta * sum(beta_denom^2)

  # sigma ~ Gamma(2, 2) on log scale with Jacobian
  # (shape-1)*log(phi) - rate*phi + log(phi)
  shape <- 2
  rate <- 2
  log_prior_sigma <- (shape - 1) * log_sigma_num - rate * sigma_num + log_sigma_num
  log_prior_sigma <- log_prior_sigma + (shape - 1) * log_sigma_denom - rate * sigma_denom + log_sigma_denom

  log_lik + log_prior_beta + log_prior_sigma
}

# Numerical gradient
numerical_grad <- function(params, eps = 1e-5, ...) {
  n <- length(params)
  grad <- numeric(n)
  for (i in 1:n) {
    params_plus <- params
    params_minus <- params
    params_plus[i] <- params_plus[i] + eps
    params_minus[i] <- params_minus[i] - eps
    grad[i] <- (log_posterior(params_plus, ...) - log_posterior(params_minus, ...)) / (2 * eps)
  }
  grad
}

# Design matrix
X <- cbind(1, df$x)

# Test at a specific parameter point
params <- c(2.0, 0.3, 4.0, 0.2, log(0.5), log(0.5))
names(params) <- c("beta_num[1]", "beta_num[2]", "beta_denom[1]", "beta_denom[2]", "log_sigma_num", "log_sigma_denom")

cat("\n=== Log-posterior at test point ===\n")
lp <- log_posterior(params, df$y, df$denom, X)
cat(sprintf("Log-posterior: %.6f\n", lp))

cat("\n=== Numerical gradient (ground truth) ===\n")
grad_num <- numerical_grad(params, y_num = df$y, y_denom = df$denom, X = X)
print(round(grad_num, 6))

# Now let's derive the analytical gradients manually and verify
cat("\n=== Analytical gradient (manual derivation) ===\n")

# For each observation:
# d(LL)/d(beta_num_j) = sum_i X[i,j] * (log(y_num[i]) - eta_num[i]) / sigma_num^2
# d(LL)/d(log_sigma_num) = sum_i [ (-1/sigma_num + z_num^2/sigma_num) * sigma_num ]
#                        = sum_i [ -1 + z_num^2 ]

sigma_num <- exp(params[5])
sigma_denom <- exp(params[6])
eta_num <- X %*% params[1:2]
eta_denom <- X %*% params[3:4]

grad_beta_num <- numeric(2)
grad_beta_denom <- numeric(2)
grad_log_sigma_num <- 0
grad_log_sigma_denom <- 0

for (i in 1:nrow(df)) {
  log_y_num <- log(df$y[i])
  log_y_denom <- log(df$denom[i])

  z_num <- (log_y_num - eta_num[i]) / sigma_num
  z_denom <- (log_y_denom - eta_denom[i]) / sigma_denom

  resid_num <- (log_y_num - eta_num[i]) / (sigma_num^2)
  resid_denom <- (log_y_denom - eta_denom[i]) / (sigma_denom^2)

  grad_beta_num <- grad_beta_num + X[i,] * resid_num
  grad_beta_denom <- grad_beta_denom + X[i,] * resid_denom

  # d(LL)/d(sigma) = (-1 + z^2) / sigma
  # d(LL)/d(log_sigma) = d(LL)/d(sigma) * sigma = -1 + z^2
  grad_log_sigma_num <- grad_log_sigma_num + (-1 + z_num^2)
  grad_log_sigma_denom <- grad_log_sigma_denom + (-1 + z_denom^2)
}

# Add priors
# beta prior: d/d(beta_j) = -tau_beta * beta_j
tau_beta <- 1 / (10^2)
grad_beta_num <- grad_beta_num - tau_beta * params[1:2]
grad_beta_denom <- grad_beta_denom - tau_beta * params[3:4]

# sigma prior: d/d(log_sigma) = (shape - 1) + 1 - rate * sigma = shape - rate * sigma
shape <- 2
rate <- 2
grad_log_sigma_num <- grad_log_sigma_num + shape - rate * sigma_num
grad_log_sigma_denom <- grad_log_sigma_denom + shape - rate * sigma_denom

grad_manual <- c(grad_beta_num, grad_beta_denom, grad_log_sigma_num, grad_log_sigma_denom)
names(grad_manual) <- names(params)

cat("Manual analytical gradient:\n")
print(round(grad_manual, 6))

cat("\n=== Comparison ===\n")
diff <- abs(grad_manual - grad_num)
cat("Max difference (should be ~0): ", max(diff), "\n")
print(data.frame(
  param = names(params),
  numerical = round(grad_num, 6),
  analytical = round(grad_manual, 6),
  diff = round(diff, 10)
))

# The analytical gradient formula I derived:
# For log_sigma: sum_i(-1 + z_i^2) + shape - rate * sigma
#
# Now check what H-mode does:
# H-mode (line 2521-2523):
#   grad_phi_num_i = (-1.0 + z_num * z_num) / sigma_num;
# Then (line 2726):
#   grad[log_phi_num_idx] += phi_num * grad_phi_num_lik;
#
# So H-mode computes:
#   sum_i ((-1 + z^2) / sigma) * sigma = sum_i (-1 + z^2)  ✓
#
# Plus prior (line 1804-1805):
#   grad = (shape - 1) - rate * phi + 1 = shape - rate * phi  ✓
#
# Wait... but H-mode ADDS prior FIRST, then ADDS likelihood contribution.
# Let me check the order...

cat("\n=== Checking H-mode order ===\n")
cat("H-mode code flow:\n")
cat("1. Prior gradient initialized: grad[log_phi_idx] = (shape-1) - rate*phi + 1\n")
cat("2. For each obs, accumulate: grad_phi_lik += (-1 + z^2) / sigma\n")
cat("3. After loop: grad[log_phi_idx] += phi * grad_phi_lik\n")
cat("\nFinal: (shape - rate*phi) + sum_i(-1 + z^2) = analytical formula\n")
cat("This looks correct!\n")

# The issue might be somewhere else. Let me check if the data is being read correctly...
cat("\n=== Testing with numdenom package ===\n")

# Fit with A-mode (should be correct)
fit_A <- tratio(
  y | denom ~ x, data = df,
  family = ratiod_lognormal(),
  control = list(iter = 10, warmup = 5, chains = 1, gradient_mode = "A", verbose = FALSE)
)

# Fit with H-mode
fit_H <- tratio(
  y | denom ~ x, data = df,
  family = ratiod_lognormal(),
  control = list(iter = 10, warmup = 5, chains = 1, gradient_mode = "H", verbose = FALSE)
)

draws_A <- as.matrix(fit_A$draws)
draws_H <- as.matrix(fit_H$draws)

cat("\nA-mode final params:\n")
print(draws_A[nrow(draws_A), ])
cat("\nH-mode final params:\n")
print(draws_H[nrow(draws_H), ])

cat("\nFirst row (should be same initialization):\n")
cat("A-mode:", draws_A[1, ], "\n")
cat("H-mode:", draws_H[1, ], "\n")
