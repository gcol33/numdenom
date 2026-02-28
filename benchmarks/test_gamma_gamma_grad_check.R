# Direct gradient check for gamma_gamma
# This test compares gradients computed by different methods at the SAME parameter point

library(numdenom)

set.seed(42)

# Small test case
N <- 50
x <- rnorm(N)

# Generate gamma data
shape_num <- 3
shape_denom <- 5
eta_num <- 1.5 + 0.3 * x
eta_denom <- 2.0 + 0.2 * x
mu_num <- exp(eta_num)
mu_denom <- exp(eta_denom)

y_num <- rgamma(N, shape = shape_num, rate = shape_num / mu_num)
y_denom <- rgamma(N, shape = shape_denom, rate = shape_denom / mu_denom)

df <- data.frame(y = y_num, denom = y_denom, x = x)

# Prepare HMC data
form <- numdenom:::ratiod_formula(y | denom ~ x, data = df)
model_type <- "gamma_gamma"
hmc_data <- numdenom:::prepare_hmc_data(form, df, ratiod_gamma_gamma(), model_type)

cat("=== HMC Data Structure ===\n")
cat("y_num_cont first 5:", head(hmc_data$y_num_cont, 5), "\n")
cat("y_denom_cont first 5:", head(hmc_data$y_denom_cont, 5), "\n")
cat("X_num dim:", dim(hmc_data$X_num), "\n")
cat("X_denom dim:", dim(hmc_data$X_denom), "\n")

# Create a test point (where we'll evaluate gradients)
# Parameter layout for gamma_gamma with no RE:
# [beta_num[1], beta_num[2], beta_denom[1], beta_denom[2], log_shape_num, log_shape_denom]
test_params <- c(
  1.5, 0.3,    # beta_num (intercept, slope)
  2.0, 0.2,    # beta_denom (intercept, slope)
  log(3.0),    # log_shape_num
  log(5.0)     # log_shape_denom
)

cat("\n=== Test parameters ===\n")
cat("beta_num: (", test_params[1], ",", test_params[2], ")\n")
cat("beta_denom: (", test_params[3], ",", test_params[4], ")\n")
cat("log_shape_num:", test_params[5], "=> shape_num =", exp(test_params[5]), "\n")
cat("log_shape_denom:", test_params[6], "=> shape_denom =", exp(test_params[6]), "\n")

# Compute log-posterior and gradients manually
# For Gamma(y | shape, rate=shape/mu):
# LL = shape*log(shape/mu) + (shape-1)*log(y) - (shape/mu)*y - lgamma(shape)
# d(LL)/d(eta) = shape * (y/mu - 1)
# d(LL)/d(log_shape) = d(LL)/d(shape) * shape
#   where d(LL)/d(shape) = log(shape/mu) + 1 + log(y) - y/mu - digamma(shape)

beta_num <- test_params[1:2]
beta_denom <- test_params[3:4]
log_shape_num <- test_params[5]
log_shape_denom <- test_params[6]
shape_num <- exp(log_shape_num)
shape_denom <- exp(log_shape_denom)

# Compute linear predictors
eta_num_test <- hmc_data$X_num %*% beta_num
eta_denom_test <- hmc_data$X_denom %*% beta_denom
mu_num_test <- exp(eta_num_test)
mu_denom_test <- exp(eta_denom_test)

# Compute log-likelihood
ll <- 0
for (i in 1:N) {
  # Numerator
  rate_num <- shape_num / mu_num_test[i]
  ll <- ll + shape_num * log(rate_num) + (shape_num - 1) * log(y_num[i]) -
        rate_num * y_num[i] - lgamma(shape_num)
  # Denominator
  rate_denom <- shape_denom / mu_denom_test[i]
  ll <- ll + shape_denom * log(rate_denom) + (shape_denom - 1) * log(y_denom[i]) -
        rate_denom * y_denom[i] - lgamma(shape_denom)
}

# Add priors
# beta ~ N(0, 10^2)
sigma_beta <- 10
for (b in beta_num) ll <- ll - 0.5 * (b / sigma_beta)^2
for (b in beta_denom) ll <- ll - 0.5 * (b / sigma_beta)^2

# log_shape ~ Gamma(shape_prior, rate_prior) on shape, transformed
# Default prior is Gamma(2, 0.5)
shape_prior <- 2
rate_prior <- 0.5
ll <- ll + (shape_prior - 1) * log_shape_num - rate_prior * shape_num + log_shape_num  # Jacobian
ll <- ll + (shape_prior - 1) * log_shape_denom - rate_prior * shape_denom + log_shape_denom

cat("\n=== Manual log-posterior ===\n")
cat("Log-posterior:", ll, "\n")

# Compute gradients
grad_manual <- numeric(6)

# d(LL)/d(beta_num)
resid_num <- shape_num * (y_num / mu_num_test - 1)
grad_manual[1] <- sum(hmc_data$X_num[, 1] * resid_num) - beta_num[1] / sigma_beta^2
grad_manual[2] <- sum(hmc_data$X_num[, 2] * resid_num) - beta_num[2] / sigma_beta^2

# d(LL)/d(beta_denom)
resid_denom <- shape_denom * (y_denom / mu_denom_test - 1)
grad_manual[3] <- sum(hmc_data$X_denom[, 1] * resid_denom) - beta_denom[1] / sigma_beta^2
grad_manual[4] <- sum(hmc_data$X_denom[, 2] * resid_denom) - beta_denom[2] / sigma_beta^2

# d(LL)/d(log_shape_num)
grad_shape_num <- 0
for (i in 1:N) {
  rate_num <- shape_num / mu_num_test[i]
  grad_shape_num <- grad_shape_num + log(rate_num) + 1 + log(y_num[i]) -
                     y_num[i] / mu_num_test[i] - digamma(shape_num)
}
# Transform to log scale and add prior
grad_manual[5] <- grad_shape_num * shape_num + (shape_prior - 1) - rate_prior * shape_num + 1

# d(LL)/d(log_shape_denom)
grad_shape_denom <- 0
for (i in 1:N) {
  rate_denom <- shape_denom / mu_denom_test[i]
  grad_shape_denom <- grad_shape_denom + log(rate_denom) + 1 + log(y_denom[i]) -
                       y_denom[i] / mu_denom_test[i] - digamma(shape_denom)
}
grad_manual[6] <- grad_shape_denom * shape_denom + (shape_prior - 1) - rate_prior * shape_denom + 1

cat("\n=== Manual gradients ===\n")
cat("d(LP)/d(beta_num[1]):", grad_manual[1], "\n")
cat("d(LP)/d(beta_num[2]):", grad_manual[2], "\n")
cat("d(LP)/d(beta_denom[1]):", grad_manual[3], "\n")
cat("d(LP)/d(beta_denom[2]):", grad_manual[4], "\n")
cat("d(LP)/d(log_shape_num):", grad_manual[5], "\n")
cat("d(LP)/d(log_shape_denom):", grad_manual[6], "\n")

# Compute numerical gradient for verification
eps <- 1e-5
grad_numerical <- numeric(6)

compute_log_post <- function(params) {
  beta_num <- params[1:2]
  beta_denom <- params[3:4]
  log_shape_num <- params[5]
  log_shape_denom <- params[6]
  shape_num <- exp(log_shape_num)
  shape_denom <- exp(log_shape_denom)

  eta_num <- hmc_data$X_num %*% beta_num
  eta_denom <- hmc_data$X_denom %*% beta_denom
  mu_num <- exp(eta_num)
  mu_denom <- exp(eta_denom)

  ll <- 0
  for (i in 1:N) {
    rate_num <- shape_num / mu_num[i]
    ll <- ll + shape_num * log(rate_num) + (shape_num - 1) * log(y_num[i]) -
          rate_num * y_num[i] - lgamma(shape_num)
    rate_denom <- shape_denom / mu_denom[i]
    ll <- ll + shape_denom * log(rate_denom) + (shape_denom - 1) * log(y_denom[i]) -
          rate_denom * y_denom[i] - lgamma(shape_denom)
  }

  sigma_beta <- 10
  for (b in beta_num) ll <- ll - 0.5 * (b / sigma_beta)^2
  for (b in beta_denom) ll <- ll - 0.5 * (b / sigma_beta)^2

  shape_prior <- 2
  rate_prior <- 0.5
  ll <- ll + (shape_prior - 1) * log_shape_num - rate_prior * shape_num + log_shape_num
  ll <- ll + (shape_prior - 1) * log_shape_denom - rate_prior * shape_denom + log_shape_denom

  return(ll)
}

for (j in 1:6) {
  params_plus <- test_params
  params_plus[j] <- test_params[j] + eps
  params_minus <- test_params
  params_minus[j] <- test_params[j] - eps
  grad_numerical[j] <- (compute_log_post(params_plus) - compute_log_post(params_minus)) / (2 * eps)
}

cat("\n=== Numerical gradients ===\n")
cat("d(LP)/d(beta_num[1]):", grad_numerical[1], "\n")
cat("d(LP)/d(beta_num[2]):", grad_numerical[2], "\n")
cat("d(LP)/d(beta_denom[1]):", grad_numerical[3], "\n")
cat("d(LP)/d(beta_denom[2]):", grad_numerical[4], "\n")
cat("d(LP)/d(log_shape_num):", grad_numerical[5], "\n")
cat("d(LP)/d(log_shape_denom):", grad_numerical[6], "\n")

cat("\n=== Manual - Numerical difference ===\n")
for (j in 1:6) {
  cat("param", j, ":", grad_manual[j] - grad_numerical[j], "\n")
}
