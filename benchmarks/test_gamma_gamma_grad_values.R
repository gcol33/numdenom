# Compare actual gradient values between modes

library(numdenom)

set.seed(42)

# Small test
N <- 20
y_num <- c(5.5, 8.2, 6.1, 7.3, 4.8, 9.1, 5.9, 7.7, 6.5, 8.0,
           5.2, 7.1, 6.8, 8.5, 5.6, 7.9, 6.3, 8.3, 5.4, 7.5)  # mean ~ 7
y_denom <- c(18, 22, 19, 21, 17, 23, 20, 21, 18, 22,
             19, 20, 21, 23, 18, 22, 19, 21, 17, 20)  # mean ~ 20

df <- data.frame(y = y_num, denom = y_denom)

# Compute gradient manually in R

# Test point
beta_num <- 1.94  # log(mean(y_num)) ~ 1.94
beta_denom <- 3.0  # log(mean(y_denom)) ~ 3.0
log_shape_num <- log(2)  # shape_num = 2
log_shape_denom <- log(2)  # shape_denom = 2

shape_num <- exp(log_shape_num)
shape_denom <- exp(log_shape_denom)

# Compute linear predictors
eta_num <- beta_num  # intercept only
eta_denom <- beta_denom
mu_num <- exp(eta_num)
mu_denom <- exp(eta_denom)

cat("=== Test point ===\n")
cat("beta_num:", beta_num, "=> mu_num:", mu_num, "\n")
cat("beta_denom:", beta_denom, "=> mu_denom:", mu_denom, "\n")
cat("log_shape_num:", log_shape_num, "=> shape_num:", shape_num, "\n")
cat("log_shape_denom:", log_shape_denom, "=> shape_denom:", shape_denom, "\n")

# Manual gradient computation

# Priors (from numdenom code)
sigma_beta <- 10
phi_prior_shape <- 2
phi_prior_rate <- 0.5

# Prior gradients
grad_beta_num_prior <- -beta_num / (sigma_beta^2)
grad_beta_denom_prior <- -beta_denom / (sigma_beta^2)
# Gamma prior: d/d(log_phi) = (alpha-1) - rate*phi + 1
grad_log_shape_num_prior <- (phi_prior_shape - 1) - phi_prior_rate * shape_num + 1
grad_log_shape_denom_prior <- (phi_prior_shape - 1) - phi_prior_rate * shape_denom + 1

cat("\n=== Prior gradients ===\n")
cat("grad_beta_num_prior:", grad_beta_num_prior, "\n")
cat("grad_beta_denom_prior:", grad_beta_denom_prior, "\n")
cat("grad_log_shape_num_prior:", grad_log_shape_num_prior, "\n")
cat("grad_log_shape_denom_prior:", grad_log_shape_denom_prior, "\n")

# Likelihood gradients
grad_beta_num_lik <- 0
grad_beta_denom_lik <- 0
grad_shape_num_lik <- 0  # gradient w.r.t. shape (not log_shape)
grad_shape_denom_lik <- 0

for (i in 1:N) {
  # Numerator gamma
  rate_num <- shape_num / mu_num
  resid_num <- shape_num * (y_num[i] / mu_num - 1.0)
  grad_shape_num_i <- log(rate_num) + 1 + log(y_num[i]) - digamma(shape_num) - y_num[i] / mu_num

  # Denominator gamma
  rate_denom <- shape_denom / mu_denom
  resid_denom <- shape_denom * (y_denom[i] / mu_denom - 1.0)
  grad_shape_denom_i <- log(rate_denom) + 1 + log(y_denom[i]) - digamma(shape_denom) - y_denom[i] / mu_denom

  # Accumulate
  grad_beta_num_lik <- grad_beta_num_lik + resid_num  # X = 1 for intercept
  grad_beta_denom_lik <- grad_beta_denom_lik + resid_denom
  grad_shape_num_lik <- grad_shape_num_lik + grad_shape_num_i
  grad_shape_denom_lik <- grad_shape_denom_lik + grad_shape_denom_i
}

cat("\n=== Likelihood gradients (w.r.t. shape, not log_shape) ===\n")
cat("grad_beta_num_lik:", grad_beta_num_lik, "\n")
cat("grad_beta_denom_lik:", grad_beta_denom_lik, "\n")
cat("grad_shape_num_lik:", grad_shape_num_lik, "\n")
cat("grad_shape_denom_lik:", grad_shape_denom_lik, "\n")

# Convert to log_shape gradients (multiply by shape)
grad_log_shape_num_lik <- shape_num * grad_shape_num_lik
grad_log_shape_denom_lik <- shape_denom * grad_shape_denom_lik

cat("\n=== Likelihood gradients (w.r.t. log_shape) ===\n")
cat("grad_log_shape_num_lik:", grad_log_shape_num_lik, "\n")
cat("grad_log_shape_denom_lik:", grad_log_shape_denom_lik, "\n")

# Total gradients
grad_beta_num_total <- grad_beta_num_prior + grad_beta_num_lik
grad_beta_denom_total <- grad_beta_denom_prior + grad_beta_denom_lik
grad_log_shape_num_total <- grad_log_shape_num_prior + grad_log_shape_num_lik
grad_log_shape_denom_total <- grad_log_shape_denom_prior + grad_log_shape_denom_lik

cat("\n=== Total gradients (prior + likelihood) ===\n")
cat("grad_beta_num:", grad_beta_num_total, "\n")
cat("grad_beta_denom:", grad_beta_denom_total, "\n")
cat("grad_log_shape_num:", grad_log_shape_num_total, "\n")
cat("grad_log_shape_denom:", grad_log_shape_denom_total, "\n")

# Numerical verification
eps <- 1e-6
params <- c(beta_num, beta_denom, log_shape_num, log_shape_denom)

log_post <- function(p) {
  bn <- p[1]; bd <- p[2]; lsn <- p[3]; lsd <- p[4]
  sn <- exp(lsn); sd <- exp(lsd)
  mn <- exp(bn); md <- exp(bd)

  ll <- 0
  for (i in 1:N) {
    # Gamma(y | shape, rate=shape/mu)
    ll <- ll + dgamma(y_num[i], shape=sn, rate=sn/mn, log=TRUE)
    ll <- ll + dgamma(y_denom[i], shape=sd, rate=sd/md, log=TRUE)
  }

  # Priors
  ll <- ll - 0.5 * (bn/sigma_beta)^2 - 0.5 * (bd/sigma_beta)^2
  # Gamma prior on shape: dgamma(shape | alpha, beta) * |d(shape)/d(log_shape)|
  ll <- ll + dgamma(sn, phi_prior_shape, phi_prior_rate, log=TRUE) + lsn  # Jacobian
  ll <- ll + dgamma(sd, phi_prior_shape, phi_prior_rate, log=TRUE) + lsd

  return(ll)
}

grad_numerical <- numeric(4)
for (j in 1:4) {
  p_plus <- params; p_plus[j] <- params[j] + eps
  p_minus <- params; p_minus[j] <- params[j] - eps
  grad_numerical[j] <- (log_post(p_plus) - log_post(p_minus)) / (2*eps)
}

cat("\n=== Numerical gradients ===\n")
cat("grad_beta_num (num):", grad_numerical[1], "\n")
cat("grad_beta_denom (num):", grad_numerical[2], "\n")
cat("grad_log_shape_num (num):", grad_numerical[3], "\n")
cat("grad_log_shape_denom (num):", grad_numerical[4], "\n")

cat("\n=== Manual - Numerical ===\n")
cat("beta_num diff:", grad_beta_num_total - grad_numerical[1], "\n")
cat("beta_denom diff:", grad_beta_denom_total - grad_numerical[2], "\n")
cat("log_shape_num diff:", grad_log_shape_num_total - grad_numerical[3], "\n")
cat("log_shape_denom diff:", grad_log_shape_denom_total - grad_numerical[4], "\n")
