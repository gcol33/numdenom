# Test if the gamma log-likelihood is computed correctly

set.seed(123)

# Test the Gamma log-likelihood formula
# Gamma(y | shape, rate) where rate = shape/mu (mean = mu)
# log p(y) = shape*log(rate) + (shape-1)*log(y) - rate*y - lgamma(shape)

gamma_loglik <- function(y, mu, shape) {
  rate <- shape / mu
  ll <- shape * log(rate) + (shape - 1) * log(y) - rate * y - lgamma(shape)
  return(ll)
}

# Verify against R's dgamma
y <- 5.0
mu <- 10.0
shape <- 3.0

# Our formula
ll_ours <- gamma_loglik(y, mu, shape)

# R's dgamma with rate parameterization
rate <- shape / mu
ll_R <- dgamma(y, shape = shape, rate = rate, log = TRUE)

cat("Log-likelihood verification:\n")
cat("  Our formula: ", ll_ours, "\n")
cat("  R's dgamma:  ", ll_R, "\n")
cat("  Difference:  ", abs(ll_ours - ll_R), "\n")

# Now test gradients
# d(LL)/d(eta) where eta = log(mu), so mu = exp(eta)
# d(LL)/d(mu) = -shape/mu + shape*y/mu^2 = (shape/mu)(y/mu - 1)
# d(LL)/d(eta) = d(LL)/d(mu) * d(mu)/d(eta) = d(LL)/d(mu) * mu
#              = shape * (y/mu - 1)

grad_eta <- function(y, mu, shape) {
  return(shape * (y / mu - 1))
}

# Verify numerically
eta <- log(mu)
eps <- 1e-6
ll_plus <- gamma_loglik(y, exp(eta + eps), shape)
ll_minus <- gamma_loglik(y, exp(eta - eps), shape)
grad_numerical <- (ll_plus - ll_minus) / (2 * eps)
grad_analytical <- grad_eta(y, mu, shape)

cat("\nGradient w.r.t. eta verification:\n")
cat("  Analytical:  ", grad_analytical, "\n")
cat("  Numerical:   ", grad_numerical, "\n")
cat("  Difference:  ", abs(grad_analytical - grad_numerical), "\n")

# Shape gradient
# d(LL)/d(shape) = log(rate) + 1 + log(y) - y/mu - digamma(shape)
#                = log(shape/mu) + 1 + log(y) - y/mu - digamma(shape)

grad_shape <- function(y, mu, shape) {
  rate <- shape / mu
  return(log(rate) + 1 + log(y) - y/mu - digamma(shape))
}

ll_plus <- gamma_loglik(y, mu, shape + eps)
ll_minus <- gamma_loglik(y, mu, shape - eps)
grad_shape_numerical <- (ll_plus - ll_minus) / (2 * eps)
grad_shape_analytical <- grad_shape(y, mu, shape)

cat("\nGradient w.r.t. shape verification:\n")
cat("  Analytical:  ", grad_shape_analytical, "\n")
cat("  Numerical:   ", grad_shape_numerical, "\n")
cat("  Difference:  ", abs(grad_shape_analytical - grad_shape_numerical), "\n")

# Now let's check if the issue is with how numdenom parameterizes shape
# numdenom uses log(shape) as the parameter, so we need d(LL)/d(log_shape)
# d(LL)/d(log_shape) = d(LL)/d(shape) * shape

grad_log_shape <- function(y, mu, shape) {
  return(grad_shape(y, mu, shape) * shape)
}

log_shape <- log(shape)
ll_plus <- gamma_loglik(y, mu, exp(log_shape + eps))
ll_minus <- gamma_loglik(y, mu, exp(log_shape - eps))
grad_log_shape_numerical <- (ll_plus - ll_minus) / (2 * eps)
grad_log_shape_analytical <- grad_log_shape(y, mu, shape)

cat("\nGradient w.r.t. log(shape) verification:\n")
cat("  Analytical:  ", grad_log_shape_analytical, "\n")
cat("  Numerical:   ", grad_log_shape_numerical, "\n")
cat("  Difference:  ", abs(grad_log_shape_analytical - grad_log_shape_numerical), "\n")
