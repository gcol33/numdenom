# Direct gradient check for gamma_gamma using C++ test function

library(numdenom)

set.seed(42)

# Simple test case
N <- 10
x <- rnorm(N)
y_num <- rgamma(N, shape = 5, rate = 5 / exp(2 + 0.3 * x))
y_denom <- rgamma(N, shape = 8, rate = 8 / exp(3 + 0.2 * x))

df <- data.frame(y = y_num, denom = y_denom, x = x)

# Use the C++ gradient check function
# First, prepare the data structures

cat("=== Gradient check with cpp_gradient_check ===\n")

# Build the data structures
parsed <- numdenom:::ratiod_formula(y | denom ~ x, data = df)
family <- ratiod_gamma_gamma()
hmc_data <- numdenom:::prepare_hmc_data(parsed, family, df, model_type = "gamma_gamma")

# Check if gradient check function is available
if (exists("cpp_test_gradient", where = "package:numdenom")) {
  cat("Found cpp_test_gradient\n")
} else {
  cat("cpp_test_gradient not exported - using ratiod with debug\n")
}

# Test at reasonable parameter values
cat("\n=== Running models to check gradient differences ===\n")

# Use very short chains to just check the first few iterations
cat("\nNumerical gradient (N) - 50 iter:\n")
fit_N <- tryCatch({
  ratiod(
    y | denom ~ x,
    data = df,
    family = ratiod_gamma_gamma(),
    iter = 50, warmup = 25, chains = 1,
    gradient_mode = "N",
    verbose = FALSE
  )
}, error = function(e) {
  cat("Error:", e$message, "\n")
  NULL
})

if (!is.null(fit_N)) {
  draws_N <- as.matrix(fit_N$draws)
  cat("beta_num[2]:", mean(draws_N[,"beta_num[2]"]), "\n")
  cat("shape_num:", mean(draws_N[,"shape_num"]), "\n")
}

cat("\nForward autodiff (A) - 50 iter:\n")
fit_A <- tryCatch({
  ratiod(
    y | denom ~ x,
    data = df,
    family = ratiod_gamma_gamma(),
    iter = 50, warmup = 25, chains = 1,
    gradient_mode = "A",
    verbose = FALSE
  )
}, error = function(e) {
  cat("Error:", e$message, "\n")
  NULL
})

if (!is.null(fit_A)) {
  draws_A <- as.matrix(fit_A$draws)
  cat("beta_num[2]:", mean(draws_A[,"beta_num[2]"]), "\n")
  cat("shape_num:", mean(draws_A[,"shape_num"]), "\n")
}

cat("\n=== Manual log-likelihood calculation ===\n")

# Calculate log-likelihood manually
log_lik_gamma <- function(y, shape, mu) {
  rate <- shape / mu
  shape * log(rate) + (shape - 1) * log(y) - rate * y - lgamma(shape)
}

# At true parameters
beta_num <- c(2, 0.3)
beta_denom <- c(3, 0.2)
shape_num_true <- 5
shape_denom_true <- 8

X <- cbind(1, df$x)
eta_num <- X %*% beta_num
eta_denom <- X %*% beta_denom
mu_num <- exp(eta_num)
mu_denom <- exp(eta_denom)

ll_num <- sum(sapply(1:N, function(i) log_lik_gamma(df$y[i], shape_num_true, mu_num[i])))
ll_denom <- sum(sapply(1:N, function(i) log_lik_gamma(df$denom[i], shape_denom_true, mu_denom[i])))

cat("Log-likelihood at true params:\n")
cat("  ll_num:", ll_num, "\n")
cat("  ll_denom:", ll_denom, "\n")
cat("  total:", ll_num + ll_denom, "\n")

# Calculate gradient of ll w.r.t. beta_num[2] numerically
eps <- 1e-6
beta_num_plus <- c(2, 0.3 + eps)
eta_num_plus <- X %*% beta_num_plus
mu_num_plus <- exp(eta_num_plus)
ll_num_plus <- sum(sapply(1:N, function(i) log_lik_gamma(df$y[i], shape_num_true, mu_num_plus[i])))

grad_beta_num_2 <- (ll_num_plus - ll_num) / eps

cat("\nNumerical gradient d(ll_num)/d(beta_num[2]):", grad_beta_num_2, "\n")

# Analytical gradient should be:
# d(ll)/d(beta) = sum over i of (d(ll_i)/d(mu_i)) * (d(mu_i)/d(eta_i)) * (d(eta_i)/d(beta))
#               = sum over i of (shape/mu - shape*y/(mu^2)) * mu * x[i]
#               = sum over i of (shape - shape*y/mu) * x[i]
#               = sum over i of shape * (1 - y/mu) * x[i]
grad_analytical <- sum(shape_num_true * (1 - df$y / mu_num) * df$x)
cat("Analytical gradient d(ll_num)/d(beta_num[2]):", grad_analytical, "\n")

cat("\nDifference:", abs(grad_beta_num_2 - grad_analytical), "\n")
