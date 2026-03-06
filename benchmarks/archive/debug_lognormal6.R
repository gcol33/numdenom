# Debug lognormal - trace through exactly what C++ receives
library(numdenom)

set.seed(42)
devtools::load_all()

N <- 50  # smaller for debugging
mu_num <- 2.0
mu_denom <- 4.0
sigma_num <- 0.5
sigma_denom <- 0.3

y_num <- exp(rnorm(N, mu_num, sigma_num))
y_denom <- exp(rnorm(N, mu_denom, sigma_denom))

dat <- data.frame(y_num = y_num, y_denom = y_denom)

# Get the hmc_data that C++ will receive
form <- numdenom:::ratiod_formula(y_num | y_denom ~ 1, data = dat)
model_type <- numdenom:::get_hmc_model_type(ratiod_lognormal())
hmc_data <- numdenom:::prepare_hmc_data(form, dat, ratiod_lognormal(), model_type)

cat("Model type:", model_type, "\n\n")

cat("=== Data arrays ===\n")
cat("y_num (integer, should be 0s for lognormal):\n")
print(head(hmc_data$y_num, 10))
cat("\ny_denom (integer, should be 0s for lognormal):\n")
print(head(hmc_data$y_denom, 10))
cat("\ny_num_cont (should match dat$y_num):\n")
print(head(hmc_data$y_num_cont, 10))
cat("\ny_denom_cont (should match dat$y_denom):\n")
print(head(hmc_data$y_denom_cont, 10))

# Check if data matches
cat("\n=== Verification ===\n")
cat("max diff y_num_cont vs dat$y_num:", max(abs(hmc_data$y_num_cont - dat$y_num)), "\n")
cat("max diff y_denom_cont vs dat$y_denom:", max(abs(hmc_data$y_denom_cont - dat$y_denom)), "\n")

# Compute what the log-likelihood SHOULD be at MLE
mle_beta_num <- mean(log(y_num))
mle_beta_denom <- mean(log(y_denom))
mle_sigma_num <- sd(log(y_num)) * sqrt((N-1)/N)
mle_sigma_denom <- sd(log(y_denom)) * sqrt((N-1)/N)

cat("\n=== MLE values ===\n")
cat("MLE beta_num:", mle_beta_num, "\n")
cat("MLE beta_denom:", mle_beta_denom, "\n")
cat("MLE sigma_num:", mle_sigma_num, "\n")
cat("MLE sigma_denom:", mle_sigma_denom, "\n")

# Compute log-likelihood at MLE
log_lik_at_mle <- function() {
  ll_num <- sum(dnorm(log(y_num), mle_beta_num, mle_sigma_num, log = TRUE)) - sum(log(y_num))
  ll_denom <- sum(dnorm(log(y_denom), mle_beta_denom, mle_sigma_denom, log = TRUE)) - sum(log(y_denom))
  ll_num + ll_denom
}
cat("\nLog-likelihood at MLE:", log_lik_at_mle(), "\n")

# What if C++ is computing ll for y_num using y_denom's parameters?
cat("\n=== Misassignment tests ===\n")
ll_wrong_order <- function() {
  # If C++ swapped num/denom in likelihood
  ll_num <- sum(dnorm(log(y_denom), mle_beta_num, mle_sigma_num, log = TRUE)) - sum(log(y_denom))
  ll_denom <- sum(dnorm(log(y_num), mle_beta_denom, mle_sigma_denom, log = TRUE)) - sum(log(y_num))
  ll_num + ll_denom
}
cat("LL if y_num/y_denom swapped:", ll_wrong_order(), "\n")

# What if phi_num and phi_denom are swapped?
ll_phi_swapped <- function() {
  ll_num <- sum(dnorm(log(y_num), mle_beta_num, mle_sigma_denom, log = TRUE)) - sum(log(y_num))
  ll_denom <- sum(dnorm(log(y_denom), mle_beta_denom, mle_sigma_num, log = TRUE)) - sum(log(y_denom))
  ll_num + ll_denom
}
cat("LL if sigma_num/sigma_denom swapped:", ll_phi_swapped(), "\n")

# Check the initialization
spatial_info <- list(type = "none", n_units = 0)
temporal_info <- list(type = "none", n_times = 0, n_groups = 0)
zi_info <- list(type = "none", X_zi = matrix(0, 0, 0), X_oi = matrix(0, 0, 0), has_zi = FALSE, has_oi = FALSE)
latent_info <- list(n_factors = 0, constraint = "sum_to_zero", shared = TRUE, sigma_rate = 1.0)
svc_info <- list(n_svc = 0, n_neighbors = 0)

q_init <- numdenom:::initialize_hmc_params_full(
  hmc_data, model_type, spatial_info, temporal_info, zi_info, latent_info, svc_info
)
cat("\n=== Initial parameter values ===\n")
cat("q_init:", q_init, "\n")
cat("\nTransformed:\n")
cat("  beta_num:", q_init[1], "\n")
cat("  beta_denom:", q_init[2], "\n")
cat("  sigma_num (=exp(q[3])):", exp(q_init[3]), "\n")
cat("  sigma_denom (=exp(q[4])):", exp(q_init[4]), "\n")
