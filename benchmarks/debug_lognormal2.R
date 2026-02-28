# Debug lognormal - test log-posterior computation directly
library(numdenom)

set.seed(42)

# Generate minimal intercept-only data for simplicity
N <- 100

# True parameters for lognormal
mu_num <- 2.0  # mean on log scale (intercept)
mu_denom <- 4.0
sigma_num <- 0.5
sigma_denom <- 0.3

# Generate responses - lognormal means log(Y) ~ Normal(mu, sigma^2)
y_num <- exp(rnorm(N, mu_num, sigma_num))
y_denom <- exp(rnorm(N, mu_denom, sigma_denom))

dat <- data.frame(y_num = y_num, y_denom = y_denom)

cat("True values:\n")
cat("  beta_num: ", mu_num, "\n")
cat("  beta_denom: ", mu_denom, "\n")
cat("  sigma_num: ", sigma_num, "\n")
cat("  sigma_denom: ", sigma_denom, "\n\n")

# Check what R expects
cat("Data statistics:\n")
cat("  mean(log(y_num)): ", mean(log(y_num)), " (expected ~", mu_num, ")\n")
cat("  mean(log(y_denom)): ", mean(log(y_denom)), " (expected ~", mu_denom, ")\n")
cat("  sd(log(y_num)): ", sd(log(y_num)), " (expected ~", sigma_num, ")\n")
cat("  sd(log(y_denom)): ", sd(log(y_denom)), " (expected ~", sigma_denom, ")\n\n")

# Compute likelihood manually at true values
manual_ll <- function(beta_num, beta_denom, sigma_num, sigma_denom) {
  ll_num <- sum(dnorm(log(y_num), beta_num, sigma_num, log = TRUE))
  ll_denom <- sum(dnorm(log(y_denom), beta_denom, sigma_denom, log = TRUE))
  # Add Jacobian for lognormal
  ll_num <- ll_num - sum(log(y_num))
  ll_denom <- ll_denom - sum(log(y_denom))
  ll_num + ll_denom
}

cat("Manual log-likelihood at true values:\n")
ll_true <- manual_ll(mu_num, mu_denom, sigma_num, sigma_denom)
cat("  LL =", ll_true, "\n")

cat("\nManual log-likelihood at wrong values (swapped):\n")
ll_swapped <- manual_ll(mu_denom, mu_num, sigma_num, sigma_denom)
cat("  LL =", ll_swapped, " (swapped num/denom betas)\n")

# MLE for comparison
cat("\nMLE estimates (from log-transformed data):\n")
cat("  beta_num MLE: ", mean(log(y_num)), "\n")
cat("  beta_denom MLE: ", mean(log(y_denom)), "\n")
cat("  sigma_num MLE: ", sd(log(y_num)) * sqrt((N-1)/N), "\n")
cat("  sigma_denom MLE: ", sd(log(y_denom)) * sqrt((N-1)/N), "\n")

# Now fit with numdenom
cat("\n=== Fitting numdenom (lognormal) ===\n")
nd_fit <- ratiod(
  y_num | y_denom ~ 1,
  data = dat,
  family = ratiod_lognormal(),
  iter = 2000,
  warmup = 1000,
  chains = 1,
  seed = 123
)

print(summary(nd_fit))

# Check what the C++ side is getting
cat("\n=== Checking C++ data structures ===\n")
form <- numdenom:::ratiod_formula(y_num | y_denom ~ 1, data = dat)
model_type <- numdenom:::get_hmc_model_type(ratiod_lognormal())
hmc_data <- numdenom:::prepare_hmc_data(form, dat, ratiod_lognormal(), model_type)

cat("Model type:", model_type, "\n")
cat("HMC data y_num_cont (first 5):", head(hmc_data$y_num_cont, 5), "\n")
cat("HMC data y_denom_cont (first 5):", head(hmc_data$y_denom_cont, 5), "\n")

# Check that y_num_cont matches y_num from our data
cat("\nVerifying data passed correctly:\n")
cat("  max(|y_num_cont - y_num|):", max(abs(hmc_data$y_num_cont - dat$y_num)), "\n")
cat("  max(|y_denom_cont - y_denom|):", max(abs(hmc_data$y_denom_cont - dat$y_denom)), "\n")
