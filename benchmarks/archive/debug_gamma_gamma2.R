# Debug script for gamma_gamma - deeper investigation
# Check if design matrix is being used correctly

library(numdenom)

set.seed(42)

# Generate minimal data
N <- 100
x <- rnorm(N)

# True parameters
beta_num <- c(2.0, 0.3)
beta_denom <- c(4.0, 0.2)
shape_num <- 2.0
shape_denom <- 3.0

# Generate responses
eta_num <- beta_num[1] + beta_num[2] * x
eta_denom <- beta_denom[1] + beta_denom[2] * x
mu_num <- exp(eta_num)
mu_denom <- exp(eta_denom)

y_num <- rgamma(N, shape = shape_num, rate = shape_num / mu_num)
y_denom <- rgamma(N, shape = shape_denom, rate = shape_denom / mu_denom)

dat <- data.frame(y_num = y_num, y_denom = y_denom, x = x)

# Parse formula
form <- numdenom:::ratiod_formula(y_num | y_denom ~ x, data = dat)
cat("Formula parsed:\n")
cat("  X_num dim:", dim(form$numerator$X), "\n")
cat("  X_denom dim:", dim(form$denominator$X), "\n")

# Check prepare_hmc_data
model_type <- numdenom:::get_hmc_model_type(ratiod_gamma_gamma())
cat("\nModel type:", model_type, "\n")

hmc_data <- numdenom:::prepare_hmc_data(form, dat, ratiod_gamma_gamma(), model_type)
cat("\nHMC data:\n")
cat("  y_num (first 5):", head(hmc_data$y_num, 5), "\n")
cat("  y_denom (first 5):", head(hmc_data$y_denom, 5), "\n")
cat("  y_num_cont (first 5):", head(hmc_data$y_num_cont, 5), "\n")
cat("  y_denom_cont (first 5):", head(hmc_data$y_denom_cont, 5), "\n")
cat("  X_num (first row):", hmc_data$X_num[1,], "\n")
cat("  X_denom (first row):", hmc_data$X_denom[1,], "\n")

# Check if X_num and X_denom are identical
cat("\n  X_num == X_denom:", all(hmc_data$X_num == hmc_data$X_denom), "\n")

# Now manually test log-likelihood
cat("\n=== Testing log-likelihood computation ===\n")

# Use numdenom's internal C++ test function if available
# Otherwise, compute manually

# Let's see what the data looks like in terms of expected linear predictor
log_mu_num_obs <- log(y_num)
cat("\nObserved log(y_num) stats: mean =", mean(log_mu_num_obs), "sd =", sd(log_mu_num_obs), "\n")
cat("True eta_num stats: mean =", mean(eta_num), "sd =", sd(eta_num), "\n")

log_mu_denom_obs <- log(y_denom)
cat("Observed log(y_denom) stats: mean =", mean(log_mu_denom_obs), "sd =", sd(log_mu_denom_obs), "\n")
cat("True eta_denom stats: mean =", mean(eta_denom), "sd =", sd(eta_denom), "\n")

# Try running with longer chains and check convergence
cat("\n=== Running with longer chains ===\n")
nd_fit <- tratio(
  y_num | y_denom ~ x,
  data = dat,
  family = ratiod_gamma_gamma(),
  control = list(iter = 4000, warmup = 2000, chains = 2, seed = 123)
)

nd_summary <- summary(nd_fit)
print(nd_summary)

cat("\n=== Checking trace plots ===\n")
draws <- as.matrix(nd_fit$draws)
cat("beta_num[1] range:", range(draws[,"beta_num[1]"]), "\n")
cat("beta_num[2] range:", range(draws[,"beta_num[2]"]), "\n")
cat("beta_denom[1] range:", range(draws[,"beta_denom[1]"]), "\n")
cat("beta_denom[2] range:", range(draws[,"beta_denom[2]"]), "\n")

# Check if there's some confusion between num and denom parameters
cat("\n=== Checking for parameter confusion ===\n")
cat("Expected intercepts: num~2, denom~4\n")
cat("Estimated: num=", mean(draws[,"beta_num[1]"]), ", denom=", mean(draws[,"beta_denom[1]"]), "\n")
cat("Ratio of intercepts (expected ~0.5): ", mean(draws[,"beta_num[1]"]) / mean(draws[,"beta_denom[1]"]), "\n")
