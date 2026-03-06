# Debug script for gamma_gamma - check likelihood and data passing
# Focus on whether eta_num and eta_denom are computed correctly

library(numdenom)

set.seed(42)

# Generate minimal intercept-only data first
N <- 100

# True parameters - INTERCEPT ONLY
beta_num_int <- 2.0
beta_denom_int <- 4.0
shape_num <- 2.0
shape_denom <- 3.0

# Generate responses
mu_num <- exp(beta_num_int)  # exp(2) ~ 7.4
mu_denom <- exp(beta_denom_int)  # exp(4) ~ 54.6

y_num <- rgamma(N, shape = shape_num, rate = shape_num / mu_num)
y_denom <- rgamma(N, shape = shape_denom, rate = shape_denom / mu_denom)

dat <- data.frame(y_num = y_num, y_denom = y_denom)

cat("=== INTERCEPT-ONLY TEST ===\n")
cat("True values:\n")
cat("  beta_num (intercept): ", beta_num_int, " => mu_num =", exp(beta_num_int), "\n")
cat("  beta_denom (intercept): ", beta_denom_int, " => mu_denom =", exp(beta_denom_int), "\n")
cat("  shape_num: ", shape_num, "\n")
cat("  shape_denom: ", shape_denom, "\n")
cat("\nData stats:\n")
cat("  y_num mean:", mean(y_num), " (expected ~", exp(beta_num_int), ")\n")
cat("  y_denom mean:", mean(y_denom), " (expected ~", exp(beta_denom_int), ")\n")
cat("  log(y_num) mean:", mean(log(y_num)), " (expected ~", beta_num_int, ")\n")
cat("  log(y_denom) mean:", mean(log(y_denom)), " (expected ~", beta_denom_int, ")\n")

# Fit model
cat("\n=== Fitting intercept-only model ===\n")
fit <- ratiod(
  y_num | y_denom ~ 1,
  data = dat,
  family = ratiod_gamma_gamma(),
  iter = 2000,
  warmup = 1000,
  chains = 2,
  seed = 123
)

print(summary(fit))

# Extract draws and check
draws <- as.matrix(fit$draws)
cat("\nEstimates vs True:\n")
cat("  beta_num[1]: estimated =", mean(draws[,"beta_num[1]"]), ", true =", beta_num_int, "\n")
cat("  beta_denom[1]: estimated =", mean(draws[,"beta_denom[1]"]), ", true =", beta_denom_int, "\n")
cat("  shape_num: estimated =", mean(draws[,"shape_num"]), ", true =", shape_num, "\n")
cat("  shape_denom: estimated =", mean(draws[,"shape_denom"]), ", true =", shape_denom, "\n")

# Also check what the C++ side thinks is happening
cat("\n=== Checking C++ data structures ===\n")
form <- numdenom:::ratiod_formula(y_num | y_denom ~ 1, data = dat)
model_type <- numdenom:::get_hmc_model_type(ratiod_gamma_gamma())
hmc_data <- numdenom:::prepare_hmc_data(form, dat, ratiod_gamma_gamma(), model_type)

cat("HMC data y_num_cont (first 5):", head(hmc_data$y_num_cont, 5), "\n")
cat("HMC data y_denom_cont (first 5):", head(hmc_data$y_denom_cont, 5), "\n")
cat("HMC data X_num:\n")
print(head(hmc_data$X_num))
cat("HMC data X_denom:\n")
print(head(hmc_data$X_denom))
