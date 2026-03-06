# Debug lognormal - test with explicit gradient checking
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

cat("=== TESTING GRADIENT CORRECTNESS ===\n\n")

# Fit with VERY SHORT chains to see initial behavior
cat("Short fit (10 iter) with verbose sampling:\n")
fit_short <- ratiod(
  y_num | y_denom ~ 1,
  data = dat,
  family = ratiod_lognormal(),
  iter = 10,
  warmup = 5,
  chains = 1,
  seed = 123
)

draws_short <- as.matrix(fit_short$draws)
cat("\nParameter values after 10 iterations:\n")
print(draws_short[1:5,])

# Now test with numerical gradients only
cat("\n=== Testing with NUMERICAL gradients only (mode='N') ===\n")
fit_numerical <- ratiod(
  y_num | y_denom ~ 1,
  data = dat,
  family = ratiod_lognormal(),
  iter = 1000,
  warmup = 500,
  chains = 1,
  seed = 123,
  gradient_mode = "N"
)

print(summary(fit_numerical))

draws_num <- as.matrix(fit_numerical$draws)
cat("\nNumerical gradient estimates:\n")
cat("  beta_num[1]: ", mean(draws_num[, "beta_num[1]"]), " (true: 2.0)\n")
cat("  beta_denom[1]: ", mean(draws_num[, "beta_denom[1]"]), " (true: 4.0)\n")
cat("  sigma_num: ", mean(draws_num[, "sigma_num"]), " (true: 0.5)\n")
cat("  sigma_denom: ", mean(draws_num[, "sigma_denom"]), " (true: 0.3)\n")

# Now test with autodiff
cat("\n=== Testing with Forward AUTODIFF (mode='A') ===\n")
fit_autodiff <- ratiod(
  y_num | y_denom ~ 1,
  data = dat,
  family = ratiod_lognormal(),
  iter = 1000,
  warmup = 500,
  chains = 1,
  seed = 123,
  gradient_mode = "A"
)

print(summary(fit_autodiff))

draws_auto <- as.matrix(fit_autodiff$draws)
cat("\nForward autodiff estimates:\n")
cat("  beta_num[1]: ", mean(draws_auto[, "beta_num[1]"]), " (true: 2.0)\n")
cat("  beta_denom[1]: ", mean(draws_auto[, "beta_denom[1]"]), " (true: 4.0)\n")
cat("  sigma_num: ", mean(draws_auto[, "sigma_num"]), " (true: 0.5)\n")
cat("  sigma_denom: ", mean(draws_auto[, "sigma_denom"]), " (true: 0.3)\n")

# Check if the issue is chains starting far from mode
cat("\n=== Checking initialization ===\n")
form <- numdenom:::ratiod_formula(y_num | y_denom ~ 1, data = dat)
model_type <- numdenom:::get_hmc_model_type(ratiod_lognormal())
hmc_data <- numdenom:::prepare_hmc_data(form, dat, ratiod_lognormal(), model_type)
q_init <- numdenom:::initialize_hmc_params_full(hmc_data, model_type)
cat("Initial q values:\n")
print(q_init)
cat("\nTransformed:\n")
cat("  beta_num[1] (log-scale mean init): ", q_init[1], "\n")
cat("  beta_denom[1]: ", q_init[2], "\n")
cat("  sigma_num (exp of init): ", exp(q_init[3]), "\n")
cat("  sigma_denom (exp of init): ", exp(q_init[4]), "\n")
