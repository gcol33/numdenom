# Debug script for gamma_gamma - test different gradient modes
# Try forcing numerical gradients to rule out gradient issues

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

cat("True values:\n")
cat("  beta_num: ", beta_num, "\n")
cat("  beta_denom: ", beta_denom, "\n")
cat("  shape_num: ", shape_num, "\n")
cat("  shape_denom: ", shape_denom, "\n\n")

# ============================================
# Test 1: Default (forward autodiff fallback)
# ============================================
cat("=== Test 1: Default gradient mode ===\n")
fit_default <- ratiod(
  y_num | y_denom ~ x,
  data = dat,
  family = ratiod_gamma_gamma(),
  iter = 2000,
  warmup = 1000,
  chains = 1,
  seed = 123
)
draws_default <- as.matrix(fit_default$draws)
cat("beta_num[1]:", mean(draws_default[,"beta_num[1]"]), "\n")
cat("beta_num[2]:", mean(draws_default[,"beta_num[2]"]), "\n")

# ============================================
# Test 2: Force numerical gradients
# ============================================
cat("\n=== Test 2: Numerical gradients (gradient_mode='N') ===\n")
fit_numerical <- ratiod(
  y_num | y_denom ~ x,
  data = dat,
  family = ratiod_gamma_gamma(),
  iter = 2000,
  warmup = 1000,
  chains = 1,
  seed = 123,
  gradient_mode = "N"
)
draws_numerical <- as.matrix(fit_numerical$draws)
cat("beta_num[1]:", mean(draws_numerical[,"beta_num[1]"]), "\n")
cat("beta_num[2]:", mean(draws_numerical[,"beta_num[2]"]), "\n")

# ============================================
# Test 3: Force tape autodiff
# ============================================
cat("\n=== Test 3: Tape autodiff (gradient_mode='A_t') ===\n")
fit_tape <- ratiod(
  y_num | y_denom ~ x,
  data = dat,
  family = ratiod_gamma_gamma(),
  iter = 2000,
  warmup = 1000,
  chains = 1,
  seed = 123,
  gradient_mode = "A_t"
)
draws_tape <- as.matrix(fit_tape$draws)
cat("beta_num[1]:", mean(draws_tape[,"beta_num[1]"]), "\n")
cat("beta_num[2]:", mean(draws_tape[,"beta_num[2]"]), "\n")

# ============================================
# Summary comparison
# ============================================
cat("\n=== SUMMARY ===\n")
cat("True beta_num[1] = 2.0, beta_num[2] = 0.3\n")
cat("Default (A):   beta_num[1] =", mean(draws_default[,"beta_num[1]"]),
    ", beta_num[2] =", mean(draws_default[,"beta_num[2]"]), "\n")
cat("Numerical (N): beta_num[1] =", mean(draws_numerical[,"beta_num[1]"]),
    ", beta_num[2] =", mean(draws_numerical[,"beta_num[2]"]), "\n")
cat("Tape (A_t):    beta_num[1] =", mean(draws_tape[,"beta_num[1]"]),
    ", beta_num[2] =", mean(draws_tape[,"beta_num[2]"]), "\n")
