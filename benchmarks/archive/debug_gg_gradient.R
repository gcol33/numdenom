# Gradient verification for gamma_gamma

library(numdenom)

set.seed(42)

# Simple test case
N <- 20
x <- rnorm(N)
true_beta_num <- c(2.0, 0.3)
true_beta_denom <- c(3.0, 0.2)
shape_num <- 5
shape_denom <- 8

# Generate data
eta_num <- cbind(1, x) %*% true_beta_num
eta_denom <- cbind(1, x) %*% true_beta_denom
mu_num <- exp(eta_num)
mu_denom <- exp(eta_denom)
y_num <- rgamma(N, shape = shape_num, rate = shape_num / mu_num)
y_denom <- rgamma(N, shape = shape_denom, rate = shape_denom / mu_denom)

df <- data.frame(y = y_num, denom = y_denom, x = x)

cat("=== Testing with gradient_mode = 'N' (numerical - reference) ===\n")
fit_num <- tratio(
  y | denom ~ x,
  data = df,
  family = ratiod_gamma_gamma(),
  control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "N", verbose = TRUE)
)

draws_num <- as.matrix(fit_num$draws)
cat("\n=== Numerical Gradient Results ===\n")
cat("beta_num[2]:", mean(draws_num[,"beta_num[2]"]), "(true: 0.3)\n")
cat("shape_num:", mean(draws_num[,"shape_num"]), "(true: 5)\n")

cat("\n=== Testing with gradient_mode = 'A' (forward autodiff) ===\n")
fit_fwd <- tratio(
  y | denom ~ x,
  data = df,
  family = ratiod_gamma_gamma(),
  control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "A", verbose = TRUE)
)

draws_fwd <- as.matrix(fit_fwd$draws)
cat("\n=== Forward Autodiff Results ===\n")
cat("beta_num[2]:", mean(draws_fwd[,"beta_num[2]"]), "(true: 0.3)\n")
cat("shape_num:", mean(draws_fwd[,"shape_num"]), "(true: 5)\n")

cat("\n=== Testing with gradient_mode = 'A_t' (tape autodiff) ===\n")
fit_tape <- tratio(
  y | denom ~ x,
  data = df,
  family = ratiod_gamma_gamma(),
  control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "A_t", verbose = TRUE)
)

draws_tape <- as.matrix(fit_tape$draws)
cat("\n=== Tape Autodiff Results ===\n")
cat("beta_num[2]:", mean(draws_tape[,"beta_num[2]"]), "(true: 0.3)\n")
cat("shape_num:", mean(draws_tape[,"shape_num"]), "(true: 5)\n")

cat("\n=== Comparison ===\n")
cat("Numerical:    beta_num[2] =", mean(draws_num[,"beta_num[2]"]), ", shape_num =", mean(draws_num[,"shape_num"]), "\n")
cat("Forward AD:   beta_num[2] =", mean(draws_fwd[,"beta_num[2]"]), ", shape_num =", mean(draws_fwd[,"shape_num"]), "\n")
cat("Tape AD:      beta_num[2] =", mean(draws_tape[,"beta_num[2]"]), ", shape_num =", mean(draws_tape[,"shape_num"]), "\n")
cat("True values:  beta_num[2] = 0.3, shape_num = 5\n")
