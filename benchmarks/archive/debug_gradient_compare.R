# Direct numerical comparison of gradients at the same parameter values
# Run with gradient_mode = "N" vs "A" and print log_post and gradients

library(numdenom)

set.seed(42)

# Minimal test case
N <- 20
x <- rnorm(N)
y_num <- rgamma(N, shape = 5, rate = 5 / exp(2 + 0.3 * x))
y_denom <- rgamma(N, shape = 8, rate = 8 / exp(3 + 0.2 * x))

df <- data.frame(y = y_num, denom = y_denom, x = x)

cat("Data summary:\n")
cat("  y_num range:", range(y_num), "\n")
cat("  y_denom range:", range(y_denom), "\n")

# Very short run - just 5 iterations to compare behavior
cat("\n=== Numerical gradient (N) - 5 iterations ===\n")
fit_N <- tratio(
  y | denom ~ x,
  data = df,
  family = ratiod_gamma_gamma(),
  control = list(iter = 5, warmup = 1, chains = 1, gradient_mode = "N", verbose = TRUE)
)

cat("\n=== Forward autodiff (A) - 5 iterations ===\n")
fit_A <- tratio(
  y | denom ~ x,
  data = df,
  family = ratiod_gamma_gamma(),
  control = list(iter = 5, warmup = 1, chains = 1, gradient_mode = "A", verbose = TRUE)
)

# Compare final parameter values
cat("\n=== Final parameter comparison ===\n")
draws_N <- as.matrix(fit_N$draws)
draws_A <- as.matrix(fit_A$draws)

cat("Numerical (N) final:\n")
cat("  beta_num[1]:", tail(draws_N[,"beta_num[1]"], 1), "\n")
cat("  beta_num[2]:", tail(draws_N[,"beta_num[2]"], 1), "\n")

cat("\nForward AD (A) final:\n")
cat("  beta_num[1]:", tail(draws_A[,"beta_num[1]"], 1), "\n")
cat("  beta_num[2]:", tail(draws_A[,"beta_num[2]"], 1), "\n")
