# Direct gradient comparison at a fixed point

library(numdenom)

set.seed(42)

# ==============================================================================
# Test LOGNORMAL only - most problematic
# ==============================================================================
cat("=== Testing LOGNORMAL gradients directly ===\n")

N <- 50
x <- rnorm(N)
y_num_ln <- rlnorm(N, meanlog = 2 + 0.3 * x, sdlog = 0.5)
y_denom_ln <- rlnorm(N, meanlog = 3 + 0.2 * x, sdlog = 0.4)

df_ln <- data.frame(y = y_num_ln, denom = y_denom_ln, x = x)

# Test with more iterations to see if they converge similarly
cat("\nLOGNORMAL - Forward AD (A) - 1000 iter:\n")
fit_A <- tratio(
  y | denom ~ x,
  data = df_ln,
  family = ratiod_lognormal(),
  control = list(iter = 1000, warmup = 500, chains = 1, gradient_mode = "A", verbose = FALSE)
)
draws_A <- as.matrix(fit_A$draws)
cat("  beta_num[1]:", mean(draws_A[,"beta_num[1]"]), "+/-", sd(draws_A[,"beta_num[1]"]), "(true: 2)\n")
cat("  beta_num[2]:", mean(draws_A[,"beta_num[2]"]), "+/-", sd(draws_A[,"beta_num[2]"]), "(true: 0.3)\n")
cat("  sigma_num:", mean(draws_A[,"sigma_num"]), "+/-", sd(draws_A[,"sigma_num"]), "(true: 0.5)\n")

cat("\nLOGNORMAL - Hand-coded (H) - 1000 iter:\n")
fit_H <- tratio(
  y | denom ~ x,
  data = df_ln,
  family = ratiod_lognormal(),
  control = list(iter = 1000, warmup = 500, chains = 1, gradient_mode = "H", verbose = FALSE)
)
draws_H <- as.matrix(fit_H$draws)
cat("  beta_num[1]:", mean(draws_H[,"beta_num[1]"]), "+/-", sd(draws_H[,"beta_num[1]"]), "(true: 2)\n")
cat("  beta_num[2]:", mean(draws_H[,"beta_num[2]"]), "+/-", sd(draws_H[,"beta_num[2]"]), "(true: 0.3)\n")
cat("  sigma_num:", mean(draws_H[,"sigma_num"]), "+/-", sd(draws_H[,"sigma_num"]), "(true: 0.5)\n")

cat("\nLOGNORMAL - Tape AD (A_t) - 1000 iter:\n")
fit_At <- tratio(
  y | denom ~ x,
  data = df_ln,
  family = ratiod_lognormal(),
  control = list(iter = 1000, warmup = 500, chains = 1, gradient_mode = "A_t", verbose = FALSE)
)
draws_At <- as.matrix(fit_At$draws)
cat("  beta_num[1]:", mean(draws_At[,"beta_num[1]"]), "+/-", sd(draws_At[,"beta_num[1]"]), "(true: 2)\n")
cat("  beta_num[2]:", mean(draws_At[,"beta_num[2]"]), "+/-", sd(draws_At[,"beta_num[2]"]), "(true: 0.3)\n")
cat("  sigma_num:", mean(draws_At[,"sigma_num"]), "+/-", sd(draws_At[,"sigma_num"]), "(true: 0.5)\n")

cat("\n=== DONE ===\n")
