# Direct comparison: numerical vs forward autodiff gradients for gamma_gamma

library(numdenom)

set.seed(42)

# Minimal test case
N <- 5
x <- rnorm(N)
y_num <- rgamma(N, shape = 5, rate = 5 / exp(2 + 0.3 * x))
y_denom <- rgamma(N, shape = 8, rate = 8 / exp(3 + 0.2 * x))

df <- data.frame(y = y_num, denom = y_denom, x = x)

cat("=== Test 1: Verify numerical gradient gives correct results ===\n")
fit_N <- ratiod(
  y | denom ~ x,
  data = df,
  family = ratiod_gamma_gamma(),
  iter = 200, warmup = 100, chains = 1,
  gradient_mode = "N",
  verbose = FALSE
)

draws_N <- as.matrix(fit_N$draws)
cat("Numerical (N):\n")
cat("  beta_num[1]:", mean(draws_N[,"beta_num[1]"]), "(true: 2)\n")
cat("  beta_num[2]:", mean(draws_N[,"beta_num[2]"]), "(true: 0.3)\n")
cat("  shape_num:", mean(draws_N[,"shape_num"]), "(true: 5)\n")

cat("\n=== Test 2: Forward autodiff (A) for comparison ===\n")
fit_A <- ratiod(
  y | denom ~ x,
  data = df,
  family = ratiod_gamma_gamma(),
  iter = 200, warmup = 100, chains = 1,
  gradient_mode = "A",
  verbose = FALSE
)

draws_A <- as.matrix(fit_A$draws)
cat("Forward AD (A):\n")
cat("  beta_num[1]:", mean(draws_A[,"beta_num[1]"]), "(true: 2)\n")
cat("  beta_num[2]:", mean(draws_A[,"beta_num[2]"]), "(true: 0.3)\n")
cat("  shape_num:", mean(draws_A[,"shape_num"]), "(true: 5)\n")

cat("\n=== Test 3: Tape autodiff (A_t) for comparison ===\n")
fit_At <- ratiod(
  y | denom ~ x,
  data = df,
  family = ratiod_gamma_gamma(),
  iter = 200, warmup = 100, chains = 1,
  gradient_mode = "A_t",
  verbose = FALSE
)

draws_At <- as.matrix(fit_At$draws)
cat("Tape AD (A_t):\n")
cat("  beta_num[1]:", mean(draws_At[,"beta_num[1]"]), "(true: 2)\n")
cat("  beta_num[2]:", mean(draws_At[,"beta_num[2]"]), "(true: 0.3)\n")
cat("  shape_num:", mean(draws_At[,"shape_num"]), "(true: 5)\n")

cat("\n=== Summary ===\n")
cat("           beta_num[1]   beta_num[2]   shape_num\n")
cat("True:      2.0           0.3           5.0\n")
cat(sprintf("N:         %.3f         %.3f         %.3f\n", 
            mean(draws_N[,"beta_num[1]"]), mean(draws_N[,"beta_num[2]"]), mean(draws_N[,"shape_num"])))
cat(sprintf("A:         %.3f         %.3f         %.3f\n",
            mean(draws_A[,"beta_num[1]"]), mean(draws_A[,"beta_num[2]"]), mean(draws_A[,"shape_num"])))
cat(sprintf("A_t:       %.3f         %.3f         %.3f\n",
            mean(draws_At[,"beta_num[1]"]), mean(draws_At[,"beta_num[2]"]), mean(draws_At[,"shape_num"])))
