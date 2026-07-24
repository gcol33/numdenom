# Direct comparison: numerical vs forward autodiff gradients for gamma_gamma
# With more data to ensure identifiability

library(numdenom)

set.seed(42)

# Larger test case for better identifiability
N <- 100
x <- rnorm(N)
y_num <- rgamma(N, shape = 5, rate = 5 / exp(2 + 0.3 * x))
y_denom <- rgamma(N, shape = 8, rate = 8 / exp(3 + 0.2 * x))

df <- data.frame(y = y_num, denom = y_denom, x = x)

cat("Data summary:\n")
cat("  N =", N, "\n")
cat("  y_num range:", range(y_num), "\n")
cat("  y_denom range:", range(y_denom), "\n")

cat("\n=== Test 1: Numerical gradient (N) ===\n")
fit_N <- tratio(
  y | denom ~ x,
  data = df,
  family = ratiod_gamma_gamma(),
  control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "N", verbose = FALSE)
)

draws_N <- as.matrix(fit_N$draws)
cat("Numerical (N):\n")
cat("  beta_num[1]:", mean(draws_N[,"beta_num[1]"]), "+/-", sd(draws_N[,"beta_num[1]"]), "(true: 2)\n")
cat("  beta_num[2]:", mean(draws_N[,"beta_num[2]"]), "+/-", sd(draws_N[,"beta_num[2]"]), "(true: 0.3)\n")
cat("  shape_num:", mean(draws_N[,"shape_num"]), "+/-", sd(draws_N[,"shape_num"]), "(true: 5)\n")

cat("\n=== Test 2: Forward autodiff (A) ===\n")
fit_A <- tratio(
  y | denom ~ x,
  data = df,
  family = ratiod_gamma_gamma(),
  control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "A", verbose = FALSE)
)

draws_A <- as.matrix(fit_A$draws)
cat("Forward AD (A):\n")
cat("  beta_num[1]:", mean(draws_A[,"beta_num[1]"]), "+/-", sd(draws_A[,"beta_num[1]"]), "(true: 2)\n")
cat("  beta_num[2]:", mean(draws_A[,"beta_num[2]"]), "+/-", sd(draws_A[,"beta_num[2]"]), "(true: 0.3)\n")
cat("  shape_num:", mean(draws_A[,"shape_num"]), "+/-", sd(draws_A[,"shape_num"]), "(true: 5)\n")

cat("\n=== Test 3: Tape autodiff (A_t) ===\n")
fit_At <- tratio(
  y | denom ~ x,
  data = df,
  family = ratiod_gamma_gamma(),
  control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "A_t", verbose = FALSE)
)

draws_At <- as.matrix(fit_At$draws)
cat("Tape AD (A_t):\n")
cat("  beta_num[1]:", mean(draws_At[,"beta_num[1]"]), "+/-", sd(draws_At[,"beta_num[1]"]), "(true: 2)\n")
cat("  beta_num[2]:", mean(draws_At[,"beta_num[2]"]), "+/-", sd(draws_At[,"beta_num[2]"]), "(true: 0.3)\n")
cat("  shape_num:", mean(draws_At[,"shape_num"]), "+/-", sd(draws_At[,"shape_num"]), "(true: 5)\n")

cat("\n=== ESS Comparison ===\n")
cat("ESS Numerical:\n")
cat("  beta_num[1]:", posterior::ess_basic(draws_N[,"beta_num[1]"]), "\n")
cat("  beta_num[2]:", posterior::ess_basic(draws_N[,"beta_num[2]"]), "\n")
cat("  shape_num:", posterior::ess_basic(draws_N[,"shape_num"]), "\n")

cat("ESS Forward AD:\n")
cat("  beta_num[1]:", posterior::ess_basic(draws_A[,"beta_num[1]"]), "\n")
cat("  beta_num[2]:", posterior::ess_basic(draws_A[,"beta_num[2]"]), "\n")
cat("  shape_num:", posterior::ess_basic(draws_A[,"shape_num"]), "\n")

cat("ESS Tape AD:\n")
cat("  beta_num[1]:", posterior::ess_basic(draws_At[,"beta_num[1]"]), "\n")
cat("  beta_num[2]:", posterior::ess_basic(draws_At[,"beta_num[2]"]), "\n")
cat("  shape_num:", posterior::ess_basic(draws_At[,"shape_num"]), "\n")
