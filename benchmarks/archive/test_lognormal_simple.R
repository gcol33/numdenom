# Simpler test of LOGNORMAL with intercept only

library(numdenom)

set.seed(123)

# ==============================================================================
# Test LOGNORMAL - intercept only (simpler case)
# ==============================================================================
cat("=== Testing LOGNORMAL intercept-only model ===\n")

N <- 200

# Generate data: log(y) ~ N(mu, sigma^2)
# True: mu_num = 2, sigma_num = 0.5
#       mu_denom = 3, sigma_denom = 0.4
y_num <- rlnorm(N, meanlog = 2, sdlog = 0.5)
y_denom <- rlnorm(N, meanlog = 3, sdlog = 0.4)

df <- data.frame(y = y_num, denom = y_denom)

cat("\nSample means of log(y):\n")
cat("  mean(log(y_num)):", mean(log(y_num)), " (true: 2)\n")
cat("  mean(log(y_denom)):", mean(log(y_denom)), " (true: 3)\n")
cat("  sd(log(y_num)):", sd(log(y_num)), " (true: 0.5)\n")
cat("  sd(log(y_denom)):", sd(log(y_denom)), " (true: 0.4)\n")

cat("\n=== Forward AD (A) ===\n")
fit_A <- tratio(
  y | denom ~ 1,  # intercept only
  data = df,
  family = ratiod_lognormal(),
  control = list(iter = 1000, warmup = 500, chains = 1, gradient_mode = "A", verbose = FALSE)
)
draws_A <- as.matrix(fit_A$draws)
cat("  beta_num[1]:", mean(draws_A[,"beta_num[1]"]), "+/-", sd(draws_A[,"beta_num[1]"]), "(true: 2)\n")
cat("  beta_denom[1]:", mean(draws_A[,"beta_denom[1]"]), "+/-", sd(draws_A[,"beta_denom[1]"]), "(true: 3)\n")
cat("  sigma_num:", mean(draws_A[,"sigma_num"]), "+/-", sd(draws_A[,"sigma_num"]), "(true: 0.5)\n")
cat("  sigma_denom:", mean(draws_A[,"sigma_denom"]), "+/-", sd(draws_A[,"sigma_denom"]), "(true: 0.4)\n")

cat("\n=== Tape AD (A_t) ===\n")
fit_At <- tratio(
  y | denom ~ 1,  # intercept only
  data = df,
  family = ratiod_lognormal(),
  control = list(iter = 1000, warmup = 500, chains = 1, gradient_mode = "A_t", verbose = FALSE)
)
draws_At <- as.matrix(fit_At$draws)
cat("  beta_num[1]:", mean(draws_At[,"beta_num[1]"]), "+/-", sd(draws_At[,"beta_num[1]"]), "(true: 2)\n")
cat("  beta_denom[1]:", mean(draws_At[,"beta_denom[1]"]), "+/-", sd(draws_At[,"beta_denom[1]"]), "(true: 3)\n")
cat("  sigma_num:", mean(draws_At[,"sigma_num"]), "+/-", sd(draws_At[,"sigma_num"]), "(true: 0.5)\n")
cat("  sigma_denom:", mean(draws_At[,"sigma_denom"]), "+/-", sd(draws_At[,"sigma_denom"]), "(true: 0.4)\n")

cat("\n=== Hand-coded (H) ===\n")
fit_H <- tratio(
  y | denom ~ 1,  # intercept only
  data = df,
  family = ratiod_lognormal(),
  control = list(iter = 1000, warmup = 500, chains = 1, gradient_mode = "H", verbose = FALSE)
)
draws_H <- as.matrix(fit_H$draws)
cat("  beta_num[1]:", mean(draws_H[,"beta_num[1]"]), "+/-", sd(draws_H[,"beta_num[1]"]), "(true: 2)\n")
cat("  beta_denom[1]:", mean(draws_H[,"beta_denom[1]"]), "+/-", sd(draws_H[,"beta_denom[1]"]), "(true: 3)\n")
cat("  sigma_num:", mean(draws_H[,"sigma_num"]), "+/-", sd(draws_H[,"sigma_num"]), "(true: 0.5)\n")
cat("  sigma_denom:", mean(draws_H[,"sigma_denom"]), "+/-", sd(draws_H[,"sigma_denom"]), "(true: 0.4)\n")

cat("\n=== DONE ===\n")
