# Test H gradients for GAMMA_GAMMA, LOGNORMAL, BETA_BINOMIAL

library(numdenom)

set.seed(42)

# ==============================================================================
# Test 1: GAMMA_GAMMA
# ==============================================================================
cat("=== Test 1: GAMMA_GAMMA ===\n")

N <- 100
x <- rnorm(N)
y_num <- rgamma(N, shape = 5, rate = 5 / exp(2 + 0.3 * x))
y_denom <- rgamma(N, shape = 8, rate = 8 / exp(3 + 0.2 * x))

df_gg <- data.frame(y = y_num, denom = y_denom, x = x)

# Test all gradient modes
cat("\nGAMMA_GAMMA - Numerical (N):\n")
fit_N <- ratiod(
  y | denom ~ x,
  data = df_gg,
  family = ratiod_gamma_gamma(),
  iter = 500, warmup = 250, chains = 1,
  gradient_mode = "N",
  verbose = FALSE
)
draws_N <- as.matrix(fit_N$draws)
cat("  beta_num[1]:", mean(draws_N[,"beta_num[1]"]), "+/-", sd(draws_N[,"beta_num[1]"]), "(true: 2)\n")
cat("  beta_num[2]:", mean(draws_N[,"beta_num[2]"]), "+/-", sd(draws_N[,"beta_num[2]"]), "(true: 0.3)\n")

cat("\nGAMMA_GAMMA - Forward AD (A):\n")
fit_A <- ratiod(
  y | denom ~ x,
  data = df_gg,
  family = ratiod_gamma_gamma(),
  iter = 500, warmup = 250, chains = 1,
  gradient_mode = "A",
  verbose = FALSE
)
draws_A <- as.matrix(fit_A$draws)
cat("  beta_num[1]:", mean(draws_A[,"beta_num[1]"]), "+/-", sd(draws_A[,"beta_num[1]"]), "(true: 2)\n")
cat("  beta_num[2]:", mean(draws_A[,"beta_num[2]"]), "+/-", sd(draws_A[,"beta_num[2]"]), "(true: 0.3)\n")

cat("\nGAMMA_GAMMA - Hand-coded (H):\n")
fit_H <- ratiod(
  y | denom ~ x,
  data = df_gg,
  family = ratiod_gamma_gamma(),
  iter = 500, warmup = 250, chains = 1,
  gradient_mode = "H",
  verbose = FALSE
)
draws_H <- as.matrix(fit_H$draws)
cat("  beta_num[1]:", mean(draws_H[,"beta_num[1]"]), "+/-", sd(draws_H[,"beta_num[1]"]), "(true: 2)\n")
cat("  beta_num[2]:", mean(draws_H[,"beta_num[2]"]), "+/-", sd(draws_H[,"beta_num[2]"]), "(true: 0.3)\n")

# ==============================================================================
# Test 2: LOGNORMAL
# ==============================================================================
cat("\n=== Test 2: LOGNORMAL ===\n")

y_num_ln <- rlnorm(N, meanlog = 2 + 0.3 * x, sdlog = 0.5)
y_denom_ln <- rlnorm(N, meanlog = 3 + 0.2 * x, sdlog = 0.4)

df_ln <- data.frame(y = y_num_ln, denom = y_denom_ln, x = x)

cat("\nLOGNORMAL - Numerical (N):\n")
fit_N <- ratiod(
  y | denom ~ x,
  data = df_ln,
  family = ratiod_lognormal(),
  iter = 500, warmup = 250, chains = 1,
  gradient_mode = "N",
  verbose = FALSE
)
draws_N <- as.matrix(fit_N$draws)
cat("  beta_num[1]:", mean(draws_N[,"beta_num[1]"]), "+/-", sd(draws_N[,"beta_num[1]"]), "(true: 2)\n")
cat("  beta_num[2]:", mean(draws_N[,"beta_num[2]"]), "+/-", sd(draws_N[,"beta_num[2]"]), "(true: 0.3)\n")

cat("\nLOGNORMAL - Forward AD (A):\n")
fit_A <- ratiod(
  y | denom ~ x,
  data = df_ln,
  family = ratiod_lognormal(),
  iter = 500, warmup = 250, chains = 1,
  gradient_mode = "A",
  verbose = FALSE
)
draws_A <- as.matrix(fit_A$draws)
cat("  beta_num[1]:", mean(draws_A[,"beta_num[1]"]), "+/-", sd(draws_A[,"beta_num[1]"]), "(true: 2)\n")
cat("  beta_num[2]:", mean(draws_A[,"beta_num[2]"]), "+/-", sd(draws_A[,"beta_num[2]"]), "(true: 0.3)\n")

cat("\nLOGNORMAL - Hand-coded (H):\n")
fit_H <- ratiod(
  y | denom ~ x,
  data = df_ln,
  family = ratiod_lognormal(),
  iter = 500, warmup = 250, chains = 1,
  gradient_mode = "H",
  verbose = FALSE
)
draws_H <- as.matrix(fit_H$draws)
cat("  beta_num[1]:", mean(draws_H[,"beta_num[1]"]), "+/-", sd(draws_H[,"beta_num[1]"]), "(true: 2)\n")
cat("  beta_num[2]:", mean(draws_H[,"beta_num[2]"]), "+/-", sd(draws_H[,"beta_num[2]"]), "(true: 0.3)\n")

# ==============================================================================
# Test 3: BETA_BINOMIAL
# ==============================================================================
cat("\n=== Test 3: BETA_BINOMIAL ===\n")

# Generate beta-binomial data
n_trials <- rep(50, N)
prob <- plogis(0.5 + 0.8 * x)  # logit(p) = 0.5 + 0.8*x
phi_bb <- 10  # concentration parameter

# Use beta-binomial generation: alpha = p*phi, beta = (1-p)*phi
alpha <- prob * phi_bb
beta_param <- (1 - prob) * phi_bb
y_bb <- rbinom(N, n_trials, rbeta(N, alpha, beta_param))

df_bb <- data.frame(y = y_bb, n = n_trials, x = x)

cat("\nBETA_BINOMIAL - Numerical (N):\n")
fit_N <- ratiod(
  y | n ~ x,
  data = df_bb,
  family = ratiod_beta_binomial(),
  iter = 500, warmup = 250, chains = 1,
  gradient_mode = "N",
  verbose = FALSE
)
draws_N <- as.matrix(fit_N$draws)
cat("  beta_num[1]:", mean(draws_N[,"beta_num[1]"]), "+/-", sd(draws_N[,"beta_num[1]"]), "(true: 0.5)\n")
cat("  beta_num[2]:", mean(draws_N[,"beta_num[2]"]), "+/-", sd(draws_N[,"beta_num[2]"]), "(true: 0.8)\n")

cat("\nBETA_BINOMIAL - Forward AD (A):\n")
fit_A <- ratiod(
  y | n ~ x,
  data = df_bb,
  family = ratiod_beta_binomial(),
  iter = 500, warmup = 250, chains = 1,
  gradient_mode = "A",
  verbose = FALSE
)
draws_A <- as.matrix(fit_A$draws)
cat("  beta_num[1]:", mean(draws_A[,"beta_num[1]"]), "+/-", sd(draws_A[,"beta_num[1]"]), "(true: 0.5)\n")
cat("  beta_num[2]:", mean(draws_A[,"beta_num[2]"]), "+/-", sd(draws_A[,"beta_num[2]"]), "(true: 0.8)\n")

cat("\nBETA_BINOMIAL - Hand-coded (H):\n")
fit_H <- ratiod(
  y | n ~ x,
  data = df_bb,
  family = ratiod_beta_binomial(),
  iter = 500, warmup = 250, chains = 1,
  gradient_mode = "H",
  verbose = FALSE
)
draws_H <- as.matrix(fit_H$draws)
cat("  beta_num[1]:", mean(draws_H[,"beta_num[1]"]), "+/-", sd(draws_H[,"beta_num[1]"]), "(true: 0.5)\n")
cat("  beta_num[2]:", mean(draws_H[,"beta_num[2]"]), "+/-", sd(draws_H[,"beta_num[2]"]), "(true: 0.8)\n")

cat("\n=== DONE ===\n")
