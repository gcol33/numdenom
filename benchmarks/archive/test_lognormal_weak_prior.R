# Test LOGNORMAL with weak prior on sigma

library(numdenom)

set.seed(42)
N <- 50

y_num <- rlnorm(N, meanlog = 2, sdlog = 0.5)
y_denom <- rlnorm(N, meanlog = 3, sdlog = 0.4)
df <- data.frame(y = y_num, denom = y_denom)

cat("Sample statistics:\n")
cat("  mean(log(y_num)):", mean(log(y_num)), " (true: 2)\n")
cat("  sd(log(y_num)):", sd(log(y_num)), " (true: 0.5)\n")

# Use very weak prior on phi (Gamma(0.5, 0.01) - nearly flat)
weak_prior <- list(phi_shape = 0.5, phi_rate = 0.01)

cat("\n=== A_t mode (weak prior) ===\n")
fit_At <- ratiod(
  y | denom ~ 1,
  data = df,
  family = ratiod_lognormal(),
  priors = weak_prior,
  iter = 500, warmup = 250, chains = 1,
  gradient_mode = "A_t",
  verbose = FALSE
)
draws_At <- as.matrix(fit_At$draws)
cat("  beta_num[1]:", mean(draws_At[,"beta_num[1]"]), "+/-", sd(draws_At[,"beta_num[1]"]), " (true: 2)\n")
cat("  sigma_num:", mean(draws_At[,"sigma_num"]), "+/-", sd(draws_At[,"sigma_num"]), " (true: 0.5)\n")

cat("\n=== H mode (weak prior) ===\n")
fit_H <- ratiod(
  y | denom ~ 1,
  data = df,
  family = ratiod_lognormal(),
  priors = weak_prior,
  iter = 500, warmup = 250, chains = 1,
  gradient_mode = "H",
  verbose = FALSE
)
draws_H <- as.matrix(fit_H$draws)
cat("  beta_num[1]:", mean(draws_H[,"beta_num[1]"]), "+/-", sd(draws_H[,"beta_num[1]"]), " (true: 2)\n")
cat("  sigma_num:", mean(draws_H[,"sigma_num"]), "+/-", sd(draws_H[,"sigma_num"]), " (true: 0.5)\n")

# Compare directly
cat("\n=== Comparison ===\n")
cat("A_t vs H difference (beta_num[1]):", abs(mean(draws_At[,"beta_num[1]"]) - mean(draws_H[,"beta_num[1]"])), "\n")
cat("A_t vs H difference (sigma_num):", abs(mean(draws_At[,"sigma_num"]) - mean(draws_H[,"sigma_num"])), "\n")

cat("\n=== DONE ===\n")
