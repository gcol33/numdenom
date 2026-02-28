# Compare numdenom with direct MLE for LOGNORMAL

library(numdenom)

set.seed(42)
N <- 50

y_num <- rlnorm(N, meanlog = 2, sdlog = 0.5)
y_denom <- rlnorm(N, meanlog = 3, sdlog = 0.4)
df <- data.frame(y = y_num, denom = y_denom)

cat("=== MLE estimates (direct calculation) ===\n")
# For lognormal, MLE of mu is mean(log(y)), MLE of sigma is SD(log(y))
log_y_num <- log(y_num)
log_y_denom <- log(y_denom)

mu_mle_num <- mean(log_y_num)
sigma_mle_num <- sd(log_y_num) * sqrt((N-1)/N)  # MLE uses N, not N-1

mu_mle_denom <- mean(log_y_denom)
sigma_mle_denom <- sd(log_y_denom) * sqrt((N-1)/N)

cat("Numerator:\n")
cat("  mu_mle:", mu_mle_num, " (true: 2)\n")
cat("  sigma_mle:", sigma_mle_num, " (true: 0.5)\n")

cat("\nDenominator:\n")
cat("  mu_mle:", mu_mle_denom, " (true: 3)\n")
cat("  sigma_mle:", sigma_mle_denom, " (true: 0.4)\n")

cat("\n=== A_t mode ===\n")
fit_At <- ratiod(
  y | denom ~ 1,
  data = df,
  family = ratiod_lognormal(),
  iter = 1000, warmup = 500, chains = 1,
  gradient_mode = "A_t",
  verbose = FALSE
)
draws_At <- as.matrix(fit_At$draws)
cat("  beta_num[1]:", mean(draws_At[,"beta_num[1]"]), " (MLE:", mu_mle_num, ")\n")
cat("  sigma_num:", mean(draws_At[,"sigma_num"]), " (MLE:", sigma_mle_num, ")\n")
cat("  beta_denom[1]:", mean(draws_At[,"beta_denom[1]"]), " (MLE:", mu_mle_denom, ")\n")
cat("  sigma_denom:", mean(draws_At[,"sigma_denom"]), " (MLE:", sigma_mle_denom, ")\n")

cat("\n=== H mode ===\n")
fit_H <- ratiod(
  y | denom ~ 1,
  data = df,
  family = ratiod_lognormal(),
  iter = 1000, warmup = 500, chains = 1,
  gradient_mode = "H",
  verbose = FALSE
)
draws_H <- as.matrix(fit_H$draws)
cat("  beta_num[1]:", mean(draws_H[,"beta_num[1]"]), " (MLE:", mu_mle_num, ")\n")
cat("  sigma_num:", mean(draws_H[,"sigma_num"]), " (MLE:", sigma_mle_num, ")\n")
cat("  beta_denom[1]:", mean(draws_H[,"beta_denom[1]"]), " (MLE:", mu_mle_denom, ")\n")
cat("  sigma_denom:", mean(draws_H[,"sigma_denom"]), " (MLE:", sigma_mle_denom, ")\n")

cat("\n=== DONE ===\n")
