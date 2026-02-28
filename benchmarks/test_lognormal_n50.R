# Test LOGNORMAL with N=50 (same as earlier failing case)

library(numdenom)

set.seed(42)  # Same seed as failing case
N <- 50

y_num <- rlnorm(N, meanlog = 2, sdlog = 0.5)
y_denom <- rlnorm(N, meanlog = 3, sdlog = 0.4)
df <- data.frame(y = y_num, denom = y_denom)

cat("Sample statistics:\n")
cat("  mean(log(y_num)):", mean(log(y_num)), " (true: 2)\n")
cat("  mean(log(y_denom)):", mean(log(y_denom)), " (true: 3)\n")
cat("  sd(log(y_num)):", sd(log(y_num)), " (true: 0.5)\n")

cat("\n=== A_t mode ===\n")
fit_At <- ratiod(
  y | denom ~ 1,
  data = df,
  family = ratiod_lognormal(),
  iter = 200, warmup = 100, chains = 1,
  gradient_mode = "A_t",
  verbose = FALSE
)
draws_At <- as.matrix(fit_At$draws)
cat("  beta_num[1]:", mean(draws_At[,"beta_num[1]"]), "+/-", sd(draws_At[,"beta_num[1]"]), "\n")
cat("  beta_num[1] range:", range(draws_At[,"beta_num[1]"]), "\n")
cat("  sigma_num:", mean(draws_At[,"sigma_num"]), "+/-", sd(draws_At[,"sigma_num"]), "\n")

cat("\n=== H mode ===\n")
fit_H <- ratiod(
  y | denom ~ 1,
  data = df,
  family = ratiod_lognormal(),
  iter = 200, warmup = 100, chains = 1,
  gradient_mode = "H",
  verbose = FALSE
)
draws_H <- as.matrix(fit_H$draws)
cat("  beta_num[1]:", mean(draws_H[,"beta_num[1]"]), "+/-", sd(draws_H[,"beta_num[1]"]), "\n")
cat("  beta_num[1] range:", range(draws_H[,"beta_num[1]"]), "\n")
cat("  sigma_num:", mean(draws_H[,"sigma_num"]), "+/-", sd(draws_H[,"sigma_num"]), "\n")

# Check if draws are stuck
cat("\n=== First 10 draws ===\n")
cat("A_t beta_num[1]:", head(draws_At[,"beta_num[1]"], 10), "\n")
cat("H beta_num[1]:", head(draws_H[,"beta_num[1]"], 10), "\n")

cat("\n=== DONE ===\n")
