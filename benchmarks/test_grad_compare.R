# Compare gradients directly for LOGNORMAL

library(numdenom)

# Create a minimal test case
set.seed(123)
N <- 10

y_num <- rlnorm(N, meanlog = 2, sdlog = 0.5)
y_denom <- rlnorm(N, meanlog = 3, sdlog = 0.4)
df <- data.frame(y = y_num, denom = y_denom)

cat("Data:\n")
cat("  y_num:", head(y_num), "\n")
cat("  y_denom:", head(y_denom), "\n")

# Run with verbose and very few iterations to see what's happening
cat("\n=== A_t mode (single step) ===\n")
fit_At <- ratiod(
  y | denom ~ 1,
  data = df,
  family = ratiod_lognormal(),
  iter = 3, warmup = 1, chains = 1,
  gradient_mode = "A_t",
  verbose = TRUE
)

cat("\n=== H mode (single step) ===\n")
fit_H <- ratiod(
  y | denom ~ 1,
  data = df,
  family = ratiod_lognormal(),
  iter = 3, warmup = 1, chains = 1,
  gradient_mode = "H",
  verbose = TRUE
)

# Compare draws
draws_At <- as.matrix(fit_At$draws)
draws_H <- as.matrix(fit_H$draws)

cat("\n=== Draw comparison ===\n")
cat("A_t draws:\n")
print(draws_At)
cat("\nH draws:\n")
print(draws_H)

# Run longer to see acceptance rates
cat("\n=== Longer runs ===\n")

cat("\nA_t mode (100 iter):\n")
fit_At_long <- ratiod(
  y | denom ~ 1,
  data = df,
  family = ratiod_lognormal(),
  iter = 100, warmup = 50, chains = 1,
  gradient_mode = "A_t",
  verbose = FALSE
)
draws_At_long <- as.matrix(fit_At_long$draws)
cat("  beta_num[1] range:", range(draws_At_long[,"beta_num[1]"]), "\n")
cat("  sigma_num range:", range(draws_At_long[,"sigma_num"]), "\n")

cat("\nH mode (100 iter):\n")
fit_H_long <- ratiod(
  y | denom ~ 1,
  data = df,
  family = ratiod_lognormal(),
  iter = 100, warmup = 50, chains = 1,
  gradient_mode = "H",
  verbose = FALSE
)
draws_H_long <- as.matrix(fit_H_long$draws)
cat("  beta_num[1] range:", range(draws_H_long[,"beta_num[1]"]), "\n")
cat("  sigma_num range:", range(draws_H_long[,"sigma_num"]), "\n")

cat("\n=== DONE ===\n")
