# Debug lognormal gradient at C++ level
# Compare A-mode and H-mode gradient values directly

library(numdenom)
set.seed(42)

# Very simple case - just 5 obs, no RE
N_OBS <- 5
x <- rnorm(N_OBS)
eta_num <- 2 + 0.3 * x
eta_denom <- 4 + 0.2 * x

df <- data.frame(
  y = rlnorm(N_OBS, eta_num, 0.5),
  denom = rlnorm(N_OBS, eta_denom, 0.5),
  x = x
)

cat("Data:\n")
print(df)

# Run just 2 iterations to see initial gradient behavior
cat("\n=== Running A-mode (2 iter) ===\n")
fit_A <- tratio(
  y | denom ~ x, data = df,
  family = ratiod_lognormal(),
  control = list(iter = 2, warmup = 1, chains = 1, gradient_mode = "A", verbose = TRUE)
)

cat("\n=== Running H-mode (2 iter) ===\n")
fit_H <- tratio(
  y | denom ~ x, data = df,
  family = ratiod_lognormal(),
  control = list(iter = 2, warmup = 1, chains = 1, gradient_mode = "H", verbose = TRUE)
)

# Compare draws
draws_A <- as.matrix(fit_A$draws)
draws_H <- as.matrix(fit_H$draws)

cat("\n=== Draw comparison ===\n")
cat("A-mode draws:\n")
print(draws_A)
cat("\nH-mode draws:\n")
print(draws_H)

# Now let's check if the dispatch is actually working
cat("\n=== Checking which mode is actually used ===\n")

# Run with longer chains to see if posteriors converge
cat("\nRunning 200 iter comparison...\n")
set.seed(123)
fit_A_long <- tratio(
  y | denom ~ x, data = df,
  family = ratiod_lognormal(),
  control = list(iter = 200, warmup = 100, chains = 1, gradient_mode = "A", verbose = FALSE)
)

set.seed(123)  # Same seed
fit_H_long <- tratio(
  y | denom ~ x, data = df,
  family = ratiod_lognormal(),
  control = list(iter = 200, warmup = 100, chains = 1, gradient_mode = "H", verbose = FALSE)
)

draws_A_long <- as.matrix(fit_A_long$draws)
draws_H_long <- as.matrix(fit_H_long$draws)

params <- c("beta_num[1]", "beta_num[2]", "beta_denom[1]", "beta_denom[2]",
            "sigma_num", "sigma_denom")

cat("\nPosterior comparison (200 iter):\n")
for (p in params) {
  mean_A <- mean(draws_A_long[, p])
  mean_H <- mean(draws_H_long[, p])
  sd_A <- sd(draws_A_long[, p])
  se <- sd_A / sqrt(nrow(draws_A_long))
  diff_se <- abs(mean_A - mean_H) / se
  cat(sprintf("  %s: A=%.4f, H=%.4f (diff=%.4f, %.1f SE)\n",
              p, mean_A, mean_H, mean_A - mean_H, diff_se))
}

# Check first iteration specifically - should be identical if gradients match
cat("\n=== First iteration values ===\n")
cat("A-mode first row:", draws_A_long[1, ], "\n")
cat("H-mode first row:", draws_H_long[1, ], "\n")

# Try with numerical gradient as reference
cat("\n=== Numerical gradient comparison ===\n")
set.seed(123)
fit_N <- tratio(
  y | denom ~ x, data = df,
  family = ratiod_lognormal(),
  control = list(iter = 200, warmup = 100, chains = 1, gradient_mode = "N", verbose = FALSE)
)
draws_N <- as.matrix(fit_N$draws)

cat("\nN-mode first row:", draws_N[1, ], "\n")

cat("\nN vs A comparison:\n")
for (p in params) {
  mean_N <- mean(draws_N[, p])
  mean_A <- mean(draws_A_long[, p])
  sd_N <- sd(draws_N[, p])
  se <- sd_N / sqrt(nrow(draws_N))
  diff_se <- abs(mean_N - mean_A) / se
  cat(sprintf("  %s: N=%.4f, A=%.4f (diff=%.4f, %.1f SE)\n",
              p, mean_N, mean_A, mean_N - mean_A, diff_se))
}

cat("\nN vs H comparison:\n")
for (p in params) {
  mean_N <- mean(draws_N[, p])
  mean_H <- mean(draws_H_long[, p])
  sd_N <- sd(draws_N[, p])
  se <- sd_N / sqrt(nrow(draws_N))
  diff_se <- abs(mean_N - mean_H) / se
  cat(sprintf("  %s: N=%.4f, H=%.4f (diff=%.4f, %.1f SE)\n",
              p, mean_N, mean_H, mean_N - mean_H, diff_se))
}
