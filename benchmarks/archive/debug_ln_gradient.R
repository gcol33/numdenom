# Direct gradient comparison for lognormal
# Compare H-mode and A-mode gradients at the same parameter values

library(numdenom)
set.seed(42)

# Simple lognormal model
N_OBS <- 20
x <- rnorm(N_OBS)
eta_num <- 2 + 0.3 * x
eta_denom <- 4 + 0.2 * x

df <- data.frame(
  y = rlnorm(N_OBS, eta_num, 0.5),
  denom = rlnorm(N_OBS, eta_denom, 0.5),
  x = x
)

# Fit one iteration just to get data structures set up correctly
cat("Fitting with A-mode to get initial params...\n")
fit_A <- tratio(
  y | denom ~ x, data = df,
  family = ratiod_lognormal(),
  control = list(iter = 10, warmup = 5, chains = 1, gradient_mode = "A", verbose = FALSE)
)

# Get the final state from A-mode
draws_A <- as.matrix(fit_A$draws)
final_row <- draws_A[nrow(draws_A), ]
cat("Final A-mode params:\n")
print(final_row)

# Now fit with H-mode starting from same initialization
cat("\nFitting with H-mode...\n")
fit_H <- tratio(
  y | denom ~ x, data = df,
  family = ratiod_lognormal(),
  control = list(iter = 10, warmup = 5, chains = 1, gradient_mode = "H", verbose = FALSE)
)

draws_H <- as.matrix(fit_H$draws)
final_row_H <- draws_H[nrow(draws_H), ]
cat("Final H-mode params:\n")
print(final_row_H)

# Check if HMC dynamics are different
cat("\n=== Trajectory Comparison ===\n")
cat("First 5 iterations (each mode):\n")
cat("A-mode beta_num[1]:", draws_A[1:5, "beta_num[1]"], "\n")
cat("H-mode beta_num[1]:", draws_H[1:5, "beta_num[1]"], "\n")

# The trajectories should be identical if gradients are correct
# (given same seed and initialization)

# Try checking if the issue is that lognormal falls through to wrong branch
cat("\n=== Checking gradient mode dispatch ===\n")

# Run with explicit modes to ensure we're testing what we think
fit_A2 <- tratio(
  y | denom ~ x, data = df,
  family = ratiod_lognormal(),
  # verbose = TRUE should show which gradient mode is used
  control = list(iter = 50, warmup = 25, chains = 1, gradient_mode = "A",
                 verbose = TRUE)
)

cat("\n")

fit_H2 <- tratio(
  y | denom ~ x, data = df,
  family = ratiod_lognormal(),
  control = list(iter = 50, warmup = 25, chains = 1, gradient_mode = "H", verbose = TRUE)
)

# Compare posteriors
cat("\n=== Posterior Comparison ===\n")
draws_A2 <- as.matrix(fit_A2$draws)
draws_H2 <- as.matrix(fit_H2$draws)

params <- c("beta_num[1]", "beta_num[2]", "beta_denom[1]", "beta_denom[2]",
            "sigma_num", "sigma_denom")

for (p in params) {
  cat(sprintf("%s: A=%.4f (SD=%.4f), H=%.4f (SD=%.4f)\n",
              p, mean(draws_A2[,p]), sd(draws_A2[,p]),
              mean(draws_H2[,p]), sd(draws_H2[,p])))
}
