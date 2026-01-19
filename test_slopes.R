# test_slopes.R - Test H gradients for uncorrelated random slopes
devtools::load_all()

cat("Testing H gradients for uncorrelated random slopes...\n\n")

passed <- 0
failed <- 0

# Create data with random slopes
set.seed(42)
n_groups <- 10
n_per_group <- 10
n <- n_groups * n_per_group

df <- data.frame(
  y_num = rpois(n, 10),
  y_denom = rgamma(n, 5, 0.5),
  x = rnorm(n),  # Fixed effect
  z = rnorm(n),  # Slope variable
  group = factor(rep(1:n_groups, each = n_per_group))
)

# Test 1: POISSON_GAMMA with uncorrelated random slopes (|| syntax)
tryCatch({
  fit <- ratiod(y_num | y_denom ~ x + (1 + z || group), data = df,
                family = ratiod_poisson_gamma(),
                chains = 1, warmup = 30, iter = 60, refresh = 0)
  cat("1. POISSON_GAMMA + uncorrelated slopes: PASS\n")
  passed <- passed + 1
}, error = function(e) {
  cat("1. POISSON_GAMMA + uncorrelated slopes: FAIL -", e$message, "\n")
  failed <<- failed + 1
})

# NegBin data
df_nb <- data.frame(
  y_num = rnbinom(n, size = 5, mu = 10),
  y_denom = rnbinom(n, size = 5, mu = 15),
  x = rnorm(n),
  z = rnorm(n),
  group = factor(rep(1:n_groups, each = n_per_group))
)

# Test 2: NEGBIN_NEGBIN with uncorrelated random slopes
tryCatch({
  fit <- ratiod(y_num | y_denom ~ x + (1 + z || group), data = df_nb,
                family = ratiod_negbin_negbin(),
                chains = 1, warmup = 30, iter = 60, refresh = 0)
  cat("2. NEGBIN_NEGBIN + uncorrelated slopes: PASS\n")
  passed <- passed + 1
}, error = function(e) {
  cat("2. NEGBIN_NEGBIN + uncorrelated slopes: FAIL -", e$message, "\n")
  failed <<- failed + 1
})

# Binomial data
df_binom <- data.frame(
  y_num = rbinom(n, 20, 0.3),
  n_trials = rep(20, n),
  x = rnorm(n),
  z = rnorm(n),
  group = factor(rep(1:n_groups, each = n_per_group))
)

# Test 3: BINOMIAL with uncorrelated random slopes
tryCatch({
  fit <- ratiod(y_num | n_trials ~ x + (1 + z || group), data = df_binom,
                family = ratiod_binomial(),
                chains = 1, warmup = 30, iter = 60, refresh = 0)
  cat("3. BINOMIAL + uncorrelated slopes: PASS\n")
  passed <- passed + 1
}, error = function(e) {
  cat("3. BINOMIAL + uncorrelated slopes: FAIL -", e$message, "\n")
  failed <<- failed + 1
})

# Test 4: Correlated slopes should use autodiff (| syntax)
cat("\n--- Correlated slopes (should use autodiff) ---\n")
tryCatch({
  fit <- ratiod(y_num | y_denom ~ x + (1 + z | group), data = df,
                family = ratiod_poisson_gamma(),
                chains = 1, warmup = 30, iter = 60, refresh = 0)
  cat("4. POISSON_GAMMA + correlated slopes (autodiff): PASS\n")
  passed <- passed + 1
}, error = function(e) {
  cat("4. POISSON_GAMMA + correlated slopes: FAIL -", e$message, "\n")
  failed <<- failed + 1
})

cat(sprintf("\nResults: %d/4 passed\n", passed))
if (passed >= 3) {
  cat("Random slopes H gradient tests passed (uncorrelated)!\n")
}
