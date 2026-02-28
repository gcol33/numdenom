# test_zi.R - Test H gradients for ZI and Hurdle models
devtools::load_all()

cat("Testing H gradients for ZI/Hurdle models...\n\n")

passed <- 0
failed <- 0

# Create data with excess zeros
set.seed(42)
n <- 50

# Generate zero-inflated data
zi_prob <- 0.3  # 30% structural zeros
count_rate <- 5

y_zi <- sapply(1:n, function(i) {
  if (runif(1) < zi_prob) return(0)
  rpois(1, count_rate)
})

df <- data.frame(
  y_num = y_zi,
  y_denom = rgamma(n, 5, 0.5),
  x = rnorm(n)
)

# Test 1: POISSON_GAMMA with ZI-Poisson
tryCatch({
  fit <- ratiod(y_num | y_denom ~ x, data = df,
                family = ratiod_poisson_gamma(),
                zi = zi_poisson(),
                chains = 1, warmup = 30, iter = 60, refresh = 0)
  cat("1. POISSON_GAMMA + ZI-Poisson: PASS\n")
  passed <- passed + 1
}, error = function(e) {
  cat("1. POISSON_GAMMA + ZI-Poisson: FAIL -", e$message, "\n")
  failed <<- failed + 1
})

# Test 2: POISSON_GAMMA with Hurdle-Poisson
tryCatch({
  fit <- ratiod(y_num | y_denom ~ x, data = df,
                family = ratiod_poisson_gamma(),
                zi = hurdle_poisson(),
                chains = 1, warmup = 30, iter = 60, refresh = 0)
  cat("2. POISSON_GAMMA + Hurdle-Poisson: PASS\n")
  passed <- passed + 1
}, error = function(e) {
  cat("2. POISSON_GAMMA + Hurdle-Poisson: FAIL -", e$message, "\n")
  failed <<- failed + 1
})

# Create data for NegBin-NegBin
df_nb <- data.frame(
  y_num = y_zi,
  y_denom = rnbinom(n, size = 5, mu = 15),
  x = rnorm(n)
)

# Test 3: NEGBIN_NEGBIN with ZI-NegBin
tryCatch({
  fit <- ratiod(y_num | y_denom ~ x, data = df_nb,
                family = ratiod_negbin_negbin(),
                zi = zi_negbin(),
                chains = 1, warmup = 30, iter = 60, refresh = 0)
  cat("3. NEGBIN_NEGBIN + ZI-NegBin: PASS\n")
  passed <- passed + 1
}, error = function(e) {
  cat("3. NEGBIN_NEGBIN + ZI-NegBin: FAIL -", e$message, "\n")
  failed <<- failed + 1
})

# Test 4: NEGBIN_NEGBIN with Hurdle-NegBin
tryCatch({
  fit <- ratiod(y_num | y_denom ~ x, data = df_nb,
                family = ratiod_negbin_negbin(),
                zi = hurdle_negbin(),
                chains = 1, warmup = 30, iter = 60, refresh = 0)
  cat("4. NEGBIN_NEGBIN + Hurdle-NegBin: PASS\n")
  passed <- passed + 1
}, error = function(e) {
  cat("4. NEGBIN_NEGBIN + Hurdle-NegBin: FAIL -", e$message, "\n")
  failed <<- failed + 1
})

cat(sprintf("\nResults: %d/4 passed\n", passed))
if (passed == 4) {
  cat("All ZI/Hurdle H gradient tests passed!\n")
}
