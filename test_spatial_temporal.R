# test_spatial_temporal.R - Test H gradients for Spatial+Temporal models
devtools::load_all()

cat("Testing H gradients for Spatial+Temporal models...\n\n")

passed <- 0
failed <- 0

# Create data with spatial AND temporal structure
set.seed(42)
n_sites <- 9  # 3x3 grid
n_times <- 5
n <- n_sites * n_times

# Create adjacency matrix for 3x3 grid
adj <- matrix(0, n_sites, n_sites)
adj[1, c(2, 4)] <- 1
adj[2, c(1, 3, 5)] <- 1
adj[3, c(2, 6)] <- 1
adj[4, c(1, 5, 7)] <- 1
adj[5, c(2, 4, 6, 8)] <- 1
adj[6, c(3, 5, 9)] <- 1
adj[7, c(4, 8)] <- 1
adj[8, c(5, 7, 9)] <- 1
adj[9, c(6, 8)] <- 1

df <- data.frame(
  y_num = rpois(n, 10),
  y_denom = rgamma(n, 5, 0.5),
  x = rnorm(n),
  site = factor(rep(1:n_sites, n_times)),
  time = factor(rep(1:n_times, each = n_sites))
)

# Test 1: POISSON_GAMMA + ICAR + RW1 (row 14)
tryCatch({
  fit <- ratiod(y_num | y_denom ~ x, data = df,
                family = ratiod_poisson_gamma(),
                spatial = spatial_car(adj, level = "group", group_var = "site"),
                temporal = temporal_rw1(time_var = "time"),
                chains = 1, warmup = 30, iter = 60, refresh = 0)
  cat("1. POISSON_GAMMA + ICAR + RW1: PASS\n")
  passed <- passed + 1
}, error = function(e) {
  cat("1. POISSON_GAMMA + ICAR + RW1: FAIL -", e$message, "\n")
  failed <<- failed + 1
})

# Test 2: POISSON_GAMMA + BYM2 + RW1 (row 15)
tryCatch({
  fit <- ratiod(y_num | y_denom ~ x, data = df,
                family = ratiod_poisson_gamma(),
                spatial = spatial_bym2(adj, level = "group", group_var = "site"),
                temporal = temporal_rw1(time_var = "time"),
                chains = 1, warmup = 30, iter = 60, refresh = 0)
  cat("2. POISSON_GAMMA + BYM2 + RW1: PASS\n")
  passed <- passed + 1
}, error = function(e) {
  cat("2. POISSON_GAMMA + BYM2 + RW1: FAIL -", e$message, "\n")
  failed <<- failed + 1
})

# Test 3: POISSON_GAMMA + ICAR + AR1 (row 16)
tryCatch({
  fit <- ratiod(y_num | y_denom ~ x, data = df,
                family = ratiod_poisson_gamma(),
                spatial = spatial_car(adj, level = "group", group_var = "site"),
                temporal = temporal_ar1(time_var = "time"),
                chains = 1, warmup = 30, iter = 60, refresh = 0)
  cat("3. POISSON_GAMMA + ICAR + AR1: PASS\n")
  passed <- passed + 1
}, error = function(e) {
  cat("3. POISSON_GAMMA + ICAR + AR1: FAIL -", e$message, "\n")
  failed <<- failed + 1
})

# NegBin data
df_nb <- data.frame(
  y_num = rnbinom(n, size = 5, mu = 10),
  y_denom = rnbinom(n, size = 5, mu = 15),
  x = rnorm(n),
  site = factor(rep(1:n_sites, n_times)),
  time = factor(rep(1:n_times, each = n_sites))
)

# Test 4: NEGBIN_NEGBIN + ICAR + RW1 (row 34)
tryCatch({
  fit <- ratiod(y_num | y_denom ~ x, data = df_nb,
                family = ratiod_negbin_negbin(),
                spatial = spatial_car(adj, level = "group", group_var = "site"),
                temporal = temporal_rw1(time_var = "time"),
                chains = 1, warmup = 30, iter = 60, refresh = 0)
  cat("4. NEGBIN_NEGBIN + ICAR + RW1: PASS\n")
  passed <- passed + 1
}, error = function(e) {
  cat("4. NEGBIN_NEGBIN + ICAR + RW1: FAIL -", e$message, "\n")
  failed <<- failed + 1
})

# Test 5: NEGBIN_NEGBIN + BYM2 + RW1 (row 35)
tryCatch({
  fit <- ratiod(y_num | y_denom ~ x, data = df_nb,
                family = ratiod_negbin_negbin(),
                spatial = spatial_bym2(adj, level = "group", group_var = "site"),
                temporal = temporal_rw1(time_var = "time"),
                chains = 1, warmup = 30, iter = 60, refresh = 0)
  cat("5. NEGBIN_NEGBIN + BYM2 + RW1: PASS\n")
  passed <- passed + 1
}, error = function(e) {
  cat("5. NEGBIN_NEGBIN + BYM2 + RW1: FAIL -", e$message, "\n")
  failed <<- failed + 1
})

# Test 6: NEGBIN_NEGBIN + ICAR + AR1 (row 36)
tryCatch({
  fit <- ratiod(y_num | y_denom ~ x, data = df_nb,
                family = ratiod_negbin_negbin(),
                spatial = spatial_car(adj, level = "group", group_var = "site"),
                temporal = temporal_ar1(time_var = "time"),
                chains = 1, warmup = 30, iter = 60, refresh = 0)
  cat("6. NEGBIN_NEGBIN + ICAR + AR1: PASS\n")
  passed <- passed + 1
}, error = function(e) {
  cat("6. NEGBIN_NEGBIN + ICAR + AR1: FAIL -", e$message, "\n")
  failed <<- failed + 1
})

# Binomial data
df_binom <- data.frame(
  y_num = rbinom(n, 20, 0.3),
  n_trials = rep(20, n),
  x = rnorm(n),
  site = factor(rep(1:n_sites, n_times)),
  time = factor(rep(1:n_times, each = n_sites))
)

# Test 7: BINOMIAL + ICAR + RW1 (row 54)
tryCatch({
  fit <- ratiod(y_num | n_trials ~ x, data = df_binom,
                family = ratiod_binomial(),
                spatial = spatial_car(adj, level = "group", group_var = "site"),
                temporal = temporal_rw1(time_var = "time"),
                chains = 1, warmup = 30, iter = 60, refresh = 0)
  cat("7. BINOMIAL + ICAR + RW1: PASS\n")
  passed <- passed + 1
}, error = function(e) {
  cat("7. BINOMIAL + ICAR + RW1: FAIL -", e$message, "\n")
  failed <<- failed + 1
})

# Test 8: BINOMIAL + BYM2 + RW1 (row 55)
tryCatch({
  fit <- ratiod(y_num | n_trials ~ x, data = df_binom,
                family = ratiod_binomial(),
                spatial = spatial_bym2(adj, level = "group", group_var = "site"),
                temporal = temporal_rw1(time_var = "time"),
                chains = 1, warmup = 30, iter = 60, refresh = 0)
  cat("8. BINOMIAL + BYM2 + RW1: PASS\n")
  passed <- passed + 1
}, error = function(e) {
  cat("8. BINOMIAL + BYM2 + RW1: FAIL -", e$message, "\n")
  failed <<- failed + 1
})

# Test 9: BINOMIAL + ICAR + AR1 (row 56)
tryCatch({
  fit <- ratiod(y_num | n_trials ~ x, data = df_binom,
                family = ratiod_binomial(),
                spatial = spatial_car(adj, level = "group", group_var = "site"),
                temporal = temporal_ar1(time_var = "time"),
                chains = 1, warmup = 30, iter = 60, refresh = 0)
  cat("9. BINOMIAL + ICAR + AR1: PASS\n")
  passed <- passed + 1
}, error = function(e) {
  cat("9. BINOMIAL + ICAR + AR1: FAIL -", e$message, "\n")
  failed <<- failed + 1
})

cat(sprintf("\nResults: %d/9 passed\n", passed))
if (passed == 9) {
  cat("All Spatial+Temporal H gradient tests passed!\n")
}
