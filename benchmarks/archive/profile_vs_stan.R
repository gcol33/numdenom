# profile_vs_stan.R
# Profile numdenom vs Stan to find where the 1.8x gap comes from.
#
# Strategy: break down runtime into components:
#   1. Gradient computation (the O(N) kernel)
#   2. NUTS overhead (tree building, U-turn checks, acceptance)
#   3. Warmup overhead (step size adaptation, mass matrix)
#   4. R↔C++ interface overhead
#
# Compare NB base (where Stan wins 1.8x) vs Binomial base (where numdenom wins 8.5x).

library(numdenom)
library(brms)
library(cmdstanr)

set.seed(42)

# ============================================================================
# PART 1: Generate test data
# ============================================================================

N <- 500
x <- rnorm(N)
X <- cbind(1, x)

# NegBin data
beta_num <- c(1.0, 0.3)
beta_denom <- c(1.5, -0.2)
phi_num <- 5.0
phi_denom <- 3.0
mu_num <- exp(X %*% beta_num)
mu_denom <- exp(X %*% beta_denom)
y_num <- rnbinom(N, size = phi_num, mu = mu_num)
y_denom <- rnbinom(N, size = phi_denom, mu = mu_denom)

df_nb <- data.frame(y_num = y_num, y_denom = y_denom, x = x)

# Binomial data
beta_bin <- c(0.5, 0.3)
n_trials <- rep(20, N)
p_bin <- plogis(X %*% beta_bin)
y_bin <- rbinom(N, size = n_trials, prob = p_bin)

df_bin <- data.frame(y = y_bin, trials = n_trials, x = x)

cat("=== Data generated (N =", N, ") ===\n\n")

# ============================================================================
# PART 2: Micro-benchmark the gradient kernel alone
# ============================================================================

cat("=== PART 2: Gradient kernel microbenchmark ===\n")

# Fit a quick model to get the C++ data structures initialized
fit_nb_quick <- tratio(
  y_num | y_denom ~ x, data = df_nb,
  family = ratiod_negbin_negbin(),
  control = list(iter = 10, warmup = 5, chains = 1)
)

fit_bin_quick <- tratio(
  y | trials ~ x, data = df_bin,
  family = ratiod_binomial(),
  control = list(iter = 10, warmup = 5, chains = 1)
)

# Now benchmark with more iterations to get stable timing
# Use very short warmup to isolate sampling overhead
n_reps <- 5

cat("\n--- NegBin base (numdenom) ---\n")
nb_times <- numeric(n_reps)
for (r in 1:n_reps) {
  t0 <- proc.time()["elapsed"]
  fit_nb <- tratio(
    y_num | y_denom ~ x, data = df_nb,
    family = ratiod_negbin_negbin(),
    control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H")
  )
  nb_times[r] <- proc.time()["elapsed"] - t0
}
cat("  Times:", round(nb_times, 2), "\n")
cat("  Median:", round(median(nb_times), 2), "s\n")

cat("\n--- PG base (numdenom) ---\n")
# PG data
lambda_num <- exp(X %*% beta_num)
y_pg_num <- rpois(N, lambda_num)
shape <- 5.0
rate <- shape / exp(X %*% c(1.5, -0.2))
y_pg_denom <- rgamma(N, shape = shape, rate = rate)
df_pg <- data.frame(y_num = y_pg_num, y_denom = y_pg_denom, x = x)

pg_times <- numeric(n_reps)
for (r in 1:n_reps) {
  t0 <- proc.time()["elapsed"]
  fit_pg <- tratio(
    y_num | y_denom ~ x, data = df_pg,
    family = ratiod_poisson_gamma(),
    control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H")
  )
  pg_times[r] <- proc.time()["elapsed"] - t0
}
cat("  Times:", round(pg_times, 2), "\n")
cat("  Median:", round(median(pg_times), 2), "s\n")

cat("\n--- Binomial base (numdenom) ---\n")
bin_times <- numeric(n_reps)
for (r in 1:n_reps) {
  t0 <- proc.time()["elapsed"]
  fit_bin <- tratio(
    y | trials ~ x, data = df_bin,
    family = ratiod_binomial(),
    control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H")
  )
  bin_times[r] <- proc.time()["elapsed"] - t0
}
cat("  Times:", round(bin_times, 2), "\n")
cat("  Median:", round(median(bin_times), 2), "s\n")

# ============================================================================
# PART 3: Compare gradient modes (isolate gradient cost vs NUTS overhead)
# ============================================================================

cat("\n=== PART 3: Gradient mode comparison (NB base) ===\n")
cat("  If H and N have similar TOTAL time, gradient is not the bottleneck.\n")
cat("  If H is much faster than N, gradient dominates.\n\n")

for (mode in c("H", "A", "N")) {
  t0 <- proc.time()["elapsed"]
  fit <- tratio(
    y_num | y_denom ~ x, data = df_nb,
    family = ratiod_negbin_negbin(),
    control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = mode)
  )
  elapsed <- proc.time()["elapsed"] - t0
  cat(sprintf("  Mode %s: %.2fs\n", mode, elapsed))
}

# ============================================================================
# PART 4: Breakdown - warmup vs sampling
# ============================================================================

cat("\n=== PART 4: Warmup vs Sampling breakdown ===\n")
cat("  Compare iter=500/warmup=250 vs iter=500/warmup=50 vs iter=250/warmup=0\n\n")

configs <- list(
  c(500, 250),
  c(500, 50),
  c(250, 0)
)

for (cfg in configs) {
  iter <- cfg[1]
  warm <- cfg[2]
  t0 <- proc.time()["elapsed"]
  fit <- tratio(
    y_num | y_denom ~ x, data = df_nb,
    family = ratiod_negbin_negbin(),
    control = list(iter = iter, warmup = warm, chains = 1, gradient_mode = "H")
  )
  elapsed <- proc.time()["elapsed"] - t0
  n_sample <- iter - warm
  cat(sprintf("  iter=%d warmup=%d (sampling=%d): %.2fs  (%.1f ms/sample iter)\n",
              iter, warm, n_sample, elapsed, 1000 * elapsed / iter))
}

# ============================================================================
# PART 5: Stan comparison (custom joint model)
# ============================================================================

cat("\n=== PART 5: Stan NB base comparison ===\n")

stan_model_file <- file.path(dirname(getwd()), "numdenom", "benchmarks", "stan", "joint_nb_base.stan")
if (!file.exists(stan_model_file)) {
  stan_model_file <- "benchmarks/stan/joint_nb_base.stan"
}

if (file.exists(stan_model_file)) {
  cat("  Using custom joint Stan model:", stan_model_file, "\n")

  mod <- cmdstan_model(stan_model_file)

  stan_data <- list(
    N = N,
    p = 2,
    X = X,
    y_num = y_num,
    y_denom = y_denom
  )

  stan_times <- numeric(n_reps)
  for (r in 1:n_reps) {
    t0 <- proc.time()["elapsed"]
    stan_fit <- mod$sample(
      data = stan_data,
      iter_sampling = 250, iter_warmup = 250,
      chains = 1, refresh = 0,
      show_messages = FALSE, show_exceptions = FALSE
    )
    stan_times[r] <- proc.time()["elapsed"] - t0
  }
  cat("  Stan times:", round(stan_times, 2), "\n")
  cat("  Stan median:", round(median(stan_times), 2), "s\n")
  cat("  numdenom median:", round(median(nb_times), 2), "s\n")
  cat("  Ratio:", round(median(nb_times) / median(stan_times), 2), "x\n")

} else {
  cat("  Stan model not found, using brms for binomial comparison\n")
}

# ============================================================================
# PART 6: brms binomial comparison (valid for binomial family)
# ============================================================================

cat("\n=== PART 6: brms Binomial comparison ===\n")

brms_bin_times <- numeric(n_reps)
for (r in 1:n_reps) {
  t0 <- proc.time()["elapsed"]
  brms_fit <- brm(
    y | trials(trials) ~ x, data = df_bin,
    family = binomial(),
    iter = 500, warmup = 250, chains = 1, refresh = 0,
    backend = "cmdstanr", silent = 2
  )
  brms_bin_times[r] <- proc.time()["elapsed"] - t0
}
cat("  brms times:", round(brms_bin_times, 2), "\n")
cat("  brms median:", round(median(brms_bin_times), 2), "s\n")
cat("  numdenom median:", round(median(bin_times), 2), "s\n")
cat("  Ratio:", round(median(bin_times) / median(brms_bin_times), 2), "x\n")

# ============================================================================
# PART 7: Scaling test — how does cost scale with N?
# ============================================================================

cat("\n=== PART 7: N-scaling (NB base, H mode) ===\n")

for (n_test in c(100, 250, 500, 1000, 2000)) {
  x_t <- rnorm(n_test)
  X_t <- cbind(1, x_t)
  mu_n <- exp(X_t %*% beta_num)
  mu_d <- exp(X_t %*% beta_denom)
  y_n <- rnbinom(n_test, size = phi_num, mu = mu_n)
  y_d <- rnbinom(n_test, size = phi_denom, mu = mu_d)
  df_t <- data.frame(y_num = y_n, y_denom = y_d, x = x_t)

  t0 <- proc.time()["elapsed"]
  fit_t <- tratio(
    y_num | y_denom ~ x, data = df_t,
    family = ratiod_negbin_negbin(),
    control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H")
  )
  elapsed <- proc.time()["elapsed"] - t0
  cat(sprintf("  N=%5d: %.2fs  (%.3f ms/obs/iter)\n",
              n_test, elapsed, 1000 * elapsed / (n_test * 500)))
}

# ============================================================================
# PART 8: Leapfrog count analysis
# ============================================================================

cat("\n=== PART 8: NUTS diagnostics (treedepth → leapfrog count) ===\n")

fit_nb_diag <- tratio(
  y_num | y_denom ~ x, data = df_nb,
  family = ratiod_negbin_negbin(),
  control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H")
)

if (!is.null(fit_nb_diag$diagnostics)) {
  diag <- fit_nb_diag$diagnostics
  if (!is.null(diag$treedepth)) {
    td <- diag$treedepth
    cat("  Mean treedepth:", round(mean(td), 1), "\n")
    cat("  Treedepth distribution:\n")
    print(table(td))
    cat("  Mean leapfrog steps (2^td - 1):", round(mean(2^td - 1), 0), "\n")
    total_lf <- sum(2^td - 1)
    cat("  Total leapfrog steps:", total_lf, "\n")
    cat("  If gradient takes ~X us, total gradient time = X *", total_lf, "us\n")
  }
}

cat("\n=== DONE ===\n")
