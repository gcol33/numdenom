suppressPackageStartupMessages({
  library(testthat)
  devtools::load_all('.', quiet = TRUE)
})

simulate_temporal_binomial <- function(n_times = 16L, reps = 12L, type = "ar1") {
  set.seed(20260603)
  rho <- 0.6
  phi <- numeric(n_times)
  phi[1] <- rnorm(1, sd = 0.4)
  for (t in 2:n_times) {
    phi[t] <- rho * phi[t - 1] + rnorm(1, sd = 0.3)
  }

  time <- rep(seq_len(n_times), each = reps)
  n    <- length(time)
  x1   <- rnorm(n)
  eta  <- 0.4 + 0.5 * x1 + phi[time]
  p    <- plogis(eta)
  nt   <- sample(15:30, n, replace = TRUE)
  y    <- rbinom(n, nt, p)

  list(
    formula = y | n_trials ~ x1,
    data    = data.frame(y = y, n_trials = nt, x1 = x1,
                          t = factor(time, levels = seq_len(n_times))),
    family  = tulpaRatio::ratiod_binomial()
  )
}

fit_one <- function(sim, use_specs, seed_val) {
  op <- options(tulpaRatio.use_specs = use_specs); on.exit(options(op), add = TRUE)
  fit <- tulpaRatio::ratiod(
    formula = sim$formula, data = sim$data, family = sim$family,
    temporal = tulpaRatio::temporal_ar1(time_var = "t"),
    mode = "hmc", iter = 3000L, warmup = 1000L, chains = 1L,
    seed = seed_val, verbose = FALSE, gradient_mode = "A_r"
  )
  fit
}

cat("--- Fitting legacy AR1 ---\n")
sim <- simulate_temporal_binomial()
fit_leg <- fit_one(sim, FALSE, 42L)
cat("--- Fitting spec AR1 ---\n")
fit_spec <- fit_one(sim, TRUE, 42L)

leg_means <- colMeans(fit_leg$draws)
spec_means <- colMeans(fit_spec$draws)

stopifnot(identical(names(leg_means), names(spec_means)))

cat("\n--- Column-by-column posterior means ---\n")
df <- data.frame(
  param = names(leg_means),
  legacy = round(leg_means, 4),
  spec = round(spec_means, 4),
  diff = round(leg_means - spec_means, 4)
)
print(df)

cat("\n--- Max abs diff: ", max(abs(leg_means - spec_means)), "\n")
cat("--- Number of columns:", ncol(fit_leg$draws), "\n")
