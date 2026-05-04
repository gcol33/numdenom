# dev_notes/parity_B1b.R
# Parity check for B1b: each ratio family fitted twice (legacy and specs)
# at the same seed, comparing posterior means. Threshold: max abs cross-
# backend diff < max(4 * within-MC, 5e-3) — same yardstick as B1a.

suppressMessages(devtools::load_all("C:/GillesC/Documents/dev/tulpaRatio", quiet = TRUE))

set.seed(20260503)

fit_one <- function(formula, data, family, use_specs, seed_val) {
  op <- options(tulpaRatio.use_specs = use_specs)
  on.exit(options(op), add = TRUE)
  fit <- tulpaRatio::ratiod(
    formula = formula, data = data, family = family,
    mode = "hmc", iter = 2000L, warmup = 500L, chains = 1L,
    seed = seed_val, verbose = FALSE, gradient_mode = "A_r"
  )
  draws <- fit$draws
  if (is.matrix(draws)) colMeans(draws) else stop("unexpected draws shape")
}

run_family <- function(name, formula, data, family) {
  cat(sprintf("\n--- %s ---\n", name))
  t1 <- Sys.time()
  legacy42 <- fit_one(formula, data, family, FALSE, 42L)
  t_legacy <- as.numeric(difftime(Sys.time(), t1, units = "secs"))
  t1 <- Sys.time()
  specs42  <- fit_one(formula, data, family, TRUE,  42L)
  t_specs  <- as.numeric(difftime(Sys.time(), t1, units = "secs"))
  legacy43 <- fit_one(formula, data, family, FALSE, 43L)
  cat(" legacy42 names: ", paste(names(legacy42), collapse=", "), "\n")
  cat(" specs42  names: ", paste(names(specs42),  collapse=", "), "\n")
  common <- intersect(names(legacy42), names(specs42))
  cross  <- max(abs(legacy42[common] - specs42[common]))
  within <- max(abs(legacy42[common] - legacy43[common]))
  cat(sprintf(" cross  = %.4g\n within = %.4g\n",
              cross, within))
  cat(sprintf(" wallclock specs/legacy = %.2fs / %.2fs (ratio %.2f)\n",
              t_specs, t_legacy, t_specs / max(t_legacy, 1e-9)))
  invisible(list(name = name, cross = cross, within = within,
                 t_specs = t_specs, t_legacy = t_legacy))
}

results <- list()

# Binomial (sanity — already in B1a)
{
  n <- 200
  x1 <- rnorm(n); x2 <- rnorm(n)
  eta <- 0.4 + 0.8 * x1 - 0.5 * x2
  p <- plogis(eta)
  n_trials <- sample(5:20, n, replace = TRUE)
  y <- rbinom(n, n_trials, p)
  dat <- data.frame(y = y, n_trials = n_trials, x1 = x1, x2 = x2)
  results$binomial <- run_family("binomial",
    y | n_trials ~ x1 + x2, dat, tulpaRatio::ratiod_binomial())
}

# Poisson-Gamma
{
  n <- 200
  x1 <- rnorm(n); x2 <- rnorm(n)
  mu_num   <- exp(1.0 + 0.4 * x1)
  mu_denom <- exp(0.5 + 0.3 * x2)
  y_num   <- rpois(n, mu_num)
  shape   <- 4.0
  y_denom <- rgamma(n, shape = shape, rate = shape / mu_denom)
  dat <- data.frame(y_num = y_num, y_denom = y_denom, x1 = x1, x2 = x2)
  results$poisson_gamma <- run_family("poisson_gamma",
    y_num | y_denom ~ x1 + x2, dat, tulpaRatio::ratiod_poisson_gamma())
}

# NegBin-Gamma
{
  n <- 200
  x1 <- rnorm(n); x2 <- rnorm(n)
  mu_num   <- exp(1.0 + 0.4 * x1)
  mu_denom <- exp(0.5 + 0.3 * x2)
  phi_num  <- 5.0
  shape    <- 4.0
  y_num   <- rnbinom(n, size = phi_num, mu = mu_num)
  y_denom <- rgamma(n, shape = shape, rate = shape / mu_denom)
  dat <- data.frame(y_num = y_num, y_denom = y_denom, x1 = x1, x2 = x2)
  results$negbin_gamma <- run_family("negbin_gamma",
    y_num | y_denom ~ x1 + x2, dat, tulpaRatio::ratiod_negbin_gamma())
}

# NegBin-NegBin
{
  n <- 200
  x1 <- rnorm(n); x2 <- rnorm(n)
  mu_num   <- exp(1.0 + 0.4 * x1)
  mu_denom <- exp(0.7 + 0.3 * x2)
  phi_num  <- 5.0; phi_denom <- 5.0
  y_num   <- rnbinom(n, size = phi_num,   mu = mu_num)
  y_denom <- rnbinom(n, size = phi_denom, mu = mu_denom)
  dat <- data.frame(y_num = y_num, y_denom = y_denom, x1 = x1, x2 = x2)
  results$negbin_negbin <- run_family("negbin_negbin",
    y_num | y_denom ~ x1 + x2, dat, tulpaRatio::ratiod_negbin_negbin())
}

# Gamma-Gamma
{
  n <- 200
  x1 <- rnorm(n); x2 <- rnorm(n)
  mu_num   <- exp(1.0 + 0.4 * x1)
  mu_denom <- exp(0.5 + 0.3 * x2)
  shape_n <- 4.0; shape_d <- 5.0
  y_num   <- rgamma(n, shape = shape_n, rate = shape_n / mu_num)
  y_denom <- rgamma(n, shape = shape_d, rate = shape_d / mu_denom)
  dat <- data.frame(y_num = y_num, y_denom = y_denom, x1 = x1, x2 = x2)
  results$gamma_gamma <- run_family("gamma_gamma",
    y_num | y_denom ~ x1 + x2, dat, tulpaRatio::ratiod_gamma_gamma())
}

# Lognormal
{
  n <- 200
  x1 <- rnorm(n); x2 <- rnorm(n)
  mu_log_num   <- 1.0 + 0.4 * x1
  mu_log_denom <- 0.5 + 0.3 * x2
  sig_n <- 0.5; sig_d <- 0.5
  y_num   <- exp(rnorm(n, mu_log_num,   sig_n))
  y_denom <- exp(rnorm(n, mu_log_denom, sig_d))
  dat <- data.frame(y_num = y_num, y_denom = y_denom, x1 = x1, x2 = x2)
  results$lognormal <- run_family("lognormal",
    y_num | y_denom ~ x1 + x2, dat, tulpaRatio::ratiod_lognormal())
}

# Beta-Binomial
{
  n <- 200
  x1 <- rnorm(n); x2 <- rnorm(n)
  eta <- 0.4 + 0.8 * x1 - 0.5 * x2
  p <- plogis(eta)
  n_trials <- sample(5:20, n, replace = TRUE)
  phi <- 10
  alpha <- p * phi; beta <- (1 - p) * phi
  y <- vapply(seq_len(n), function(i) {
    pp <- rbeta(1, alpha[i], beta[i])
    rbinom(1, n_trials[i], pp)
  }, integer(1))
  dat <- data.frame(y = y, n_trials = n_trials, x1 = x1, x2 = x2)
  results$beta_binomial <- run_family("beta_binomial",
    y | n_trials ~ x1 + x2, dat, tulpaRatio::ratiod_beta_binomial())
}

cat("\n=== Summary ===\n")
for (r in results) {
  cat(sprintf("%-15s cross=%.4g  within=%.4g  pass=%s  ratio=%.2f\n",
              r$name, r$cross, r$within,
              r$cross < max(4 * r$within, 5e-3),
              r$t_specs / max(r$t_legacy, 1e-9)))
}
