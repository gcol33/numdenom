# test-specs-binomial.R
# Smoke test for the B1a LikelihoodSpec PoC.
# Confirms that the feature flag routes through the spec path and produces
# posterior means consistent with the legacy backend (within MC noise).

test_that("B1a LikelihoodSpec PoC matches legacy binomial within MC noise", {
  skip_on_cran()

  set.seed(20260503)
  n <- 200
  x1 <- rnorm(n); x2 <- rnorm(n)
  eta <- 0.4 + 0.8 * x1 - 0.5 * x2
  p <- plogis(eta)
  n_trials <- sample(5:20, n, replace = TRUE)
  y <- rbinom(n, n_trials, p)
  dat <- data.frame(y = y, n_trials = n_trials, x1 = x1, x2 = x2)

  fit_one <- function(use_specs, seed_val) {
    op <- options(tulpaRatio.use_specs = use_specs)
    on.exit(options(op), add = TRUE)
    fit <- tulpaRatio::tratio(
      formula = y | n_trials ~ x1 + x2, data = dat,
      family = tulpaRatio::ratiod_binomial(), mode = "hmc",
      control = list(iter = 2000L, warmup = 500L, chains = 1L, seed = seed_val, verbose = FALSE, gradient_mode = "A_r")
    )
    draws <- fit$draws
    if (is.matrix(draws)) colMeans(draws) else stop("unexpected draws shape")
  }

  legacy42 <- fit_one(FALSE, 42L)
  specs42  <- fit_one(TRUE,  42L)
  legacy43 <- fit_one(FALSE, 43L)

  expect_named(specs42, names(legacy42))

  cross  <- max(abs(legacy42 - specs42))
  within <- max(abs(legacy42 - legacy43))

  # Cross-backend posterior-mean drift must stay within 4x the within-backend
  # MC noise (which is itself a single-seed estimate, so we keep the cap loose).
  expect_lt(cross, max(4 * within, 5e-3))
})
