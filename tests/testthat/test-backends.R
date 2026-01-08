# Tests for alternative backends (PG and Laplace)

test_that("PG sampler produces valid PG(1, z) samples", {
  skip_on_cran()

  set.seed(123)

  # Test PG(1, 0) - should have mean ~0.25
  x <- quotr:::cpp_rpg1(rep(0, 5000))
  expect_true(all(x > 0))
  expect_equal(mean(x), 0.25, tolerance = 0.02)

  # Test PG(1, 2) - should have mean ~0.19
  x <- quotr:::cpp_rpg1(rep(2, 5000))
  mean_theory <- tanh(1) / 4  # tanh(z/2) / (2z)

  expect_equal(mean(x), mean_theory, tolerance = 0.02)
})

test_that("PG sampler works for PG(b, z)", {
  skip_on_cran()

  set.seed(456)

  # PG(5, 1) should be sum of 5 PG(1, 1)
  b <- rep(5L, 1000)
  z <- rep(1, 1000)
  x <- quotr:::cpp_rpg(b, z)

  expect_true(all(x > 0))
  # Mean should be approximately 5 * PG(1, 1) mean
  expected_mean <- 5 * tanh(0.5) / 2
  expect_equal(mean(x), expected_mean, tolerance = 0.1)
})

test_that("PG Gibbs sampler recovers parameters", {
  skip_on_cran()

  set.seed(42)

  # Generate data
  n_groups <- 10
  n_per_group <- 20
  N <- n_groups * n_per_group

  true_beta <- c(0.5, -0.3)
  true_sigma <- 0.6

  x <- rnorm(N)
  X <- cbind(1, x)
  group <- rep(1:n_groups, each = n_per_group)
  true_re <- rnorm(n_groups, 0, true_sigma)

  eta <- X %*% true_beta + true_re[group]
  trials <- rep(10L, N)
  y <- rbinom(N, trials, plogis(eta))

  # Fit
  fit <- quotr:::cpp_pg_binomial_gibbs(
    y = as.integer(y),
    n = trials,
    X = X,
    group = as.integer(group),
    n_groups = n_groups,
    n_iter = 1500,
    n_warmup = 500,
    thin = 1,
    verbose = FALSE
  )

  # Check recovery (within reasonable tolerance for stochastic data)
  beta_post <- colMeans(fit$beta)
  expect_equal(beta_post[1], true_beta[1], tolerance = 0.5)
  expect_equal(beta_post[2], true_beta[2], tolerance = 0.3)

  # RE correlation should be high
  re_post <- colMeans(fit$re)
  expect_gt(cor(re_post, true_re), 0.8)
})

test_that("Laplace backend finds correct mode", {
  skip_on_cran()

  set.seed(123)

  # Simple binomial data without RE
  N <- 100
  x <- rnorm(N)
  X <- cbind(1, x)
  true_beta <- c(-0.5, 0.4)

  eta <- X %*% true_beta
  trials <- rep(10L, N)
  y <- rbinom(N, trials, plogis(eta))

  # Fit with Laplace (no RE)
  result <- quotr:::cpp_laplace_fit(
    y = as.integer(y),
    n = as.integer(trials),
    X = X,
    re_idx = as.numeric(rep(1, N)),
    n_re_groups = 0L,
    sigma_re = 1.0,
    family = "binomial"
  )

  expect_true(result$converged)
  # Tolerances are wide because of stochastic data generation
  expect_equal(result$mode[1], true_beta[1], tolerance = 0.4)
  expect_equal(result$mode[2], true_beta[2], tolerance = 0.3)
})

test_that("Laplace backend works with random effects", {
  skip_on_cran()

  set.seed(42)

  # Data with RE
  n_groups <- 15
  n_per_group <- 10
  N <- n_groups * n_per_group

  true_beta <- c(-0.3, 0.5)
  true_sigma <- 0.7

  x <- rnorm(N)
  X <- cbind(1, x)
  group <- rep(1:n_groups, each = n_per_group)
  true_re <- rnorm(n_groups, 0, true_sigma)

  eta <- X %*% true_beta + true_re[group]
  trials <- rep(15L, N)
  y <- rbinom(N, trials, plogis(eta))

  # Fit
  result <- quotr:::cpp_laplace_fit(
    y = as.integer(y),
    n = as.integer(trials),
    X = X,
    re_idx = as.numeric(group),
    n_re_groups = as.integer(n_groups),
    sigma_re = true_sigma,
    family = "binomial"
  )

  expect_true(result$converged)

  # Fixed effects should be close
  p <- ncol(X)
  expect_equal(result$mode[1], true_beta[1], tolerance = 0.4)
  expect_equal(result$mode[2], true_beta[2], tolerance = 0.3)

  # RE should correlate with true values
  re_est <- result$mode[(p + 1):(p + n_groups)]
  expect_gt(cor(re_est, true_re), 0.7)
})

test_that("can_use_pg_backend correctly identifies binomial family", {
  expect_true(quotr:::can_use_pg_backend(quotr_binomial()))
  expect_false(quotr:::can_use_pg_backend(quotr_negbin_negbin()))
  expect_false(quotr:::can_use_pg_backend(quotr_poisson_gamma()))
})

test_that("can_use_laplace_backend accepts all families", {
  expect_true(quotr:::can_use_laplace_backend(quotr_binomial()))
  expect_true(quotr:::can_use_laplace_backend(quotr_negbin_negbin()))
  expect_true(quotr:::can_use_laplace_backend(quotr_poisson_gamma()))
})
