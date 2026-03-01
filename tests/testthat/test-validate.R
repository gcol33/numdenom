# Tests for validation and model comparison functions (R/validate.R)

test_that("pp_check generic dispatches correctly", {
  # pp_check is a generic
  expect_true(is.function(pp_check))
  expect_true("pp_check.ratiod_fit" %in% methods("pp_check"))
})

test_that("prior_predict is not yet implemented", {
  expect_error(
    prior_predict(
      formula = y ~ x,
      family = ratiod_negbin_negbin(),
      data = data.frame(y = 1:10, x = rnorm(10))
    ),
    "not yet implemented"
  )
})

test_that("ratiod_compare requires at least two models", {
  skip_on_cran()
  skip_if_not_installed("loo")

  set.seed(123)
  n <- 30
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 3, rate = 1),
    x = rnorm(n)
  )

  fit <- ratiod(
    count | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    iter = 100,
    warmup = 50,
    chains = 1
  )

  expect_error(
    ratiod_compare(fit),
    "At least two models"
  )
})

test_that("ratiod_compare validates model class", {
  skip_if_not_installed("loo")

  expect_error(
    ratiod_compare("not_a_fit", "also_not_a_fit"),
    "ratiod_fit"
  )
})

test_that("ratiod_average requires at least two models", {
  skip_on_cran()
  skip_if_not_installed("loo")

  set.seed(456)
  n <- 30
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 3, rate = 1),
    x = rnorm(n)
  )

  fit <- ratiod(
    count | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    iter = 100,
    warmup = 50,
    chains = 1
  )

  expect_error(
    ratiod_average(fit),
    "At least two models"
  )
})

test_that("ratiod_average validates model class", {
  skip_if_not_installed("loo")

  expect_error(
    ratiod_average("not_a_fit", "also_not_a_fit"),
    "ratiod_fit"
  )
})

test_that("average_predictions works on log scale for ratios", {
  set.seed(789)
  # Create fake prediction matrices
  predictions <- list(
    matrix(rep(2, 50), nrow = 10, ncol = 5),
    matrix(rep(8, 50), nrow = 10, ncol = 5)
  )
  weights <- c(0.5, 0.5)

  result <- numdenom:::average_predictions(predictions, weights, type = "ratio", summary = FALSE)

  # Geometric mean of 2 and 8 with equal weights = 4
  expect_equal(result[1, 1], 4, tolerance = 0.001)
})

test_that("average_predictions works for numerator/denominator", {
  set.seed(101)
  predictions <- list(
    matrix(rep(2, 50), nrow = 10, ncol = 5),
    matrix(rep(8, 50), nrow = 10, ncol = 5)
  )
  weights <- c(0.5, 0.5)

  result <- numdenom:::average_predictions(predictions, weights, type = "numerator", summary = FALSE)

  # Arithmetic mean of 2 and 8 with equal weights = 5
  expect_equal(result[1, 1], 5, tolerance = 0.001)
})

test_that("average_predictions returns summary when requested", {
  set.seed(102)
  predictions <- list(
    matrix(rnorm(50, mean = 10), nrow = 10, ncol = 5),
    matrix(rnorm(50, mean = 12), nrow = 10, ncol = 5)
  )
  weights <- c(0.6, 0.4)

  result <- numdenom:::average_predictions(predictions, weights, type = "numerator", summary = TRUE)

  expect_true(is.data.frame(result))
  expect_true("mean" %in% names(result))
  expect_true("sd" %in% names(result))
  expect_true("q2.5" %in% names(result))
  expect_true("q97.5" %in% names(result))
  expect_equal(nrow(result), 5)
})

test_that("print.ratiod_average works", {
  # Create mock ratiod_average object
  avg <- structure(
    list(
      weights = c(model1 = 0.7, model2 = 0.3),
      predictions = data.frame(
        mean = c(1.2, 1.5, 1.8),
        sd = c(0.1, 0.15, 0.2),
        q2.5 = c(1.0, 1.2, 1.4),
        q97.5 = c(1.4, 1.8, 2.2)
      ),
      models = c("model1", "model2"),
      type = "ratio",
      weights_method = "loo",
      n_models = 2
    ),
    class = "ratiod_average"
  )

  output <- capture.output(print(avg))
  expect_true(any(grepl("ratiod model averaging", output)))
  expect_true(any(grepl("Method:", output)))
  expect_true(any(grepl("Model weights:", output)))
})

test_that("fitted.ratiod_average extracts predictions", {
  avg <- structure(
    list(
      predictions = data.frame(mean = 1:3, sd = rep(0.1, 3))
    ),
    class = "ratiod_average"
  )

  result <- fitted(avg)
  expect_equal(result, avg$predictions)
})

test_that("weights.ratiod_average extracts weights", {
  avg <- structure(
    list(
      weights = c(m1 = 0.6, m2 = 0.4)
    ),
    class = "ratiod_average"
  )

  result <- weights(avg)
  expect_equal(result, c(m1 = 0.6, m2 = 0.4))
})

test_that("loo.ratiod_fit method exists", {
  # Just verify the method exists
  expect_true(exists("loo.ratiod_fit", mode = "function"))
})

test_that("waic.ratiod_fit method exists", {
  # Just verify the method exists
  expect_true(exists("waic.ratiod_fit", mode = "function"))
})

test_that("pp_check_single requires bayesplot", {
  skip_if_not_installed("bayesplot")

  # This is an internal function - just verify it exists
  expect_true(exists("pp_check_single", where = asNamespace("numdenom"), mode = "function"))
})

# ---------------------------------------------------------------------------
# Integration tests for loo.ratiod_fit and waic.ratiod_fit
# Note: These require log_lik draws which are not yet produced by HMC backend
# ---------------------------------------------------------------------------

test_that("loo.ratiod_fit errors gracefully without log_lik", {
  skip_on_cran()
  skip_if_not_installed("loo")

  set.seed(123)
  n <- 20
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n)
  )

  fit <- ratiod(
    count | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    iter = 100,
    warmup = 50,
    chains = 1,
    verbose = FALSE
  )

  # HMC backend doesn't produce log_lik, so this should error
  expect_error(loo::loo(fit), "Log-likelihood not found")
})

test_that("waic.ratiod_fit errors gracefully without log_lik", {
  skip_on_cran()
  skip_if_not_installed("loo")

  set.seed(456)
  n <- 20
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n)
  )

  fit <- ratiod(
    count | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    iter = 100,
    warmup = 50,
    chains = 1,
    verbose = FALSE
  )

  # HMC backend doesn't produce log_lik, so this should error
  expect_error(loo::waic(fit), "Log-likelihood not found")
})

test_that("ratio() extracts ratio posterior draws", {
  skip_on_cran()

  set.seed(333)
  n <- 20
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n)
  )

  fit <- ratiod(
    count | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    mode = "hmc",
    iter = 100,
    warmup = 50,
    chains = 1,
    verbose = FALSE
  )

  # ratio() should work with HMC fits
  r <- ratio(fit)
  expect_s3_class(r, "ratiod_ratio")
  expect_true(is.matrix(r$draws))
  expect_equal(ncol(r$draws), n)
})

test_that("print.ratiod_average handles matrix predictions", {
  # Create mock ratiod_average object with matrix predictions
  avg <- structure(
    list(
      weights = c(model1 = 0.6, model2 = 0.4),
      predictions = matrix(rnorm(30), nrow = 10, ncol = 3),
      models = c("model1", "model2"),
      type = "ratio",
      weights_method = "loo",
      n_models = 2
    ),
    class = "ratiod_average"
  )

  output <- capture.output(print(avg))
  expect_true(any(grepl("draws x", output)))
  expect_true(any(grepl("observations", output)))
})
