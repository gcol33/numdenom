# Tests for response validation (R/validate_response.R)

test_that("validate_response accepts valid count data", {
  # Non-negative integers for Poisson/NegBin/Binomial
  valid_counts <- c(0L, 1L, 5L, 10L, 100L)

  expect_silent(tulpaRatio:::validate_response(valid_counts, "poisson", "test"))
  expect_silent(tulpaRatio:::validate_response(valid_counts, "neg_binomial_2", "test"))
  expect_silent(tulpaRatio:::validate_response(valid_counts, "binomial", "test"))
})

test_that("validate_response rejects negative counts", {
  invalid_counts <- c(-1, 0, 5, 10)

  expect_error(
    tulpaRatio:::validate_response(invalid_counts, "poisson", "numerator"),
    "non-negative"
  )
})

test_that("validate_response rejects non-integer counts", {
  non_integers <- c(0.5, 1.2, 3.7)

  expect_error(
    tulpaRatio:::validate_response(non_integers, "poisson", "numerator"),
    "integer counts"
  )
})

test_that("validate_response accepts valid gamma data", {
  valid_gamma <- c(0.1, 1.5, 2.3, 10.0)

  expect_silent(tulpaRatio:::validate_response(valid_gamma, "gamma", "denominator"))
})

test_that("validate_response rejects non-positive gamma data", {
  invalid_gamma <- c(0, 1.5, 2.3)

  expect_error(
    tulpaRatio:::validate_response(invalid_gamma, "gamma", "denominator"),
    "positive"
  )

  # Negative values
  expect_error(
    tulpaRatio:::validate_response(c(-1, 1, 2), "gamma", "denominator"),
    "positive"
  )
})

test_that("validate_response rejects non-numeric data", {
  expect_error(
    tulpaRatio:::validate_response(c("a", "b", "c"), "poisson", "test"),
    "numeric"
  )

  expect_error(
    tulpaRatio:::validate_response(c("x", "y"), "gamma", "test"),
    "numeric"
  )
})
