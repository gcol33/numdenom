test_that("quotr_formula parses basic formulas", {
  df <- data.frame(
    y_num = rpois(100, 10),
    y_denom = rpois(100, 20),
    x = rnorm(100),
    site = rep(1:10, each = 10)
  )

  f <- quotr_formula(
    numerator = y_num ~ x + (1 | site),
    denominator = y_denom ~ x + (1 | site),
    data = df
  )

  expect_s3_class(f, "quotr_formula")
  expect_equal(f$numerator$response_var, "y_num")
  expect_equal(f$denominator$response_var, "y_denom")
  expect_equal(ncol(f$numerator$X), 2)  # intercept + x
  expect_length(f$numerator$random_effects, 1)
})

test_that("offset() is rejected", {
  df <- data.frame(
    y_num = rpois(100, 10),
    y_denom = rpois(100, 20),
    x = rnorm(100),
    effort = runif(100, 1, 10)
  )

  expect_error(
    quotr_formula(
      numerator = y_num ~ x + offset(log(effort)),
      denominator = y_denom ~ x,
      data = df
    ),
    "offset\\(\\) is not allowed"
  )
})

test_that("shared = ~ 0 triggers warning", {
  df <- data.frame(
    y_num = rpois(100, 10),
    y_denom = rpois(100, 20),
    x = rnorm(100),
    site = rep(1:10, each = 10)
  )

  expect_warning(
    quotr_formula(
      numerator = y_num ~ x + (1 | site),
      denominator = y_denom ~ x + (1 | site),
      shared = ~ 0,
      data = df
    ),
    "independence"
  )
})

test_that("random effects are correctly parsed", {
  df <- data.frame(
    y_num = rpois(100, 10),
    y_denom = rpois(100, 20),
    x = rnorm(100),
    site = rep(1:10, each = 10),
    observer = rep(1:5, 20)
  )

  f <- quotr_formula(
    numerator = y_num ~ x + (1 | site) + (1 | observer),
    denominator = y_denom ~ (1 | site),
    data = df
  )

  # Should have 2 RE in numerator, 1 in denominator
  expect_length(f$numerator$random_effects, 2)
  expect_length(f$denominator$random_effects, 1)

  # Site should be inferred as shared
  expect_equal(f$shared$type, "inferred")
})

test_that("model matrix is built correctly", {
  df <- data.frame(
    y = rpois(100, 10),
    x1 = rnorm(100),
    x2 = rnorm(100),
    cat = factor(rep(c("A", "B"), 50))
  )

  f <- quotr_formula(
    numerator = y ~ x1 + x2 + cat,
    denominator = y ~ 1,
    data = df
  )

  # Intercept + x1 + x2 + catB
  expect_equal(ncol(f$numerator$X), 4)
  expect_true("(Intercept)" %in% colnames(f$numerator$X))
})
