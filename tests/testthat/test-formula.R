test_that("combined formula syntax works: num | denom ~ x", {
  df <- data.frame(
    y_num = rpois(100, 10),
    y_denom = rpois(100, 20),
    x = rnorm(100),
    site = rep(1:10, each = 10)
  )

  f <- quotr_formula(
    formula = y_num | y_denom ~ x + (1 | site),
    data = df
  )

  expect_s3_class(f, "quotr_formula")
  expect_equal(f$numerator$response_var, "y_num")
  expect_equal(f$denominator$response_var, "y_denom")
  expect_equal(ncol(f$numerator$X), 2)  # intercept + x
  expect_equal(ncol(f$denominator$X), 2)  # same predictors
  expect_length(f$numerator$random_effects, 1)
  expect_length(f$denominator$random_effects, 1)
})

test_that("combined formula with additional process-specific terms", {
  df <- data.frame(
    y_num = rpois(100, 10),
    y_denom = rpois(100, 20),
    x = rnorm(100),
    z = rnorm(100),
    site = rep(1:10, each = 10)
  )

  f <- quotr_formula(
    formula = y_num | y_denom ~ (1 | site),
    formula_num = ~ x,
    formula_denom = ~ z,
    data = df
  )

  # Numerator: intercept + x
  expect_equal(ncol(f$numerator$X), 2)
  expect_true("x" %in% colnames(f$numerator$X))

  # Denominator: intercept + z
  expect_equal(ncol(f$denominator$X), 2)
  expect_true("z" %in% colnames(f$denominator$X))
})

test_that("separate formula syntax works", {
  df <- data.frame(
    y_num = rpois(100, 10),
    y_denom = rpois(100, 20),
    x = rnorm(100),
    site = rep(1:10, each = 10)
  )

  f <- quotr_formula(
    formula = y_num ~ x + (1 | site),
    formula_denom = y_denom ~ (1 | site),
    data = df
  )

  expect_equal(f$numerator$response_var, "y_num")
  expect_equal(f$denominator$response_var, "y_denom")
  expect_equal(ncol(f$numerator$X), 2)  # intercept + x
  expect_equal(ncol(f$denominator$X), 1)  # intercept only
})

test_that("separate formula requires formula_denom", {
  df <- data.frame(
    y_num = rpois(100, 10),
    x = rnorm(100)
  )

  expect_error(
    quotr_formula(
      formula = y_num ~ x,
      data = df
    ),
    "formula_denom.*required"
  )
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
      formula = y_num | y_denom ~ x + offset(log(effort)),
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
      formula = y_num | y_denom ~ x + (1 | site),
      shared = ~ 0,
      data = df
    ),
    "independence"
  )
})

test_that("random effects are correctly inferred as shared", {
  df <- data.frame(
    y_num = rpois(100, 10),
    y_denom = rpois(100, 20),
    x = rnorm(100),
    site = rep(1:10, each = 10)
  )

  f <- quotr_formula(
    formula = y_num | y_denom ~ x + (1 | site),
    data = df
  )

  # Site should be inferred as shared (appears in both)
  expect_equal(f$shared$type, "inferred")
  expect_length(f$shared$random_effects, 1)
})

test_that("different RE in each process are handled", {
  df <- data.frame(
    y_num = rpois(100, 10),
    y_denom = rpois(100, 20),
    x = rnorm(100),
    site = rep(1:10, each = 10),
    observer = rep(1:5, 20)
  )

  f <- quotr_formula(
    formula = y_num ~ x + (1 | site) + (1 | observer),
    formula_denom = y_denom ~ (1 | site),
    data = df
  )

  # Numerator has 2 RE, denominator has 1
  expect_length(f$numerator$random_effects, 2)
  expect_length(f$denominator$random_effects, 1)

  # Only site is shared
  expect_equal(f$shared$type, "inferred")
  expect_length(f$shared$random_effects, 1)
  expect_equal(f$shared$random_effects[[1]]$group_var, "site")
})

test_that("model matrix is built correctly with factors", {
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20),
    x1 = rnorm(100),
    x2 = rnorm(100),
    cat = factor(rep(c("A", "B"), 50))
  )

  f <- quotr_formula(
    formula = y | n ~ x1 + x2 + cat,
    data = df
  )

  # Intercept + x1 + x2 + catB
  expect_equal(ncol(f$numerator$X), 4)
  expect_true("(Intercept)" %in% colnames(f$numerator$X))
  expect_true("catB" %in% colnames(f$numerator$X))
})

test_that("missing variables trigger errors", {
  df <- data.frame(
    y_num = rpois(100, 10),
    x = rnorm(100)
  )

  expect_error(
    quotr_formula(
      formula = y_num | missing_var ~ x,
      data = df
    ),
    "not found in data"
  )
})
