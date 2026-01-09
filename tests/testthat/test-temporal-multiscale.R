# Tests for temporal_multiscale()

test_that("temporal_multiscale creates valid object with defaults", {
  tm <- temporal_multiscale("year")

  expect_s3_class(tm, "ratiod_temporal_multiscale")
  expect_s3_class(tm, "ratiod_temporal")
  expect_equal(tm$type, "multiscale")
  expect_equal(tm$time_var, "year")
  expect_equal(tm$trend, "rw2")
  expect_equal(tm$short_term, "ar1")
  expect_null(tm$seasonal)
  expect_true(tm$shared)
})

test_that("temporal_multiscale accepts all component types", {
  # RW1 trend
  tm1 <- temporal_multiscale("year", trend = "rw1")
  expect_equal(tm1$trend, "rw1")

  # No trend
  tm2 <- temporal_multiscale("year", trend = "none", seasonal = 12)
  expect_equal(tm2$trend, "none")

  # IID short-term
  tm3 <- temporal_multiscale("year", short_term = "iid")
  expect_equal(tm3$short_term, "iid")

  # No short-term
 tm4 <- temporal_multiscale("year", short_term = "none")
  expect_equal(tm4$short_term, "none")
})

test_that("temporal_multiscale accepts seasonal specification", {
  tm <- temporal_multiscale("month", seasonal = 12)

  expect_equal(tm$seasonal, 12L)
  expect_true("seasonal" %in% tm$components)
})

test_that("temporal_multiscale components list is correct", {
  # All components
  tm1 <- temporal_multiscale("t", trend = "rw2", seasonal = 12, short_term = "ar1")
  expect_equal(tm1$components, c("trend", "seasonal", "short_term"))

  # Just trend
  tm2 <- temporal_multiscale("t", trend = "rw1", seasonal = NULL, short_term = "none")
  expect_equal(tm2$components, c("trend"))

  # Just seasonal
  tm3 <- temporal_multiscale("t", trend = "none", seasonal = 4, short_term = "none")
  expect_equal(tm3$components, c("seasonal"))

  # Trend + short-term
  tm4 <- temporal_multiscale("t", trend = "rw2", seasonal = NULL, short_term = "iid")
  expect_equal(tm4$components, c("trend", "short_term"))
})

test_that("temporal_multiscale rejects invalid time_var", {
  expect_error(temporal_multiscale(123), "character string")
  expect_error(temporal_multiscale(c("a", "b")), "single character string")
})

test_that("temporal_multiscale rejects invalid seasonal", {
  expect_error(temporal_multiscale("t", seasonal = 1), ">= 2")
  expect_error(temporal_multiscale("t", seasonal = 0), ">= 2")
  expect_error(temporal_multiscale("t", seasonal = -5), ">= 2")
})

test_that("temporal_multiscale rejects all-none components", {
  expect_error(
    temporal_multiscale("t", trend = "none", seasonal = NULL, short_term = "none"),
    "At least one"
  )
})

test_that("temporal_multiscale print method works", {
  tm <- temporal_multiscale("year", trend = "rw2", seasonal = 12, short_term = "ar1")
  output <- capture.output(print(tm))

  expect_true(any(grepl("Multi-Scale", output)))
  expect_true(any(grepl("Trend", output)))
  expect_true(any(grepl("RW2", output)))
  expect_true(any(grepl("Seasonal", output)))
  expect_true(any(grepl("12", output)))
  expect_true(any(grepl("AR\\(1\\)", output)))
})

# Tests for validate_temporal_multiscale

test_that("validate_temporal_multiscale computes indices", {
  tm <- temporal_multiscale("year", trend = "rw2")

  df <- data.frame(
    year = rep(2010:2020, each = 5),
    x = rnorm(55)
  )

  validated <- validate_temporal_multiscale(tm, df)

  expect_equal(validated$n_times, 11)
  expect_equal(length(validated$time_index), 55)
  expect_equal(validated$time_levels, as.character(2010:2020))
})

test_that("validate_temporal_multiscale handles groups", {
  tm <- temporal_multiscale("year", group_var = "site")

  df <- data.frame(
    year = rep(2010:2015, 3),
    site = rep(c("A", "B", "C"), each = 6)
  )

  validated <- validate_temporal_multiscale(tm, df)

  expect_equal(validated$n_times, 6)
  expect_equal(validated$n_groups, 3)
  expect_equal(validated$group_levels, c("A", "B", "C"))
})

test_that("validate_temporal_multiscale errors for missing time variable", {
  tm <- temporal_multiscale("year")
  df <- data.frame(time = 1:10)

  expect_error(validate_temporal_multiscale(tm, df), "year.*not found")
})

test_that("validate_temporal_multiscale errors for RW2 with < 3 time points", {
  tm <- temporal_multiscale("year", trend = "rw2")
  df <- data.frame(year = c(2020, 2021))

  expect_error(validate_temporal_multiscale(tm, df), "at least 3 time points")
})

test_that("validate_temporal_multiscale warns for short series with seasonal", {
  tm <- temporal_multiscale("month", trend = "none", seasonal = 12, short_term = "none")
  df <- data.frame(month = 1:6)

  expect_warning(validate_temporal_multiscale(tm, df), "less than seasonal period")
})

test_that("validate_temporal_multiscale builds precision structures", {
  tm <- temporal_multiscale("year", trend = "rw2", seasonal = 12, short_term = "ar1")

  df <- data.frame(year = rep(1:20, 5))

  validated <- validate_temporal_multiscale(tm, df)

  expect_true("trend" %in% names(validated$precision_structures))
  expect_true("seasonal" %in% names(validated$precision_structures))
  expect_true("short_term" %in% names(validated$precision_structures))

  expect_equal(validated$precision_structures$trend$type, "rw2")
  expect_equal(validated$precision_structures$seasonal$type, "rw1")
  expect_true(validated$precision_structures$seasonal$cyclic)
  expect_equal(validated$precision_structures$short_term$type, "ar1")
})

test_that("validate_temporal_multiscale calculates n_temporal_params correctly", {
  # Trend (T=10) + seasonal (P=4) + short-term (T=10) = 24
  tm <- temporal_multiscale("t", trend = "rw1", seasonal = 4, short_term = "iid")
  df <- data.frame(t = 1:10)

  validated <- validate_temporal_multiscale(tm, df)
  expect_equal(validated$n_temporal_params, 10 + 4 + 10)

  # With 3 groups: 24 * 3 = 72
  tm2 <- temporal_multiscale("t", trend = "rw1", seasonal = 4, short_term = "iid",
                             group_var = "g")
  df2 <- data.frame(t = rep(1:10, 3), g = rep(c("A", "B", "C"), each = 10))

  validated2 <- validate_temporal_multiscale(tm2, df2)
  expect_equal(validated2$n_temporal_params, (10 + 4 + 10) * 3)
})

test_that("validate_temporal_multiscale handles factor time variable", {
  tm <- temporal_multiscale("period")

  df <- data.frame(
    period = factor(rep(c("Q1", "Q2", "Q3", "Q4"), 5),
                    levels = c("Q1", "Q2", "Q3", "Q4"))
  )

  validated <- validate_temporal_multiscale(tm, df)

  expect_equal(validated$n_times, 4)
  expect_equal(validated$time_levels, c("Q1", "Q2", "Q3", "Q4"))
})
