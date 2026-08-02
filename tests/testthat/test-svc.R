# Tests for Spatially-Varying Coefficients (SVC)

test_that("spatial_svc() creates valid specification with formula coords", {
  svc <- spatial_svc(~ lon + lat, terms = 1)

  expect_s3_class(svc, "ratiod_svc")
  expect_s3_class(svc, "ratiod_spatial")
  expect_equal(svc$type, "svc")
  expect_equal(svc$coord_vars, c("lon", "lat"))
  expect_equal(svc$cov, "exponential")
  expect_equal(svc$nn, 15)
  expect_true(svc$shared)
  expect_true(svc$scale)
})

test_that("spatial_svc() creates valid specification with character coords", {
  svc <- spatial_svc(c("x", "y"), terms = c(1, 2))

  expect_equal(svc$coord_vars, c("x", "y"))
  expect_equal(svc$terms_spec$type, "index")
  expect_equal(svc$terms_spec$indices, c(1L, 2L))
})

test_that("spatial_svc() accepts different term specifications", {
  # By index
  svc1 <- spatial_svc(~ lon + lat, terms = c(1, 3))
  expect_equal(svc1$terms_spec$type, "index")
  expect_equal(svc1$terms_spec$indices, c(1L, 3L))

  # By name
  svc2 <- spatial_svc(~ lon + lat, terms = c("(Intercept)", "depth"))
  expect_equal(svc2$terms_spec$type, "names")
  expect_equal(svc2$terms_spec$names, c("(Intercept)", "depth"))

  # By formula
  svc3 <- spatial_svc(~ lon + lat, terms = ~ 1 + depth)
  expect_equal(svc3$terms_spec$type, "formula")
})

test_that("spatial_svc() accepts different covariance functions", {
  expect_equal(spatial_svc(~ x + y, cov = "exponential")$cov, "exponential")
  expect_equal(spatial_svc(~ x + y, cov = "matern")$cov, "matern")
  expect_equal(spatial_svc(~ x + y, cov = "gaussian")$cov, "gaussian")
  expect_equal(spatial_svc(~ x + y, cov = "spherical")$cov, "spherical")
})

test_that("spatial_svc() validates inputs", {
  # Bad coords formula

  expect_error(spatial_svc(~ lon, terms = 1), "exactly 2 coordinate")

  # Bad coords type
  expect_error(spatial_svc(123, terms = 1), "must be a formula")

  # Bad terms
  expect_error(spatial_svc(~ x + y, terms = TRUE), "must be a formula")

  # Bad nn
  expect_error(spatial_svc(~ x + y, nn = -1), "positive integer")
  expect_error(spatial_svc(~ x + y, nn = c(10, 20)), "positive integer")
})

test_that("spatial_svc() warns for non-shared SVCs", {
  expect_warning(
    spatial_svc(~ lon + lat, terms = 1, shared = FALSE),
    "confounded ratio"
  )
})

test_that("compute_nngp_neighbors() computes correct neighbors", {
  # Simple grid
  coords <- expand.grid(x = 1:3, y = 1:3)
  coords <- as.matrix(coords)

  neighbors <- tulpaRatio:::compute_nngp_neighbors(coords, k = 4)

  expect_equal(nrow(neighbors$nn_idx), 9)
  expect_equal(ncol(neighbors$nn_idx), 4)
  expect_equal(length(neighbors$nn_order), 9)

  # First observation should have no neighbors
  expect_true(all(neighbors$nn_idx[1, ] == 0))

  # Later observations should have neighbors
  expect_true(any(neighbors$nn_idx[5, ] > 0))
})

test_that("validate_svc() resolves term indices correctly", {
  svc <- spatial_svc(~ lon + lat, terms = c("(Intercept)", "depth"))

  # Mock data
  df <- data.frame(
    lon = runif(20),
    lat = runif(20),
    depth = rnorm(20)
  )

  # Mock design matrix
  X <- model.matrix(~ depth, data = df)

  validated <- tulpaRatio:::validate_svc(svc, df, X)

  expect_equal(validated$svc_indices, c(1L, 2L))
  expect_equal(validated$svc_names, c("(Intercept)", "depth"))
  expect_equal(validated$n_obs, 20)
  expect_equal(validated$n_svc, 2)
})

test_that("validate_svc() errors on missing columns", {
  svc <- spatial_svc(~ lon + lat, terms = 1)

  # Missing coordinate column
  df_bad <- data.frame(x = 1:10, lat = runif(10))
  X <- matrix(1, nrow = 10, ncol = 1)

  expect_error(tulpaRatio:::validate_svc(svc, df_bad, X), "lon")
})

test_that("validate_svc() errors on missing term names", {
  svc <- spatial_svc(~ lon + lat, terms = "nonexistent")

  df <- data.frame(lon = runif(10), lat = runif(10), x = rnorm(10))
  X <- model.matrix(~ x, data = df)
  colnames(X) <- c("(Intercept)", "x")

  expect_error(tulpaRatio:::validate_svc(svc, df, X), "nonexistent")
})

test_that("print.ratiod_svc() works", {
  svc <- spatial_svc(~ lon + lat, terms = c(1, 2), cov = "matern", nn = 20)

  expect_output(print(svc), "spatially-varying")
  expect_output(print(svc), "Coordinates:")
  expect_output(print(svc), "Covariance: matern")
  expect_output(print(svc), "Neighbors.*20")
})

test_that("validate_spatial() skips SVC objects",
{
  svc <- spatial_svc(~ lon + lat, terms = 1)
  df <- data.frame(lon = 1:5, lat = 1:5)

  # Should not error - SVC validation is separate
  expect_silent(tulpaRatio:::validate_spatial(svc, df))
})
