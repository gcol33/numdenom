# Tests for spatial_rsr() - Restricted Spatial Regression

test_that("spatial_rsr wraps spatial_gp correctly", {
  gp <- spatial_gp(~ lon + lat)
  rsr <- spatial_rsr(gp, restrict_to = ~ x1 + x2)

  expect_s3_class(rsr, "ratiod_rsr")
  expect_s3_class(rsr, "ratiod_gp")
  expect_s3_class(rsr, "ratiod_spatial")
  expect_true(rsr$rsr)
  expect_equal(rsr$rsr_formula, ~ x1 + x2)
})

test_that("spatial_rsr wraps spatial_car correctly", {
  adj <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), 3, 3)
  car <- spatial_car(adj, level = "obs")
  rsr <- spatial_rsr(car, restrict_to = ~ temp)

  expect_s3_class(rsr, "ratiod_rsr")
  expect_s3_class(rsr, "ratiod_spatial")
  expect_true(rsr$rsr)
})

test_that("spatial_rsr wraps spatial_multiscale correctly", {
  ms <- spatial_multiscale(~ lon + lat)
  rsr <- spatial_rsr(ms, restrict_to = ~ depth)

  expect_s3_class(rsr, "ratiod_rsr")
  expect_s3_class(rsr, "ratiod_multiscale")
  expect_true(rsr$rsr)
})

test_that("spatial_rsr rejects non-spatial input", {
  expect_error(spatial_rsr("not spatial", ~ x), "tulpaRatio spatial specification")
  expect_error(spatial_rsr(list(a = 1), ~ x), "tulpaRatio spatial specification")
})

test_that("spatial_rsr rejects non-formula restrict_to", {
  gp <- spatial_gp(~ lon + lat)
  expect_error(spatial_rsr(gp, "x1"), "must be a formula")
  expect_error(spatial_rsr(gp, c("x1", "x2")), "must be a formula")
})

test_that("spatial_rsr print method works", {
  gp <- spatial_gp(~ lon + lat)
  rsr <- spatial_rsr(gp, restrict_to = ~ depth + temp)

  output <- capture.output(print(rsr))

  expect_true(any(grepl("Gaussian Process", output)))
  expect_true(any(grepl("Restricted Spatial Regression", output)))
  expect_true(any(grepl("depth", output)))
  expect_true(any(grepl("temp", output)))
})

# Tests for compute_rsr_projection

test_that("compute_rsr_projection creates valid projection matrix", {
  # Simple design matrix
  set.seed(123)
  X <- cbind(1, rnorm(20), rnorm(20))

  P_perp <- compute_rsr_projection(X)

  expect_equal(dim(P_perp), c(20, 20))

  # P_perp should be idempotent: P_perp %*% P_perp = P_perp
  expect_equal(P_perp %*% P_perp, P_perp, tolerance = 1e-10)

  # P_perp should be symmetric
  expect_equal(P_perp, t(P_perp), tolerance = 1e-10)
})

test_that("compute_rsr_projection orthogonalizes correctly", {
  set.seed(123)
  n <- 30
  X <- cbind(1, rnorm(n), rnorm(n))

  P_perp <- compute_rsr_projection(X)

  # Random vector
  w <- rnorm(n)

  # Projected vector should be orthogonal to X
  w_projected <- P_perp %*% w

  # X' %*% w_projected should be (approximately) zero
  orthogonality <- t(X) %*% w_projected
  expect_equal(as.vector(orthogonality), rep(0, 3), tolerance = 1e-10)
})

test_that("compute_rsr_projection warns for high-dimensional case", {
  # More columns than rows
  X <- matrix(rnorm(50), nrow = 5, ncol = 10)

  expect_warning(compute_rsr_projection(X), "More covariates than observations")
})

# Tests for validate_rsr

test_that("validate_rsr computes projection matrix", {
  gp <- spatial_gp(~ x + y)
  rsr <- spatial_rsr(gp, restrict_to = ~ z1 + z2)

  df <- data.frame(
    x = rnorm(20),
    y = rnorm(20),
    z1 = rnorm(20),
    z2 = rnorm(20)
  )

  validated <- validate_rsr(rsr, df, NULL)

  expect_true(!is.null(validated$rsr_projection))
  expect_equal(dim(validated$rsr_projection), c(20, 20))
  expect_equal(validated$rsr_vars, c("z1", "z2"))
})

test_that("validate_rsr errors for missing variables", {
  gp <- spatial_gp(~ x + y)
  rsr <- spatial_rsr(gp, restrict_to = ~ missing_var)

  df <- data.frame(x = 1:10, y = 1:10)

  expect_error(validate_rsr(rsr, df, NULL), "not found in data")
})

test_that("validate_rsr returns non-RSR spatial unchanged", {
  gp <- spatial_gp(~ x + y)
  df <- data.frame(x = 1:10, y = 1:10)

  # Non-RSR spatial should pass through unchanged
  result <- validate_rsr(gp, df, NULL)
  expect_identical(result, gp)
})

# Tests for apply_rsr_projection

test_that("apply_rsr_projection applies projection correctly", {
  set.seed(123)
  n <- 20
  X <- cbind(1, rnorm(n))
  P_perp <- compute_rsr_projection(X)

  w <- rnorm(n)
  w_rsr <- apply_rsr_projection(w, P_perp)

  expect_length(w_rsr, n)

  # Result should be orthogonal to X
  expect_equal(as.vector(t(X) %*% w_rsr), c(0, 0), tolerance = 1e-10)
})

test_that("apply_rsr_projection is idempotent", {
  set.seed(456)
  n <- 15
  X <- cbind(1, rnorm(n), rnorm(n))
  P_perp <- compute_rsr_projection(X)

  w <- rnorm(n)

  # First projection
  w1 <- apply_rsr_projection(w, P_perp)

  # Second projection should give same result
  w2 <- apply_rsr_projection(w1, P_perp)

  expect_equal(w1, w2, tolerance = 1e-10)
})
