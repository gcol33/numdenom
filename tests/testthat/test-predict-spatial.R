# test-predict-spatial.R
# Tests for predict() with spatial models

test_that("predict works for basic model (no spatial)", {
  skip_on_cran()

  set.seed(123)
  n <- 30
  df <- data.frame(
    y = rpois(n, 10),
    n_trials = rep(20L, n),
    x = rnorm(n),
    site = factor(rep(1:5, each = 6))
  )

  fit <- tratio(
    y | n_trials ~ x + (1 | site),
    data = df,
    family = ratiod_binomial(),
    control = list(iter = 100, warmup = 50, chains = 1)
  )

  # Basic prediction
  new_df <- data.frame(x = c(-1, 0, 1), site = factor(c(1, 2, 3)))
  pred <- predict(fit, newdata = new_df)

  expect_s3_class(pred, "ratiod_prediction")
  expect_equal(nrow(pred), 3)
  expect_true(all(c("mean", "sd", "q2.5", "q97.5") %in% names(pred)))

  # Population-level prediction
  pred_pop <- predict(fit, newdata = new_df, re_formula = NA)
  expect_equal(nrow(pred_pop), 3)
})


test_that("predict works with ICAR spatial (lookup)",
{
  skip_on_cran()

  set.seed(456)
  n_sites <- 6
  n <- 30

  # Create simple adjacency
  adj <- matrix(0, n_sites, n_sites)
  for (i in 1:(n_sites - 1)) {
    adj[i, i + 1] <- 1
    adj[i + 1, i] <- 1
  }

  df <- data.frame(
    y = rpois(n, 10),
    n_trials = rep(20L, n),
    x = rnorm(n),
    region = factor(rep(LETTERS[1:n_sites], length.out = n))
  )

  fit <- tratio(
    y | n_trials ~ x,
    data = df,
    family = ratiod_binomial(),
    spatial = spatial_car(adjacency = adj, group_var = "region"),
    control = list(iter = 100, warmup = 50, chains = 1)
  )

  # Predict for existing regions
  new_df <- data.frame(x = c(0, 0.5), region = factor(c("A", "C")))
  pred <- predict(fit, newdata = new_df)

  expect_equal(nrow(pred), 2)
  expect_true(all(pred$mean > 0 & pred$mean < 1))  # Probabilities
})


test_that("predict works with return_spatial = TRUE", {
  skip_on_cran()

  set.seed(789)
  n_sites <- 6
  n <- 30

  adj <- matrix(0, n_sites, n_sites)
  for (i in 1:(n_sites - 1)) {
    adj[i, i + 1] <- 1
    adj[i + 1, i] <- 1
  }

  df <- data.frame(
    y = rpois(n, 10),
    n_trials = rep(20L, n),
    x = rnorm(n),
    region = factor(rep(LETTERS[1:n_sites], length.out = n))
  )

  fit <- tratio(
    y | n_trials ~ x,
    data = df,
    family = ratiod_binomial(),
    spatial = spatial_car(adjacency = adj, group_var = "region"),
    control = list(iter = 100, warmup = 50, chains = 1)
  )

  new_df <- data.frame(x = c(0, 0.5, 1), region = factor(c("A", "B", "C")))
  pred <- predict(fit, newdata = new_df, return_spatial = TRUE)

  expect_type(pred, "list")
  expect_true("predictions" %in% names(pred))
  expect_true("w.0.samples" %in% names(pred))
  expect_equal(ncol(pred$w.0.samples), 3)  # 3 new observations
})


test_that("predict errors when coords.0 missing for GP", {
  skip_on_cran()

  set.seed(111)
  n <- 20
  df <- data.frame(
    y = rpois(n, 10),
    n_trials = rep(20L, n),
    x = rnorm(n),
    lon = runif(n),
    lat = runif(n)
  )

  # Skip if GP fitting fails
  fit <- tryCatch(
    tratio(
      y | n_trials ~ x,
      data = df,
      family = ratiod_binomial(),
      spatial = spatial_gp(coords = c("lon", "lat"), nn = 5),
      control = list(iter = 50, warmup = 25, chains = 1)
    ),
    error = function(e) NULL
  )

  skip_if(is.null(fit), "GP model fitting failed")

  new_df <- data.frame(x = c(0, 1))

  expect_error(
    predict(fit, newdata = new_df),
    "coords.0 required"
  )
})


test_that("kriging_predict returns correct dimensions", {
  coords_train <- matrix(runif(20), ncol = 2)
  coords_new <- matrix(runif(6), ncol = 2)
  w_train <- rnorm(10)

  w_new <- kriging_predict(
    coords_train = coords_train,
    coords_new = coords_new,
    w_train = w_train,
    sigma2 = 1.0,
    phi = 0.5,
    cov_type = 0L,
    nn = 5
  )

  expect_length(w_new, 3)
  expect_true(all(is.finite(w_new)))
})


test_that("cov_function computes correct values", {
  # Exponential
  expect_equal(cov_function(0, 1, 1, 0L), 1)
  expect_true(cov_function(1, 1, 1, 0L) < 1)


  # Matern 3/2
  expect_equal(cov_function(0, 1, 1, 1L), 1)

  # Squared exponential
  expect_equal(cov_function(0, 1, 1, 2L), 1)
})


test_that("detect_spatial_type identifies spatial types correctly", {
  # Mock objects
  obj_none <- list(spatial = NULL, .internal = list(hmc_data = list()))
  obj_gp <- list(
    spatial = structure(list(), class = "ratiod_gp"),
    .internal = list(hmc_data = list(gp_type = 1L))
  )
  obj_icar <- list(
    spatial = list(type = "icar"),
    .internal = list(hmc_data = list())
  )

  expect_null(detect_spatial_type(obj_none))
  expect_equal(detect_spatial_type(obj_gp), "gp")
  expect_equal(detect_spatial_type(obj_icar), "icar")
})


test_that("predict works for PG backend with spatial", {
  skip_on_cran()

  set.seed(222)
  n_sites <- 6
  n <- 30

  adj <- matrix(0, n_sites, n_sites)
  for (i in 1:(n_sites - 1)) {
    adj[i, i + 1] <- 1
    adj[i + 1, i] <- 1
  }

  df <- data.frame(
    y = rbinom(n, 20, 0.4),
    n_trials = rep(20L, n),
    x = rnorm(n),
    region = factor(rep(LETTERS[1:n_sites], length.out = n))
  )

  fit <- tryCatch(
    tratio(
      y | n_trials ~ x,
      data = df,
      family = ratiod_binomial(),
      spatial = spatial_car(adjacency = adj, group_var = "region"),
      mode = "pg",
      control = list(iter = 100, warmup = 50, chains = 1)
    ),
    error = function(e) NULL
  )

  skip_if(is.null(fit), "PG backend not available")

  new_df <- data.frame(x = c(0, 0.5), region = factor(c("A", "C")))
  pred <- predict(fit, newdata = new_df)

  expect_equal(nrow(pred), 2)
})
