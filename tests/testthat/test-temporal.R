# test-temporal.R
# Tests for temporal structure specification and HMC integration

test_that("temporal_rw1 creates correct structure", {
  temp <- temporal_rw1("time_var")

  expect_s3_class(temp, "quotr_temporal")
  expect_equal(temp$type, "rw1")
  expect_equal(temp$time_var, "time_var")
  expect_false(temp$cyclic)
  expect_true(temp$shared)
})


test_that("temporal_rw1 with cyclic option", {
  temp <- temporal_rw1("month", cyclic = TRUE)

  expect_s3_class(temp, "quotr_temporal")
  expect_equal(temp$type, "rw1")
  expect_true(temp$cyclic)
})


test_that("temporal_rw2 creates correct structure", {
  temp <- temporal_rw2("year")

  expect_s3_class(temp, "quotr_temporal")
  expect_equal(temp$type, "rw2")
  expect_equal(temp$time_var, "year")
  expect_false(temp$cyclic)
  expect_true(temp$shared)
})


test_that("temporal_ar1 creates correct structure", {
  temp <- temporal_ar1("time_var")

  expect_s3_class(temp, "quotr_temporal")
  expect_equal(temp$type, "ar1")
  expect_equal(temp$time_var, "time_var")
  expect_true(temp$shared)
})


test_that("temporal with panel group", {
  temp <- temporal_rw1("year", group_var = "site")

  expect_s3_class(temp, "quotr_temporal")
  expect_equal(temp$type, "rw1")
  expect_equal(temp$time_var, "year")
  expect_equal(temp$group_var, "site")
})


test_that("temporal shared=FALSE works", {
  temp <- temporal_rw1("time_var", shared = FALSE)

  expect_s3_class(temp, "quotr_temporal")
  expect_false(temp$shared)
})


test_that("temporal print method works", {
  temp <- temporal_rw1("year")
  output <- capture.output(print(temp))
  expect_true(any(grepl("temporal", output, ignore.case = TRUE)))
  expect_true(any(grepl("RW1", output)))
})


test_that("validate_temporal extracts time indices", {
  df <- data.frame(
    y = rbinom(20, 10, 0.5),
    n = rep(10, 20),
    time = rep(1:5, 4)
  )

  temp <- temporal_rw1("time")
  validated <- validate_temporal(temp, df)

  expect_equal(length(validated$time_index), 20)
  expect_equal(validated$n_times, 5)
  expect_equal(validated$n_groups, 1)  # No panel group
})


test_that("validate_temporal with panel groups", {
  df <- data.frame(
    y = rbinom(30, 10, 0.5),
    n = rep(10, 30),
    time = rep(1:5, 6),
    site = rep(1:3, each = 10)
  )

  temp <- temporal_rw1("time", group_var = "site")
  validated <- validate_temporal(temp, df)

  expect_equal(length(validated$time_index), 30)
  expect_equal(validated$n_times, 5)
  expect_equal(validated$n_groups, 3)
  expect_equal(validated$n_temporal_params, 15)  # 5 times * 3 groups
})


test_that("HMC with temporal RW1 runs", {
  skip_on_cran()

  set.seed(333)
  n_times <- 8
  n_per_time <- 20
  n <- n_times * n_per_time

  # Simulate temporal effect (random walk)
  temporal_effect <- cumsum(rnorm(n_times, 0, 0.2))
  temporal_effect <- temporal_effect - mean(temporal_effect)

  time_idx <- rep(1:n_times, each = n_per_time)
  x <- rnorm(n)
  p <- plogis(-0.5 + 0.5 * x + temporal_effect[time_idx])
  trials <- rpois(n, 20) + 10
  successes <- rbinom(n, trials, p)

  X <- cbind(1, x)

  # Parameters: 2 beta_num + 1 beta_denom + 1 log_tau_temporal + 8 temporal = 12
  result <- quotr:::cpp_hmc_fit(
    q_init = rep(0, 12),
    y_num = as.integer(successes),
    y_denom = as.integer(trials),
    y_denom_cont = rep(0.0, n),
    X_num = X,
    X_denom = matrix(1, n, 1),
    re_group = rep(0L, n),
    n_re_groups = 0L,
    model_type_str = "binomial",
    spatial_type_str = "none",
    spatial_group = rep(0L, n),
    n_spatial_units = 0L,
    adj_row_ptr = 0L,
    adj_col_idx = integer(0),
    n_neighbors = integer(0),
    bym2_scale_factor = 1.0,
    temporal_type_str = "rw1",
    temporal_time_idx = as.integer(time_idx),
    temporal_group_idx = rep(1L, n),  # Single group
    n_times = n_times,
    n_temporal_groups = 1L,
    n_temporal_params = n_times,
    temporal_cyclic = FALSE,
    temporal_shared = TRUE,
    tau_temporal_shape = 2.0,
    tau_temporal_rate = 0.5,
    sigma_beta = 10.0,
    sigma_re_scale = 2.5,
    phi_prior_shape = 2.0,
    phi_prior_rate = 0.1,
    tau_spatial_shape = 1.0,
    tau_spatial_rate = 0.01,
    zi_type_str = "none",
    X_zi = matrix(0, nrow = n, ncol = 1),
    zi_prior_sd = 10.0,
    n_iter = 800L,
    n_warmup = 400L,
    L = 10L,
    n_chains = 1L,
    seed = 333L,
    n_threads = 1L,
    verbose = FALSE
  )

  expect_equal(ncol(result$samples), 12)
  expect_equal(nrow(result$samples), 400)

  # Check temporal precision is positive
  log_tau_idx <- 4
  tau_est <- mean(exp(result$samples[, log_tau_idx]))
  expect_true(tau_est > 0)

  # Check temporal effects sum approximately to zero (soft constraint)
  temporal_cols <- 5:12
  temporal_means <- colMeans(result$samples[, temporal_cols])
  expect_true(abs(sum(temporal_means)) < 1.0)  # Reasonable constraint
})


test_that("HMC with temporal AR1 runs", {
  skip_on_cran()

  set.seed(444)
  n_times <- 10
  n_per_time <- 15
  n <- n_times * n_per_time

  # Simulate AR1 process
  rho_true <- 0.7
  temporal_effect <- numeric(n_times)
  temporal_effect[1] <- rnorm(1, 0, 0.3)
  for (t in 2:n_times) {
    temporal_effect[t] <- rho_true * temporal_effect[t-1] + rnorm(1, 0, 0.2)
  }

  time_idx <- rep(1:n_times, each = n_per_time)
  x <- rnorm(n)
  p <- plogis(-0.3 + 0.4 * x + temporal_effect[time_idx])
  trials <- rpois(n, 15) + 10
  successes <- rbinom(n, trials, p)

  X <- cbind(1, x)

  # Parameters: 2 beta_num + 1 beta_denom + 1 log_tau + 1 logit_rho + 10 temporal = 15
  result <- quotr:::cpp_hmc_fit(
    q_init = rep(0, 15),
    y_num = as.integer(successes),
    y_denom = as.integer(trials),
    y_denom_cont = rep(0.0, n),
    X_num = X,
    X_denom = matrix(1, n, 1),
    re_group = rep(0L, n),
    n_re_groups = 0L,
    model_type_str = "binomial",
    spatial_type_str = "none",
    spatial_group = rep(0L, n),
    n_spatial_units = 0L,
    adj_row_ptr = 0L,
    adj_col_idx = integer(0),
    n_neighbors = integer(0),
    bym2_scale_factor = 1.0,
    temporal_type_str = "ar1",
    temporal_time_idx = as.integer(time_idx),
    temporal_group_idx = rep(1L, n),
    n_times = n_times,
    n_temporal_groups = 1L,
    n_temporal_params = n_times,
    temporal_cyclic = FALSE,
    temporal_shared = TRUE,
    tau_temporal_shape = 2.0,
    tau_temporal_rate = 0.5,
    sigma_beta = 10.0,
    sigma_re_scale = 2.5,
    phi_prior_shape = 2.0,
    phi_prior_rate = 0.1,
    tau_spatial_shape = 1.0,
    tau_spatial_rate = 0.01,
    zi_type_str = "none",
    X_zi = matrix(0, nrow = n, ncol = 1),
    zi_prior_sd = 10.0,
    n_iter = 800L,
    n_warmup = 400L,
    L = 10L,
    n_chains = 1L,
    seed = 444L,
    n_threads = 1L,
    verbose = FALSE
  )

  expect_equal(ncol(result$samples), 15)
  expect_equal(nrow(result$samples), 400)

  # Check rho is estimated (transformed from logit)
  logit_rho_idx <- 5  # After 2 beta_num + 1 beta_denom + 1 log_tau
  rho_samples <- plogis(result$samples[, logit_rho_idx])
  rho_mean <- mean(rho_samples)
  expect_true(rho_mean > 0 && rho_mean < 1)
})


test_that("HMC with cyclic RW1 (seasonal) runs", {
  skip_on_cran()

  set.seed(555)
  n_months <- 12
  n_years <- 3
  n_per_month <- 5
  n <- n_months * n_years * n_per_month

  # Simulate seasonal effect
  seasonal_effect <- sin(2 * pi * (1:n_months) / 12) * 0.3

  time_idx <- rep(rep(1:n_months, n_years), each = n_per_month)
  x <- rnorm(n)
  p <- plogis(-0.2 + 0.3 * x + seasonal_effect[time_idx])
  trials <- rpois(n, 20) + 10
  successes <- rbinom(n, trials, p)

  X <- cbind(1, x)

  # Parameters: 2 beta_num + 1 beta_denom + 1 log_tau + 12 temporal = 16
  result <- quotr:::cpp_hmc_fit(
    q_init = rep(0, 16),
    y_num = as.integer(successes),
    y_denom = as.integer(trials),
    y_denom_cont = rep(0.0, n),
    X_num = X,
    X_denom = matrix(1, n, 1),
    re_group = rep(0L, n),
    n_re_groups = 0L,
    model_type_str = "binomial",
    spatial_type_str = "none",
    spatial_group = rep(0L, n),
    n_spatial_units = 0L,
    adj_row_ptr = 0L,
    adj_col_idx = integer(0),
    n_neighbors = integer(0),
    bym2_scale_factor = 1.0,
    temporal_type_str = "rw1",
    temporal_time_idx = as.integer(time_idx),
    temporal_group_idx = rep(1L, n),
    n_times = n_months,
    n_temporal_groups = 1L,
    n_temporal_params = n_months,
    temporal_cyclic = TRUE,  # Cyclic for seasonal pattern
    temporal_shared = TRUE,
    tau_temporal_shape = 2.0,
    tau_temporal_rate = 0.5,
    sigma_beta = 10.0,
    sigma_re_scale = 2.5,
    phi_prior_shape = 2.0,
    phi_prior_rate = 0.1,
    tau_spatial_shape = 1.0,
    tau_spatial_rate = 0.01,
    zi_type_str = "none",
    X_zi = matrix(0, nrow = n, ncol = 1),
    zi_prior_sd = 10.0,
    n_iter = 600L,
    n_warmup = 300L,
    L = 10L,
    n_chains = 1L,
    seed = 555L,
    n_threads = 1L,
    verbose = FALSE
  )

  expect_equal(ncol(result$samples), 16)
  expect_equal(nrow(result$samples), 300)
})


test_that("HMC with panel temporal (group-specific) runs", {
  skip_on_cran()

  set.seed(666)
  n_times <- 5
  n_groups <- 3
  n_per <- 10
  n <- n_times * n_groups * n_per

  # Simulate group-specific temporal effects
  temporal_effects <- matrix(0, n_times, n_groups)
  for (g in 1:n_groups) {
    temporal_effects[, g] <- cumsum(rnorm(n_times, 0, 0.15))
    temporal_effects[, g] <- temporal_effects[, g] - mean(temporal_effects[, g])
  }

  time_idx <- rep(rep(1:n_times, each = n_per), n_groups)
  group_idx <- rep(1:n_groups, each = n_times * n_per)
  x <- rnorm(n)

  # Linear predictor
  eta <- -0.4 + 0.5 * x
  for (i in seq_along(eta)) {
    eta[i] <- eta[i] + temporal_effects[time_idx[i], group_idx[i]]
  }
  p <- plogis(eta)
  trials <- rpois(n, 15) + 5
  successes <- rbinom(n, trials, p)

  X <- cbind(1, x)

  # Parameters: 2 beta_num + 1 beta_denom + 1 log_tau + 15 temporal (5*3) = 19
  result <- quotr:::cpp_hmc_fit(
    q_init = rep(0, 19),
    y_num = as.integer(successes),
    y_denom = as.integer(trials),
    y_denom_cont = rep(0.0, n),
    X_num = X,
    X_denom = matrix(1, n, 1),
    re_group = rep(0L, n),
    n_re_groups = 0L,
    model_type_str = "binomial",
    spatial_type_str = "none",
    spatial_group = rep(0L, n),
    n_spatial_units = 0L,
    adj_row_ptr = 0L,
    adj_col_idx = integer(0),
    n_neighbors = integer(0),
    bym2_scale_factor = 1.0,
    temporal_type_str = "rw1",
    temporal_time_idx = as.integer(time_idx),
    temporal_group_idx = as.integer(group_idx),
    n_times = n_times,
    n_temporal_groups = n_groups,
    n_temporal_params = n_times * n_groups,
    temporal_cyclic = FALSE,
    temporal_shared = TRUE,
    tau_temporal_shape = 2.0,
    tau_temporal_rate = 0.5,
    sigma_beta = 10.0,
    sigma_re_scale = 2.5,
    phi_prior_shape = 2.0,
    phi_prior_rate = 0.1,
    tau_spatial_shape = 1.0,
    tau_spatial_rate = 0.01,
    zi_type_str = "none",
    X_zi = matrix(0, nrow = n, ncol = 1),
    zi_prior_sd = 10.0,
    n_iter = 600L,
    n_warmup = 300L,
    L = 10L,
    n_chains = 1L,
    seed = 666L,
    n_threads = 1L,
    verbose = FALSE
  )

  expect_equal(ncol(result$samples), 19)
  expect_equal(nrow(result$samples), 300)
})


test_that("HMC with temporal runs multiple chains in parallel", {
  skip_on_cran()

  set.seed(777)
  n_times <- 6
  n_per_time <- 15
  n <- n_times * n_per_time

  # Simulate temporal effect (random walk)
  temporal_effect <- cumsum(rnorm(n_times, 0, 0.2))
  temporal_effect <- temporal_effect - mean(temporal_effect)

  time_idx <- rep(1:n_times, each = n_per_time)
  x <- rnorm(n)
  p <- plogis(-0.4 + 0.4 * x + temporal_effect[time_idx])
  trials <- rpois(n, 15) + 10
  successes <- rbinom(n, trials, p)

  X <- cbind(1, x)

  # Parameters: 2 beta_num + 1 beta_denom + 1 log_tau_temporal + 6 temporal = 10
  result <- quotr:::cpp_hmc_fit(
    q_init = rep(0, 10),
    y_num = as.integer(successes),
    y_denom = as.integer(trials),
    y_denom_cont = rep(0.0, n),
    X_num = X,
    X_denom = matrix(1, n, 1),
    re_group = rep(0L, n),
    n_re_groups = 0L,
    model_type_str = "binomial",
    spatial_type_str = "none",
    spatial_group = rep(0L, n),
    n_spatial_units = 0L,
    adj_row_ptr = 0L,
    adj_col_idx = integer(0),
    n_neighbors = integer(0),
    bym2_scale_factor = 1.0,
    temporal_type_str = "rw1",
    temporal_time_idx = as.integer(time_idx),
    temporal_group_idx = rep(1L, n),
    n_times = n_times,
    n_temporal_groups = 1L,
    n_temporal_params = n_times,
    temporal_cyclic = FALSE,
    temporal_shared = TRUE,
    tau_temporal_shape = 2.0,
    tau_temporal_rate = 0.5,
    sigma_beta = 10.0,
    sigma_re_scale = 2.5,
    phi_prior_shape = 2.0,
    phi_prior_rate = 0.1,
    tau_spatial_shape = 1.0,
    tau_spatial_rate = 0.01,
    zi_type_str = "none",
    X_zi = matrix(0, nrow = n, ncol = 1),
    zi_prior_sd = 10.0,
    n_iter = 400L,
    n_warmup = 200L,
    L = 10L,
    n_chains = 2L,  # Multiple chains in parallel
    seed = 777L,
    n_threads = 1L,
    verbose = FALSE
  )

  # Multi-chain returns lists
  expect_true(is.list(result$samples))
  expect_equal(length(result$samples), 2)

  # Each chain has correct dimensions
  expect_equal(nrow(result$samples[[1]]), 200)
  expect_equal(nrow(result$samples[[2]]), 200)
  expect_equal(ncol(result$samples[[1]]), 10)
  expect_equal(ncol(result$samples[[2]]), 10)

  # Results should be different between chains (different random seeds)
  chain1_mean <- colMeans(result$samples[[1]])
  chain2_mean <- colMeans(result$samples[[2]])
  expect_false(all(chain1_mean == chain2_mean))
})
