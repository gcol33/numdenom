# Tests for HMC backend

test_that("HMC backend produces valid samples", {
  skip_on_cran()

  set.seed(123)

  N <- 50
  x <- rnorm(N)
  X <- cbind(1, x)
  true_beta <- c(0.5, -0.3)

  eta <- X %*% true_beta
  trials <- rep(10L, N)
  y <- rbinom(N, trials, plogis(eta))

  # Run HMC for a short time
  q_init <- rep(0.0, 3)  # 2 beta_num + 1 beta_denom

  fit <- ratiod:::cpp_hmc_simple(
    q_init = q_init,
    y_num = as.integer(y),
    y_denom = trials,
    y_denom_cont = rep(0.0, N),
    X_num = X,
    X_denom = matrix(1, N, 1),
    re_group = as.integer(rep(0, N)),
    n_re_groups = 0L,
    model_type = "binomial",
    sigma_beta = 10.0,
    sigma_re_scale = 2.5,
    phi_prior_shape = 2.0,
    phi_prior_rate = 0.1,
    n_iter = 500L,
    n_warmup = 250L,
    L = 15L,
    adapt = TRUE,
    verbose = FALSE,
    seed = 123L
  )

  expect_true(is.list(fit))
  expect_equal(nrow(fit$samples), 250)  # 500 - 250 warmup
  expect_equal(ncol(fit$samples), 3)    # 2 beta_num + 1 beta_denom

  # Check that samples are reasonable (not NaN or Inf)
  expect_true(all(is.finite(fit$samples)))
  expect_true(all(is.finite(fit$log_prob)))
})

test_that("HMC backend recovers parameters for binomial model", {
  skip_on_cran()

  set.seed(42)

  # Generate data
  N <- 100
  x <- rnorm(N)
  X <- cbind(1, x)
  true_beta <- c(-0.3, 0.5)

  eta <- X %*% true_beta
  trials <- rep(15L, N)
  y <- rbinom(N, trials, plogis(eta))

  # Run NUTS
  q_init <- rep(0.0, 3)  # 2 for numerator + 1 for denominator

  fit <- ratiod:::cpp_nuts_fit(
    q_init = q_init,
    y_num = as.integer(y),
    y_denom = trials,
    y_denom_cont = rep(0.0, N),
    X_num = X,
    X_denom = matrix(1, N, 1),
    re_group = as.integer(rep(0, N)),
    n_re_groups = 0L,
    model_type = "binomial",
    n_iter = 1000L,
    n_warmup = 500L,
    max_treedepth = 10L,
    adapt = TRUE,
    verbose = FALSE,
    seed = 42L
  )

  # Check parameter recovery
  beta_post <- colMeans(fit$samples[, 1:2])
  expect_equal(beta_post[1], true_beta[1], tolerance = 0.4)
  expect_equal(beta_post[2], true_beta[2], tolerance = 0.3)
})

test_that("HMC backend works with random effects", {
  skip_on_cran()

  set.seed(123)

  # Generate data with RE
  n_groups <- 10
  n_per_group <- 10
  N <- n_groups * n_per_group

  true_beta <- c(0.2, -0.4)
  true_sigma <- 0.5

  x <- rnorm(N)
  X <- cbind(1, x)
  group <- rep(1:n_groups, each = n_per_group)
  true_re <- rnorm(n_groups, 0, true_sigma)

  eta <- X %*% true_beta + true_re[group]
  trials <- rep(10L, N)
  y <- rbinom(N, trials, plogis(eta))

  # Number of parameters: 2 (beta_num) + 1 (beta_denom) + 1 (log_sigma_re) + 10 (RE)
  n_params <- 2 + 1 + 1 + n_groups
  q_init <- rep(0.0, n_params)

  fit <- ratiod:::cpp_nuts_fit(
    q_init = q_init,
    y_num = as.integer(y),
    y_denom = trials,
    y_denom_cont = rep(0.0, N),
    X_num = X,
    X_denom = matrix(1, N, 1),
    re_group = as.integer(group),
    n_re_groups = n_groups,
    model_type = "binomial",
    n_iter = 1000L,
    n_warmup = 500L,
    max_treedepth = 10L,
    adapt = TRUE,
    verbose = FALSE,
    seed = 1234L
  )

  # Check that sigma_re is recovered
  log_sigma_idx <- 4  # After beta_num(2) + beta_denom(1)
  sigma_re_samples <- exp(fit$samples[, log_sigma_idx])
  expect_equal(mean(sigma_re_samples), true_sigma, tolerance = 0.3)

  # Check RE correlation
  re_start <- 5
  re_samples <- fit$samples[, re_start:(re_start + n_groups - 1)]
  re_post <- colMeans(re_samples)
  expect_gt(cor(re_post, true_re), 0.5)
})

test_that("HMC backend works for negbin_negbin model", {
  skip_on_cran()

  set.seed(456)

  # Generate NegBin-NegBin data
  N <- 80
  x <- rnorm(N)
  X <- cbind(1, x)
  true_beta_num <- c(2.0, 0.3)
  true_beta_denom <- c(2.5, -0.2)
  true_phi <- 5.0

  eta_num <- X %*% true_beta_num
  eta_denom <- X %*% true_beta_denom
  mu_num <- exp(eta_num)
  mu_denom <- exp(eta_denom)

  y_num <- rnbinom(N, size = true_phi, mu = mu_num)
  y_denom <- rnbinom(N, size = true_phi, mu = mu_denom)

  # Parameters: beta_num(2) + beta_denom(2) + log_phi_num + log_phi_denom = 6
  q_init <- c(rep(0.0, 4), log(5), log(5))  # Initialize phi at 5

  # Use cpp_hmc_fit which has working step-size adaptation
  fit <- ratiod:::cpp_hmc_fit(
    q_init = q_init,
    y_num = as.integer(y_num),
    y_denom = as.integer(y_denom),
    y_denom_cont = rep(0.0, N),
    X_num = X,
    X_denom = X,
    re_group = rep(0L, N),
    n_re_groups = 0L,
    # Multi-term RE parameters
    n_re_terms = 0L,
    re_group_matrix = matrix(0L, nrow = N, ncol = 1),
    re_n_groups_vec = 0L,
    # Random slopes parameters
    has_re_slopes = FALSE,
    has_re_correlated_slopes = FALSE,
    re_n_coefs_vec = integer(0),
    re_correlated_vec = logical(0),
    re_n_chol_vec = integer(0),
    slope_matrices_list = list(),
    model_type_str = "negbin_negbin",
    spatial_type_str = "none",
    spatial_group = rep(0L, N),
    n_spatial_units = 0L,
    adj_row_ptr = 0L,
    adj_col_idx = integer(0),
    n_neighbors = integer(0),
    bym2_scale_factor = 1.0,
    temporal_type_str = "none",
    temporal_time_idx = rep(0L, N),
    temporal_group_idx = rep(0L, N),
    n_times = 0L,
    n_temporal_groups = 0L,
    n_temporal_params = 0L,
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
    X_zi = matrix(0, nrow = N, ncol = 1),
    zi_prior_sd = 10.0,
    n_iter = 1000L,
    n_warmup = 500L,
    L = 10L,
    n_chains = 1L,
    seed = 456L,
    n_threads = 1L,
    verbose = FALSE
  )

  expect_true(all(is.finite(fit$samples)))
  expect_equal(nrow(fit$samples), 500)
  expect_equal(ncol(fit$samples), 6)

  # Check intercepts are in right ballpark
  beta_num_post <- colMeans(fit$samples[, 1:2])
  expect_equal(beta_num_post[1], true_beta_num[1], tolerance = 0.5)
})

test_that("HMC backend works for poisson_gamma model", {
  skip_on_cran()

  set.seed(789)

  # Generate Poisson-Gamma data (CPUE-like)
  N <- 60
  x <- rnorm(N)
  X <- cbind(1, x)
  true_beta_num <- c(2.0, 0.4)
  true_beta_denom <- c(1.5, 0.2)
  true_shape <- 3.0

  eta_num <- X %*% true_beta_num
  eta_denom <- X %*% true_beta_denom

  y_num <- rpois(N, exp(eta_num))
  y_denom <- rgamma(N, shape = true_shape, rate = true_shape / exp(eta_denom))

  # Parameters: beta_num(2) + beta_denom(2) + log_shape = 5
  q_init <- c(rep(0.0, 4), log(3))

  # Use cpp_hmc_fit which has working step-size adaptation
  fit <- ratiod:::cpp_hmc_fit(
    q_init = q_init,
    y_num = as.integer(y_num),
    y_denom = as.integer(rep(1, N)),  # placeholder
    y_denom_cont = y_denom,
    X_num = X,
    X_denom = X,
    re_group = rep(0L, N),
    n_re_groups = 0L,
    # Multi-term RE parameters
    n_re_terms = 0L,
    re_group_matrix = matrix(0L, nrow = N, ncol = 1),
    re_n_groups_vec = 0L,
    # Random slopes parameters
    has_re_slopes = FALSE,
    has_re_correlated_slopes = FALSE,
    re_n_coefs_vec = integer(0),
    re_correlated_vec = logical(0),
    re_n_chol_vec = integer(0),
    slope_matrices_list = list(),
    model_type_str = "poisson_gamma",
    spatial_type_str = "none",
    spatial_group = rep(0L, N),
    n_spatial_units = 0L,
    adj_row_ptr = 0L,
    adj_col_idx = integer(0),
    n_neighbors = integer(0),
    bym2_scale_factor = 1.0,
    temporal_type_str = "none",
    temporal_time_idx = rep(0L, N),
    temporal_group_idx = rep(0L, N),
    n_times = 0L,
    n_temporal_groups = 0L,
    n_temporal_params = 0L,
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
    X_zi = matrix(0, nrow = N, ncol = 1),
    zi_prior_sd = 10.0,
    n_iter = 1000L,
    n_warmup = 500L,
    L = 10L,
    n_chains = 1L,
    seed = 789L,
    n_threads = 1L,
    verbose = FALSE
  )

  expect_true(all(is.finite(fit$samples)))
  expect_equal(nrow(fit$samples), 500)
  expect_equal(ncol(fit$samples), 5)
})

test_that("NUTS diagnostics are reasonable", {
  skip_on_cran()

  set.seed(111)

  N <- 50
  trials <- rep(10L, N)
  y <- rbinom(N, trials, 0.5)

  q_init <- c(0.0, 0.0)

  fit <- ratiod:::cpp_nuts_fit(
    q_init = q_init,
    y_num = as.integer(y),
    y_denom = trials,
    y_denom_cont = rep(0.0, N),
    X_num = matrix(1, N, 1),
    X_denom = matrix(1, N, 1),
    re_group = as.integer(rep(0, N)),
    n_re_groups = 0L,
    model_type = "binomial",
    n_iter = 500L,
    n_warmup = 250L,
    max_treedepth = 10L,
    adapt = TRUE,
    verbose = FALSE,
    seed = 111L
  )

  # Check step size is reasonable
  expect_true(fit$epsilon > 0)
  expect_true(fit$epsilon < 10)

  # Check acceptance rate
  avg_accept <- mean(fit$accept_prob)
  expect_true(avg_accept > 0.3)  # Should be reasonably high

  # Check divergences are low
  n_div <- sum(fit$divergent)
  expect_true(n_div < nrow(fit$samples) * 0.1)  # Less than 10% divergent
})
