# Tests for HMC backend
# Note: Tests for deprecated cpp_hmc_simple and cpp_nuts_fit have been removed
# as those functions were moved to src/deprecated/ during C++ refactoring.

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
    has_latent = FALSE,
    latent_n_factors = 0L,
    latent_shared = FALSE,
    latent_scale = TRUE,
    latent_constraint = 0L,
    latent_sigma_prior_rate = 0.0,
    st_params = list(
      has_spatiotemporal = FALSE,
      type = "none",
      shared = TRUE,
      n_spatial = 0L,
      n_times = 0L,
      n_params = 0L,
      s_idx = integer(0),
      t_idx = integer(0),
      st_flat = integer(0),
      temporal_type = "rw1",
      temporal_cyclic = FALSE,
      adj_row_ptr = integer(0),
      adj_col_idx = integer(0),
      sigma2_prior_U = 1.0,
      sigma2_prior_alpha = 0.01
    ),
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
    has_latent = FALSE,
    latent_n_factors = 0L,
    latent_shared = FALSE,
    latent_scale = TRUE,
    latent_constraint = 0L,
    latent_sigma_prior_rate = 0.0,
    st_params = list(
      has_spatiotemporal = FALSE,
      type = "none",
      shared = TRUE,
      n_spatial = 0L,
      n_times = 0L,
      n_params = 0L,
      s_idx = integer(0),
      t_idx = integer(0),
      st_flat = integer(0),
      temporal_type = "rw1",
      temporal_cyclic = FALSE,
      adj_row_ptr = integer(0),
      adj_col_idx = integer(0),
      sigma2_prior_U = 1.0,
      sigma2_prior_alpha = 0.01
    ),
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

# ---------------------------------------------------------------------------
# Additional HMC integration tests with ratiod()
# ---------------------------------------------------------------------------

test_that("ratiod() fits poisson_gamma model with HMC backend", {
  skip_on_cran()

  set.seed(123)
  n <- 30
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n)
  )

  fit <- ratiod(
    count | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    backend = "hmc",
    iter = 200,
    warmup = 100,
    chains = 1,
    verbose = FALSE
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "hmc")
  expect_true(is.array(fit$draws))
})

test_that("ratiod() fits binomial model with HMC backend", {
  skip_on_cran()

  set.seed(456)
  n <- 30
  trials <- sample(10:20, n, replace = TRUE)
  df <- data.frame(
    successes = rbinom(n, trials, 0.4),
    trials = trials,
    x = rnorm(n)
  )

  fit <- ratiod(
    successes | trials ~ x,
    data = df,
    family = ratiod_binomial(),
    backend = "hmc",
    iter = 200,
    warmup = 100,
    chains = 1,
    verbose = FALSE
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "hmc")
  expect_true(is.array(fit$draws))
})

test_that("ratiod() fits negbin_negbin model with HMC backend", {
  skip_on_cran()

  set.seed(789)
  n <- 30
  df <- data.frame(
    y_num = rnbinom(n, size = 5, mu = 10),
    y_denom = rnbinom(n, size = 5, mu = 15),
    x = rnorm(n)
  )

  fit <- ratiod(
    y_num | y_denom ~ x,
    data = df,
    family = ratiod_negbin_negbin(),
    backend = "hmc",
    iter = 200,
    warmup = 100,
    chains = 1,
    verbose = FALSE
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "hmc")
})

test_that("ratiod() with random effects using HMC backend", {
  skip_on_cran()

  set.seed(111)
  n_groups <- 5
  n_per_group <- 6
  n <- n_groups * n_per_group

  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n),
    site = factor(rep(1:n_groups, each = n_per_group))
  )

  fit <- ratiod(
    count | effort ~ x + (1 | site),
    data = df,
    family = ratiod_poisson_gamma(),
    backend = "hmc",
    iter = 200,
    warmup = 100,
    chains = 1,
    verbose = FALSE
  )

  expect_s3_class(fit, "ratiod_fit")
  # Check that draws have sigma parameters (3D array or matrix)
  draws <- fit$draws
  if (is.array(draws) && length(dim(draws)) == 3) {
    param_names <- dimnames(draws)[[3]]
  } else if (is.matrix(draws)) {
    param_names <- colnames(draws)
  } else {
    param_names <- names(draws)
  }
  expect_true(any(grepl("sigma", param_names, ignore.case = TRUE)) ||
              length(param_names) > 5)  # Has RE parameters
})

test_that("summary.ratiod_fit returns correct structure", {
  skip_on_cran()

  set.seed(222)
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
    backend = "hmc",
    iter = 100,
    warmup = 50,
    chains = 1,
    verbose = FALSE
  )

  summ <- summary(fit)

  expect_true(is.data.frame(summ) || is.list(summ))
})

test_that("print.ratiod_fit works without error", {
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
    backend = "hmc",
    iter = 100,
    warmup = 50,
    chains = 1,
    verbose = FALSE
  )

  output <- capture.output(print(fit))
  expect_true(length(output) > 0)
})

test_that("ratio() works with HMC fit", {
  skip_on_cran()

  set.seed(444)
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
    backend = "hmc",
    iter = 100,
    warmup = 50,
    chains = 1,
    verbose = FALSE
  )

  r <- ratio(fit)

  expect_s3_class(r, "ratiod_ratio")
  expect_true(is.matrix(r$draws))
  expect_equal(ncol(r$draws), n)
})

test_that("ratio_contrast() works with HMC fit", {
  skip_on_cran()

  set.seed(555)
  n <- 20
  df <- data.frame(
    count = rpois(n, lambda = 10),
    effort = rgamma(n, shape = 5, rate = 1),
    x = rnorm(n),
    group = factor(rep(c("A", "B"), each = n/2))
  )

  fit <- ratiod(
    count | effort ~ x + group,
    data = df,
    family = ratiod_poisson_gamma(),
    backend = "hmc",
    iter = 100,
    warmup = 50,
    chains = 1,
    verbose = FALSE
  )

  # ratio_contrast takes a formula for contrast
  contrast <- ratio_contrast(fit, contrast = ~ group)

  expect_true(is.numeric(contrast) || is.list(contrast) || inherits(contrast, "ratiod_contrast"))
})
