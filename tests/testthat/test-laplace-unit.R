# Unit tests for Laplace backend helper functions

# ---------------------------------------------------------------------------
# get_laplace_family tests
# ---------------------------------------------------------------------------

test_that("get_laplace_family returns binomial for binomial family", {
  family <- ratiod_binomial()
  result <- tulpaRatio:::get_laplace_family(family)
  expect_equal(result, "binomial")
})

test_that("get_laplace_family returns negbin for negbin family", {
  family <- ratiod_negbin_negbin()
  result <- tulpaRatio:::get_laplace_family(family)
  expect_equal(result, "negbin")
})

test_that("get_laplace_family returns poisson for poisson_gamma family", {
  family <- ratiod_poisson_gamma()
  result <- tulpaRatio:::get_laplace_family(family)
  expect_equal(result, "poisson")
})

test_that("get_laplace_family errors on unsupported family", {
  # Create a mock unsupported family
  mock_family <- list(
    numerator = list(distribution = "unknown_dist")
  )
  class(mock_family) <- "ratiod_family"

  expect_error(
    tulpaRatio:::get_laplace_family(mock_family),
    "Unsupported family for Laplace backend"
  )
})

# ---------------------------------------------------------------------------
# extract_re_for_laplace tests
# ---------------------------------------------------------------------------

test_that("extract_re_for_laplace with no random effects", {
  set.seed(111)
  n <- 10
  df <- data.frame(
    count = rpois(n, 5),
    effort = rgamma(n, 3, 1),
    x = rnorm(n)
  )

  f <- ratiod_formula(count | effort ~ x, data = df)
  re_info <- tulpaRatio:::extract_re_for_laplace(f)

  expect_equal(re_info$n_groups, 0L)
  expect_equal(re_info$n_re_terms, 0L)
  expect_null(re_info$group_var)
  expect_false(re_info$has_slopes)
  expect_equal(length(re_info$group_idx), n)
})

test_that("extract_re_for_laplace with single RE term", {
  set.seed(222)
  n <- 15
  df <- data.frame(
    count = rpois(n, 5),
    effort = rgamma(n, 3, 1),
    x = rnorm(n),
    site = factor(rep(1:3, each = 5))
  )

  f <- ratiod_formula(count | effort ~ x + (1 | site), data = df)
  re_info <- tulpaRatio:::extract_re_for_laplace(f)

  expect_equal(re_info$n_groups, 3L)
  expect_equal(re_info$n_re_terms, 1L)
  expect_equal(re_info$group_var, "site")
  expect_false(re_info$has_slopes)
  expect_equal(length(re_info$group_idx), n)
})

test_that("extract_re_for_laplace with slopes warns", {
  set.seed(333)
  n <- 15
  df <- data.frame(
    count = rpois(n, 5),
    effort = rgamma(n, 3, 1),
    x = rnorm(n),
    site = factor(rep(1:3, each = 5))
  )

  f <- ratiod_formula(count | effort ~ x + (x | site), data = df)

  expect_warning(
    re_info <- tulpaRatio:::extract_re_for_laplace(f),
    "Random slopes not yet fully supported"
  )

  expect_true(re_info$has_slopes)
})

test_that("extract_re_for_laplace with multiple RE terms", {
  set.seed(444)
  n <- 18
  df <- data.frame(
    count = rpois(n, 5),
    effort = rgamma(n, 3, 1),
    x = rnorm(n),
    site = factor(rep(1:3, each = 6)),
    year = factor(rep(1:2, times = 9))
  )

  f <- ratiod_formula(count | effort ~ x + (1 | site) + (1 | year), data = df)
  re_info <- tulpaRatio:::extract_re_for_laplace(f)

  expect_equal(re_info$n_re_terms, 2L)
  expect_equal(re_info$total_groups, 5L)  # 3 sites + 2 years
  expect_true(!is.null(re_info$re_terms))
  expect_equal(length(re_info$re_terms), 2)
})

# ---------------------------------------------------------------------------
# can_use_laplace_backend tests
# ---------------------------------------------------------------------------

test_that("can_use_laplace_backend returns TRUE for all families", {
  expect_true(tulpaRatio:::can_use_laplace_backend(ratiod_binomial()))
  expect_true(tulpaRatio:::can_use_laplace_backend(ratiod_negbin_negbin()))
  expect_true(tulpaRatio:::can_use_laplace_backend(ratiod_poisson_gamma()))
})

# ---------------------------------------------------------------------------
# prepare_spatial_for_laplace tests
# ---------------------------------------------------------------------------

test_that("prepare_spatial_for_laplace extracts adjacency structure", {
  set.seed(555)
  n_sites <- 4
  n_per_site <- 3
  n <- n_sites * n_per_site

  adj <- matrix(0, n_sites, n_sites)
  for (i in 1:(n_sites - 1)) adj[i, i + 1] <- adj[i + 1, i] <- 1

  df <- data.frame(
    count = rpois(n, 5),
    effort = rgamma(n, 3, 1),
    site = factor(rep(1:n_sites, each = n_per_site))
  )

  spatial <- spatial_car(adj, level = "group", group_var = "site")

  # Need a formula for prepare_spatial_for_laplace
  formula <- ratiod_formula(count | effort ~ 1, data = df)

  result <- tulpaRatio:::prepare_spatial_for_laplace(spatial, df, formula)

  expect_equal(result$n_units, n_sites)
  expect_equal(length(result$group_idx), n)
  expect_true(all(result$group_idx >= 1 & result$group_idx <= n_sites))
  expect_equal(length(result$n_neighbors), n_sites)
})

test_that("prepare_spatial_for_laplace errors without group_var", {
  spatial <- list(adj_matrix = matrix(0, 3, 3))
  df <- data.frame(x = 1:10)
  formula <- list()

  expect_error(
    tulpaRatio:::prepare_spatial_for_laplace(spatial, df, formula),
    "group_var"
  )
})

test_that("prepare_spatial_for_laplace errors when group_var not in data", {
  spatial <- list(
    adj_matrix = matrix(0, 3, 3),
    group_var = "missing_var"
  )
  df <- data.frame(x = 1:10, site = factor(1:10))
  formula <- list()

  expect_error(
    tulpaRatio:::prepare_spatial_for_laplace(spatial, df, formula),
    "not found in data"
  )
})

test_that("prepare_spatial_for_laplace errors without adjacency matrix", {
  spatial <- list(group_var = "site")
  df <- data.frame(site = factor(1:5))
  formula <- list()

  expect_error(
    tulpaRatio:::prepare_spatial_for_laplace(spatial, df, formula),
    "adj_matrix"
  )
})

# ---------------------------------------------------------------------------
# compute_hessian_at_mode tests
# ---------------------------------------------------------------------------

test_that("compute_hessian_at_mode returns symmetric matrix", {
  set.seed(666)
  n <- 10
  y <- rpois(n, 5)
  n_trials <- rep(10L, n)
  X <- cbind(1, rnorm(n))
  re_idx <- as.numeric(rep(1:2, each = 5))
  n_re_groups <- 2L
  mode <- c(0.5, -0.2, 0.1, -0.1)  # 2 fixed + 2 RE

  result <- tulpaRatio:::compute_hessian_at_mode(
    y = y,
    n_trials = n_trials,
    X = X,
    re_idx = re_idx,
    n_re_groups = n_re_groups,
    mode = mode,
    family = "binomial",
    phi = 1.0,
    sigma_re = 1.0
  )

  H <- result$H
  expect_equal(nrow(H), 4)
  expect_equal(ncol(H), 4)
  # Hessian should be symmetric
  expect_equal(H, t(H), tolerance = 1e-10)
})

test_that("compute_hessian_at_mode works for negbin family", {
  set.seed(777)
  n <- 10
  y <- rnbinom(n, size = 5, mu = 10)
  n_trials <- rep(1L, n)
  X <- cbind(1, rnorm(n))
  re_idx <- rep(1, n)
  n_re_groups <- 0L
  mode <- c(2.0, 0.5)

  result <- tulpaRatio:::compute_hessian_at_mode(
    y = y,
    n_trials = n_trials,
    X = X,
    re_idx = re_idx,
    n_re_groups = n_re_groups,
    mode = mode,
    family = "negbin",
    phi = 5.0,
    sigma_re = 1.0
  )

  expect_equal(nrow(result$H), 2)
  expect_equal(ncol(result$H), 2)
})

test_that("compute_hessian_at_mode works for poisson family", {
  set.seed(888)
  n <- 10
  y <- rpois(n, 10)
  n_trials <- rep(1L, n)
  X <- cbind(1, rnorm(n))
  re_idx <- rep(1, n)
  n_re_groups <- 0L
  mode <- c(2.3, 0.1)

  result <- tulpaRatio:::compute_hessian_at_mode(
    y = y,
    n_trials = n_trials,
    X = X,
    re_idx = re_idx,
    n_re_groups = n_re_groups,
    mode = mode,
    family = "poisson",
    phi = 1.0,
    sigma_re = 1.0
  )

  expect_equal(nrow(result$H), 2)
  expect_equal(ncol(result$H), 2)
})

# ---------------------------------------------------------------------------
# convert_laplace_to_ratiod_fit tests
# ---------------------------------------------------------------------------

test_that("convert_laplace_to_ratiod_fit creates valid ratiod_fit", {
  set.seed(999)
  n <- 10
  p <- 2
  n_re <- 0
  n_samples <- 100

  df <- data.frame(
    count = rpois(n, 5),
    effort = rgamma(n, 3, 1),
    x = rnorm(n)
  )

  formula <- ratiod_formula(count | effort ~ x, data = df)
  X <- model.matrix(~ x, data = df)
  samples <- matrix(rnorm(n_samples * p), nrow = n_samples, ncol = p)

  result <- list(
    mode = rnorm(p),
    sigma_re_opt = 1.0,
    log_marginal = -50.0
  )

  re_info <- list(
    n_groups = 0L,
    group_idx = rep(1, n),
    group_var = NULL
  )

  fit <- tulpaRatio:::convert_laplace_to_ratiod_fit(
    samples = samples,
    result = result,
    formula = formula,
    data = df,
    family = ratiod_poisson_gamma(),
    X = X,
    re_info = re_info,
    n_samples = n_samples
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "laplace")
  expect_equal(fit$n_save, n_samples)
  expect_equal(nrow(fit$draws), n_samples)
  expect_equal(ncol(fit$draws), p)
})

test_that("convert_laplace_to_ratiod_fit handles random effects", {
  set.seed(1000)
  n <- 12
  p <- 2
  n_re <- 3
  n_samples <- 50

  df <- data.frame(
    count = rpois(n, 5),
    effort = rgamma(n, 3, 1),
    x = rnorm(n),
    site = factor(rep(1:3, each = 4))
  )

  formula <- ratiod_formula(count | effort ~ x + (1 | site), data = df)
  X <- model.matrix(~ x, data = df)
  samples <- matrix(rnorm(n_samples * (p + n_re)), nrow = n_samples, ncol = p + n_re)

  result <- list(
    mode = rnorm(p + n_re),
    sigma_re_opt = 0.5,
    log_marginal = -60.0
  )

  re_info <- list(
    n_groups = n_re,
    group_idx = as.numeric(df$site),
    group_var = "site"
  )

  fit <- tulpaRatio:::convert_laplace_to_ratiod_fit(
    samples = samples,
    result = result,
    formula = formula,
    data = df,
    family = ratiod_poisson_gamma(),
    X = X,
    re_info = re_info,
    n_samples = n_samples
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_true("sigma_re" %in% colnames(fit$draws))
  expect_equal(unique(fit$draws[, "sigma_re"]), 0.5)
})

# ---------------------------------------------------------------------------
# compute_hessian_spatial tests
# ---------------------------------------------------------------------------

test_that("compute_hessian_spatial returns correct dimensions", {
  set.seed(1111)
  n <- 12
  y <- rpois(n, 5)
  n_trials <- rep(10L, n)
  X <- cbind(1, rnorm(n))
  re_idx <- rep(1, n)
  n_re_groups <- 0L
  spatial_idx <- as.integer(rep(1:4, each = 3))
  n_spatial_units <- 4L

  # Simple chain adjacency (0-based indexing for C++ compatibility)
  # Site 1 neighbors: [2], Site 2: [1,3], Site 3: [2,4], Site 4: [3]
  adj_row_ptr <- c(0L, 1L, 3L, 5L, 6L)  # 0-based row pointers
  adj_col_idx <- c(1L, 0L, 2L, 1L, 3L, 2L)  # 0-based neighbor indices
  n_neighbors <- c(1L, 2L, 2L, 1L)

  p <- ncol(X)
  n_x <- p + n_re_groups + n_spatial_units
  mode <- rnorm(n_x)

  result <- tulpaRatio:::compute_hessian_spatial(
    y = y,
    n_trials = n_trials,
    X = X,
    re_idx = re_idx,
    n_re_groups = n_re_groups,
    spatial_idx = spatial_idx,
    n_spatial_units = n_spatial_units,
    adj_row_ptr = adj_row_ptr,
    adj_col_idx = adj_col_idx,
    n_neighbors = n_neighbors,
    mode = mode,
    family = "binomial",
    phi = 1.0,
    sigma_re = 1.0,
    tau_spatial = 1.0
  )

  expect_equal(nrow(result$H), n_x)
  expect_equal(ncol(result$H), n_x)
})

# ---------------------------------------------------------------------------
# convert_laplace_spatial_to_ratiod_fit tests
# ---------------------------------------------------------------------------

test_that("convert_laplace_spatial_to_ratiod_fit creates valid fit", {
  set.seed(1212)
  n <- 12
  p <- 2
  n_re <- 0
  n_spatial <- 4
  n_samples <- 50

  df <- data.frame(
    count = rpois(n, 5),
    effort = rgamma(n, 3, 1),
    x = rnorm(n),
    site = factor(rep(1:4, each = 3))
  )

  formula <- ratiod_formula(count | effort ~ x, data = df)
  X <- model.matrix(~ x, data = df)

  samples <- matrix(rnorm(n_samples * (p + n_re + n_spatial)),
                    nrow = n_samples, ncol = p + n_re + n_spatial)

  result <- list(
    mode = rnorm(p + n_re + n_spatial),
    log_marginal = -40.0
  )

  re_info <- list(
    n_groups = 0L,
    group_idx = rep(1, n),
    group_var = NULL
  )

  spatial_info <- list(
    n_units = n_spatial,
    group_idx = as.integer(df$site)
  )

  fit <- tulpaRatio:::convert_laplace_spatial_to_ratiod_fit(
    samples = samples,
    result = result,
    formula = formula,
    data = df,
    family = ratiod_poisson_gamma(),
    X = X,
    re_info = re_info,
    spatial_info = spatial_info,
    n_samples = n_samples
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_equal(fit$backend, "laplace")
  expect_true(any(grepl("spatial", colnames(fit$draws))))
})

# ---------------------------------------------------------------------------
# prior_predict tests
# ---------------------------------------------------------------------------

test_that("prior_predict raises not implemented error", {
  expect_error(
    prior_predict(
      formula = ~ 1,
      family = ratiod_binomial(),
      data = data.frame(x = 1:10)
    ),
    "not yet implemented"
  )
})

# ---------------------------------------------------------------------------
# ratiod_compare error handling tests
# ---------------------------------------------------------------------------

test_that("ratiod_compare errors with single model", {
  fit <- list()
  class(fit) <- "ratiod_fit"

  expect_error(
    ratiod_compare(fit),
    "At least two models"
  )
})

test_that("ratiod_compare errors with non-ratiod objects", {
  expect_error(
    ratiod_compare(list(a = 1), list(b = 2)),
    "must be ratiod_fit"
  )
})

# ---------------------------------------------------------------------------
# ratiod_average error handling tests
# ---------------------------------------------------------------------------

test_that("ratiod_average errors with single model", {
  fit <- list()
  class(fit) <- "ratiod_fit"

  expect_error(
    ratiod_average(fit),
    "At least two models"
  )
})

test_that("ratiod_average errors with non-ratiod objects", {
  expect_error(
    ratiod_average(list(a = 1), list(b = 2)),
    "must be ratiod_fit"
  )
})

# ---------------------------------------------------------------------------
# print.ratiod_average tests
# ---------------------------------------------------------------------------

test_that("print.ratiod_average works", {
  avg_obj <- list(
    weights = c(m1 = 0.6, m2 = 0.4),
    predictions = data.frame(
      mean = 1:3,
      sd = 0.1
    ),
    models = c("m1", "m2"),
    type = "ratio",
    weights_method = "loo",
    n_models = 2
  )
  class(avg_obj) <- "ratiod_average"

  output <- capture.output(print(avg_obj))
  expect_true(any(grepl("ratiod model averaging", output)))
  expect_true(any(grepl("Model weights", output)))
  expect_true(any(grepl("m1", output)))
})

test_that("print.ratiod_average handles matrix predictions", {
  avg_obj <- list(
    weights = c(m1 = 0.6, m2 = 0.4),
    predictions = matrix(1:20, nrow = 10, ncol = 2),
    models = c("m1", "m2"),
    type = "ratio",
    weights_method = "loo",
    n_models = 2
  )
  class(avg_obj) <- "ratiod_average"

  output <- capture.output(print(avg_obj))
  expect_true(any(grepl("draws x", output)))
})

# ---------------------------------------------------------------------------
# fitted.ratiod_average tests
# ---------------------------------------------------------------------------

test_that("fitted.ratiod_average extracts predictions", {
  preds <- data.frame(mean = 1:5, sd = 0.1)
  avg_obj <- list(predictions = preds)
  class(avg_obj) <- "ratiod_average"

  result <- fitted(avg_obj)
  expect_equal(result, preds)
})

# ---------------------------------------------------------------------------
# weights.ratiod_average tests
# ---------------------------------------------------------------------------

test_that("weights.ratiod_average extracts weights", {
  w <- c(m1 = 0.7, m2 = 0.3)
  avg_obj <- list(weights = w)
  class(avg_obj) <- "ratiod_average"

  result <- weights(avg_obj)
  expect_equal(result, w)
})
