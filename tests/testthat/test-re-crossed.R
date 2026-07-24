# Tests for crossed random effects: (1 | site) + (1 | year)

test_that("formula parsing handles multiple RE terms", {
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20),
    x = rnorm(100),
    site = factor(rep(1:10, each = 10)),
    year = factor(rep(1:5, 20))
  )

  f <- ratiod_formula(y | n ~ x + (1 | site) + (1 | year), data = df)

  # Should have 2 RE terms

  expect_equal(length(f$numerator$random_effects), 2)
  expect_equal(f$numerator$random_effects[[1]]$group_var, "site")
  expect_equal(f$numerator$random_effects[[2]]$group_var, "year")
  expect_equal(f$numerator$random_effects[[1]]$n_groups, 10)
  expect_equal(f$numerator$random_effects[[2]]$n_groups, 5)
})


test_that("extract_re_for_hmc handles multiple RE terms", {
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20),
    x = rnorm(100),
    site = factor(rep(1:10, each = 10)),
    year = factor(rep(1:5, 20))
  )

  f <- ratiod_formula(y | n ~ x + (1 | site) + (1 | year), data = df)
  re_info <- tulpaRatio:::extract_re_for_hmc(f)

  # Should have multi-term structure

  expect_equal(re_info$n_re_terms, 2)
  expect_equal(length(re_info$re_terms), 2)
  expect_equal(re_info$total_groups, 15)  # 10 + 5

  # Check group index matrix
  expect_equal(nrow(re_info$group_idx_matrix), 100)
  expect_equal(ncol(re_info$group_idx_matrix), 2)

  # Check offsets (field renamed to re_offset for slopes support)
  expect_equal(re_info$re_terms[[1]]$re_offset, 0)
  expect_equal(re_info$re_terms[[2]]$re_offset, 10)
})


test_that("extract_re_for_laplace handles multiple RE terms", {
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20),
    x = rnorm(100),
    site = factor(rep(1:10, each = 10)),
    year = factor(rep(1:5, 20))
  )

  f <- ratiod_formula(y | n ~ x + (1 | site) + (1 | year), data = df)
  re_info <- tulpaRatio:::extract_re_for_laplace(f)

  expect_equal(re_info$n_re_terms, 2)
  expect_equal(re_info$total_groups, 15)
})


test_that("extract_re_from_data (PG) handles multiple RE terms", {
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20),
    x = rnorm(100),
    site = factor(rep(1:10, each = 10)),
    year = factor(rep(1:5, 20))
  )

  f <- ratiod_formula(y | n ~ x + (1 | site) + (1 | year), data = df)
  re_info <- tulpaRatio:::extract_re_from_data(f, df)

  expect_equal(re_info$n_re_terms, 2)
  expect_equal(re_info$total_groups, 15)
})


test_that("single RE term still works with new structure", {
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20),
    x = rnorm(100),
    site = factor(rep(1:10, each = 10))
  )

  f <- ratiod_formula(y | n ~ x + (1 | site), data = df)
  re_info <- tulpaRatio:::extract_re_for_hmc(f)

  expect_equal(re_info$n_re_terms, 1)
  expect_equal(re_info$n_groups, 10)
  expect_equal(re_info$total_groups, 10)
  expect_false(re_info$has_slopes)
})


test_that("random slopes are detected in RE structure", {
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20),
    x = rnorm(100),
    site = factor(rep(1:10, each = 10))
  )

  f <- ratiod_formula(y | n ~ x + (1 + x | site), data = df)
  re_info <- tulpaRatio:::extract_re_for_hmc(f)

  # Should detect slopes
  expect_true(re_info$has_slopes)
  expect_equal(re_info$n_re_terms, 1)
  expect_equal(re_info$re_terms[[1]]$n_coefs, 2)  # intercept + x slope
  expect_equal(re_info$re_terms[[1]]$slope_vars, "x")
  # Total params: 10 groups * 2 coefs = 20 RE effects
  expect_equal(re_info$total_re_params, 20)
  # Sigma params: 2 (one for intercept, one for x slope)
  expect_equal(re_info$total_sigma_params, 2)
})


test_that("random slopes work in HMC backend", {
  skip_on_cran()

  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20) + 10,
    x = rnorm(100),
    site = factor(rep(1:10, each = 10))
  )

  # Random slopes should now work
  fit <- tratio(
    y | n ~ x + (1 + x | site),
    data = df,
    family = ratiod_binomial(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1)
  )

  expect_s3_class(fit, "ratiod_fit")

  # Check parameter names
  draws <- as.data.frame(fit$draws)
  params <- names(draws)
  expect_true(any(grepl("sigma_re\\[1,intercept\\]", params)))
  expect_true(any(grepl("sigma_re\\[1,x\\]", params)))
})


test_that("three crossed RE terms work", {
  df <- data.frame(
    y = rpois(200, 10),
    n = rpois(200, 20),
    x = rnorm(200),
    site = factor(rep(1:10, 20)),
    year = factor(rep(1:5, each = 40)),
    observer = factor(rep(1:4, 50))
  )

  f <- ratiod_formula(y | n ~ x + (1 | site) + (1 | year) + (1 | observer), data = df)
  re_info <- tulpaRatio:::extract_re_for_hmc(f)

  expect_equal(re_info$n_re_terms, 3)
  expect_equal(re_info$total_groups, 19)  # 10 + 5 + 4

  # Check offsets (field renamed to re_offset for slopes support)
  expect_equal(re_info$re_terms[[1]]$re_offset, 0)
  expect_equal(re_info$re_terms[[2]]$re_offset, 10)
  expect_equal(re_info$re_terms[[3]]$re_offset, 15)
})


test_that("no RE returns correct empty structure", {
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20),
    x = rnorm(100)
  )

  f <- ratiod_formula(y | n ~ x, data = df)
  re_info <- tulpaRatio:::extract_re_for_hmc(f)

  expect_equal(re_info$n_re_terms, 0)
  expect_equal(re_info$n_groups, 0)
  expect_equal(length(re_info$re_terms), 0)
})


test_that("prepare_hmc_data includes multi-term RE info", {
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20),
    x = rnorm(100),
    site = factor(rep(1:10, each = 10)),
    year = factor(rep(1:5, 20))
  )

  f <- ratiod_formula(y | n ~ x + (1 | site) + (1 | year), data = df)
  family <- ratiod_binomial()

  hmc_data <- tulpaRatio:::prepare_hmc_data(f, df, family, "binomial")

  expect_equal(hmc_data$n_re_terms, 2)
  expect_equal(hmc_data$total_re_groups, 15)
  expect_equal(length(hmc_data$re_terms), 2)
})


# Integration tests - actual model fitting

test_that("HMC fits model with crossed RE terms", {
  skip_on_cran()

  set.seed(12345)
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20) + 10,
    x = rnorm(100),
    site = factor(rep(1:10, each = 10)),
    year = factor(rep(1:5, 20))
  )

  # Should run without error
  fit <- tratio(
    y | n ~ x + (1 | site) + (1 | year),
    data = df,
    family = ratiod_binomial(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")
  expect_true("samples" %in% names(fit))

  # Should have RE parameters for both terms
  param_names <- colnames(fit$samples[[1]])

  # Count sigma_re parameters (should be 2)
  sigma_re_params <- grep("^sigma_re", param_names, value = TRUE)
  expect_equal(length(sigma_re_params), 2)

  # Should have RE effects for all 15 groups (10 sites + 5 years)
  re_params <- grep("^re\\[", param_names, value = TRUE)
  expect_equal(length(re_params), 15)
})


test_that("HMC with three crossed RE terms runs", {
  skip_on_cran()

  set.seed(54321)
  df <- data.frame(
    y = rpois(200, 10),
    n = rpois(200, 20) + 10,
    x = rnorm(200),
    site = factor(rep(1:10, 20)),
    year = factor(rep(1:5, each = 40)),
    observer = factor(rep(1:4, 50))
  )

  fit <- tratio(
    y | n ~ x + (1 | site) + (1 | year) + (1 | observer),
    data = df,
    family = ratiod_binomial(),
    mode = "hmc",
    control = list(iter = 100, warmup = 50, chains = 1, verbose = FALSE)
  )

  expect_s3_class(fit, "ratiod_fit")

  # Should have 3 sigma_re parameters
  param_names <- colnames(fit$samples[[1]])
  sigma_re_params <- grep("^sigma_re", param_names, value = TRUE)
  expect_equal(length(sigma_re_params), 3)

  # Should have 19 RE effects (10 + 5 + 4)
  re_params <- grep("^re\\[", param_names, value = TRUE)
  expect_equal(length(re_params), 19)
})
