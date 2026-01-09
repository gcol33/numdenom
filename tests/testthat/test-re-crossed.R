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
  re_info <- ratiod:::extract_re_for_hmc(f)

  # Should have multi-term structure

  expect_equal(re_info$n_re_terms, 2)
  expect_equal(length(re_info$re_terms), 2)
  expect_equal(re_info$total_groups, 15)  # 10 + 5

  # Check group index matrix
  expect_equal(nrow(re_info$group_idx_matrix), 100)
  expect_equal(ncol(re_info$group_idx_matrix), 2)

  # Check offsets
  expect_equal(re_info$re_terms[[1]]$offset, 0)
  expect_equal(re_info$re_terms[[2]]$offset, 10)
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
  re_info <- ratiod:::extract_re_for_laplace(f)

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
  re_info <- ratiod:::extract_re_from_data(f, df)

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
  re_info <- ratiod:::extract_re_for_hmc(f)

  expect_equal(re_info$n_re_terms, 1)
  expect_equal(re_info$n_groups, 10)
  expect_equal(re_info$total_groups, 10)
  expect_false(re_info$has_slopes)
})


test_that("random slopes trigger warning", {
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20),
    x = rnorm(100),
    site = factor(rep(1:10, each = 10))
  )

  f <- ratiod_formula(y | n ~ x + (1 + x | site), data = df)

  # Should warn about unsupported slopes
  expect_warning(
    ratiod:::extract_re_for_hmc(f),
    regexp = "Random slopes not yet fully supported"
  )
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
  re_info <- ratiod:::extract_re_for_hmc(f)

  expect_equal(re_info$n_re_terms, 3)
  expect_equal(re_info$total_groups, 19)  # 10 + 5 + 4

  # Check offsets
  expect_equal(re_info$re_terms[[1]]$offset, 0)
  expect_equal(re_info$re_terms[[2]]$offset, 10)
  expect_equal(re_info$re_terms[[3]]$offset, 15)
})


test_that("no RE returns correct empty structure", {
  df <- data.frame(
    y = rpois(100, 10),
    n = rpois(100, 20),
    x = rnorm(100)
  )

  f <- ratiod_formula(y | n ~ x, data = df)
  re_info <- ratiod:::extract_re_for_hmc(f)

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

  hmc_data <- ratiod:::prepare_hmc_data(f, df, family, "binomial")

  expect_equal(hmc_data$n_re_terms, 2)
  expect_equal(hmc_data$total_re_groups, 15)
  expect_equal(length(hmc_data$re_terms), 2)
})
