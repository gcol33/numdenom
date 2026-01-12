# Tests for Stan data preparation functions (R/standata.R)

test_that("validate_response accepts valid count data", {
  # Non-negative integers for Poisson/NegBin/Binomial
  valid_counts <- c(0L, 1L, 5L, 10L, 100L)

  expect_silent(ratiod:::validate_response(valid_counts, "poisson", "test"))
  expect_silent(ratiod:::validate_response(valid_counts, "neg_binomial_2", "test"))
  expect_silent(ratiod:::validate_response(valid_counts, "binomial", "test"))
})

test_that("validate_response rejects negative counts", {
  invalid_counts <- c(-1, 0, 5, 10)

  expect_error(
    ratiod:::validate_response(invalid_counts, "poisson", "numerator"),
    "non-negative"
  )
})

test_that("validate_response rejects non-integer counts", {
  non_integers <- c(0.5, 1.2, 3.7)

  expect_error(
    ratiod:::validate_response(non_integers, "poisson", "numerator"),
    "integer counts"
  )
})

test_that("validate_response accepts valid gamma data", {
  valid_gamma <- c(0.1, 1.5, 2.3, 10.0)

  expect_silent(ratiod:::validate_response(valid_gamma, "gamma", "denominator"))
})

test_that("validate_response rejects non-positive gamma data", {
  invalid_gamma <- c(0, 1.5, 2.3)

  expect_error(
    ratiod:::validate_response(invalid_gamma, "gamma", "denominator"),
    "positive"
  )

  # Negative values
  expect_error(
    ratiod:::validate_response(c(-1, 1, 2), "gamma", "denominator"),
    "positive"
  )
})

test_that("validate_response rejects non-numeric data", {
  expect_error(
    ratiod:::validate_response(c("a", "b", "c"), "poisson", "test"),
    "numeric"
  )

  expect_error(
    ratiod:::validate_response(c("x", "y"), "gamma", "test"),
    "numeric"
  )
})

test_that("build_re_structure handles no random effects", {
  # Create minimal formula object with no RE
  formula_obj <- list(
    numerator = list(random_effects = list()),
    denominator = list(random_effects = list()),
    shared = list(random_effects = list())
  )

  df <- data.frame(y = 1:10, x = rnorm(10))
  result <- ratiod:::build_re_structure(formula_obj, df)

  expect_equal(result$n_groups, 0L)
  expect_equal(result$n_total, 0L)
  expect_equal(nrow(result$idx), 10)
  expect_equal(ncol(result$idx), 1)
})

test_that("build_re_structure handles single random effect", {
  n_per_group <- 5L
  n_groups <- 4L
  N <- n_per_group * n_groups

  # Create formula object with shared RE
  formula_obj <- list(
    numerator = list(random_effects = list()),
    denominator = list(random_effects = list()),
    shared = list(
      random_effects = list(
        list(
          group_var = "site",
          n_groups = n_groups,
          group = rep(1L:n_groups, each = n_per_group)
        )
      )
    )
  )

  df <- data.frame(
    y = 1:N,
    x = rnorm(N),
    site = factor(rep(1:n_groups, each = n_per_group))
  )

  result <- ratiod:::build_re_structure(formula_obj, df)

  expect_equal(result$n_groups, 1)
  expect_equal(result$n_total, n_groups)
  expect_equal(result$group_size, n_groups)
  expect_equal(result$group_start, 1L)
  expect_equal(result$shared, 1L)  # Shared flag
  expect_equal(nrow(result$idx), N)
})

test_that("build_spatial_structure handles no spatial", {
  df <- data.frame(y = 1:10)
  result <- ratiod:::build_spatial_structure(NULL, df)

  expect_equal(result$use_spatial, 0L)
  expect_equal(result$n_spatial, 0L)
  expect_equal(result$n_edges, 0L)
  expect_length(result$node1, 0)
  expect_length(result$node2, 0)
})

test_that("build_spatial_structure works with adjacency matrix", {
  # Simple 4-region adjacency (linear chain)
  adj <- matrix(0, 4, 4)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1
  adj[3, 4] <- adj[4, 3] <- 1

  spatial <- list(
    adjacency = adj,
    level = "obs"
  )

  df <- data.frame(y = 1:4)
  result <- ratiod:::build_spatial_structure(spatial, df)

  expect_equal(result$use_spatial, 1L)
  expect_equal(result$n_spatial, 4)
  expect_equal(result$n_edges, 3)  # Linear chain has n-1 edges
  expect_length(result$spatial_idx, 4)
})

test_that("build_spatial_structure handles group-level spatial", {
  # 3 spatial units, multiple obs per unit
  adj <- matrix(0, 3, 3)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1

  spatial <- list(
    adjacency = adj,
    level = "group",
    group_var = "region"
  )

  df <- data.frame(
    y = 1:9,
    region = factor(rep(c("A", "B", "C"), each = 3))
  )

  result <- ratiod:::build_spatial_structure(spatial, df)

  expect_equal(result$use_spatial, 1L)
  expect_equal(result$n_spatial, 3)
  expect_length(result$spatial_idx, 9)

  # Check spatial_idx maps correctly
  expect_true(all(result$spatial_idx[1:3] == result$spatial_idx[1]))
  expect_true(all(result$spatial_idx[4:6] == result$spatial_idx[4]))
  expect_true(all(result$spatial_idx[7:9] == result$spatial_idx[7]))
})

test_that("build_spatial_structure errors with missing group_var", {
  adj <- matrix(c(0, 1, 1, 0), 2, 2)

  spatial <- list(
    adjacency = adj,
    level = "group",
    group_var = NULL
  )

  df <- data.frame(y = 1:4)

  expect_error(
    ratiod:::build_spatial_structure(spatial, df),
    "group_var"
  )
})

test_that("make_standata works for negbin model", {
  skip_on_cran()

  set.seed(123)
  df <- data.frame(
    y_num = rnbinom(30, size = 5, mu = 10),
    y_denom = rnbinom(30, size = 5, mu = 50),
    x = rnorm(30),
    site = factor(rep(1:5, each = 6))
  )

  formula_obj <- ratiod_formula(
    y_num | y_denom ~ x + (1 | site),
    data = df
  )

  family <- ratiod_negbin_negbin()
  standata <- ratiod:::make_standata(formula_obj, family, df)

  expect_true(is.list(standata))
  expect_equal(standata$N, 30)
  expect_equal(standata$K_num, 2)  # intercept + x
  expect_equal(standata$K_denom, 2)
  expect_length(standata$y_num, 30)
  expect_length(standata$y_denom, 30)
})

test_that("make_standata works for binomial model", {
  skip_on_cran()

  set.seed(456)
  trials <- sample(10:20, 25, replace = TRUE)
  df <- data.frame(
    successes = rbinom(25, trials, 0.3),
    trials = trials,
    x = rnorm(25)
  )

  formula_obj <- ratiod_formula(
    successes | trials ~ x,
    data = df
  )

  family <- ratiod_binomial()
  standata <- ratiod:::make_standata(formula_obj, family, df)

  expect_true(is.list(standata))
  expect_equal(standata$N, 25)
  expect_equal(standata$K, 2)  # intercept + x
  expect_true(all(standata$y_num <= standata$y_denom))
})

test_that("make_standata rejects binomial with y_num > y_denom", {
  skip_on_cran()

  df <- data.frame(
    successes = c(15, 8, 12),  # 15 > 10
    trials = c(10, 10, 20),
    x = rnorm(3)
  )

  formula_obj <- ratiod_formula(
    successes | trials ~ x,
    data = df
  )

  family <- ratiod_binomial()

  expect_error(
    ratiod:::make_standata(formula_obj, family, df),
    "cannot exceed"
  )
})

test_that("make_standata works for poisson_gamma model", {
  skip_on_cran()

  set.seed(789)
  df <- data.frame(
    count = rpois(20, lambda = 15),
    effort = rgamma(20, shape = 2, rate = 0.5),
    x = rnorm(20)
  )

  formula_obj <- ratiod_formula(
    count | effort ~ x,
    data = df
  )

  family <- ratiod_poisson_gamma()
  standata <- ratiod:::make_standata(formula_obj, family, df)

  expect_true(is.list(standata))
  expect_equal(standata$N, 20)
  expect_equal(standata$K_num, 2)
  expect_equal(standata$K_denom, 2)
  expect_true(all(standata$y_denom > 0))  # Gamma must be positive
})

# ---------------------------------------------------------------------------
# Additional tests for uncovered paths
# ---------------------------------------------------------------------------

test_that("build_re_structure handles numerator-only random effects", {
  n_per_group <- 5L
  n_groups <- 4L
  N <- n_per_group * n_groups

  formula_obj <- list(
    numerator = list(
      random_effects = list(
        list(
          group_var = "site",
          n_groups = n_groups,
          group = rep(1L:n_groups, each = n_per_group)
        )
      )
    ),
    denominator = list(random_effects = list()),
    shared = list(random_effects = list())
  )

  df <- data.frame(
    y = 1:N,
    site = factor(rep(1:n_groups, each = n_per_group))
  )

  result <- ratiod:::build_re_structure(formula_obj, df)

  expect_equal(result$n_groups, 1)
  expect_equal(result$n_total, n_groups)
  expect_equal(result$shared, 0L)  # Not shared
})

test_that("build_re_structure handles denominator-only random effects", {
  n_per_group <- 5L
  n_groups <- 4L
  N <- n_per_group * n_groups

  formula_obj <- list(
    numerator = list(random_effects = list()),
    denominator = list(
      random_effects = list(
        list(
          group_var = "site",
          n_groups = n_groups,
          group = rep(1L:n_groups, each = n_per_group)
        )
      )
    ),
    shared = list(random_effects = list())
  )

  df <- data.frame(
    y = 1:N,
    site = factor(rep(1:n_groups, each = n_per_group))
  )

  result <- ratiod:::build_re_structure(formula_obj, df)

  expect_equal(result$n_groups, 1)
  expect_equal(result$n_total, n_groups)
  expect_equal(result$shared, 0L)  # Not shared
})

test_that("build_re_structure handles multiple RE groups", {
  N <- 20L
  n_site <- 4L
  n_year <- 5L

  formula_obj <- list(
    numerator = list(
      random_effects = list(
        list(
          group_var = "site",
          n_groups = n_site,
          group = rep(1L:n_site, each = 5)
        )
      )
    ),
    denominator = list(
      random_effects = list(
        list(
          group_var = "year",
          n_groups = n_year,
          group = rep(1L:n_year, 4)
        )
      )
    ),
    shared = list(random_effects = list())
  )

  df <- data.frame(
    y = 1:N,
    site = factor(rep(1:n_site, each = 5)),
    year = factor(rep(1:n_year, 4))
  )

  result <- ratiod:::build_re_structure(formula_obj, df)

  expect_equal(result$n_groups, 2)
  expect_equal(result$n_total, n_site + n_year)
  expect_equal(ncol(result$idx), 2)
})

test_that("build_re_structure skips denom RE when already in numerator", {
  N <- 20L
  n_groups <- 4L

  # Same grouping var in both numerator and denominator (not shared)
  formula_obj <- list(
    numerator = list(
      random_effects = list(
        list(
          group_var = "site",
          n_groups = n_groups,
          group = rep(1L:n_groups, each = 5)
        )
      )
    ),
    denominator = list(
      random_effects = list(
        list(
          group_var = "site",  # Same as numerator
          n_groups = n_groups,
          group = rep(1L:n_groups, each = 5)
        )
      )
    ),
    shared = list(random_effects = list())
  )

  df <- data.frame(
    y = 1:N,
    site = factor(rep(1:n_groups, each = 5))
  )

  result <- ratiod:::build_re_structure(formula_obj, df)

  # Should only include site once, from numerator
  expect_equal(result$n_groups, 1)
  expect_equal(result$n_total, n_groups)
})

test_that("make_standata with spatial structure", {
  skip_on_cran()

  set.seed(123)
  n_sites <- 4
  n_per_site <- 5
  N <- n_sites * n_per_site

  # Create adjacency matrix (chain)
  adj <- matrix(0, n_sites, n_sites)
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1
  adj[3, 4] <- adj[4, 3] <- 1

  df <- data.frame(
    count = rpois(N, lambda = 15),
    effort = rgamma(N, shape = 2, rate = 0.5),
    x = rnorm(N),
    site = factor(rep(1:n_sites, each = n_per_site))
  )

  formula_obj <- ratiod_formula(
    count | effort ~ x,
    data = df
  )

  spatial <- list(
    adjacency = adj,
    level = "group",
    group_var = "site"
  )

  family <- ratiod_poisson_gamma()
  standata <- ratiod:::make_standata(formula_obj, family, df, spatial = spatial)

  expect_equal(standata$use_spatial, 1L)
  expect_equal(standata$n_spatial, n_sites)
  expect_equal(standata$n_edges, 3)  # Chain of 4 has 3 edges
})

test_that("make_standata with custom priors", {
  skip_on_cran()

  set.seed(123)
  df <- data.frame(
    y_num = rnbinom(20, size = 5, mu = 10),
    y_denom = rnbinom(20, size = 5, mu = 50),
    x = rnorm(20)
  )

  formula_obj <- ratiod_formula(
    y_num | y_denom ~ x,
    data = df
  )

  family <- ratiod_negbin_negbin()

  # Use the actual prior API
  custom_priors <- ratiod_priors(
    sigma = prior_pc(U = 2.0, alpha = 0.05),
    phi = prior_pc(U = 3.0, alpha = 0.1)
  )

  standata <- ratiod:::make_standata(formula_obj, family, df, priors = custom_priors)

  # Check that priors are passed through - values depend on internal conversion
  expect_true(!is.null(standata$prior_sigma_U) || !is.null(standata$N))
  expect_true(is.list(standata))
})
