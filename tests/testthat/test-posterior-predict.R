# Posterior predictive replicates, the check built on them, and the generics
# they are reached through.

pp_ctl <- list(iter = 100, warmup = 50, chains = 1)

pp_data <- function(n = 50, seed = 5) {
  set.seed(seed)
  data.frame(
    count = rpois(n, 12),
    total = rpois(n, 80),
    x = rnorm(n),
    site = factor(rep(seq_len(n / 5), each = 5))
  )
}

pp_quiet_fit <- function(...) {
  suppressWarnings(suppressMessages(utils::capture.output(fit <- tratio(...))))
  fit
}


test_that("posterior_predict replicates both arms of a count ratio", {
  skip_on_cran()
  df <- pp_data()
  fit <- pp_quiet_fit(count | total ~ x + (1 | site), data = df,
                      family = ratiod_negbin_negbin(), control = pp_ctl)

  reps <- posterior_predict(fit, ndraws = 20)

  expect_equal(dim(reps$y_num_rep), c(20L, nrow(df)))
  expect_equal(dim(reps$y_denom_rep), c(20L, nrow(df)))
  expect_equal(reps$y_num, as.numeric(df$count))
  expect_equal(reps$y_denom, as.numeric(df$total))

  # Replicates are drawn, not copied
  expect_false(identical(reps$y_num_rep[1, ], reps$y_num_rep[2, ]))
  expect_true(all(reps$y_num_rep >= 0))

  # and they land in the neighbourhood of the data they replicate
  expect_lt(abs(mean(reps$y_num_rep) - mean(df$count)) / mean(df$count), 0.5)
})


test_that("a binomial fit replicates counts within its trials", {
  skip_on_cran()
  df <- pp_data()
  df$hits <- rbinom(nrow(df), size = 20, prob = 0.3)
  df$trials <- 20L
  fit <- pp_quiet_fit(hits | trials ~ x, data = df, family = ratiod_binomial(),
                      control = pp_ctl)

  reps <- posterior_predict(fit, ndraws = 10)
  expect_true(all(reps$y_num_rep >= 0 & reps$y_num_rep <= 20))
  # The denominator of a binomial fit is its trial counts, which are data
  expect_equal(unique(as.vector(reps$y_denom_rep)), 20)
})


test_that("pp_check plots a fit instead of erroring on its draws matrix", {
  skip_on_cran()
  skip_if_not_installed("bayesplot")
  df <- pp_data()
  fit <- pp_quiet_fit(count | total ~ x + (1 | site), data = df,
                      family = ratiod_negbin_negbin(), control = pp_ctl)

  for (type in c("dens_overlay", "scatter", "intervals", "stat")) {
    p <- pp_check(fit, type = type, ndraws = 20)
    expect_s3_class(p, "ggplot")
  }

  expect_s3_class(pp_check(fit, component = "denominator", ndraws = 20), "ggplot")
})


test_that("the generics still serve objects this package knows nothing about", {
  skip_if_not_installed("posterior")
  skip_if_not_installed("bayesplot")

  dm <- posterior::draws_matrix(alpha = rnorm(10), beta = rnorm(10))
  expect_s3_class(as_draws(dm), "draws_matrix")

  y <- rnorm(20)
  yrep <- matrix(rnorm(200), 10, 20)
  expect_s3_class(pp_check(y, yrep, fun = bayesplot::ppc_dens_overlay), "ggplot")

  expect_identical(as_draws, posterior::as_draws)
  expect_identical(pp_check, bayesplot::pp_check)
  expect_identical(posterior_predict, tulpa::posterior_predict)
})


test_that("the tidy-draws verbs are this package's own names", {
  skip_on_cran()
  df <- pp_data()
  fit <- pp_quiet_fit(count | total ~ x + (1 | site), data = df,
                      family = ratiod_negbin_negbin(), control = pp_ctl)

  wide <- draws_wide(fit, beta_num)
  expect_true(all(c(".chain", ".iteration", ".draw") %in% names(wide)))

  long <- draws_long(fit, beta_num)
  expect_true(all(c(".variable", ".value") %in% names(long)))

  interval <- draws_interval(fit, beta_num)
  expect_true(nrow(interval) > 0)

  # the names tidybayes owns are not taken over
  expect_false(exists("spread_draws", envir = asNamespace("tulpaRatio")))
  expect_false(exists("gather_draws", envir = asNamespace("tulpaRatio")))
  expect_false(exists("point_interval", envir = asNamespace("tulpaRatio")))
})
