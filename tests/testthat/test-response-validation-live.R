# The response checks have to fire on the path a user actually takes.
#
# They used to live only in R/standata.R, which nothing under R/ called, so
# binomial data with the numerator above the denominator reached the
# likelihood: portable_lchoose() returns -Inf for k > n, the log posterior is
# flat everywhere, and the sampler explored that instead of the fit stopping.
# These tests go through tratio(), so they pin the guard where it is reached.

vl_binom_data <- function(n = 20, seed = 1) {
  set.seed(seed)
  data.frame(
    succ = rbinom(n, 10, 0.4),
    trials = rep(10L, n),
    x = rnorm(n)
  )
}

vl_count_data <- function(n = 20, seed = 2) {
  set.seed(seed)
  data.frame(
    count = rpois(n, 5),
    total = rpois(n, 50) + 10,
    x = rnorm(n)
  )
}

# tratio() validates before it dispatches, so these need no sampling. A tiny
# control is set anyway, in case a future change moves the check later.
vl_ctrl <- list(iter = 2, warmup = 1, chains = 1)

test_that("a numerator above its denominator is refused, not sampled", {
  df <- vl_binom_data()
  df$succ[3] <- 15L

  expect_error(
    tratio(succ | trials ~ x, data = df, family = ratiod_binomial(),
           control = vl_ctrl),
    "cannot exceed denominator"
  )
})

test_that("the error names how many observations are wrong", {
  df <- vl_binom_data()
  df$succ[c(2, 5, 9)] <- 12L

  expect_error(
    tratio(succ | trials ~ x, data = df, family = ratiod_binomial(),
           control = vl_ctrl),
    "3 of 20"
  )
})

test_that("the same check covers the beta-binomial family", {
  df <- vl_binom_data(seed = 3)
  df$succ[1] <- 11L

  expect_error(
    tratio(succ | trials ~ x, data = df, family = ratiod_beta_binomial(),
           control = vl_ctrl),
    "cannot exceed denominator"
  )
})

test_that("a negative count is refused", {
  df <- vl_count_data()
  df$count[4] <- -1L

  expect_error(
    tratio(count | total ~ x, data = df, family = ratiod_negbin_negbin(),
           control = vl_ctrl),
    "non-negative"
  )
})

test_that("a non-integer count is refused rather than truncated", {
  # as.integer() in prepare_hmc_data() would silently make this 5.
  df <- vl_count_data(seed = 4)
  df$count <- as.numeric(df$count)
  df$count[2] <- 5.7

  expect_error(
    tratio(count | total ~ x, data = df, family = ratiod_negbin_negbin(),
           control = vl_ctrl),
    "integer counts"
  )
})

test_that("a bad denominator is caught on the arms that model one", {
  df <- vl_count_data(seed = 5)
  df$total[6] <- -3L

  expect_error(
    tratio(count | total ~ x, data = df, family = ratiod_negbin_negbin(),
           control = vl_ctrl),
    "non-negative"
  )
})

test_that("valid data passes the check", {
  # The validator itself must not raise; the fit beyond it is not the subject.
  df <- vl_count_data(seed = 6)
  spec <- ratiod_formula(count | total ~ x, data = df)
  expect_silent(validate_ratio_responses(spec, ratiod_negbin_negbin()))

  dfb <- vl_binom_data(seed = 7)
  specb <- ratiod_formula(succ | trials ~ x, data = dfb)
  expect_silent(validate_ratio_responses(specb, ratiod_binomial()))
})

test_that("a continuous arm is checked for type but not clamped away", {
  # gamma / lognormal backends warn and clamp non-positive values, which is
  # their documented behaviour, so the front door must not error on them.
  set.seed(8)
  df <- data.frame(
    count = rpois(20, 5),
    total = c(0, rgamma(19, 2, 1)),
    x = rnorm(20)
  )
  spec <- ratiod_formula(count | total ~ x, data = df)
  expect_silent(validate_ratio_responses(spec, ratiod_poisson_gamma()))

  df$total <- as.character(df$total)
  spec2 <- ratiod_formula(count | total ~ x, data = df)
  expect_error(validate_ratio_responses(spec2, ratiod_poisson_gamma()),
               "must be numeric")
})

test_that("the validator is a no-op without a family object", {
  df <- vl_count_data(seed = 9)
  spec <- ratiod_formula(count | total ~ x, data = df)
  expect_silent(validate_ratio_responses(spec, "not a family"))
})
