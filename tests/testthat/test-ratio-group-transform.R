# ratio() aggregates on the natural scale, then transforms.
#
# aggregate_by_group() takes an unconditional geometric mean,
# exp(rowMeans(log(draws))), which is only defined on the natural scale, and
# passes a group of one row through untouched. Transforming first therefore
# logged the multi-row groups twice -- NaN wherever the ratio is below 1 --
# while leaving the singleton groups correct, so one matrix could hold both.

fake_ratio_fit <- function(ratio_draws, group) {
  structure(
    list(
      ratio_draws = ratio_draws,
      data = data.frame(g = group),
      backend = "hmc"
    ),
    class = c("ratiod_fit")
  )
}

# Group A has three rows, group B has one: the two branches of
# aggregate_by_group() in a single call.
rg_fit <- function(seed = 1, n_draws = 50) {
  set.seed(seed)
  # Ratios straddling 1, so a double log would be NaN on part of the matrix.
  draws <- matrix(exp(rnorm(n_draws * 4, mean = 0, sd = 1)), n_draws, 4)
  fake_ratio_fit(draws, group = c("a", "a", "a", "b"))
}

test_that("a grouped log ratio is the log of the grouped ratio", {
  fit <- rg_fit()

  identity_by <- ratio(fit, by = "g")$draws
  log_by <- ratio(fit, type = "log", by = "g")$draws

  expect_equal(log_by, log(identity_by))
})

test_that("grouping a log-scale request produces no NaN", {
  fit <- rg_fit()
  d <- ratio(fit, type = "log", by = "g")$draws

  expect_false(anyNA(d))
  expect_true(all(is.finite(d)))
})

test_that("the two group sizes are treated consistently", {
  fit <- rg_fit()

  natural <- ratio(fit, by = "g")$draws
  logged <- ratio(fit, type = "log", by = "g")$draws

  # Both branches of aggregate_by_group() must satisfy the same identity: the
  # singleton group is where the pre-fix code happened to be right, so a test
  # on it alone would have passed throughout.
  expect_equal(logged[, "ratio[a]"], log(natural[, "ratio[a]"]))
  expect_equal(logged[, "ratio[b]"], log(natural[, "ratio[b]"]))

  # The singleton group is the untransformed column it came from.
  expect_equal(natural[, "ratio[b]"], fit$ratio_draws[, 4])
})

test_that("the group value is the geometric mean of its members", {
  fit <- rg_fit()
  natural <- ratio(fit, by = "g")$draws

  expected_a <- exp(rowMeans(log(fit$ratio_draws[, 1:3])))
  expect_equal(unname(natural[, "ratio[a]"]), unname(expected_a))
})

test_that("a grouped logit ratio is the logit of the grouped ratio", {
  set.seed(2)
  # Proportions, so logit is defined and no warning is raised.
  draws <- matrix(stats::runif(50 * 4, 0.1, 0.9), 50, 4)
  fit <- fake_ratio_fit(draws, group = c("a", "a", "a", "b"))

  natural <- ratio(fit, by = "g")$draws
  logit_by <- ratio(fit, type = "logit", by = "g")$draws

  expect_equal(logit_by, log(natural / (1 - natural)))
  expect_false(anyNA(logit_by))
})

test_that("ungrouped transforms are unchanged", {
  fit <- rg_fit()

  # ratio() names the columns it returns; the values are the input's.
  expect_equal(unname(ratio(fit, type = "log")$draws), log(fit$ratio_draws))
  expect_equal(unname(ratio(fit)$draws), fit$ratio_draws)
  expect_equal(colnames(ratio(fit)$draws), paste0("ratio[", 1:4, "]"))
})

test_that("the reported group count matches the aggregated columns", {
  fit <- rg_fit()
  r <- ratio(fit, type = "log", by = "g")

  expect_equal(r$n_obs, 2L)
  expect_equal(ncol(r$draws), 2L)
  expect_equal(r$type, "log")
})
