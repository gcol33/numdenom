test_that("ratiod_negbin_negbin creates valid family", {
  fam <- ratiod_negbin_negbin()

  expect_s3_class(fam, "ratiod_family")
  expect_equal(fam$name, "negbin_negbin")
  expect_equal(fam$numerator$distribution, "neg_binomial_2")
  expect_equal(fam$denominator$distribution, "neg_binomial_2")
})

test_that("ratiod_binomial creates valid family", {
  fam <- ratiod_binomial()

  expect_s3_class(fam, "ratiod_family")
  expect_equal(fam$numerator$distribution, "binomial")
  expect_true(fam$denominator_known)
})

test_that("ratiod_binomial with denominator_known = FALSE", {
  fam <- ratiod_binomial(denominator_known = FALSE)

  expect_equal(fam$name, "binomial_binomial")
  expect_false(fam$denominator_known)
})

test_that("ratiod_poisson_gamma creates valid family", {
  fam <- ratiod_poisson_gamma()

  expect_s3_class(fam, "ratiod_family")
  expect_equal(fam$numerator$distribution, "poisson")
  expect_equal(fam$denominator$distribution, "gamma")
})

test_that("invalid link functions are rejected", {
  expect_error(
    ratiod_negbin_negbin(link_num = "identity"),
    "not supported"
  )

  expect_error(
    ratiod_binomial(link = "log"),
    "not supported"
  )
})

# Tests for v0.9.0 extended families

test_that("ratiod_beta_binomial creates valid family", {
  fam <- ratiod_beta_binomial()

  expect_s3_class(fam, "ratiod_family")
  expect_equal(fam$name, "beta_binomial_fixed")
  expect_equal(fam$numerator$distribution, "beta_binomial")
  expect_equal(fam$numerator$link, "logit")
  expect_true(fam$denominator_known)
  expect_true(fam$overdispersed)
})

test_that("ratiod_beta_binomial with denominator_known = FALSE", {
  fam <- ratiod_beta_binomial(denominator_known = FALSE)

  expect_equal(fam$name, "beta_binomial")
  expect_false(fam$denominator_known)
})

test_that("ratiod_beta_binomial accepts valid link functions", {
  expect_s3_class(ratiod_beta_binomial(link = "logit"), "ratiod_family")
  expect_s3_class(ratiod_beta_binomial(link = "probit"), "ratiod_family")
  expect_s3_class(ratiod_beta_binomial(link = "cloglog"), "ratiod_family")

  expect_error(ratiod_beta_binomial(link = "log"), "not supported")
})

test_that("ratiod_gamma_gamma creates valid family", {
  fam <- ratiod_gamma_gamma()

  expect_s3_class(fam, "ratiod_family")
  expect_equal(fam$name, "gamma_gamma")
  expect_equal(fam$numerator$distribution, "gamma")
  expect_equal(fam$denominator$distribution, "gamma")
  expect_equal(fam$numerator$link, "log")
  expect_equal(fam$denominator$link, "log")
})

test_that("ratiod_gamma_gamma accepts valid link functions", {
  expect_s3_class(ratiod_gamma_gamma(link_num = "log"), "ratiod_family")
  expect_s3_class(ratiod_gamma_gamma(link_num = "identity"), "ratiod_family")
  expect_s3_class(ratiod_gamma_gamma(link_num = "inverse"), "ratiod_family")

  expect_error(ratiod_gamma_gamma(link_num = "logit"), "not supported")
})

test_that("ratiod_lognormal creates valid family", {
  fam <- ratiod_lognormal()

  expect_s3_class(fam, "ratiod_family")
  expect_equal(fam$name, "lognormal_lognormal")
  expect_equal(fam$numerator$distribution, "lognormal")
  expect_equal(fam$denominator$distribution, "lognormal")
  expect_false(fam$denom_fixed)
})

test_that("ratiod_lognormal with denom_fixed = TRUE", {
  fam <- ratiod_lognormal(denom_fixed = TRUE)

  expect_equal(fam$name, "lognormal_fixed")
  expect_true(fam$denom_fixed)
  expect_equal(fam$denominator$distribution, "fixed")
  expect_null(fam$denominator$link)
})

test_that("ratiod_lognormal accepts valid link functions", {
  expect_s3_class(ratiod_lognormal(link_num = "log"), "ratiod_family")
  expect_s3_class(ratiod_lognormal(link_num = "identity"), "ratiod_family")

  expect_error(ratiod_lognormal(link_num = "logit"), "not supported")
})

test_that("family print methods work", {
  expect_output(print(ratiod_beta_binomial()), "beta_binomial")
  expect_output(print(ratiod_gamma_gamma()), "gamma_gamma")
  expect_output(print(ratiod_lognormal()), "lognormal")
})
