test_that("quotr_negbin_negbin creates valid family", {
  fam <- quotr_negbin_negbin()

  expect_s3_class(fam, "quotr_family")
  expect_equal(fam$name, "negbin_negbin")
  expect_equal(fam$numerator$distribution, "neg_binomial_2")
  expect_equal(fam$denominator$distribution, "neg_binomial_2")
})

test_that("quotr_binomial creates valid family", {
  fam <- quotr_binomial()

  expect_s3_class(fam, "quotr_family")
  expect_equal(fam$numerator$distribution, "binomial")
  expect_true(fam$denominator_known)
})

test_that("quotr_binomial with denominator_known = FALSE", {
  fam <- quotr_binomial(denominator_known = FALSE)

  expect_equal(fam$name, "binomial_binomial")
  expect_false(fam$denominator_known)
})

test_that("quotr_poisson_gamma creates valid family", {
  fam <- quotr_poisson_gamma()

  expect_s3_class(fam, "quotr_family")
  expect_equal(fam$numerator$distribution, "poisson")
  expect_equal(fam$denominator$distribution, "gamma")
})

test_that("invalid link functions are rejected", {
  expect_error(
    quotr_negbin_negbin(link_num = "identity"),
    "not supported"
  )

  expect_error(
    quotr_binomial(link = "log"),
    "not supported"
  )
})
