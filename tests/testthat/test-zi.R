# test-zi.R
# Tests for zero-inflation specification and helper functions

test_that("zi_poisson creates correct structure", {
  zi <- zi_poisson()

  expect_s3_class(zi, "ratiod_zi")
  expect_equal(zi$type, "zi_poisson")
  expect_null(zi$formula)
  expect_equal(zi$distribution, "poisson")
})


test_that("zi_poisson with formula", {
  zi <- zi_poisson(~ habitat)

  expect_s3_class(zi, "ratiod_zi")
  expect_equal(zi$type, "zi_poisson")
  expect_true(inherits(zi$formula, "formula"))
})


test_that("zi_negbin creates correct structure", {
  zi <- zi_negbin()

  expect_s3_class(zi, "ratiod_zi")
  expect_equal(zi$type, "zi_negbin")
  expect_null(zi$formula)
  expect_equal(zi$distribution, "negbin")
})


test_that("hurdle_poisson creates correct structure", {
  zi <- hurdle_poisson()

  expect_s3_class(zi, "ratiod_zi")
  expect_equal(zi$type, "hurdle_poisson")
  expect_null(zi$formula)
  expect_equal(zi$distribution, "poisson")
})


test_that("hurdle_negbin creates correct structure", {
  zi <- hurdle_negbin()

  expect_s3_class(zi, "ratiod_zi")
  expect_equal(zi$type, "hurdle_negbin")
  expect_null(zi$formula)
  expect_equal(zi$distribution, "negbin")
})


test_that("print.ratiod_zi works", {
  zi <- zi_poisson(~ habitat)
  output <- capture.output(print(zi))
  expect_true(any(grepl("Zero-Inflation", output)))
  expect_true(any(grepl("Poisson", output)))
})


test_that("validate_zi returns NULL for NULL input", {
  result <- validate_zi(NULL)
  expect_null(result)
})


test_that("validate_zi passes valid ratiod_zi", {
  zi <- zi_poisson()
  result <- validate_zi(zi)
  expect_s3_class(result, "ratiod_zi")
})


test_that("validate_zi rejects non-ratiod_zi objects", {
  expect_error(
    validate_zi(list(type = "zi_poisson")),
    "ratiod_zi object"
  )
})


test_that("validate_zi checks formula variables exist", {
  zi <- zi_poisson(~ nonexistent_var)
  df <- data.frame(x = 1:10, y = 10:1)

  expect_error(
    validate_zi(zi, df),
    "not in data"
  )
})


test_that("prepare_zi_for_hmc returns none for NULL zi", {
  zi_info <- ratiod:::prepare_zi_for_hmc(NULL, data.frame(), 10)

  expect_equal(zi_info$type, "none")
  expect_equal(nrow(zi_info$X_zi), 10)
  expect_equal(ncol(zi_info$X_zi), 1)
})


test_that("prepare_zi_for_hmc creates design matrix for intercept-only", {
  zi <- zi_poisson()
  df <- data.frame(x = 1:10, y = 10:1)

  zi_info <- ratiod:::prepare_zi_for_hmc(zi, df, 10)

  expect_equal(zi_info$type, "zi_poisson")
  expect_equal(nrow(zi_info$X_zi), 10)
  expect_equal(ncol(zi_info$X_zi), 1)
  expect_true(all(zi_info$X_zi[, 1] == 1))  # Intercept
})


test_that("prepare_zi_for_hmc creates design matrix with predictors", {
  zi <- zi_poisson(~ x)
  df <- data.frame(x = 1:10, y = 10:1)

  zi_info <- ratiod:::prepare_zi_for_hmc(zi, df, 10)

  expect_equal(zi_info$type, "zi_poisson")
  expect_equal(nrow(zi_info$X_zi), 10)
  expect_equal(ncol(zi_info$X_zi), 2)  # Intercept + x
  expect_equal(zi_info$p_zi, 2)
})


test_that("ratiod_zinegbin creates ZI family", {
  fam <- ratiod_zinegbin()

  expect_s3_class(fam, "ratiod_family_zi")
  expect_true(fam$zero_inflated)
  expect_equal(fam$zi_type, "mixture")
})


test_that("ratiod_hurdle_negbin creates hurdle family", {
  fam <- ratiod_hurdle_negbin()

  expect_s3_class(fam, "ratiod_family_zi")
  expect_true(fam$zero_inflated)
  expect_equal(fam$zi_type, "hurdle")
})


test_that("is_zi_family correctly identifies ZI families", {
  expect_true(ratiod:::is_zi_family(ratiod_zinegbin()))
  expect_true(ratiod:::is_zi_family(ratiod_zipois()))
  expect_false(ratiod:::is_zi_family(ratiod_negbin_negbin()))
})


test_that("is_hurdle_family correctly identifies hurdle families", {
  expect_true(ratiod:::is_hurdle_family(ratiod_hurdle_negbin()))
  expect_true(ratiod:::is_hurdle_family(ratiod_hurdle_pois()))
  expect_false(ratiod:::is_hurdle_family(ratiod_zinegbin()))
})
