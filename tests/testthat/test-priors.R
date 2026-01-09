# Tests for prior specification (v0.9.0)

test_that("prior_normal creates valid prior", {
  p <- prior_normal(0, 2.5)

  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$distribution, "normal")
  expect_equal(p$mean, 0)
  expect_equal(p$sd, 2.5)
})

test_that("prior_normal validates sd", {
  expect_error(prior_normal(0, -1), "positive")
  expect_error(prior_normal(0, 0), "positive")
})

test_that("prior_half_normal creates valid prior", {
  p <- prior_half_normal(1)

  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$distribution, "half_normal")
  expect_equal(p$sd, 1)
})

test_that("prior_half_cauchy creates valid prior", {
  p <- prior_half_cauchy(2.5)

  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$distribution, "half_cauchy")
  expect_equal(p$scale, 2.5)
})

test_that("prior_gamma creates valid prior", {
  p <- prior_gamma(2, 0.1)

  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$distribution, "gamma")
  expect_equal(p$shape, 2)
  expect_equal(p$rate, 0.1)
})

test_that("prior_gamma validates parameters", {
  expect_error(prior_gamma(-1, 0.1), "positive")
  expect_error(prior_gamma(2, -0.1), "positive")
})

test_that("prior_exponential creates valid prior", {
  p <- prior_exponential(1)

  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$distribution, "exponential")
  expect_equal(p$rate, 1)
})

test_that("prior_beta creates valid prior", {
  p <- prior_beta(2, 2)

  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$distribution, "beta")
  expect_equal(p$alpha, 2)
  expect_equal(p$beta, 2)
})

test_that("prior_beta validates parameters", {
  expect_error(prior_beta(-1, 2), "positive")
  expect_error(prior_beta(2, 0), "positive")
})

test_that("prior_pc creates valid prior", {
  p <- prior_pc(U = 1, alpha = 0.01)

  expect_s3_class(p, "ratiod_prior")
  expect_equal(p$distribution, "pc")
  expect_equal(p$U, 1)
  expect_equal(p$alpha, 0.01)
  expect_equal(p$rate, -log(0.01) / 1)
})

test_that("prior_pc validates parameters", {
  expect_error(prior_pc(U = -1), "positive")
  expect_error(prior_pc(alpha = 0), "in \\(0, 1\\)")
  expect_error(prior_pc(alpha = 1), "in \\(0, 1\\)")
  expect_error(prior_pc(alpha = 1.5), "in \\(0, 1\\)")
})

test_that("ratiod_priors creates valid priors object", {
  p <- ratiod_priors()

  expect_s3_class(p, "ratiod_priors")
  expect_s3_class(p$beta, "ratiod_prior")
  expect_s3_class(p$sigma, "ratiod_prior")
  expect_s3_class(p$phi, "ratiod_prior")
  expect_s3_class(p$rho_temporal, "ratiod_prior")
  expect_s3_class(p$rho_spatial, "ratiod_prior")
})

test_that("ratiod_priors uses correct defaults", {
  p <- ratiod_priors()

  # Beta: Normal(0, 2.5)
  expect_equal(p$beta$distribution, "normal")
  expect_equal(p$beta$mean, 0)
  expect_equal(p$beta$sd, 2.5)

  # Sigma: PC prior
  expect_equal(p$sigma$distribution, "pc")
  expect_equal(p$sigma$U, 1.0)
  expect_equal(p$sigma$alpha, 0.01)

  # Phi: PC prior
  expect_equal(p$phi$distribution, "pc")
  expect_equal(p$phi$U, 10.0)

  # Rho temporal: Beta(2, 2)
  expect_equal(p$rho_temporal$distribution, "beta")
  expect_equal(p$rho_temporal$alpha, 2)

  # Rho spatial: Beta(1, 1) - uniform
  expect_equal(p$rho_spatial$distribution, "beta")
  expect_equal(p$rho_spatial$alpha, 1)
})

test_that("ratiod_priors accepts custom priors", {
  p <- ratiod_priors(
    beta = prior_normal(0, 1),
    sigma = prior_half_cauchy(2.5),
    rho_temporal = prior_beta(5, 2)
  )

  expect_equal(p$beta$sd, 1)
  expect_equal(p$sigma$distribution, "half_cauchy")
  expect_equal(p$rho_temporal$alpha, 5)
})

test_that("ratiod_priors rejects invalid priors", {
  expect_error(
    ratiod_priors(beta = "normal(0, 1)"),
    "must be a prior object"
  )

  expect_error(
    ratiod_priors(sigma = list(distribution = "normal")),
    "must be a prior object"
  )
})

test_that("ratiod_priors_legacy works for backwards compatibility", {
  p <- ratiod_priors_legacy(sigma_U = 0.5, sigma_alpha = 0.05, beta_sd = 1)

  expect_s3_class(p, "ratiod_priors")
  expect_equal(p$beta$sd, 1)
  expect_equal(p$sigma$U, 0.5)
  expect_equal(p$sigma$alpha, 0.05)
})

test_that("prior print methods work", {
  expect_output(print(prior_normal(0, 2.5)), "Normal")
  expect_output(print(prior_half_cauchy(2.5)), "Half-Cauchy")
  expect_output(print(prior_pc(1, 0.01)), "PC prior")
  expect_output(print(prior_beta(2, 2)), "Beta")
  expect_output(print(ratiod_priors()), "ratiod prior specification")
})
