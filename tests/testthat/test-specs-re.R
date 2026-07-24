# test-specs-re.R
# B1d Step 1 parity test: single-term, intercept-only random effects routed
# through the LikelihoodSpec path produce posterior means within 4x within-MC
# noise of the legacy backend at the same seed.
#
# Scope: binomial with random group intercept (non-centered, default). Other
# families and parameterisations follow the same wiring; this is the smallest
# end-to-end check that validates the bridge / R-side reorder.

simulate_re_binomial <- function(n = 200, n_groups = 8) {
  set.seed(20260601)
  group <- sample.int(n_groups, n, replace = TRUE)
  re_true <- rnorm(n_groups, sd = 0.6)
  x1 <- rnorm(n)
  eta <- 0.4 + 0.8 * x1 + re_true[group]
  p   <- plogis(eta)
  nt  <- sample(5:20, n, replace = TRUE)
  y   <- rbinom(n, nt, p)
  list(
    formula = y | n_trials ~ x1 + (1 | grp),
    data    = data.frame(y = y, n_trials = nt, x1 = x1,
                          grp = factor(group)),
    family  = tulpaRatio::ratiod_binomial()
  )
}

fit_one_re <- function(spec, use_specs, seed_val) {
  op <- options(tulpaRatio.use_specs = use_specs); on.exit(options(op), add = TRUE)
  fit <- tulpaRatio::tratio(
    formula = spec$formula, data = spec$data, family = spec$family,
    mode = "hmc",
    control = list(iter = 2000L, warmup = 500L, chains = 1L, seed = seed_val, verbose = FALSE, gradient_mode = "A_r")
  )
  colMeans(fit$draws)
}

test_that("B1d spec path matches legacy for binomial + random intercept", {
  skip_on_cran()
  spec <- simulate_re_binomial()
  legacy42 <- fit_one_re(spec, FALSE, 42L)
  specs42  <- fit_one_re(spec, TRUE,  42L)
  legacy43 <- fit_one_re(spec, FALSE, 43L)

  expect_named(specs42, names(legacy42))
  cross  <- max(abs(legacy42 - specs42))
  within <- max(abs(legacy42 - legacy43))
  expect_lt(cross, max(4 * within, 5e-3))
})
