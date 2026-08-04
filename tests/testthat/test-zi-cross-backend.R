# test-zi-cross-backend.R
#
# Cross-backend ZI/hurdle/OI/ZOIB coverage for #33: VI/ESS/SGHMC previously
# auto-detected a ZI/hurdle family from a bare "zi"/"hurdle" string, which
# collapsed every variant to the same (or wrong) likelihood and left the
# four binomial-family ZI variants (zi_binomial, hurdle_binomial,
# oi_binomial, zoib) completely unreachable through those three backends.
#
# For each variant this file checks that mode = "vi"/"ess"/"sghmc":
#  - fits without error
#  - reports the correct, specific ZI/OI coefficient (not silently zero or
#    absent), agreeing with the mode = "hmc" fit on the same data/seed
#
# oi_binomial and zoib's OI component is checked on all four backends: VI,
# ESS, and SGHMC each read data.p_oi/X_oi_flat and expose beta_oi[...] in
# their draws, matching HMC.

fit_zi_coefs <- function(formula, data, family, mode, zi = NULL, seed = 20260803) {
  args <- list(
    formula = formula, data = data, family = family, mode = mode,
    control = list(iter = 400L, warmup = 200L, chains = 1L, seed = seed,
                   verbose = FALSE)
  )
  if (!is.null(zi)) args$zi <- zi
  fit <- do.call(tratio, args)

  pn <- colnames(fit$draws)
  zi_cols <- pn[grepl("^beta_zi\\[", pn)]
  oi_cols <- pn[grepl("^beta_oi\\[", pn)]
  list(
    zi = if (length(zi_cols) > 0) colMeans(fit$draws[, zi_cols, drop = FALSE]) else NULL,
    oi = if (length(oi_cols) > 0) colMeans(fit$draws[, oi_cols, drop = FALSE]) else NULL
  )
}

# ============================================================================
# Count-response families, explicit zi = (previously: bare "zi"/"hurdle"
# collapsed zi_negbin/hurdle_negbin to the same likelihood as zi_poisson on
# ESS/SGHMC, and VI mis-dispatched zi_negbin as ZI_POISSON)
# ============================================================================

simulate_count_zi <- function(n = 150) {
  set.seed(910)
  x <- rnorm(n)
  pi0 <- plogis(-1.0)
  is_zero <- runif(n) < pi0
  list(
    formula = count | total ~ x,
    data = data.frame(
      count = ifelse(is_zero, 0L, rnbinom(n, size = 5, mu = 10)),
      total = rnbinom(n, size = 5, mu = 50),
      x = x
    )
  )
}

for (backend in c("vi", "ess", "sghmc")) {
  test_that(sprintf("zi_negbin (explicit zi=) fits a nonzero, HMC-consistent ZI logit via %s", backend), {
    skip_on_cran()
    spec <- simulate_count_zi()

    ref <- fit_zi_coefs(spec$formula, spec$data, ratiod_negbin_negbin(), "hmc", zi = zi_negbin())
    got <- fit_zi_coefs(spec$formula, spec$data, ratiod_negbin_negbin(), backend, zi = zi_negbin())

    expect_false(is.null(got$zi))
    expect_true(is.finite(got$zi[[1]]))
    expect_lt(abs(got$zi[[1]] - ref$zi[[1]]), 1.5)
  })

  test_that(sprintf("ratiod_zinegbin family auto-detect fits the negbin ZI likelihood (not poisson) via %s", backend), {
    skip_on_cran()
    spec <- simulate_count_zi()

    ref <- fit_zi_coefs(spec$formula, spec$data, ratiod_zinegbin(), "hmc")
    got <- fit_zi_coefs(spec$formula, spec$data, ratiod_zinegbin(), backend)

    expect_false(is.null(got$zi))
    expect_true(is.finite(got$zi[[1]]))
    expect_lt(abs(got$zi[[1]] - ref$zi[[1]]), 1.5)
  })
}

# ============================================================================
# Binomial-family ZI/hurdle: family auto-detection (previously unreachable
# on any of VI/ESS/SGHMC -- no path existed at all before #33)
# ============================================================================

simulate_binomial_zi <- function(n = 150) {
  set.seed(911)
  x <- rnorm(n)
  trials <- sample(10:20, n, replace = TRUE)
  p_true <- plogis(0.4 + 0.3 * x)
  is_zero <- runif(n) < 0.3
  list(
    formula = succ | trials ~ x,
    data = data.frame(
      succ = ifelse(is_zero, 0L, rbinom(n, trials, p_true)),
      trials = trials,
      x = x
    )
  )
}

for (backend in c("vi", "ess", "sghmc")) {
  test_that(sprintf("ratiod_zibinomial family auto-detect fits a nonzero, HMC-consistent ZI logit via %s", backend), {
    skip_on_cran()
    spec <- simulate_binomial_zi()

    ref <- fit_zi_coefs(spec$formula, spec$data, ratiod_zibinomial(), "hmc")
    got <- fit_zi_coefs(spec$formula, spec$data, ratiod_zibinomial(), backend)

    expect_false(is.null(got$zi))
    expect_true(is.finite(got$zi[[1]]))
    expect_lt(abs(got$zi[[1]] - ref$zi[[1]]), 1.5)
  })

  test_that(sprintf("explicit zi_binomial() workaround fits the same way via %s", backend), {
    skip_on_cran()
    spec <- simulate_binomial_zi()

    ref <- fit_zi_coefs(spec$formula, spec$data, ratiod_binomial(), "hmc", zi = zi_binomial())
    got <- fit_zi_coefs(spec$formula, spec$data, ratiod_binomial(), backend, zi = zi_binomial())

    expect_false(is.null(got$zi))
    expect_true(is.finite(got$zi[[1]]))
    expect_lt(abs(got$zi[[1]] - ref$zi[[1]]), 1.5)
  })

  test_that(sprintf("ratiod_hurdle_binomial family auto-detect fits a nonzero, HMC-consistent hurdle logit via %s", backend), {
    skip_on_cran()
    spec <- simulate_binomial_zi()

    ref <- fit_zi_coefs(spec$formula, spec$data, ratiod_hurdle_binomial(), "hmc")
    got <- fit_zi_coefs(spec$formula, spec$data, ratiod_hurdle_binomial(), backend)

    expect_false(is.null(got$zi))
    expect_true(is.finite(got$zi[[1]]))
    expect_lt(abs(got$zi[[1]] - ref$zi[[1]]), 1.5)
  })
}

# ============================================================================
# One-inflated and ZOIB binomial families: full check on VI (data plumbing
# fixed here); ESS/SGHMC checked only where their existing, documented
# `p_oi = 0` hardcoding doesn't apply (zoib's ZI component).
# ============================================================================

simulate_oi_binomial <- function(n = 150) {
  set.seed(912)
  x <- rnorm(n)
  trials <- sample(10:20, n, replace = TRUE)
  p_true <- plogis(0.3 + 0.2 * x)
  succ <- rbinom(n, trials, p_true)
  is_oi <- runif(n) < 0.3
  succ[is_oi] <- trials[is_oi]
  list(
    formula = succ | trials ~ x,
    data = data.frame(succ = succ, trials = trials, x = x)
  )
}

test_that("ratiod_oibinomial family auto-detect fits a nonzero, HMC-consistent OI logit via vi", {
  skip_on_cran()
  spec <- simulate_oi_binomial()

  ref <- fit_zi_coefs(spec$formula, spec$data, ratiod_oibinomial(), "hmc")
  got <- fit_zi_coefs(spec$formula, spec$data, ratiod_oibinomial(), "vi")

  expect_false(is.null(got$oi))
  expect_true(is.finite(got$oi[[1]]))
  expect_lt(abs(got$oi[[1]] - ref$oi[[1]]), 1.5)
})


test_that("explicit oi_binomial() workaround fits via vi", {
  skip_on_cran()
  spec <- simulate_oi_binomial()

  ref <- fit_zi_coefs(spec$formula, spec$data, ratiod_binomial(), "hmc", zi = oi_binomial())
  got <- fit_zi_coefs(spec$formula, spec$data, ratiod_binomial(), "vi", zi = oi_binomial())

  expect_false(is.null(got$oi))
  expect_true(is.finite(got$oi[[1]]))
  expect_lt(abs(got$oi[[1]] - ref$oi[[1]]), 1.5)
})


for (backend in c("ess", "sghmc")) {
  test_that(sprintf("ratiod_oibinomial family auto-detect fits a nonzero, HMC-consistent OI logit via %s", backend), {
    skip_on_cran()
    spec <- simulate_oi_binomial()

    ref <- fit_zi_coefs(spec$formula, spec$data, ratiod_oibinomial(), "hmc")
    got <- fit_zi_coefs(spec$formula, spec$data, ratiod_oibinomial(), backend)

    expect_false(is.null(got$oi))
    expect_true(is.finite(got$oi[[1]]))
    expect_lt(abs(got$oi[[1]] - ref$oi[[1]]), 1.5)
  })
}


simulate_zoib <- function(n = 150) {
  set.seed(913)
  x <- rnorm(n)
  trials <- sample(10:20, n, replace = TRUE)
  p_true <- plogis(0.3 + 0.2 * x)
  u <- runif(n)
  pi0 <- 0.2
  pi1 <- 0.25
  succ <- vapply(seq_len(n), function(i) {
    if (u[i] < pi0) return(0L)
    if (u[i] < pi0 + (1 - pi0) * pi1) return(as.integer(trials[i]))
    as.integer(rbinom(1, trials[i], p_true[i]))
  }, integer(1))
  list(
    formula = succ | trials ~ x,
    data = data.frame(succ = succ, trials = trials, x = x)
  )
}

test_that("ratiod_zoibinomial family auto-detect fits nonzero, HMC-consistent ZI and OI logits via vi", {
  skip_on_cran()
  spec <- simulate_zoib()

  ref <- fit_zi_coefs(spec$formula, spec$data, ratiod_zoibinomial(), "hmc")
  got <- fit_zi_coefs(spec$formula, spec$data, ratiod_zoibinomial(), "vi")

  expect_false(is.null(got$zi))
  expect_false(is.null(got$oi))
  expect_true(is.finite(got$zi[[1]]))
  expect_true(is.finite(got$oi[[1]]))
  expect_lt(abs(got$zi[[1]] - ref$zi[[1]]), 1.5)
  expect_lt(abs(got$oi[[1]] - ref$oi[[1]]), 1.5)
})


for (backend in c("ess", "sghmc")) {
  test_that(sprintf("ratiod_zoibinomial family auto-detect fits nonzero, HMC-consistent ZI and OI logits via %s", backend), {
    skip_on_cran()
    spec <- simulate_zoib()

    ref <- fit_zi_coefs(spec$formula, spec$data, ratiod_zoibinomial(), "hmc")
    got <- fit_zi_coefs(spec$formula, spec$data, ratiod_zoibinomial(), backend)

    expect_false(is.null(got$zi))
    expect_false(is.null(got$oi))
    expect_true(is.finite(got$zi[[1]]))
    expect_true(is.finite(got$oi[[1]]))
    expect_lt(abs(got$zi[[1]] - ref$zi[[1]]), 1.5)
    expect_lt(abs(got$oi[[1]] - ref$oi[[1]]), 1.5)
  })
}
