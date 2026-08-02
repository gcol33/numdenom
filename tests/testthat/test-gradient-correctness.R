# The log posterior and the analytic gradient are written out separately, once
# per gradient mode and once per specialized path. When a prior term is added to
# one and not the other, the sampler still runs and still returns numbers: it
# simply samples a density whose gradient points somewhere else. Shape and class
# assertions cannot see that. These compare the analytic gradient against
# central differences of the log posterior it is supposed to be the gradient of.

FIELDS <- c("icar", "bym2", "rw1", "rw2", "icar_rw1")
# The spatial field carried alongside a second structure, which routes to the
# multi-feature gradient paths rather than the main one. Those paths identified
# the field by a soft penalty while the log posterior hard-centred it, so their
# gradient described a different density; nothing reached them until these.
MULTI_FIELDS <- c("icar_ms", "bym2_ms", "icar_st", "bym2_st")
# Structures nothing reached before, because make_model built none of them. The
# templated log posterior expressed no HSGP field and no random slopes at all,
# and carried only one intercept-only random-effect term; its TVC precision took
# a different prior from the analytic one. Everything the parameter layout
# allocates for belongs here, so that a structure cannot be added to one density
# and not the other without a red test.
STRUCTURE_FIELDS <- c("re", "re_crossed", "re_slopes", "re_slopes_corr",
                      "hsgp", "tvc")
# Marginalized out by locating the field's mode and adding a Laplace correction,
# which is not a closed-form function of the parameters, so the templated density
# cannot express it and the autodiff modes must refuse rather than differentiate
# something else.
COLLAPSED_FIELDS <- c("icar_collapsed", "bym2_collapsed")
# A collapsed spatial field alongside a companion temporal RW1 term. The inner
# Laplace mode-finding held beta and re_vals fixed but never added the temporal
# offset to eta, so phi* was the mode of the wrong conditional and log det(H)
# was evaluated at the wrong point (gcol33/tulpaRatio#27); separately, the
# temporal block's own gradient only picked up the envelope/data-likelihood
# term and not log det(H)'s own dependence on the temporal offset through the
# curvature W_data.
COLLAPSED_TEMPORAL_FIELDS <- c("icar_collapsed_rw1", "bym2_collapsed_rw1")
# Proper CAR (rho estimated from the data, gcol33/tulpaRatio#31). Reaches
# compute_gradient_composite only (can_use_analytical_gradient excludes it,
# same as every other spatial type but ICAR/BYM2), and its log-determinant
# has no templated expression, so -- like the collapsed fields -- it has no
# "arena" case.
CAR_PROPER_FIELDS <- c("car_proper")
# A Type IV (Kronecker) spatiotemporal interaction with no accompanying
# additive spatial or temporal field -- the only structured term is the
# interaction itself, matching gcol33/tulpaRatio#24's repro. MULTI_FIELDS'
# icar_st/bym2_st reach compute_gradient_spatiotemporal_handcoded too, but
# only ever build a Type I interaction; the Kronecker stencil in the Type IV
# branch (both its centered and non-centered forms) had never been checked
# against finite differences before #24.
ST_IV_FIELDS <- c("st4", "st4_nc")
ALL_FIELDS <- c(FIELDS, MULTI_FIELDS, STRUCTURE_FIELDS, ST_IV_FIELDS)
MODES <- c("handcoded", "arena")
AUTODIFF_MODES <- c("arena", "forward", "tape")
# The same three under the names the front door takes.
AUTODIFF_MODES_STR <- c("A_r", "A", "A_t")

# Relative deviation per parameter, with an absolute allowance first.
#
# A central difference of a log posterior of magnitude L with step h carries a
# roundoff error of about eps*L/h, which for L ~ 1e3-1e4 and h = 1e-5 is around
# 1e-7. A purely relative criterion demands more than that from any entry whose
# true value is small: the HSGP basis coefficients run to 6e-4, where 1e-4
# relative is 6e-8, below what the difference can resolve. Sweeping h on that
# field traces the expected V (1.2e-4 at h = 1e-3, 1.5e-7 at h = 1e-5, 6.3e-6 at
# h = 1e-7) rather than the flat line a missing term would give. A term actually
# absent from one side is wrong by order 1, so the allowance cannot hide one.
GRAD_ATOL <- 1e-6

rel_dev <- function(a, f, atol = GRAD_ATOL) {
  pmax(0, abs(a - f) - atol) / pmax(1e-6, pmax(abs(a), abs(f)))
}

for (field in FIELDS) {
  for (mode in MODES) {
    test_that(sprintf("analytic gradient matches finite differences (%s, %s)", field, mode), {
      r <- tulpaRatio:::cpp_gradient_check(field, mode = mode)
      expect_gt(r$n_params, 0)
      dev <- rel_dev(r$analytic, r$finite_diff)
      worst <- which.max(dev)
      expect_lt(
        max(dev),
        1e-4,
        label = sprintf(
          "%s/%s: worst parameter %d (block %s), analytic %.6f vs finite-diff %.6f",
          field, mode, worst, r$block[worst], r$analytic[worst], r$finite_diff[worst]
        )
      )
    })
  }
}

# Collapsed + temporal only ever reaches the specialized H-mode gradient
# (compute_gradient_icar_collapsed): the collapsed marginal has no autodiff
# expression, so unlike FIELDS above there is no "arena" case to loop over.
for (field in COLLAPSED_TEMPORAL_FIELDS) {
  test_that(sprintf("analytic gradient matches finite differences (%s, handcoded)", field), {
    r <- tulpaRatio:::cpp_gradient_check(field, mode = "handcoded")
    expect_gt(r$n_params, 0)
    dev <- rel_dev(r$analytic, r$finite_diff)
    worst <- which.max(dev)
    expect_lt(
      max(dev),
      1e-4,
      label = sprintf(
        "%s: worst parameter %d (block %s), analytic %.6f vs finite-diff %.6f",
        field, worst, r$block[worst], r$analytic[worst], r$finite_diff[worst]
      )
    )
  })
}

for (field in CAR_PROPER_FIELDS) {
  test_that(sprintf("analytic gradient matches finite differences (%s, handcoded)", field), {
    r <- tulpaRatio:::cpp_gradient_check(field, mode = "handcoded")
    expect_gt(r$n_params, 0)
    dev <- rel_dev(r$analytic, r$finite_diff)
    worst <- which.max(dev)
    expect_lt(
      max(dev),
      1e-4,
      label = sprintf(
        "%s: worst parameter %d (block %s), analytic %.6f vs finite-diff %.6f",
        field, worst, r$block[worst], r$analytic[worst], r$finite_diff[worst]
      )
    )
  })
}

for (field in c(MULTI_FIELDS, STRUCTURE_FIELDS, ST_IV_FIELDS)) {
  for (mode in MODES) {
    test_that(sprintf("analytic gradient matches finite differences (%s, %s)", field, mode), {
      r <- tulpaRatio:::cpp_gradient_check(field, mode = mode)
      expect_gt(r$n_params, 0)
      dev <- rel_dev(r$analytic, r$finite_diff)
      worst <- which.max(dev)
      expect_lt(
        max(dev),
        1e-4,
        label = sprintf(
          "%s/%s: worst parameter %d (block %s), analytic %.6f vs finite-diff %.6f",
          field, mode, worst, r$block[worst], r$analytic[worst], r$finite_diff[worst]
        )
      )
    })
  }
}

# Every autodiff mode differentiates the same templated log posterior, so a
# structure missing from it is missing from all of them at once, and the runtime
# gradient check cannot see it: for those modes it differences that same density,
# so the gradient and its numerical reference agree on the same wrong value. A
# missing structure contributes nothing to the linear predictor and carries no
# prior, so its parameters come out exactly zero, which is what separates an
# absent term from a wrong one. The spatiotemporal block, the HSGP field and the
# random slopes were each absent this way.
for (field in ALL_FIELDS) {
  for (mode in AUTODIFF_MODES) {
    test_that(sprintf("autodiff carries every block (%s, %s)", field, mode), {
      r <- tulpaRatio:::cpp_gradient_check(field, mode = mode)
      zero <- r$analytic == 0
      expect_equal(
        sum(zero), 0L,
        label = sprintf("%s/%s: zero-gradient blocks %s", field, mode,
                        paste(unique(r$block[zero]), collapse = ", "))
      )
      expect_lt(max(rel_dev(r$analytic, r$finite_diff)), 1e-4)
    })
  }
}

# A temporal effect that is not shared enters the numerator only. Both the
# log posterior and the gradient have to drop the denominator's residual,
# and only a family with a real denominator can tell the two apart, so the
# binomial cases above cannot reach this. The handcoded path kept adding
# the denominator term while compute_log_post did not, which makes NUTS
# sample a density whose gradient points elsewhere.
for (field in c("rw1", "rw2", "icar_rw1")) {
  for (mode in MODES) {
    for (shared in c(TRUE, FALSE)) {
      test_that(sprintf("analytic gradient matches finite differences (%s, %s, poisson_gamma, shared=%s)",
                        field, mode, shared), {
        r <- tulpaRatio:::cpp_gradient_check(field, mode = mode,
                                             family = "poisson_gamma",
                                             temporal_shared = shared)
        dev <- rel_dev(r$analytic, r$finite_diff)
        worst <- which.max(dev)
        expect_lt(
          max(dev),
          1e-4,
          label = sprintf(
            "%s/%s/shared=%s: worst parameter %d (block %s), analytic %.6f vs finite-diff %.6f",
            field, mode, shared, worst, r$block[worst],
            r$analytic[worst], r$finite_diff[worst]
          )
        )
      })
    }
  }
}

test_that("an unshared temporal effect changes the density it is meant to change", {
  # Guards the test above: if temporal_shared stopped reaching the likelihood
  # altogether, the shared and unshared cases would both pass while testing
  # the same thing.
  for (field in c("rw1", "icar_rw1")) {
    lp_shared <- tulpaRatio:::cpp_logpost_at(field, family = "poisson_gamma",
                                             temporal_shared = TRUE)
    lp_unshared <- tulpaRatio:::cpp_logpost_at(field, family = "poisson_gamma",
                                               temporal_shared = FALSE)
    expect_gt(abs(lp_shared - lp_unshared), 1.0, label = sprintf("field = %s", field))
  }
  # A binomial model has no denominator likelihood, so the flag is inert there.
  expect_equal(
    tulpaRatio:::cpp_logpost_at("rw1", family = "binomial", temporal_shared = TRUE),
    tulpaRatio:::cpp_logpost_at("rw1", family = "binomial", temporal_shared = FALSE)
  )
})

test_that("the two log posteriors are the same function, not the same shape", {
  # compute_log_post and compute_log_post_impl<double> must return the same
  # number, not merely have the same gradient. They differed by the binomial
  # coefficient, which the eta-form likelihood drops as constant in eta, and per
  # structure by whatever that structure was missing; a gradient comparison alone
  # sees neither, since both are constant in the parameters at a fixed point.
  for (field in ALL_FIELDS) {
    r <- tulpaRatio:::cpp_gradient_check(field, mode = "arena")
    expect_true(is.na(r$impl_gap), label = sprintf("field = %s", field))
    expect_equal(r$log_post, r$log_post_impl, tolerance = 1e-8,
                 info = sprintf("field = %s", field))
  }
})

test_that("the value each mode reports is the density its gradient describes", {
  # NUTS consumes the fused value as log_prob, so a mode whose value disagrees
  # with its own gradient drives the Hamiltonian off a different density.
  for (mode in c("arena", "handcoded")) {
    for (field in ALL_FIELDS) {
      r <- tulpaRatio:::cpp_gradient_check(field, mode = mode)
      expect_equal(r$log_post, r$log_post_mode, tolerance = 1e-8,
                   info = sprintf("%s/%s", field, mode))
    }
  }
})

test_that("the collapsed parameterizations are declared, not silently mis-differentiated", {
  # The templated density cannot express them, so it must say so and the
  # autodiff modes must refuse. Left undeclared, the gradient of every parameter
  # is taken against a posterior with no spatial marginal in it.
  for (field in c(COLLAPSED_FIELDS, COLLAPSED_TEMPORAL_FIELDS)) {
    r <- tulpaRatio:::cpp_gradient_check(field, mode = "arena")
    expect_false(is.na(r$impl_gap), label = sprintf("field = %s", field))
  }
})

test_that("proper CAR is declared, not silently mis-differentiated", {
  # Its log-determinant needs a dense Cholesky with no templated expression
  # (gcol33/tulpaRatio#31), so the autodiff modes must refuse rather than
  # differentiate a density with no spatial marginal in it.
  for (field in CAR_PROPER_FIELDS) {
    r <- tulpaRatio:::cpp_gradient_check(field, mode = "arena")
    expect_false(is.na(r$impl_gap), label = sprintf("field = %s", field))
  }
})

test_that("proper CAR refuses an autodiff gradient mode at the front door", {
  skip_on_cran()
  set.seed(1)
  N <- 200L
  S <- 16L
  side <- ceiling(sqrt(S))
  g <- expand.grid(lon = seq_len(side), lat = seq_len(side))[seq_len(S), ]
  A <- matrix(0, S, S)
  for (i in seq_len(S)) for (j in seq_len(S)) {
    if (i != j && sqrt((g$lon[i] - g$lon[j])^2 + (g$lat[i] - g$lat[j])^2) <= 1.5)
      A[i, j] <- 1
  }
  dimnames(A) <- list(as.character(seq_len(S)), as.character(seq_len(S)))
  x <- rnorm(N)
  trials <- sample(10:40, N, replace = TRUE)
  df <- data.frame(
    y = rbinom(N, trials, plogis(0.5 + 0.3 * x)), trials = trials, x = x,
    site = factor(rep(seq_len(S), length.out = N))
  )
  fit_with <- function(mode) {
    tratio(y | trials ~ x, data = df, family = ratiod_binomial(),
           spatial = spatial_car(A, level = "group", group_var = "site", proper = TRUE),
           control = list(iter = 30, warmup = 15, chains = 1, verbose = FALSE,
                          gradient_mode = mode))
  }
  for (mode in AUTODIFF_MODES_STR) {
    expect_error(fit_with(mode), "proper CAR", info = mode)
  }
  # The analytic density carries the prior, so H (and its AUTO fallback) must
  # still fit without a gradient-mismatch warning.
  expect_warning(fit_h <- fit_with("H"), NA)
  expect_identical(fit_h$backend, "hmc")
})

test_that("a collapsed field refuses an autodiff gradient mode at the front door", {
  skip_on_cran()
  # The refusal has to reach the user, not just the predicate: the field is
  # marginalized out, so the templated density has no slot for it and every other
  # parameter would be differentiated against a posterior with no spatial
  # marginal. A temporal term is what routes this to the sampler rather than to
  # the Gibbs backend, which spatial-only models take.
  set.seed(1)
  N <- 200L
  S <- 16L
  side <- ceiling(sqrt(S))
  g <- expand.grid(lon = seq_len(side), lat = seq_len(side))[seq_len(S), ]
  A <- matrix(0, S, S)
  for (i in seq_len(S)) for (j in seq_len(S)) {
    if (i != j && sqrt((g$lon[i] - g$lon[j])^2 + (g$lat[i] - g$lat[j])^2) <= 1.5)
      A[i, j] <- 1
  }
  dimnames(A) <- list(as.character(seq_len(S)), as.character(seq_len(S)))
  x <- rnorm(N)
  trials <- sample(10:40, N, replace = TRUE)
  df <- data.frame(
    y = rbinom(N, trials, plogis(0.5 + 0.3 * x)), trials = trials, x = x,
    site = factor(rep(seq_len(S), length.out = N)),
    time = factor(rep(seq_len(8), length.out = N))
  )
  fit_with <- function(mode) {
    tratio(y | trials ~ x, data = df, family = ratiod_binomial(),
           spatial = spatial_car(A, level = "group", group_var = "site",
                                 parameterization = "collapsed"),
           temporal = temporal_rw1("time"),
           control = list(iter = 30, warmup = 15, chains = 1, verbose = FALSE,
                          gradient_mode = mode))
  }
  for (mode in AUTODIFF_MODES_STR) {
    expect_error(fit_with(mode), "collapsed ICAR", info = mode)
  }
  # The analytic density carries the marginal, so H must still fit, and must
  # not fall back to a gradient-mismatch warning (gcol33/tulpaRatio#27): the
  # inner Laplace mode-finding has to see the temporal offset the same way
  # the log posterior does.
  expect_warning(fit_h <- fit_with("H"), NA)
  expect_identical(fit_h$backend, "hmc")
})

test_that("centring makes the spatial field's mean invisible to the likelihood", {
  # A hard sum-to-zero constraint means the intercept, not the field, carries
  # the level. Shifting every phi by a constant must leave the gradient of the
  # fixed effects untouched.
  for (field in c("icar", "bym2", "icar_rw1", MULTI_FIELDS)) {
    raw <- tulpaRatio:::cpp_gradient_check(field, mode = "handcoded", precenter = FALSE)
    cen <- tulpaRatio:::cpp_gradient_check(field, mode = "handcoded", precenter = TRUE)
    beta <- which(raw$block == "beta_num")
    expect_equal(raw$analytic[beta], cen$analytic[beta], tolerance = 1e-8,
                 info = sprintf("field = %s", field))
  }
})
