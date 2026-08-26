# The log posterior and the analytic gradient are written out separately, once
# per gradient mode and once per specialized path. When a prior term is added to
# one and not the other, the sampler still runs and still returns numbers: it
# simply samples a density whose gradient points somewhere else. Shape and class
# assertions cannot see that. These compare the analytic gradient against
# central differences of the log posterior it is supposed to be the gradient of.

# temporal_ar1 is the temporal block's own AR1 arm, which carries a rho the rw1
# and rw2 arms have no counterpart for and so reaches gradient terms neither of
# them does.
FIELDS <- c("icar", "bym2", "rw1", "rw2", "temporal_ar1", "icar_rw1")
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
                      "hsgp", "tvc", "tvc_ar1",
                      # TemporalType::IID was implemented for TVC's log-prior
                      # and gradient in both H and autodiff modes, but blocked
                      # from R by temporal_tvc()'s match.arg and never checked
                      # against finite differences (gcol33/tulpaRatio#32).
                      "tvc_iid")
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
# The interaction's other two structured margins. Type II applies the temporal
# precision within each spatial unit -- its RW2 and AR1 arms are the ones
# compute_gradient_composite had no branch for at all -- and Type III applies
# ICAR at each time point. Both reach the shared interaction gradient
# (src/st_prior_grad.h) through a different margin than Type IV does.
ST_OTHER_FIELDS <- c("st2", "st2_rw2", "st2_ar1", "st3")
# The AR1 time margin, centered and non-centered. Its precision R(rho) enters
# both the quadratic form and the log-determinant, and its correlation is the
# only hyperparameter the layout allocates logit_rho_st_idx for
# (gcol33/tulpaRatio#66).
ST_IV_AR1_FIELDS <- c("st4_ar1", "st4_ar1_nc")
# The GP interaction types: a continuous covariance over the S x T grid instead
# of a GMRF stencil on it (gcol33/tulpaRatio#68). Nothing built one before, and
# what the gradient did with the block was leave it at zero and subtract the
# sum-to-zero push -- the field felt neither its data nor its own prior. Its
# precision, its two ranges and the field itself all reach st_prior_grad.h's GP
# branch. stgp_latent pairs the interaction with a latent factor, which is what
# routes it away from compute_gradient_spatiotemporal_handcoded and into the
# composite, so both callers of that branch are checked; that pairing is also
# what caught compute_gradient_latent_handcoded being selected for a model
# carrying an interaction it never writes (gcol33/tulpaRatio#71).
ST_GP_FIELDS <- c("stgp", "stgp_matern", "stgp_gneiting", "stgp_latent")
# The kernels resolve_gradient_fn can select that make_model() could not build a
# model for at all: the GP, the multi-scale GP, the SVC pair, the temporal GP,
# multi-scale temporal and latent factors (gcol33/tulpaRatio#45). Under the
# default gradient mode NUTS runs on these with no offline check, which is how
# the GP covariance derivative (gcol33/tulpaRatio#42) and the AR1 precision
# gradient (gcol33/tulpaRatio#43) both reached a release.
KERNEL_FIELDS <- c("gp", "gp_matern", "gp_gaussian", "gp_spherical",
                   "gp_temporal", "msgp", "msgp_temporal", "svc", "svc_hsgp",
                   "temporal_gp", "ms_temporal", "latent")
# Fields the templated density declares a gap for, so like the collapsed areal
# fields they have no autodiff case and are checked against compute_log_post
# alone.
#   gp_collapsed -- marginalizes its field out, as the collapsed areal fields do.
#   gp_nc        -- spatial_gp(parameterization = "noncentered"): the sampled
#                   parameters are z ~ N(0, I) and the field is
#                   w = L(sigma2, phi) z, so both hyperparameters reach eta
#                   through the transform. compute_log_post_impl has no
#                   non-centred branch and says so (gcol33/tulpaRatio#26).
KERNEL_COLLAPSED_FIELDS <- c("gp_collapsed", "gp_nc")
# Cases the harness can build whose gradient does not match the density it
# reports. Each names the defect it is blocked on; the case is written out so
# the fix has a test to turn green, and the deviation each measures today is
# recorded in the issue.
BLOCKED_FIELDS <- character(0)
# Combinations resolve_gradient_fn used to hand to a specialized function that
# writes only one of the two blocks (gcol33/tulpaRatio#71). Each is now routed
# by its feature mask, and each of these says the function it lands on writes
# both. gp_tgp, svc_ms and gp_slopes are not on #71's list: they are what the
# mask separates once a block is named rather than negated one guard at a time
# -- a GP margin and a GMRF margin are different temporal blocks, a multiscale
# temporal term is not an SVC term, and no specialized function writes the
# random-slope block at all.
# gp_slopes_corr and gp_crossed are the RE side of the same separation
# (gcol33/tulpaRatio#72). The GP entry point declared its RE block single-term
# and slope-free, so neither model could reach the sampler at all; now that they
# can, the mask has to keep them off compute_gradient_gp_handcoded, which reads
# the legacy single-term block. gp_crossed is what GF_RE_MULTI is for -- a
# crossed term carries no slope, so GF_RE_SLOPES does not see it.
DISPATCH_MASK_FIELDS <- c("gp_st4", "gp_stgp", "gp_temporal_st4", "msgp_st4",
                          "tvc_st4", "temporal_gp_st4", "gp_tgp", "svc_ms",
                          "gp_slopes", "gp_slopes_corr", "gp_crossed")
# The same, for a collapsed field. Its marginal is an inner Laplace at a mode
# that moves with every other block in eta, which nothing but the collapsed
# kernels' own companion-temporal path carries, so the dispatch returns the
# numerical gradient of the density rather than a specialized function that
# would leave the interaction at zero. Like the other collapsed fields these
# have no autodiff case: the templated density cannot express the marginal.
DISPATCH_MASK_COLLAPSED_FIELDS <- c("gp_collapsed_st4", "icar_collapsed_st4",
                                    "bym2_collapsed_st4")
ALL_FIELDS <- c(FIELDS, MULTI_FIELDS, STRUCTURE_FIELDS, ST_IV_FIELDS,
                ST_OTHER_FIELDS, ST_IV_AR1_FIELDS, ST_GP_FIELDS, KERNEL_FIELDS,
                DISPATCH_MASK_FIELDS)
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

for (field in DISPATCH_MASK_COLLAPSED_FIELDS) {
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

# What each model dispatches TO, rather than whether its gradient happens to
# come out right. A guard written as a chain of negations passes the deviation
# check the moment the function it wrongly selects is fixed to write the second
# block; this is what says the model reaches the function that was meant for it
# (gcol33/tulpaRatio#71).
test_that("resolve_gradient_fn sends each model to the function written for it", {
  expected <- c(
    # One structure: the specialized function keeps it.
    gp = "gp", gp_temporal = "gp_temporal", msgp = "msgp",
    msgp_temporal = "msgp_temporal", hsgp = "hsgp",
    svc = "svc", svc_hsgp = "svc_hsgp", tvc = "tvc",
    temporal_gp = "temporal_gp", ms_temporal = "ms_temporal",
    latent = "latent", st4 = "spatiotemporal", icar_st = "spatiotemporal",
    icar_collapsed = "icar_collapsed", bym2_collapsed = "icar_collapsed",
    gp_collapsed = "gp_collapsed",
    icar_collapsed_rw1 = "icar_collapsed", bym2_collapsed_rw1 = "icar_collapsed",
    # Two blocks no specialized function writes together.
    gp_st4 = "composite", gp_stgp = "composite",
    gp_temporal_st4 = "composite", msgp_st4 = "composite",
    tvc_st4 = "composite", temporal_gp_st4 = "composite",
    gp_tgp = "composite", svc_ms = "composite", gp_slopes = "composite",
    gp_slopes_corr = "composite", gp_crossed = "composite",
    stgp_latent = "composite",
    # A collapsed marginal alongside a second block.
    gp_collapsed_st4 = "numerical", icar_collapsed_st4 = "numerical",
    bym2_collapsed_st4 = "numerical"
  )
  got <- vapply(names(expected), tulpaRatio:::cpp_gradient_dispatch, character(1))
  expect_equal(got, expected)
})

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

for (field in c(MULTI_FIELDS, STRUCTURE_FIELDS, ST_IV_FIELDS,
                ST_OTHER_FIELDS, ST_IV_AR1_FIELDS, ST_GP_FIELDS,
                DISPATCH_MASK_FIELDS)) {
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

# Zero-inflation and hurdle structures on the count-response families. The
# templated density read zi_type only inside its ModelType::BINOMIAL arm, so
# NEGBIN_NEGBIN, POISSON_GAMMA and NEGBIN_GAMMA called the plain
# negbin/poisson likelihood whatever zi_type said (gcol33/tulpaRatio#34): ESS,
# which evaluates that density directly, and the three autodiff gradient modes,
# which differentiate it, all sampled a posterior with no ZI term in it while
# still estimating a ZI coefficient. Nothing built a count-response ZI model
# here before, so neither density nor the analytic gradient was ever checked on
# one.
ZI_COUNT_CASES <- list(
  list(family = "poisson_gamma", zi = "zi_poisson"),
  list(family = "poisson_gamma", zi = "hurdle_poisson"),
  list(family = "negbin_gamma",  zi = "zi_negbin"),
  list(family = "negbin_gamma",  zi = "hurdle_negbin"),
  list(family = "negbin_negbin", zi = "zi_negbin"),
  list(family = "negbin_negbin", zi = "hurdle_negbin")
)

for (case in ZI_COUNT_CASES) {
  for (mode in MODES) {
    test_that(sprintf("analytic gradient matches finite differences (rw1, %s, %s, %s)",
                      mode, case$family, case$zi), {
      r <- tulpaRatio:::cpp_gradient_check("rw1", mode = mode,
                                           family = case$family, zi = case$zi)
      # A ZI block that was never allocated would leave every assertion below
      # true of a model with no zero-inflation in it.
      expect_true("beta_zi" %in% r$block)
      expect_true(all(r$analytic[r$block == "beta_zi"] != 0))
      dev <- rel_dev(r$analytic, r$finite_diff)
      worst <- which.max(dev)
      expect_lt(
        max(dev),
        1e-4,
        label = sprintf(
          "%s/%s/%s: worst parameter %d (block %s), analytic %.6f vs finite-diff %.6f",
          case$family, case$zi, mode, worst, r$block[worst],
          r$analytic[worst], r$finite_diff[worst]
        )
      )
    })
  }
}

test_that("a count-response ZI or hurdle term reaches both log posteriors", {
  # Before #34 the templated density returned the same number for zi_poisson,
  # hurdle_poisson and no zero-inflation at all, and differed from the analytic
  # density by between 1.7 and 46 log units across these six.
  for (case in ZI_COUNT_CASES) {
    r <- tulpaRatio:::cpp_gradient_check("rw1", mode = "arena",
                                         family = case$family, zi = case$zi)
    info <- sprintf("%s / %s", case$family, case$zi)
    expect_true(is.na(r$impl_gap), label = info)
    expect_equal(r$log_post, r$log_post_impl, tolerance = 1e-8, info = info)
  }
})

test_that("each count ZI structure is a density of its own", {
  # Guards the test above. Were the ZI term to reach neither density, all three
  # of these would return one number and the comparison would hold while
  # comparing nothing.
  for (family in c("poisson_gamma", "negbin_gamma", "negbin_negbin")) {
    poisson <- identical(family, "poisson_gamma")
    lp <- function(zi) tulpaRatio:::cpp_logpost_at("rw1", family = family, zi = zi)
    none <- lp("none")
    inflated <- lp(if (poisson) "zi_poisson" else "zi_negbin")
    hurdle <- lp(if (poisson) "hurdle_poisson" else "hurdle_negbin")
    expect_gt(abs(inflated - none), 1.0, label = family)
    expect_gt(abs(hurdle - none), 1.0, label = family)
    expect_gt(abs(inflated - hurdle), 1.0, label = family)
  }
})

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

for (field in KERNEL_FIELDS) {
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

for (field in KERNEL_COLLAPSED_FIELDS) {
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

# Every AR1 correlation in the sweep above is drawn centred on rho = 0, well
# inside its range, so nothing reaches the regime the floor on 1 - rho^2
# exists for. A path that floors a different quantity, at a different point,
# or not at all agrees with its siblings everywhere except there, which is
# where a long-memory fit ends up. At near_unit_rho the two densities were
# 3.14 apart on temporal_ar1, 27.2 on tvc_ar1, 30.0 on st4_ar1 and 1.9e-03 on
# ms_temporal, and the handcoded gradient of the first two was not finite at
# all one step further out (gcol33/tulpaRatio#59).
for (field in c("temporal_ar1", "tvc_ar1", "ms_temporal", "st4_ar1",
                "st4_ar1_nc", "st2_ar1")) {
  test_that(sprintf("the densities agree where the 1 - rho^2 floor binds (%s)", field), {
    r <- tulpaRatio:::cpp_gradient_check(field, mode = "handcoded",
                                         near_unit_rho = TRUE)
    expect_true(is.finite(r$log_post), label = sprintf("%s: log_post", field))
    expect_true(all(is.finite(r$analytic)),
                label = sprintf("%s: analytic gradient", field))
    expect_equal(r$log_post, r$log_post_impl, tolerance = 1e-8,
                 info = sprintf("field = %s", field))
  })
}

for (field in names(BLOCKED_FIELDS)) {
  test_that(sprintf("analytic gradient matches finite differences (%s, handcoded)", field), {
    skip(paste("blocked on", BLOCKED_FIELDS[[field]]))
    r <- tulpaRatio:::cpp_gradient_check(field, mode = "handcoded")
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

test_that("a field the harness builds no structure for is an error, not a pass", {
  # Every branch in make_model() tests the field name, so an unrecognised one
  # falls through all of them and yields a model with no structured field: the
  # check would then compare the gradient of a plain GLM and report success for
  # a kernel it never reached.
  expect_error(tulpaRatio:::cpp_gradient_check("no_such_field"), "unknown field")
  expect_error(tulpaRatio:::cpp_logpost_at("no_such_field"), "unknown field")
})

test_that("the collapsed parameterizations are declared, not silently mis-differentiated", {
  # The templated density cannot express them, so it must say so and the
  # autodiff modes must refuse. Left undeclared, the gradient of every parameter
  # is taken against a posterior with no spatial marginal in it.
  for (field in c(COLLAPSED_FIELDS, COLLAPSED_TEMPORAL_FIELDS,
                  KERNEL_COLLAPSED_FIELDS)) {
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

test_that("a non-centred GP fit keeps its handcoded gradient (gcol33/tulpaRatio#57)", {
  skip_on_cran()
  # resolve_gradient_fn sends any GP model at the default gradient mode to
  # compute_gradient_gp_handcoded, so the non-centred parameterization is the
  # default path for these fits and not an opt-in kernel. Two defects made that
  # gradient disagree with its density: nngp_nc_backward read a C_mat it never
  # filled, which zeroed dC/dphi and so dropped both the -dC*alpha term from
  # dalpha and the alpha'dC*alpha term from dd_dphi; and the caller subtracted
  # z after nngp_nc_backward had already seeded grad_z with -z, applying the
  # N(0, 1) prior twice. The runtime check caught the first at log_phi and fell
  # back to numerical gradients, so the fits were slow rather than wrong -- this
  # asserts there is nothing left to fall back from.
  set.seed(11)
  n_loc <- 30L
  reps <- 4L
  co <- cbind(runif(n_loc), runif(n_loc))
  S <- 0.8 * exp(-as.matrix(dist(co)) / 0.25)
  w <- as.vector(t(chol(S + diag(1e-8, n_loc))) %*% rnorm(n_loc))
  loc <- rep(seq_len(n_loc), each = reps)
  x <- rnorm(n_loc * reps)
  n <- rep(25L, n_loc * reps)
  df <- data.frame(
    y = rbinom(n_loc * reps, n, plogis(-0.3 + 0.6 * x + w[loc])),
    n = n, x = x, lon = co[loc, 1], lat = co[loc, 2]
  )
  for (par in c("centered", "noncentered")) {
    expect_warning(
      tratio(y | n ~ x, data = df, family = ratiod_binomial(), mode = "hmc",
             spatial = spatial_gp(~ lon + lat, parameterization = par, nn = 5),
             control = list(iter = 60, warmup = 30, chains = 1,
                            verbose = FALSE, seed = 7)),
      NA
    )
  }
})
