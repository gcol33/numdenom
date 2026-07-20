# The log posterior and the analytic gradient are written out separately, once
# per gradient mode and once per specialized path. When a prior term is added to
# one and not the other, the sampler still runs and still returns numbers: it
# simply samples a density whose gradient points somewhere else. Shape and class
# assertions cannot see that. These compare the analytic gradient against
# central differences of the log posterior it is supposed to be the gradient of.

FIELDS <- c("icar", "bym2", "rw1", "rw2", "icar_rw1")
MODES <- c("handcoded", "arena")

# Relative deviation per parameter, floored so that near-zero entries do not
# dominate the comparison.
rel_dev <- function(a, f) {
  abs(a - f) / pmax(1e-6, pmax(abs(a), abs(f)))
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

test_that("analytic and autodiff log posteriors agree up to a constant", {
  # Both implementations omit the same normalizing constant, so the offset
  # between compute_log_post and each mode's fused value must not depend on
  # the model. A model-dependent offset means the two densities differ.
  # The fused value each mode reports must satisfy this, not just the autodiff
  # one: NUTS consumes it as log_prob, so a mode whose value disagrees with its
  # own gradient drives the Hamiltonian off a different density.
  for (mode in c("arena", "handcoded")) {
    offsets <- vapply(FIELDS, function(field) {
      r <- tulpaRatio:::cpp_gradient_check(field, mode = mode)
      r$log_post - r$log_post_mode
    }, numeric(1))
    expect_lt(max(offsets) - min(offsets), 1e-6)
  }
})

test_that("centring makes the spatial field's mean invisible to the likelihood", {
  # A hard sum-to-zero constraint means the intercept, not the field, carries
  # the level. Shifting every phi by a constant must leave the gradient of the
  # fixed effects untouched.
  for (field in c("icar", "bym2", "icar_rw1")) {
    raw <- tulpaRatio:::cpp_gradient_check(field, mode = "handcoded", precenter = FALSE)
    cen <- tulpaRatio:::cpp_gradient_check(field, mode = "handcoded", precenter = TRUE)
    beta <- which(raw$block == "beta_num")
    expect_equal(raw$analytic[beta], cen$analytic[beta], tolerance = 1e-8,
                 info = sprintf("field = %s", field))
  }
})
