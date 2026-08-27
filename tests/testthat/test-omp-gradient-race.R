# Gradient functions against the concurrency a fit actually runs them under.
#
# Chains run in parallel over ONE ModelData. A gradient function that keeps its
# intermediates in that shared object has every chain writing the same memory,
# so the gradient it returns depends on where the other chains happened to be.
# NUTS then samples a density its own gradient does not describe: the step size
# adapts toward zero, every chain freezes near its warmup endpoint at a
# different constant, and rhat lands in the single digits. A single-point
# finite-difference check agrees the whole time, because one thread never races
# itself -- which is how this reached a release twice, for TVC
# (gcol33/tulpaRatio#23) and for SVC (#30).
#
# cpp_gradient_race() evaluates a field's gradient at several points serially
# and then concurrently, and reports how far the two disagree. Per-thread
# scratch makes them agree exactly, so anything but 0 is a shared buffer.

# The collapsed fields carry state from one call to the next on purpose -- the
# NNGP coefficients for the last (sigma2, phi), the location-to-observation map,
# and a Newton mode that the next call warm-starts from. The cached structure is
# keyed on the model it was built for (gcol33/tulpaRatio#76), and the probe
# clears the warm start before every evaluation, so their gradient is a function
# of the point like every other field's and they are swept with the rest.

test_that("no gradient function keeps its scratch in the shared ModelData", {
  skip_on_cran()
  skip_if(tulpaRatio:::cpp_num_procs() < 4, "needs 4 cores to interleave chains")

  fields <- tulpaRatio:::cpp_gradient_fields()
  expect_gt(length(fields), 50)

  # Every field the C++ harness builds a structure for, so a field added there
  # is raced without a second list to keep in step.
  dev <- vapply(fields, function(f) {
    tulpaRatio:::cpp_gradient_race(f, n_points = 8L, n_threads = 4L, rounds = 8L)
  }, numeric(1))

  racing <- names(dev)[dev != 0]
  expect_identical(
    racing, character(0),
    info = paste0(
      "these gradients differ between a serial and a concurrent pass over one ",
      "ModelData, so they write scratch that every chain thread shares:\n",
      paste(sprintf("  %s: max deviation %.3e", racing, dev[racing]),
            collapse = "\n"))
  )
})

test_that("the race probe measures the SVC gradient it was written for (#30)", {
  skip_on_cran()
  skip_if(tulpaRatio:::cpp_num_procs() < 4, "needs 4 cores to interleave chains")

  # A probe that reached no gradient at all would report 0 for every field.
  # SVC is the path #30 was about, so assert the concurrent pass produces the
  # same numbers as the serial one AND that those numbers are a real gradient.
  expect_identical(tulpaRatio:::cpp_gradient_dispatch("svc"), "svc")
  expect_equal(tulpaRatio:::cpp_gradient_race("svc", n_points = 8L,
                                              n_threads = 4L, rounds = 16L), 0)

  # Before the fix this same call returned a deviation of 93.
  g <- tulpaRatio:::cpp_gradient_check("svc")
  expect_true(all(is.finite(g$analytic)))
  expect_gt(max(abs(g$analytic)), 0)
})

test_that("a collapsed workspace drops a previous model's structure (#76)", {
  skip_on_cran()
  skip_if(tulpaRatio:::cpp_num_procs() < 4, "needs 4 cores to interleave chains")

  # prime_other_model evaluates a second model of the same dimensions at the
  # same point on the calling thread first. Its NNGP coefficients are a
  # different set -- the coordinates come from a different seed -- but the
  # workspace's old cache key, the number of GP locations, cannot tell the two
  # apart, and (sigma2, phi) match because the point does. The serial pass then
  # runs on that thread while the concurrent pass adds workers that build the
  # current model's structure, and the two disagree by whatever the foreign
  # coefficients are worth.
  collapsed <- c("gp_collapsed", "gp_collapsed_st4")
  dev <- vapply(collapsed, function(f) {
    tulpaRatio:::cpp_gradient_race(f, n_points = 4L, n_threads = 4L, rounds = 4L,
                                   prime_other_model = TRUE)
  }, numeric(1))

  stale <- names(dev)[dev != 0]
  expect_identical(
    stale, character(0),
    info = paste0(
      "these gradients change when the thread last evaluated a different ",
      "model, so their workspace cache is not keyed on the model:\n",
      paste(sprintf("  %s: max deviation %.3e", stale, dev[stale]),
            collapse = "\n"))
  )
})
