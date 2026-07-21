# The chain-parallel helper: every chain runs, and `cores` bounds how many run
# at once. Both are invisible in the draws -- a fit that serializes its chains
# returns exactly what a parallel one returns, only slower.

combos <- list(
  list(chains = 4L, cap = 4L),
  list(chains = 4L, cap = 2L),
  list(chains = 4L, cap = 1L),
  list(chains = 2L, cap = 2L),
  list(chains = 8L, cap = 3L),
  list(chains = 1L, cap = 1L)
)

test_that("every chain runs exactly once whatever the core budget", {
  for (cc in combos) {
    expect_equal(
      tulpaRatio:::cpp_chain_visits(cc$chains, cc$cap),
      cc$chains,
      info = sprintf("chains=%d cap=%d", cc$chains, cc$cap)
    )
  }
})

test_that("concurrency never exceeds the core budget", {
  for (cc in combos) {
    expect_lte(
      tulpaRatio:::cpp_chain_concurrency(cc$chains, cc$cap),
      min(cc$chains, cc$cap)
    )
  }
})

test_that("concurrency reaches the core budget", {
  skip_if(tulpaRatio:::cpp_num_procs() < 8, "needs 8 cores to observe 8/3")
  for (cc in combos) {
    expect_equal(
      tulpaRatio:::cpp_chain_concurrency(cc$chains, cc$cap),
      min(cc$chains, cc$cap),
      info = sprintf("chains=%d cap=%d", cc$chains, cc$cap)
    )
  }
})

test_that("a narrow fit after a wide one still honours its own budget", {
  skip_if(tulpaRatio:::cpp_num_procs() < 8, "needs 8 cores")
  # The team is grown by the first call and never shrinks, so the second call
  # meets a team wider than its budget and must still cap itself.
  expect_equal(tulpaRatio:::cpp_chain_concurrency(8L, 8L), 8L)
  expect_equal(tulpaRatio:::cpp_chain_concurrency(2L, 2L), 2L)
  expect_equal(tulpaRatio:::cpp_chain_concurrency(4L, 4L), 4L)
})

test_that("a backend that moves the OpenMP thread count does not serialize chains", {
  skip_if(tulpaRatio:::cpp_num_procs() < 4, "needs 4 cores")
  # laplace_core and the Polya-Gamma samplers call omp_set_num_threads() and
  # leave it there. A chain team read from that value collapses to 1 and every
  # later fit in the session runs its chains one at a time (#8).
  set.seed(1)
  m <- 200L
  invisible(tulpaRatio:::cpp_laplace_fit(
    y = as.numeric(rbinom(m, 20, 0.4)), n = rep(20, m),
    X = cbind(1, rnorm(m)), re_idx = as.integer(rep(0:9, length.out = m)),
    n_re_groups = 10L, sigma_re = 1.0, family = "binomial", phi = 1.0,
    max_iter = 50L, tol = 1e-6, n_threads = 1L
  ))
  expect_equal(tulpaRatio:::cpp_get_max_threads(), 1L)
  expect_equal(tulpaRatio:::cpp_chain_concurrency(4L, 4L), 4L)
})
