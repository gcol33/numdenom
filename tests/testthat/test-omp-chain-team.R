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

test_that("a within-chain region reaches its width even nested inside a chain team", {
  skip_if(tulpaRatio:::cpp_num_procs() < 4, "needs 4 cores")
  # Every chain body runs inside the chain-parallel region, so a region within
  # a chain is a nested one, and OpenMP leaves nested teams at a single thread
  # unless the active-level limit is raised (#17): `cores` split as a per-chain
  # budget could never reach the gradient/likelihood loops that read it.
  r <- tulpaRatio:::cpp_within_chain_team(n_chains = 2L, max_concurrent = 2L, want = 4L)
  expect_equal(unname(r["outside"]), 4L)
  expect_equal(unname(r["inside"]), 4L)
})

test_that("a moved OpenMP thread count does not serialize chains", {
  skip_if(tulpaRatio:::cpp_num_procs() < 4, "needs 4 cores")
  # A chain team read from the nthreads-var collapses to 1 whenever something
  # has moved that var, and every later fit in the session then runs its chains
  # one at a time (#8). The team is sized from the core budget instead, so the
  # var can sit anywhere.
  restore <- tulpaRatio:::cpp_get_max_threads()
  on.exit(tulpaRatio:::cpp_set_max_threads(restore), add = TRUE)

  tulpaRatio:::cpp_set_max_threads(1L)
  expect_equal(tulpaRatio:::cpp_get_max_threads(), 1L)
  expect_equal(tulpaRatio:::cpp_chain_concurrency(4L, 4L), 4L)
  expect_equal(tulpaRatio:::cpp_chain_visits(4L, 4L), 4L)
})
