# test-benchmark.R
# Tests for benchmark utilities

test_that("quotr_threads returns positive integer", {
  n_threads <- quotr_threads()

  expect_true(is.numeric(n_threads))
  expect_true(n_threads >= 1)
  expect_true(n_threads == as.integer(n_threads))
})


test_that("quotr_benchmark runs on small data", {
  skip_if_not_installed("quotr")

  # Very small benchmark for testing
  bench <- quotr_benchmark(
    N = 30,
    p = 2,
    n_groups = 0,
    n_iter = 100,
    n_warmup = 50,
    verbose = FALSE
  )

  expect_s3_class(bench, "quotr_benchmark")
  expect_true(bench$timing$elapsed_seconds > 0)
  expect_true(bench$timing$samples_per_second > 0)
  expect_equal(bench$config$N, 30)
  expect_equal(bench$config$p, 2)
})


test_that("quotr_benchmark handles random effects", {
  skip_if_not_installed("quotr")

  bench <- quotr_benchmark(
    N = 50,
    p = 2,
    n_groups = 5,
    n_iter = 100,
    n_warmup = 50,
    verbose = FALSE
  )

  expect_s3_class(bench, "quotr_benchmark")
  expect_equal(bench$config$n_groups, 5)
  # Should have more parameters with random effects
  expect_true(bench$config$n_params > 4)
})


test_that("print.quotr_benchmark works", {
  bench <- list(
    config = list(
      N = 100,
      p = 3,
      n_groups = 5,
      n_params = 10,
      n_iter = 500,
      n_warmup = 250,
      n_chains = 1,
      n_threads = 1
    ),
    timing = list(
      elapsed_seconds = 10.5,
      samples_per_second = 23.8,
      iterations_per_second = 47.6
    ),
    diagnostics = list(
      n_divergent = 2,
      divergent_pct = 0.8
    )
  )
  class(bench) <- c("quotr_benchmark", "list")

  output <- capture.output(print(bench))

  expect_true(any(grepl("quotr Benchmark", output)))
  expect_true(any(grepl("100", output)))  # N
  expect_true(any(grepl("10.5", output)))  # elapsed
})
