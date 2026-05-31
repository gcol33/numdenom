test_that("tulpa engine ABI is available", {
  # Track tulpa's TULPA_ABI_VERSION (currently 32; see
  # tulpa/inst/include/tulpa/model_data.h). Bump here whenever tulpa
  # bumps the ABI.
  expect_equal(tulpaRatio:::cpp_tulpa_abi_version(), 32L)
})

test_that("tulpa PG binomial bridge returns legacy result shape", {
  set.seed(1)
  X <- cbind(1, x = seq(-1, 1, length.out = 12))
  group <- rep(1:3, each = 4)
  n <- rep(10L, 12)
  y <- rbinom(12, n, plogis(0.2 + 0.4 * X[, 2]))

  fit <- tulpaRatio:::cpp_tulpa_pg_binomial_gibbs(
    y = as.integer(y),
    n = as.integer(n),
    X = X,
    group = as.integer(group),
    n_groups = 3L,
    n_iter = 8L,
    n_warmup = 4L,
    thin = 2L,
    store_eta = TRUE,
    verbose = FALSE,
    n_threads = 1L
  )

  expect_equal(dim(fit$beta), c(2L, 2L))
  expect_equal(dim(fit$re), c(2L, 3L))
  expect_equal(length(fit$sigma_re), 2L)
  expect_equal(dim(fit$eta), c(2L, 12L))
})
