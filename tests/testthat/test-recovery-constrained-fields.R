# Recovery against simulated truth for models carrying a constrained
# (rank-deficient) field. Shape and class assertions pass just as happily when
# the intercept is unidentified, which is how a missing sum-to-zero constraint
# reached a release: nothing asserted that the intercept lands near the value it
# was simulated from.

TRUE_INTERCEPT <- 0.5
TRUE_SLOPE <- 0.3

sim_data <- function(seed, N = 500, S = 50, TT = 20) {
  set.seed(seed)
  x <- rnorm(N)
  site <- factor(rep(seq_len(S), length.out = N))
  time <- factor(rep(seq_len(TT), length.out = N))
  trials <- sample(10:50, N, replace = TRUE)
  eta <- TRUE_INTERCEPT + TRUE_SLOPE * x
  data.frame(y = rbinom(N, trials, plogis(eta)), trials = trials,
             x = x, site = site, time = time, spatial_site = site,
             lon = runif(N), lat = runif(N))
}

grid_adj <- function(S) {
  n <- ceiling(sqrt(S))
  g <- expand.grid(lon = seq_len(n), lat = seq_len(n))[seq_len(S), ]
  A <- matrix(0, S, S)
  for (i in seq_len(S)) for (j in seq_len(S)) {
    if (i != j && sqrt((g$lon[i] - g$lon[j])^2 + (g$lat[i] - g$lat[j])^2) <= 1.5)
      A[i, j] <- 1
  }
  dimnames(A) <- list(as.character(seq_len(S)), as.character(seq_len(S)))
  A
}

fit_one <- function(df, ...) {
  f <- tratio(y | trials ~ x, data = df, family = ratiod_binomial(), ...,
              control = list(iter = 1000, warmup = 500, chains = 4, verbose = FALSE))
  dr <- as.matrix(f$draws)
  per <- nrow(dr) / f$chains
  summarise <- function(v) {
    m <- matrix(dr[, v], nrow = per, ncol = f$chains)
    list(mean = mean(dr[, v]),
         lo = unname(quantile(dr[, v], 0.025)),
         hi = unname(quantile(dr[, v], 0.975)),
         rhat = posterior::rhat(m),
         ess = posterior::ess_bulk(m))
  }
  list(backend = f$backend,
       intercept = summarise("beta_num[1]"),
       slope = summarise("beta_num[2]"))
}

test_that("RW1 recovers the intercept and slope with calibrated intervals", {
  skip_on_cran()
  skip_if_not_installed("posterior")

  seeds <- c(11, 22, 33)
  cover_i <- 0L; cover_s <- 0L
  for (s in seeds) {
    r <- fit_one(sim_data(s), temporal = temporal_rw1("time"))
    expect_identical(r$backend, "hmc")

    # Convergence must hold before recovery means anything.
    expect_lt(r$intercept$rhat, 1.05)
    expect_lt(r$slope$rhat, 1.05)
    expect_gt(r$intercept$ess, 400)

    # The intercept is the quantity a missing sum-to-zero constraint destroys.
    expect_equal(r$intercept$mean, TRUE_INTERCEPT, tolerance = 0.08)
    expect_equal(r$slope$mean, TRUE_SLOPE, tolerance = 0.08)

    cover_i <- cover_i + (r$intercept$lo <= TRUE_INTERCEPT && TRUE_INTERCEPT <= r$intercept$hi)
    cover_s <- cover_s + (r$slope$lo <= TRUE_SLOPE && TRUE_SLOPE <= r$slope$hi)
  }
  expect_gte(cover_i, length(seeds) - 1L)
  expect_gte(cover_s, length(seeds) - 1L)
})

test_that("ICAR combined with RW1 recovers the intercept", {
  skip_on_cran()
  skip_if_not_installed("posterior")
  r <- fit_one(sim_data(11), spatial = spatial_car(grid_adj(50), level = "group",
                                                   group_var = "spatial_site"),
               temporal = temporal_rw1("time"))
  expect_lt(r$intercept$rhat, 1.05)
  expect_equal(r$intercept$mean, TRUE_INTERCEPT, tolerance = 0.08)
})

test_that("multiscale trend and seasonal recover the intercept", {
  skip_on_cran()
  skip_if_not_installed("posterior")

  # Trend (RW1) and seasonal (cyclic RW1) are both intrinsic and both land on
  # the same linear predictor, so each carries a constant null direction that is
  # unidentified against the intercept AND against the other. With neither
  # pinned the intercept is free to move along a two-dimensional ridge; the
  # shape of the fitted series is unaffected, which is why only a recovery
  # assertion on the intercept catches it.
  r <- fit_one(sim_data(11),
               temporal = temporal_multiscale("time", trend = "rw1",
                                              seasonal = 4, short_term = "none"))
  expect_lt(r$intercept$rhat, 1.05)
  expect_equal(r$intercept$mean, TRUE_INTERCEPT, tolerance = 0.08)
  expect_equal(r$slope$mean, TRUE_SLOPE, tolerance = 0.08)
})

test_that("multiscale RW2 trend recovers the intercept", {
  skip_on_cran()
  skip_if_not_installed("posterior")

  r <- fit_one(sim_data(22),
               temporal = temporal_multiscale("time", trend = "rw2",
                                              seasonal = NULL, short_term = "none"))
  expect_lt(r$intercept$rhat, 1.05)
  expect_equal(r$intercept$mean, TRUE_INTERCEPT, tolerance = 0.08)
})

test_that("a TVC term recovers the intercept and its own slope", {
  skip_on_cran()
  skip_if_not_installed("posterior")

  # A TVC trajectory is an intrinsic RW over time per (group, term). Its mean is
  # unidentified against the fixed coefficient on the same covariate, so the
  # quantity that moves when the pin is absent is the SLOPE rather than the
  # intercept.
  r <- fit_one(sim_data(11),
               temporal = temporal_tvc("time", terms = "x", structure = "rw1"))
  expect_lt(r$intercept$rhat, 1.05)
  expect_lt(r$slope$rhat, 1.05)
  expect_equal(r$intercept$mean, TRUE_INTERCEPT, tolerance = 0.1)
  expect_equal(r$slope$mean, TRUE_SLOPE, tolerance = 0.1)
})

# No SVC case here. An NNGP field's prior is proper, so its mean constraint is a
# ridge rather than an identification pin, and its constant is still the
# unexplained 1/n_obs it has always had (gcol33/tulpaRatio#25). A fit small
# enough to belong in a test file takes minutes, so the case belongs with the
# constant it is meant to check.

test_that("a spatiotemporal interaction recovers the intercept", {
  skip_on_cran()
  skip_if_not_installed("posterior")

  # The interaction is pinned along both margins. A space margin sums S terms
  # and a time margin T, so the two carry their own constants; sharing one
  # leaves whichever margin is longer under-identified. Kept to 9 x 5 so the
  # interaction has fewer coefficients than the data has observations.
  #
  # This data carries no true space-time signal, so the interaction's
  # posterior for tau drifts to large values and, under the centered
  # parameterization, dragged rhat to 1.14 (and to 2.05 over a 4x longer run,
  # with the per-chain SD shrinking) -- a funnel between tau and the field's
  # conditional scale that gcol33/tulpaRatio#24 traced to no single step size
  # spanning both regimes. parameterization = "noncentered" samples a tau-free
  # z and reconstructs the interaction as z / sqrt(tau), which decouples the
  # two: rhat drops to 1.004 (intercept) / 1.000 (slope) at the same 1000-iter
  # budget this test uses.
  df <- sim_data(11, S = 9, TT = 5)
  r <- fit_one(df,
               spatiotemporal = spatiotemporal(
                 spatial = spatial_car(grid_adj(9), level = "group",
                                       group_var = "spatial_site"),
                 temporal = temporal_rw1("time"),
                 type = "IV",
                 parameterization = "noncentered"))
  expect_equal(r$intercept$mean, TRUE_INTERCEPT, tolerance = 0.1)
  expect_equal(r$slope$mean, TRUE_SLOPE, tolerance = 0.1)
  expect_lt(r$intercept$rhat, 1.05)
  expect_lt(r$slope$rhat, 1.05)
})

test_that("proper CAR recovers rho and the intercept (gcol33/tulpaRatio#31)", {
  skip_on_cran()
  skip_if_not_installed("posterior")
  skip_if_not_installed("MASS")

  # Simulate directly from Q(rho) = D - rho*W so the field itself, not just
  # the regression coefficients, carries a known truth to recover.
  S <- 36L
  A <- grid_adj(S)
  D <- diag(rowSums(A))
  rho_true <- 0.75
  tau_true <- 3.0
  set.seed(44)
  Q <- D - rho_true * A
  phi_true <- as.numeric(MASS::mvrnorm(1, mu = rep(0, S), Sigma = solve(tau_true * Q)))

  n_per <- 25L
  N <- S * n_per
  site <- rep(seq_len(S), each = n_per)
  x <- rnorm(N)
  trials <- sample(15:50, N, replace = TRUE)
  eta <- TRUE_INTERCEPT + TRUE_SLOPE * x + phi_true[site]
  df <- data.frame(y = rbinom(N, trials, plogis(eta)), trials = trials, x = x,
                   site = factor(site))

  f <- tratio(y | trials ~ x, data = df, family = ratiod_binomial(),
             spatial = spatial_car(A, level = "group", group_var = "site", proper = TRUE),
             mode = "hmc",
             control = list(iter = 1500, warmup = 750, chains = 4, verbose = FALSE))
  expect_identical(f$backend, "hmc")

  dr <- as.matrix(f$draws)
  expect_true("rho_spatial" %in% colnames(dr))
  expect_true("tau_spatial" %in% colnames(dr))
  per <- nrow(dr) / f$chains
  summarise <- function(v) {
    m <- matrix(dr[, v], nrow = per, ncol = f$chains)
    list(mean = mean(dr[, v]),
         lo = unname(quantile(dr[, v], 0.025)),
         hi = unname(quantile(dr[, v], 0.975)),
         rhat = posterior::rhat(m))
  }
  r_int <- summarise("beta_num[1]")
  r_rho <- summarise("rho_spatial")
  r_tau <- summarise("tau_spatial")

  expect_lt(r_int$rhat, 1.05)
  expect_lt(r_rho$rhat, 1.1)
  expect_equal(r_int$mean, TRUE_INTERCEPT, tolerance = 0.15)

  # rho is weakly identified even at this size (a well-known property of
  # proper CAR, not specific to this implementation), so the interval, not
  # the point estimate, is what recovery means here.
  expect_true(r_rho$lo <= rho_true && rho_true <= r_rho$hi,
              label = sprintf("rho 95%% CI [%.3f, %.3f] vs true %.2f",
                              r_rho$lo, r_rho$hi, rho_true))
  expect_true(r_tau$lo <= tau_true && tau_true <= r_tau$hi,
              label = sprintf("tau 95%% CI [%.3f, %.3f] vs true %.2f",
                              r_tau$lo, r_tau$hi, tau_true))
})

test_that("a spatial-only model reaches a sampler that reports its chains", {
  skip_on_cran()
  # Spatial-only models route to the Gibbs backend, which must honour chains=
  # and expose the spatial field so the convergence diagnostics above apply to
  # them too.
  df <- sim_data(11)
  f <- tratio(y | trials ~ x, data = df, family = ratiod_binomial(),
              spatial = spatial_car(grid_adj(50), level = "group",
                                    group_var = "spatial_site"),
              control = list(iter = 1000, warmup = 500, chains = 4, verbose = FALSE))
  expect_identical(as.integer(f$chains), 4L)
  expect_true(any(grepl("^phi_spatial\\[", colnames(as.matrix(f$draws)))))
})
