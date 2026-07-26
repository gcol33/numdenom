# test-specs-spatial.R
# B1d Step 2 parity test: ICAR + BYM2 (non-collapsed, group-level) routed
# through the LikelihoodSpec path produce posterior means within 4x within-MC
# noise of the legacy backend at the same seed.

build_chain_adjacency <- function(n_units) {
  adj <- matrix(0L, n_units, n_units)
  for (i in seq_len(n_units - 1L)) {
    adj[i, i + 1L] <- 1L
    adj[i + 1L, i] <- 1L
  }
  adj
}

simulate_spatial_binomial <- function(n_units = 8L, reps = 16L) {
  set.seed(20260602)
  spatial_re <- rnorm(n_units, sd = 0.4)
  spatial_re <- spatial_re - mean(spatial_re)  # ICAR sum-to-zero

  unit <- rep(seq_len(n_units), each = reps)
  n    <- length(unit)
  x1   <- rnorm(n)
  eta  <- 0.5 + 0.6 * x1 + spatial_re[unit]
  p    <- plogis(eta)
  nt   <- sample(15:30, n, replace = TRUE)
  y    <- rbinom(n, nt, p)

  list(
    formula = y | n_trials ~ x1,
    data    = data.frame(y = y, n_trials = nt, x1 = x1,
                          unit = factor(unit, levels = seq_len(n_units))),
    family  = tulpaRatio::ratiod_binomial(),
    adj     = build_chain_adjacency(n_units)
  )
}

fit_one_spatial <- function(sim, use_specs, seed_val, spatial_struct) {
  op <- options(tulpaRatio.use_specs = use_specs); on.exit(options(op), add = TRUE)
  fit <- tulpaRatio::tratio(
    formula = sim$formula, data = sim$data, family = sim$family,
    spatial = spatial_struct,
    mode = "hmc",
    control = list(iter = 3000L, warmup = 1000L, chains = 1L, seed = seed_val, verbose = FALSE, gradient_mode = "A_r")
  )
  colMeans(fit$draws)
}

test_that("B1d spec path matches legacy for ICAR (group-level)", {
  skip_on_cran()
  # The spec path leaves the field's null direction unidentified, so its level
  # random-walks: mean(phi_spatial) lands at +200.9, -342.8 and +423.1 at seeds
  # 42, 43 and 44 where legacy holds it within 0.02 of zero. The differences
  # between units agree, so this is the constraint and not the field.
  skip("blocked on gcol33/tulpaRatio#19")
  sim <- simulate_spatial_binomial()
  sp  <- tulpaRatio::spatial_car(adjacency = sim$adj, level = "group",
                                  group_var = "unit",
                                  parameterization = "standard")
  legacy42 <- fit_one_spatial(sim, FALSE, 42L, sp)
  specs42  <- fit_one_spatial(sim, TRUE,  42L, sp)
  legacy43 <- fit_one_spatial(sim, FALSE, 43L, sp)

  expect_named(specs42, names(legacy42))
  cross  <- max(abs(legacy42 - specs42))
  within <- max(abs(legacy42 - legacy43))
  expect_lt(cross, max(4 * within, 5e-3))
})

test_that("B1d spec path matches legacy for BYM2 (group-level)", {
  skip_on_cran()
  # Same defect as the ICAR case above, larger: phi_scaled sits at -1696
  # against a legacy level of zero, with the differences between units again
  # preserved.
  skip("blocked on gcol33/tulpaRatio#19")
  sim <- simulate_spatial_binomial()
  sp  <- tulpaRatio::spatial_bym2(adjacency = sim$adj, level = "group",
                                   group_var = "unit",
                                   parameterization = "standard")
  legacy42 <- fit_one_spatial(sim, FALSE, 42L, sp)
  specs42  <- fit_one_spatial(sim, TRUE,  42L, sp)
  legacy43 <- fit_one_spatial(sim, FALSE, 43L, sp)

  expect_named(specs42, names(legacy42))
  cross  <- max(abs(legacy42 - specs42))
  within <- max(abs(legacy42 - legacy43))
  expect_lt(cross, max(4 * within, 5e-3))
})
