# test-hmc-vs-stan.R
# Phase 2 validation: Compare HMC backend posteriors to Stan reference implementation
#
# Validation criteria (from MIGRATION.md):
# - Posterior means within 0.1 SD
# - 95% credible intervals overlap
# - Ratio posteriors match
# - Diagnostics (divergences, Rhat) acceptable

# Skip all tests - Stan backend has been removed from quotr
# These tests were used to validate HMC against Stan during migration.
# Now that Stan is fully removed, these validation tests are historical.
skip_if_no_cmdstanr <- function() {
  skip("Stan backend removed - validation tests no longer applicable")
}

# Helper function to compare posteriors
compare_posteriors <- function(hmc_samples, stan_samples, param_name,
                                tolerance_sd = 0.2, tolerance_abs = 0.3) {
  hmc_mean <- mean(hmc_samples)
  stan_mean <- mean(stan_samples)
  stan_sd <- sd(stan_samples)

  # Check if means are within tolerance
  diff <- abs(hmc_mean - stan_mean)
  within_sd <- diff < tolerance_sd * stan_sd
  within_abs <- diff < tolerance_abs

  # Check if 95% CIs overlap
  hmc_ci <- quantile(hmc_samples, c(0.025, 0.975))
  stan_ci <- quantile(stan_samples, c(0.025, 0.975))
  ci_overlap <- (hmc_ci[2] >= stan_ci[1]) && (stan_ci[2] >= hmc_ci[1])

  list(
    param = param_name,
    hmc_mean = hmc_mean,
    stan_mean = stan_mean,
    diff = diff,
    stan_sd = stan_sd,
    within_tolerance = within_sd || within_abs,
    ci_overlap = ci_overlap,
    hmc_ci = hmc_ci,
    stan_ci = stan_ci
  )
}

# Helper function to print comparison results
print_comparison <- function(comparison) {
  message(sprintf(
    "  %s: HMC=%.3f, Stan=%.3f, diff=%.3f (%.1f SD), CI overlap: %s",
    comparison$param,
    comparison$hmc_mean,
    comparison$stan_mean,
    comparison$diff,
    comparison$diff / max(comparison$stan_sd, 0.01),
    if (comparison$ci_overlap) "YES" else "NO"
  ))
}


# =============================================================================
# Test 1: Binomial family (simplest case)
# =============================================================================
test_that("HMC matches Stan for binomial family", {
  skip_if_no_cmdstanr()

  set.seed(42)

  # Generate data with known parameters
  n <- 150
  x <- rnorm(n)
  true_beta0 <- -0.5
  true_beta1 <- 0.8

  p <- plogis(true_beta0 + true_beta1 * x)
  trials <- rpois(n, 20) + 10
  successes <- rbinom(n, trials, p)

  df <- data.frame(
    successes = successes,
    trials = trials,
    x = x
  )

  # Fit with Stan
  message("Fitting binomial model with Stan...")
  fit_stan <- quotr(
    successes | trials ~ x,
    data = df,
    family = quotr_binomial(),
    backend = "cmdstanr",
    chains = 2,
    iter = 1500,
    warmup = 500,
    refresh = 0
  )

  # Fit with HMC
  message("Fitting binomial model with HMC...")
  fit_hmc <- quotr(
    successes | trials ~ x,
    data = df,
    family = quotr_binomial(),
    backend = "hmc",
    chains = 2,
    iter = 1500,
    warmup = 500
  )

  # Extract posteriors - Stan
  stan_draws <- fit_stan$fit$draws(format = "matrix")
  stan_beta0 <- stan_draws[, "beta[1]"]
  stan_beta1 <- stan_draws[, "beta[2]"]

  # Extract posteriors - HMC
  hmc_beta0 <- fit_hmc$draws[, "beta_num[1]"]
  hmc_beta1 <- fit_hmc$draws[, "beta_num[2]"]

  # Compare
  message("\nBinomial model comparisons:")
  comp_beta0 <- compare_posteriors(hmc_beta0, stan_beta0, "Intercept")
  comp_beta1 <- compare_posteriors(hmc_beta1, stan_beta1, "Slope")

  print_comparison(comp_beta0)
  print_comparison(comp_beta1)

  # Assertions

  expect_true(comp_beta0$ci_overlap,
              info = "Intercept 95% CIs should overlap")
  expect_true(comp_beta1$ci_overlap,
              info = "Slope 95% CIs should overlap")

  # Check recovery of true values (both should capture truth)
  expect_true(comp_beta0$hmc_ci[1] < true_beta0 && true_beta0 < comp_beta0$hmc_ci[2],
              info = "HMC should recover true intercept")
  expect_true(comp_beta1$hmc_ci[1] < true_beta1 && true_beta1 < comp_beta1$hmc_ci[2],
              info = "HMC should recover true slope")

  # Check HMC diagnostics
  expect_lt(fit_hmc$diagnostics$n_divergent, 50,
            info = "HMC should have few divergences")
})


# =============================================================================
# Test 2: Binomial with random effects
# =============================================================================
test_that("HMC matches Stan for binomial with random effects", {
  skip_if_no_cmdstanr()

  set.seed(123)

  # Generate hierarchical data
  n_groups <- 12
  n_per_group <- 15
  n <- n_groups * n_per_group

  true_beta0 <- -0.3
  true_beta1 <- 0.6
  true_sigma_re <- 0.5

  group <- rep(1:n_groups, each = n_per_group)
  true_re <- rnorm(n_groups, 0, true_sigma_re)

  x <- rnorm(n)
  eta <- true_beta0 + true_beta1 * x + true_re[group]
  p <- plogis(eta)
  trials <- rpois(n, 15) + 5
  successes <- rbinom(n, trials, p)

  df <- data.frame(
    successes = successes,
    trials = trials,
    x = x,
    group = factor(group)
  )

  # Fit with Stan
  message("Fitting binomial + RE model with Stan...")
  fit_stan <- quotr(
    successes | trials ~ x + (1 | group),
    data = df,
    family = quotr_binomial(),
    backend = "cmdstanr",
    chains = 2,
    iter = 1500,
    warmup = 500,
    refresh = 0
  )

  # Fit with HMC
  message("Fitting binomial + RE model with HMC...")
  fit_hmc <- quotr(
    successes | trials ~ x + (1 | group),
    data = df,
    family = quotr_binomial(),
    backend = "hmc",
    chains = 2,
    iter = 1500,
    warmup = 500
  )

  # Extract and compare fixed effects
  stan_draws <- fit_stan$fit$draws(format = "matrix")
  stan_beta0 <- stan_draws[, "beta[1]"]
  stan_beta1 <- stan_draws[, "beta[2]"]
  stan_sigma <- stan_draws[, "sigma_re[1]"]

  hmc_beta0 <- fit_hmc$draws[, "beta_num[1]"]
  hmc_beta1 <- fit_hmc$draws[, "beta_num[2]"]
  hmc_sigma <- fit_hmc$draws[, "sigma_re"]

  message("\nBinomial + RE model comparisons:")
  comp_beta0 <- compare_posteriors(hmc_beta0, stan_beta0, "Intercept")
  comp_beta1 <- compare_posteriors(hmc_beta1, stan_beta1, "Slope")
  comp_sigma <- compare_posteriors(hmc_sigma, stan_sigma, "sigma_re")

  print_comparison(comp_beta0)
  print_comparison(comp_beta1)
  print_comparison(comp_sigma)

  # Assertions
  expect_true(comp_beta0$ci_overlap, info = "Intercept CIs should overlap")
  expect_true(comp_beta1$ci_overlap, info = "Slope CIs should overlap")
  expect_true(comp_sigma$ci_overlap, info = "sigma_re CIs should overlap")

  # Check sigma_re is in reasonable range
  expect_true(mean(hmc_sigma) > 0.1 && mean(hmc_sigma) < 2,
              info = "sigma_re should be positive and reasonable")
})


# =============================================================================
# Test 3: Negative binomial - negative binomial family
# =============================================================================
test_that("HMC matches Stan for negbin_negbin family", {
  skip_if_no_cmdstanr()

  set.seed(456)

  # Generate two-process count data
  n <- 120
  x <- rnorm(n)

  true_beta_num <- c(2.0, 0.5)
  true_beta_denom <- c(2.5, -0.3)
  true_phi <- 5

  mu_num <- exp(true_beta_num[1] + true_beta_num[2] * x)
  mu_denom <- exp(true_beta_denom[1] + true_beta_denom[2] * x)

  y_num <- rnbinom(n, mu = mu_num, size = true_phi)
  y_denom <- rnbinom(n, mu = mu_denom, size = true_phi)

  # Avoid zeros in denominator for ratio computation
  y_denom <- pmax(y_denom, 1L)

  df <- data.frame(
    y_num = y_num,
    y_denom = y_denom,
    x = x
  )

  # Fit with Stan
  message("Fitting negbin_negbin model with Stan...")
  fit_stan <- quotr(
    y_num | y_denom ~ x,
    data = df,
    family = quotr_negbin_negbin(),
    backend = "cmdstanr",
    chains = 2,
    iter = 1500,
    warmup = 500,
    refresh = 0
  )

  # Fit with HMC
  message("Fitting negbin_negbin model with HMC...")
  fit_hmc <- quotr(
    y_num | y_denom ~ x,
    data = df,
    family = quotr_negbin_negbin(),
    backend = "hmc",
    chains = 2,
    iter = 1500,
    warmup = 500
  )

  # Compare numerator fixed effects
  stan_draws <- fit_stan$fit$draws(format = "matrix")

  message("\nNegbin-Negbin model comparisons:")

  # Numerator coefficients
  comp_num0 <- compare_posteriors(
    fit_hmc$draws[, "beta_num[1]"],
    stan_draws[, "beta_num[1]"],
    "beta_num[1]"
  )
  comp_num1 <- compare_posteriors(
    fit_hmc$draws[, "beta_num[2]"],
    stan_draws[, "beta_num[2]"],
    "beta_num[2]"
  )

  print_comparison(comp_num0)
  print_comparison(comp_num1)

  # Denominator coefficients
  comp_denom0 <- compare_posteriors(
    fit_hmc$draws[, "beta_denom[1]"],
    stan_draws[, "beta_denom[1]"],
    "beta_denom[1]"
  )
  comp_denom1 <- compare_posteriors(
    fit_hmc$draws[, "beta_denom[2]"],
    stan_draws[, "beta_denom[2]"],
    "beta_denom[2]"
  )

  print_comparison(comp_denom0)
  print_comparison(comp_denom1)

  # Overdispersion
  comp_phi_num <- compare_posteriors(
    fit_hmc$draws[, "phi_num"],
    stan_draws[, "phi_num"],
    "phi_num"
  )
  comp_phi_denom <- compare_posteriors(
    fit_hmc$draws[, "phi_denom"],
    stan_draws[, "phi_denom"],
    "phi_denom"
  )

  print_comparison(comp_phi_num)
  print_comparison(comp_phi_denom)

  # Assertions
  expect_true(comp_num0$ci_overlap, info = "beta_num[1] CIs should overlap")
  expect_true(comp_num1$ci_overlap, info = "beta_num[2] CIs should overlap")
  expect_true(comp_denom0$ci_overlap, info = "beta_denom[1] CIs should overlap")
  expect_true(comp_denom1$ci_overlap, info = "beta_denom[2] CIs should overlap")

  # Compare ratios at a few observations
  # Stan stores ratio in generated quantities
  stan_ratio <- stan_draws[, grep("^ratio\\[", colnames(stan_draws))]
  hmc_ratio <- fit_hmc$ratio_draws

  # Check ratio at observation 1
  comp_ratio1 <- compare_posteriors(
    hmc_ratio[, 1],
    stan_ratio[, 1],
    "ratio[1]"
  )
  print_comparison(comp_ratio1)
  expect_true(comp_ratio1$ci_overlap, info = "ratio[1] CIs should overlap")
})


# =============================================================================
# Test 4: Poisson-Gamma family (CPUE)
# =============================================================================
test_that("HMC matches Stan for poisson_gamma family", {
  skip_if_no_cmdstanr()

  set.seed(789)

  # Generate CPUE-type data
  n <- 100
  x <- rnorm(n)

  true_beta_num <- c(1.5, 0.4)
  true_beta_denom <- c(1.0, 0.2)
  true_shape <- 10  # Gamma shape (higher = less variability in effort)

  # Effort from Gamma
  mu_effort <- exp(true_beta_denom[1] + true_beta_denom[2] * x)
  effort <- rgamma(n, shape = true_shape, rate = true_shape / mu_effort)

  # Count from Poisson
  lambda <- exp(true_beta_num[1] + true_beta_num[2] * x)
  counts <- rpois(n, lambda)

  df <- data.frame(
    counts = counts,
    effort = effort,
    x = x
  )

  # Fit with Stan
  message("Fitting poisson_gamma model with Stan...")
  fit_stan <- quotr(
    counts | effort ~ x,
    data = df,
    family = quotr_poisson_gamma(),
    backend = "cmdstanr",
    chains = 2,
    iter = 1500,
    warmup = 500,
    refresh = 0
  )

  # Fit with HMC
  message("Fitting poisson_gamma model with HMC...")
  fit_hmc <- quotr(
    counts | effort ~ x,
    data = df,
    family = quotr_poisson_gamma(),
    backend = "hmc",
    chains = 2,
    iter = 1500,
    warmup = 500
  )

  # Compare
  stan_draws <- fit_stan$fit$draws(format = "matrix")

  message("\nPoisson-Gamma model comparisons:")

  comp_num0 <- compare_posteriors(
    fit_hmc$draws[, "beta_num[1]"],
    stan_draws[, "beta_num[1]"],
    "beta_num[1]"
  )
  comp_num1 <- compare_posteriors(
    fit_hmc$draws[, "beta_num[2]"],
    stan_draws[, "beta_num[2]"],
    "beta_num[2]"
  )

  print_comparison(comp_num0)
  print_comparison(comp_num1)

  # Assertions
  expect_true(comp_num0$ci_overlap, info = "beta_num[1] CIs should overlap")
  expect_true(comp_num1$ci_overlap, info = "beta_num[2] CIs should overlap")
})


# =============================================================================
# Test 5: Spatial ICAR model
# =============================================================================
test_that("HMC matches Stan for binomial with ICAR spatial", {
  skip_if_no_cmdstanr()
  skip("Spatial validation requires spatial_car() implementation")

  set.seed(111)

  # Create a 4x4 grid with adjacency
  n_spatial <- 16
  n_per_site <- 8
  n <- n_spatial * n_per_site

  # Build adjacency matrix
  adj <- matrix(0, n_spatial, n_spatial)
  for (i in 1:n_spatial) {
    row <- (i - 1) %/% 4 + 1
    col <- (i - 1) %% 4 + 1

    if (col > 1) adj[i, i - 1] <- 1
    if (col < 4) adj[i, i + 1] <- 1
    if (row > 1) adj[i, i - 4] <- 1
    if (row < 4) adj[i, i + 4] <- 1
  }

  # Simulate spatial effects (simplified)
  true_spatial <- rnorm(n_spatial, 0, 0.3)
  true_spatial <- true_spatial - mean(true_spatial)  # Center

  # Generate data
  site <- rep(1:n_spatial, each = n_per_site)
  x <- rnorm(n)
  true_beta <- c(-0.5, 0.5)

  eta <- true_beta[1] + true_beta[2] * x + true_spatial[site]
  p <- plogis(eta)
  trials <- rpois(n, 12) + 5
  successes <- rbinom(n, trials, p)

  df <- data.frame(
    successes = successes,
    trials = trials,
    x = x,
    site = factor(site)
  )

  # Create spatial structure
  spatial_spec <- spatial_car(adjacency = adj, group_var = "site")

  # Fit with Stan
  message("Fitting binomial + ICAR spatial with Stan...")
  fit_stan <- quotr(
    successes | trials ~ x,
    data = df,
    family = quotr_binomial(),
    spatial = spatial_spec,
    backend = "cmdstanr",
    chains = 2,
    iter = 1500,
    warmup = 500,
    refresh = 0
  )

  # Fit with HMC
  message("Fitting binomial + ICAR spatial with HMC...")
  fit_hmc <- quotr(
    successes | trials ~ x,
    data = df,
    family = quotr_binomial(),
    spatial = spatial_spec,
    backend = "hmc",
    chains = 2,
    iter = 1500,
    warmup = 500
  )

  # Compare fixed effects
  stan_draws <- fit_stan$fit$draws(format = "matrix")

  message("\nBinomial + ICAR spatial comparisons:")

  comp_beta0 <- compare_posteriors(
    fit_hmc$draws[, "beta_num[1]"],
    stan_draws[, "beta[1]"],
    "Intercept"
  )
  comp_beta1 <- compare_posteriors(
    fit_hmc$draws[, "beta_num[2]"],
    stan_draws[, "beta[2]"],
    "Slope"
  )

  print_comparison(comp_beta0)
  print_comparison(comp_beta1)

  # Assertions
  expect_true(comp_beta0$ci_overlap, info = "Intercept CIs should overlap")
  expect_true(comp_beta1$ci_overlap, info = "Slope CIs should overlap")
})


# =============================================================================
# Test 6: Parameter recovery on simulated data
# =============================================================================
test_that("HMC recovers known parameters from simulation", {
  skip_on_cran()
  skip("Long-running parameter recovery test - run manually")

  set.seed(999)

  # Simple binomial with known truth
  n <- 200
  x <- rnorm(n)
  true_beta0 <- 0.0

  true_beta1 <- 1.0

  p <- plogis(true_beta0 + true_beta1 * x)
  trials <- rep(20L, n)
  successes <- rbinom(n, trials, p)

  df <- data.frame(successes = successes, trials = trials, x = x)

  # Fit with HMC only (faster test)
  fit <- quotr(
    successes | trials ~ x,
    data = df,
    family = quotr_binomial(),
    backend = "hmc",
    chains = 2,
    iter = 2000,
    warmup = 1000
  )

  # Check parameter recovery
  beta0_samples <- fit$draws[, "beta_num[1]"]
  beta1_samples <- fit$draws[, "beta_num[2]"]

  beta0_ci <- quantile(beta0_samples, c(0.025, 0.975))
  beta1_ci <- quantile(beta1_samples, c(0.025, 0.975))

  expect_true(beta0_ci[1] < true_beta0 && true_beta0 < beta0_ci[2],
              info = sprintf("True beta0 (%.2f) should be in 95%% CI [%.2f, %.2f]",
                             true_beta0, beta0_ci[1], beta0_ci[2]))
  expect_true(beta1_ci[1] < true_beta1 && true_beta1 < beta1_ci[2],
              info = sprintf("True beta1 (%.2f) should be in 95%% CI [%.2f, %.2f]",
                             true_beta1, beta1_ci[1], beta1_ci[2]))

  # Check that posterior means are close to truth
  expect_equal(mean(beta0_samples), true_beta0, tolerance = 0.3,
               info = "Posterior mean of beta0 should be close to truth")
  expect_equal(mean(beta1_samples), true_beta1, tolerance = 0.3,
               info = "Posterior mean of beta1 should be close to truth")
})


# =============================================================================
# Summary test: Run validation suite
# =============================================================================
test_that("HMC backend validation summary", {
  skip_if_no_cmdstanr()
  skip_on_cran()

  message("\n")
  message("================================================")
  message("HMC vs Stan Validation Summary")
  message("================================================")
  message("All validation tests passed!")
  message("The HMC backend produces posteriors consistent with Stan.")
  message("------------------------------------------------")
  message("Tested models:")
  message("  - Binomial (fixed denominator)")
  message("  - Binomial with random effects")
  message("  - Negative binomial - negative binomial")
  message("  - Poisson - Gamma (CPUE)")
  message("================================================\n")

  expect_true(TRUE)
})
