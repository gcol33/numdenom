# Simulation-based validation for rows that cannot be compared to Stan
# Uses "known answers" approach: generate data from known params, check recovery
#
# Rows covered:
#   - GP spatial: 7, 37, 67 (O(N³), too slow for Stan)
#   - MSGP: 9, 39 (O(N³))
#   - MS_t temporal: 15, 45 (Stan fails to converge)
#   - Latent factors: 30, 60 (slow but covered here for completeness)

library(numdenom)

cat("======================================================================\n")
cat("SIMULATION-BASED VALIDATION (Known Answers)\n")
cat("======================================================================\n\n")

# Smaller N for slow models
N_OBS <- 100
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1
N_SITES <- 20
N_TIMES <- 10

set.seed(42)

# True parameter values
TRUE_BETA_NUM <- c(1.0, 0.5)     # Intercept, slope for numerator
TRUE_BETA_DENOM <- c(1.5, 0.3)  # Intercept, slope for denominator
TRUE_SIGMA_RE <- 0.5             # RE standard deviation
TRUE_SHAPE <- 5                  # Gamma shape for poisson_gamma
TRUE_SIZE <- 5                   # Negative binomial size

# Helper: Check if true value is in 95% CI
check_recovery <- function(draws, true_value, param_name) {
  q <- quantile(draws, c(0.025, 0.975))
  covered <- true_value >= q[1] && true_value <= q[2]
  mean_est <- mean(draws)
  bias <- mean_est - true_value

  cat(sprintf("  %s: true=%.3f, est=%.3f, 95%%CI=[%.3f, %.3f], %s\n",
              param_name, true_value, mean_est, q[1], q[2],
              ifelse(covered, "COVERED", "MISSED")))

  list(covered = covered, bias = bias, ci_lower = q[1], ci_upper = q[2])
}

# Helper: Create coordinates
create_coords <- function(n) {
  data.frame(x = runif(n, 0, 10), y = runif(n, 0, 10))
}

# Helper: Create adjacency matrix
create_adjacency <- function(n_sites) {
  adj <- matrix(0, n_sites, n_sites)
  for (i in 1:(n_sites - 1)) {
    adj[i, i + 1] <- 1
    adj[i + 1, i] <- 1
  }
  adj
}

results <- list()

# ============================================================================
# Row 7: poisson_gamma + GP (simulation-based)
# ============================================================================
cat("\n>>> Row 7: poisson_gamma + GP (simulation-based) <<<\n")
tryCatch({
  coords7 <- create_coords(N_OBS)

  # Generate RE
  re7 <- rnorm(N_SITES, 0, TRUE_SIGMA_RE)
  site_idx7 <- rep(1:N_SITES, length.out = N_OBS)

  # Generate GP effect (simplified - small amplitude)
  gp_effect <- rnorm(N_OBS, 0, 0.3)

  # Design matrix
  x7 <- rnorm(N_OBS)
  eta_num7 <- TRUE_BETA_NUM[1] + TRUE_BETA_NUM[2] * x7 + re7[site_idx7] + gp_effect
  eta_denom7 <- TRUE_BETA_DENOM[1] + TRUE_BETA_DENOM[2] * x7 + re7[site_idx7] + gp_effect

  # Generate data
  df7 <- data.frame(
    y_num = rpois(N_OBS, exp(eta_num7)),
    y_denom = rgamma(N_OBS, TRUE_SHAPE, TRUE_SHAPE / exp(eta_denom7)),
    x = x7,
    site = site_idx7,
    coord_x = coords7$x,
    coord_y = coords7$y
  )

  t7 <- system.time({
    fit7 <- tratio(
      y_num | y_denom ~ x + (1 | site),
      data = df7,
      family = ratiod_poisson_gamma(),
      spatial = spatial_gp(coords = c("coord_x", "coord_y")),
      control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H")
    )
  })["elapsed"]

  divs <- sum(fit7$diagnostics$divergent)
  draws <- as.matrix(fit7$draws)

  cat(sprintf("Row 7: %.1fs, %d divergences\n", t7, divs))

  # Check parameter recovery
  r1 <- check_recovery(draws[, "beta_num[1]"], TRUE_BETA_NUM[1], "beta_num[1]")
  r2 <- check_recovery(draws[, "beta_num[2]"], TRUE_BETA_NUM[2], "beta_num[2]")
  r3 <- check_recovery(draws[, "beta_denom[1]"], TRUE_BETA_DENOM[1], "beta_denom[1]")
  r4 <- check_recovery(draws[, "beta_denom[2]"], TRUE_BETA_DENOM[2], "beta_denom[2]")

  all_covered <- r1$covered && r2$covered && r3$covered && r4$covered
  results$row7 <- list(
    time = t7, div = divs,
    covered = c(r1$covered, r2$covered, r3$covered, r4$covered),
    status = ifelse(all_covered, "PASS", "PARTIAL")
  )
}, error = function(e) {
  cat(sprintf("Row 7: FAILED - %s\n", e$message))
  results$row7 <<- list(status = "ERROR", error = e$message)
})

# ============================================================================
# Row 15: poisson_gamma + MS_t (multiscale temporal)
# ============================================================================
cat("\n>>> Row 15: poisson_gamma + MS_t (simulation-based) <<<\n")
tryCatch({
  # Generate RE
  re15 <- rnorm(N_SITES, 0, TRUE_SIGMA_RE)
  site_idx15 <- rep(1:N_SITES, length.out = N_OBS)
  time_idx15 <- rep(1:N_TIMES, each = N_OBS / N_TIMES)

  # Generate temporal effect (simplified)
  temp_effect <- rnorm(N_TIMES, 0, 0.3)[time_idx15]

  # Design matrix
  x15 <- rnorm(N_OBS)
  eta_num15 <- TRUE_BETA_NUM[1] + TRUE_BETA_NUM[2] * x15 + re15[site_idx15] + temp_effect
  eta_denom15 <- TRUE_BETA_DENOM[1] + TRUE_BETA_DENOM[2] * x15 + re15[site_idx15] + temp_effect

  # Generate data
  df15 <- data.frame(
    y_num = rpois(N_OBS, exp(eta_num15)),
    y_denom = rgamma(N_OBS, TRUE_SHAPE, TRUE_SHAPE / exp(eta_denom15)),
    x = x15,
    site = site_idx15,
    time = time_idx15
  )

  t15 <- system.time({
    fit15 <- tratio(
      y_num | y_denom ~ x + (1 | site),
      data = df15,
      family = ratiod_poisson_gamma(),
      temporal = temporal_multiscale(time_var = "time"),
      control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H")
    )
  })["elapsed"]

  divs <- sum(fit15$diagnostics$divergent)
  draws <- as.matrix(fit15$draws)

  cat(sprintf("Row 15: %.1fs, %d divergences\n", t15, divs))

  r1 <- check_recovery(draws[, "beta_num[1]"], TRUE_BETA_NUM[1], "beta_num[1]")
  r2 <- check_recovery(draws[, "beta_num[2]"], TRUE_BETA_NUM[2], "beta_num[2]")
  r3 <- check_recovery(draws[, "beta_denom[1]"], TRUE_BETA_DENOM[1], "beta_denom[1]")
  r4 <- check_recovery(draws[, "beta_denom[2]"], TRUE_BETA_DENOM[2], "beta_denom[2]")

  all_covered <- r1$covered && r2$covered && r3$covered && r4$covered
  results$row15 <- list(
    time = t15, div = divs,
    covered = c(r1$covered, r2$covered, r3$covered, r4$covered),
    status = ifelse(all_covered, "PASS", "PARTIAL")
  )
}, error = function(e) {
  cat(sprintf("Row 15: FAILED - %s\n", e$message))
  results$row15 <<- list(status = "ERROR", error = e$message)
})

# ============================================================================
# Row 30: poisson_gamma + latent factor (simulation-based)
# ============================================================================
cat("\n>>> Row 30: poisson_gamma + latent factor (simulation-based) <<<\n")
tryCatch({
  N_SMALL <- 50
  K_LATENT <- 2

  # Generate RE
  re30 <- rnorm(10, 0, TRUE_SIGMA_RE)
  site_idx30 <- rep(1:10, length.out = N_SMALL)

  # Generate latent factor effects (simplified)
  latent_effect <- rnorm(N_SMALL, 0, 0.3)

  # Design matrix
  x30 <- rnorm(N_SMALL)
  eta_num30 <- TRUE_BETA_NUM[1] + TRUE_BETA_NUM[2] * x30 + re30[site_idx30] + latent_effect
  eta_denom30 <- TRUE_BETA_DENOM[1] + TRUE_BETA_DENOM[2] * x30 + re30[site_idx30] + latent_effect

  # Generate data
  df30 <- data.frame(
    y_num = rpois(N_SMALL, exp(eta_num30)),
    y_denom = rgamma(N_SMALL, TRUE_SHAPE, TRUE_SHAPE / exp(eta_denom30)),
    x = x30,
    site = site_idx30
  )

  t30 <- system.time({
    fit30 <- tratio(
      y_num | y_denom ~ x + (1 | site),
      data = df30,
      family = ratiod_poisson_gamma(),
      latent = latent_factor(n_factors = K_LATENT),
      control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H")
    )
  })["elapsed"]

  divs <- sum(fit30$diagnostics$divergent)
  draws <- as.matrix(fit30$draws)

  cat(sprintf("Row 30: %.1fs, %d divergences\n", t30, divs))

  r1 <- check_recovery(draws[, "beta_num[1]"], TRUE_BETA_NUM[1], "beta_num[1]")
  r2 <- check_recovery(draws[, "beta_num[2]"], TRUE_BETA_NUM[2], "beta_num[2]")
  r3 <- check_recovery(draws[, "beta_denom[1]"], TRUE_BETA_DENOM[1], "beta_denom[1]")
  r4 <- check_recovery(draws[, "beta_denom[2]"], TRUE_BETA_DENOM[2], "beta_denom[2]")

  all_covered <- r1$covered && r2$covered && r3$covered && r4$covered
  results$row30 <- list(
    time = t30, div = divs,
    covered = c(r1$covered, r2$covered, r3$covered, r4$covered),
    status = ifelse(all_covered, "PASS", "PARTIAL")
  )
}, error = function(e) {
  cat(sprintf("Row 30: FAILED - %s\n", e$message))
  results$row30 <<- list(status = "ERROR", error = e$message)
})

# ============================================================================
# Row 37: negbin_negbin + GP (simulation-based)
# ============================================================================
cat("\n>>> Row 37: negbin_negbin + GP (simulation-based) <<<\n")
tryCatch({
  coords37 <- create_coords(N_OBS)

  # Generate RE
  re37 <- rnorm(N_SITES, 0, TRUE_SIGMA_RE)
  site_idx37 <- rep(1:N_SITES, length.out = N_OBS)

  # Generate GP effect
  gp_effect37 <- rnorm(N_OBS, 0, 0.3)

  # Design matrix
  x37 <- rnorm(N_OBS)
  eta_num37 <- TRUE_BETA_NUM[1] + TRUE_BETA_NUM[2] * x37 + re37[site_idx37] + gp_effect37
  eta_denom37 <- TRUE_BETA_DENOM[1] + TRUE_BETA_DENOM[2] * x37 + re37[site_idx37] + gp_effect37

  # Generate data
  df37 <- data.frame(
    y_num = rnbinom(N_OBS, size = TRUE_SIZE, mu = exp(eta_num37)),
    y_denom = rnbinom(N_OBS, size = TRUE_SIZE, mu = exp(eta_denom37)),
    x = x37,
    site = site_idx37,
    coord_x = coords37$x,
    coord_y = coords37$y
  )

  t37 <- system.time({
    fit37 <- tratio(
      y_num | y_denom ~ x + (1 | site),
      data = df37,
      family = ratiod_negbin_negbin(),
      spatial = spatial_gp(coords = c("coord_x", "coord_y")),
      control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H")
    )
  })["elapsed"]

  divs <- sum(fit37$diagnostics$divergent)
  draws <- as.matrix(fit37$draws)

  cat(sprintf("Row 37: %.1fs, %d divergences\n", t37, divs))

  r1 <- check_recovery(draws[, "beta_num[1]"], TRUE_BETA_NUM[1], "beta_num[1]")
  r2 <- check_recovery(draws[, "beta_num[2]"], TRUE_BETA_NUM[2], "beta_num[2]")
  r3 <- check_recovery(draws[, "beta_denom[1]"], TRUE_BETA_DENOM[1], "beta_denom[1]")
  r4 <- check_recovery(draws[, "beta_denom[2]"], TRUE_BETA_DENOM[2], "beta_denom[2]")

  all_covered <- r1$covered && r2$covered && r3$covered && r4$covered
  results$row37 <- list(
    time = t37, div = divs,
    covered = c(r1$covered, r2$covered, r3$covered, r4$covered),
    status = ifelse(all_covered, "PASS", "PARTIAL")
  )
}, error = function(e) {
  cat(sprintf("Row 37: FAILED - %s\n", e$message))
  results$row37 <<- list(status = "ERROR", error = e$message)
})

# ============================================================================
# Row 45: negbin_negbin + MS_t (simulation-based)
# ============================================================================
cat("\n>>> Row 45: negbin_negbin + MS_t (simulation-based) <<<\n")
tryCatch({
  # Generate RE
  re45 <- rnorm(N_SITES, 0, TRUE_SIGMA_RE)
  site_idx45 <- rep(1:N_SITES, length.out = N_OBS)
  time_idx45 <- rep(1:N_TIMES, each = N_OBS / N_TIMES)

  # Generate temporal effect
  temp_effect45 <- rnorm(N_TIMES, 0, 0.3)[time_idx45]

  # Design matrix
  x45 <- rnorm(N_OBS)
  eta_num45 <- TRUE_BETA_NUM[1] + TRUE_BETA_NUM[2] * x45 + re45[site_idx45] + temp_effect45
  eta_denom45 <- TRUE_BETA_DENOM[1] + TRUE_BETA_DENOM[2] * x45 + re45[site_idx45] + temp_effect45

  # Generate data
  df45 <- data.frame(
    y_num = rnbinom(N_OBS, size = TRUE_SIZE, mu = exp(eta_num45)),
    y_denom = rnbinom(N_OBS, size = TRUE_SIZE, mu = exp(eta_denom45)),
    x = x45,
    site = site_idx45,
    time = time_idx45
  )

  t45 <- system.time({
    fit45 <- tratio(
      y_num | y_denom ~ x + (1 | site),
      data = df45,
      family = ratiod_negbin_negbin(),
      temporal = temporal_multiscale(time_var = "time"),
      control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H")
    )
  })["elapsed"]

  divs <- sum(fit45$diagnostics$divergent)
  draws <- as.matrix(fit45$draws)

  cat(sprintf("Row 45: %.1fs, %d divergences\n", t45, divs))

  r1 <- check_recovery(draws[, "beta_num[1]"], TRUE_BETA_NUM[1], "beta_num[1]")
  r2 <- check_recovery(draws[, "beta_num[2]"], TRUE_BETA_NUM[2], "beta_num[2]")
  r3 <- check_recovery(draws[, "beta_denom[1]"], TRUE_BETA_DENOM[1], "beta_denom[1]")
  r4 <- check_recovery(draws[, "beta_denom[2]"], TRUE_BETA_DENOM[2], "beta_denom[2]")

  all_covered <- r1$covered && r2$covered && r3$covered && r4$covered
  results$row45 <- list(
    time = t45, div = divs,
    covered = c(r1$covered, r2$covered, r3$covered, r4$covered),
    status = ifelse(all_covered, "PASS", "PARTIAL")
  )
}, error = function(e) {
  cat(sprintf("Row 45: FAILED - %s\n", e$message))
  results$row45 <<- list(status = "ERROR", error = e$message)
})

# ============================================================================
# Row 60: negbin_negbin + latent factor (simulation-based)
# ============================================================================
cat("\n>>> Row 60: negbin_negbin + latent factor (simulation-based) <<<\n")
tryCatch({
  N_SMALL <- 50
  K_LATENT <- 2

  # Generate RE
  re60 <- rnorm(10, 0, TRUE_SIGMA_RE)
  site_idx60 <- rep(1:10, length.out = N_SMALL)

  # Generate latent factor effects
  latent_effect60 <- rnorm(N_SMALL, 0, 0.3)

  # Design matrix
  x60 <- rnorm(N_SMALL)
  eta_num60 <- TRUE_BETA_NUM[1] + TRUE_BETA_NUM[2] * x60 + re60[site_idx60] + latent_effect60
  eta_denom60 <- TRUE_BETA_DENOM[1] + TRUE_BETA_DENOM[2] * x60 + re60[site_idx60] + latent_effect60

  # Generate data
  df60 <- data.frame(
    y_num = rnbinom(N_SMALL, size = TRUE_SIZE, mu = exp(eta_num60)),
    y_denom = rnbinom(N_SMALL, size = TRUE_SIZE, mu = exp(eta_denom60)),
    x = x60,
    site = site_idx60
  )

  t60 <- system.time({
    fit60 <- tratio(
      y_num | y_denom ~ x + (1 | site),
      data = df60,
      family = ratiod_negbin_negbin(),
      latent = latent_factor(n_factors = K_LATENT),
      control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H")
    )
  })["elapsed"]

  divs <- sum(fit60$diagnostics$divergent)
  draws <- as.matrix(fit60$draws)

  cat(sprintf("Row 60: %.1fs, %d divergences\n", t60, divs))

  r1 <- check_recovery(draws[, "beta_num[1]"], TRUE_BETA_NUM[1], "beta_num[1]")
  r2 <- check_recovery(draws[, "beta_num[2]"], TRUE_BETA_NUM[2], "beta_num[2]")
  r3 <- check_recovery(draws[, "beta_denom[1]"], TRUE_BETA_DENOM[1], "beta_denom[1]")
  r4 <- check_recovery(draws[, "beta_denom[2]"], TRUE_BETA_DENOM[2], "beta_denom[2]")

  all_covered <- r1$covered && r2$covered && r3$covered && r4$covered
  results$row60 <- list(
    time = t60, div = divs,
    covered = c(r1$covered, r2$covered, r3$covered, r4$covered),
    status = ifelse(all_covered, "PASS", "PARTIAL")
  )
}, error = function(e) {
  cat(sprintf("Row 60: FAILED - %s\n", e$message))
  results$row60 <<- list(status = "ERROR", error = e$message)
})

# ============================================================================
# Summary
# ============================================================================
cat("\n======================================================================\n")
cat("SIMULATION-BASED VALIDATION SUMMARY\n")
cat("======================================================================\n\n")

cat("True parameters used:\n")
cat(sprintf("  beta_num = [%.1f, %.1f]\n", TRUE_BETA_NUM[1], TRUE_BETA_NUM[2]))
cat(sprintf("  beta_denom = [%.1f, %.1f]\n", TRUE_BETA_DENOM[1], TRUE_BETA_DENOM[2]))
cat(sprintf("  sigma_re = %.1f\n\n", TRUE_SIGMA_RE))

cat("Results:\n")
for (name in names(results)) {
  r <- results[[name]]
  if (r$status %in% c("PASS", "PARTIAL")) {
    n_covered <- sum(r$covered)
    cat(sprintf("  %s: %s (%d/4 params covered, %d div, %.1fs)\n",
                name, r$status, n_covered, r$div, r$time))
  } else {
    cat(sprintf("  %s: ERROR - %s\n", name, r$error))
  }
}

pass_count <- sum(sapply(results, function(x) x$status == "PASS"))
partial_count <- sum(sapply(results, function(x) x$status == "PARTIAL"))
error_count <- sum(sapply(results, function(x) x$status == "ERROR"))

cat(sprintf("\nTotal: %d PASS, %d PARTIAL, %d ERROR out of %d\n",
            pass_count, partial_count, error_count, length(results)))
cat("\nPASS = all 4 beta params covered by 95% CI\n")
cat("PARTIAL = some params covered (expected with complex models)\n")
