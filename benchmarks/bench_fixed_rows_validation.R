# Validation for FIXED rows (39, 52, 53) - MSGP and HSGP+RW1
# These were marked as FIXED in gradient_methods.md but need validation

library(numdenom)
set.seed(20260208)

# Standard parameters - smaller N for GP models
N_OBS <- 100  # Smaller for O(N³) models
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1
N_SITES <- 20  # Smaller for spatial
N_TIMES <- 10  # Smaller for temporal

message(paste(rep("=", 60), collapse=""))
message("FIXED ROWS VALIDATION")
message("Row 39: negbin_negbin + RE + MSGP")
message("Row 52: negbin_negbin + RE + HSGP + RW1")
message("Row 53: negbin_negbin + RE + MSGP + RW1")
message(paste(rep("=", 60), collapse=""))

# True parameters
true_params <- list(
  beta_num = c(1.5, 0.3),
  beta_denom = c(2.0, -0.2),
  sigma_re = 0.3,
  size_num = 5,
  size_denom = 5,
  sigma_spatial = 0.4,
  tau_temporal = 2.0
)

# Helper to validate parameter recovery
validate_param <- function(draws, param_name, true_value, threshold_sd = 2) {
  post_mean <- mean(draws)
  post_sd <- sd(draws)
  ci_lower <- quantile(draws, 0.025)
  ci_upper <- quantile(draws, 0.975)
  diff_sd <- abs(post_mean - true_value) / post_sd

  pass <- diff_sd < threshold_sd && true_value >= ci_lower && true_value <= ci_upper

  list(
    param = param_name,
    true = true_value,
    post_mean = post_mean,
    post_sd = post_sd,
    diff_sd = diff_sd,
    ci_covers = true_value >= ci_lower && true_value <= ci_upper,
    pass = pass
  )
}

# Generate base data with coordinates at observation level
generate_data <- function(has_spatial = FALSE, has_temporal = FALSE) {
  # Design matrix with coordinates at observation level
  df <- data.frame(
    site = rep(1:N_SITES, each = if(has_temporal) N_TIMES else N_OBS/N_SITES),
    x = rnorm(N_OBS),
    x_coord = runif(N_OBS),  # Observation-level coordinates
    y_coord = runif(N_OBS)
  )
  if (has_temporal) {
    df$time <- rep(1:N_TIMES, N_SITES)
  }

  # Random effects
  re <- rnorm(N_SITES, 0, true_params$sigma_re)

  # Spatial effects - approximate using distance to origin (simplified)
  if (has_spatial) {
    dist_to_origin <- sqrt(df$x_coord^2 + df$y_coord^2)
    spatial <- rnorm(N_OBS, 0, true_params$sigma_spatial) *
               exp(-dist_to_origin)  # Simple spatial correlation
  } else {
    spatial <- rep(0, N_OBS)
  }

  # Temporal effects (RW1-like)
  if (has_temporal) {
    temporal <- cumsum(rnorm(N_TIMES, 0, 1/sqrt(true_params$tau_temporal)))
    temporal <- temporal - mean(temporal)  # Center
  } else {
    temporal <- rep(0, max(1, N_TIMES))
  }

  # Linear predictors
  eta_num <- true_params$beta_num[1] + true_params$beta_num[2] * df$x +
             re[df$site] + spatial
  eta_denom <- true_params$beta_denom[1] + true_params$beta_denom[2] * df$x +
               re[df$site] + spatial

  if (has_temporal) {
    eta_num <- eta_num + temporal[df$time]
    eta_denom <- eta_denom + temporal[df$time]
  }

  # Generate responses (NegBin)
  mu_num <- exp(eta_num)
  mu_denom <- exp(eta_denom)

  df$y_num <- rnbinom(N_OBS, size = true_params$size_num, mu = mu_num)
  df$y_denom <- rnbinom(N_OBS, size = true_params$size_denom, mu = mu_denom)

  # Ensure non-zero denominator
  df$y_denom <- pmax(df$y_denom, 1)

  df
}

# ===== Row 39: negbin_negbin + RE + MSGP =====
message("\n>>> Row 39: negbin_negbin + RE + MSGP <<<")

tryCatch({
  data39 <- generate_data(has_spatial = TRUE, has_temporal = FALSE)

  time39 <- system.time({
    fit39 <- ratiod(
      y_num | y_denom ~ x + (1 | site),
      data = data39,
      family = ratiod_negbin_negbin(),
      spatial = spatial_multiscale(coords = ~ x_coord + y_coord),
      iter = N_ITER,
      warmup = N_WARMUP,
      chains = N_CHAINS,
      gradient_mode = "H"
    )
  })["elapsed"]

  message(sprintf("Time: %.1fs", time39))

  # Extract draws and validate
  draws <- posterior::as_draws_df(fit39$draws)

  results39 <- list(
    validate_param(draws$`beta_num[1]`, "beta_num[1]", true_params$beta_num[1]),
    validate_param(draws$`beta_num[2]`, "beta_num[2]", true_params$beta_num[2]),
    validate_param(draws$`beta_denom[1]`, "beta_denom[1]", true_params$beta_denom[1]),
    validate_param(draws$`beta_denom[2]`, "beta_denom[2]", true_params$beta_denom[2])
  )

  all_pass39 <- all(sapply(results39, function(r) r$pass))
  message(sprintf("Row 39 validation: %s", if(all_pass39) "PASS" else "FAIL"))
  for (r in results39) {
    message(sprintf("  %s: true=%.3f, post=%.3f (%.2f SD)",
                    r$param, r$true, r$post_mean, r$diff_sd))
  }
}, error = function(e) {
  message(sprintf("Row 39 ERROR: %s", e$message))
})

# ===== Row 52: negbin_negbin + RE + HSGP + RW1 =====
message("\n>>> Row 52: negbin_negbin + RE + HSGP + RW1 <<<")

tryCatch({
  data52 <- generate_data(has_spatial = TRUE, has_temporal = TRUE)

  time52 <- system.time({
    fit52 <- ratiod(
      y_num | y_denom ~ x + (1 | site),
      data = data52,
      family = ratiod_negbin_negbin(),
      spatial = spatial_hsgp(coords = ~ x_coord + y_coord),
      temporal = temporal_rw1(time_var = "time"),
      iter = N_ITER,
      warmup = N_WARMUP,
      chains = N_CHAINS,
      gradient_mode = "H"
    )
  })["elapsed"]

  message(sprintf("Time: %.1fs", time52))

  # Extract draws and validate
  draws <- posterior::as_draws_df(fit52$draws)

  results52 <- list(
    validate_param(draws$`beta_num[1]`, "beta_num[1]", true_params$beta_num[1]),
    validate_param(draws$`beta_num[2]`, "beta_num[2]", true_params$beta_num[2]),
    validate_param(draws$`beta_denom[1]`, "beta_denom[1]", true_params$beta_denom[1]),
    validate_param(draws$`beta_denom[2]`, "beta_denom[2]", true_params$beta_denom[2])
  )

  all_pass52 <- all(sapply(results52, function(r) r$pass))
  message(sprintf("Row 52 validation: %s", if(all_pass52) "PASS" else "FAIL"))
  for (r in results52) {
    message(sprintf("  %s: true=%.3f, post=%.3f (%.2f SD)",
                    r$param, r$true, r$post_mean, r$diff_sd))
  }
}, error = function(e) {
  message(sprintf("Row 52 ERROR: %s", e$message))
})

# ===== Row 53: negbin_negbin + RE + MSGP + RW1 =====
message("\n>>> Row 53: negbin_negbin + RE + MSGP + RW1 <<<")

tryCatch({
  data53 <- generate_data(has_spatial = TRUE, has_temporal = TRUE)

  time53 <- system.time({
    fit53 <- ratiod(
      y_num | y_denom ~ x + (1 | site),
      data = data53,
      family = ratiod_negbin_negbin(),
      spatial = spatial_multiscale(coords = ~ x_coord + y_coord),
      temporal = temporal_rw1(time_var = "time"),
      iter = N_ITER,
      warmup = N_WARMUP,
      chains = N_CHAINS,
      gradient_mode = "H"
    )
  })["elapsed"]

  message(sprintf("Time: %.1fs", time53))

  # Extract draws and validate
  draws <- posterior::as_draws_df(fit53$draws)

  results53 <- list(
    validate_param(draws$`beta_num[1]`, "beta_num[1]", true_params$beta_num[1]),
    validate_param(draws$`beta_num[2]`, "beta_num[2]", true_params$beta_num[2]),
    validate_param(draws$`beta_denom[1]`, "beta_denom[1]", true_params$beta_denom[1]),
    validate_param(draws$`beta_denom[2]`, "beta_denom[2]", true_params$beta_denom[2])
  )

  all_pass53 <- all(sapply(results53, function(r) r$pass))
  message(sprintf("Row 53 validation: %s", if(all_pass53) "PASS" else "FAIL"))
  for (r in results53) {
    message(sprintf("  %s: true=%.3f, post=%.3f (%.2f SD)",
                    r$param, r$true, r$post_mean, r$diff_sd))
  }
}, error = function(e) {
  message(sprintf("Row 53 ERROR: %s", e$message))
})

message(paste(rep("=", 60), collapse=""))
message("FIXED ROWS VALIDATION COMPLETE")
message(paste(rep("=", 60), collapse=""))
