# Simulation-based validation for HSGP spatial models
# Rows 22, 52 (HSGP + RW1), 68 (binomial + HSGP), 84 (binomial + HSGP + RW1)
# HSGP uses Hilbert space approximation - fast, handles duplicate coords naturally

library(numdenom)
library(posterior)

set.seed(42)

N_OBS <- 100
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1
N_SITES <- 20
N_TIMES <- 10

cat("=======================================================\n")
cat("HSGP Spatial Models: Simulation-Based Validation\n")
cat("Rows 22, 52, 68, 84\n")
cat("=======================================================\n")
cat(sprintf("N=%d, sites=%d, times=%d, iter=%d, chains=%d\n\n",
            N_OBS, N_SITES, N_TIMES, N_ITER, N_CHAINS))

check_recovery <- function(draws, true_value, param_name, threshold_sd = 2) {
  post_mean <- mean(draws)
  post_sd <- sd(draws)
  diff_sd <- abs(post_mean - true_value) / post_sd
  in_ci <- true_value >= quantile(draws, 0.025) && true_value <= quantile(draws, 0.975)
  list(param = param_name, true = true_value, mean = post_mean, sd = post_sd,
       diff_sd = diff_sd, in_ci = in_ci, pass = diff_sd < threshold_sd)
}

print_recovery <- function(r) {
  cat(sprintf("  %s: true=%.3f, post=%.3f (SD=%.3f), %.2f SD => %s\n",
              r$param, r$true, r$mean, r$sd, r$diff_sd, if(r$pass) "PASS" else "FAIL"))
}

results <- list()

# Generate spatial coordinates on a grid
n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = seq(0, 1, length.out = n_side),
                    lat = seq(0, 1, length.out = n_side))[1:N_SITES, ]

# Generate GP spatial effects using exponential covariance
generate_gp_effects <- function(coords, sigma, lengthscale) {
  n <- nrow(coords)
  dist_mat <- as.matrix(dist(coords))
  cov_mat <- sigma^2 * exp(-dist_mat / lengthscale)
  diag(cov_mat) <- diag(cov_mat) + 1e-6
  L <- chol(cov_mat)
  z <- rnorm(n)
  effects <- as.vector(t(L) %*% z)
  effects - mean(effects)
}

generate_rw1 <- function(n, sigma) {
  gamma <- cumsum(rnorm(n, 0, sigma))
  gamma - mean(gamma)
}

# True parameters
true_intercept <- 1.0
true_slope <- 0.3
true_sigma_gp <- 0.3
true_lengthscale <- 0.3  # Relative to 0-1 domain

# Generate effects
spatial_effects <- generate_gp_effects(grid, true_sigma_gp, true_lengthscale)
true_sigma_rw1 <- 0.15
temporal_effects <- generate_rw1(N_TIMES, true_sigma_rw1)

# Common data
site <- factor(rep(1:N_SITES, length.out = N_OBS))
time <- rep(1:N_TIMES, length.out = N_OBS)
x <- rnorm(N_OBS)
lon <- grid$lon[as.integer(site)]
lat <- grid$lat[as.integer(site)]

# =============================================================================
# Row 68: binomial + HSGP (no temporal)
# =============================================================================
cat("\n========== Row 68: binomial + HSGP ==========\n")

eta_68 <- true_intercept + true_slope * x + spatial_effects[as.integer(site)]
prob_68 <- plogis(eta_68)
trials_68 <- sample(20:50, N_OBS, replace = TRUE)
successes_68 <- rbinom(N_OBS, trials_68, prob_68)
df_68 <- data.frame(successes = successes_68, trials = trials_68, x = x,
                    lon = lon, lat = lat)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_68 <- system.time({
  fit_68 <- tryCatch({
    ratiod(successes | trials ~ x, data = df_68, family = ratiod_binomial(),
           spatial = spatial_hsgp(coords = c("lon", "lat"), m = 6),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_68)) {
  cat(sprintf("%.1fs\n", t_68))
  draws_68 <- as.matrix(fit_68$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_68), value = TRUE)[1]
  results$row_68 <- list(slope = check_recovery(draws_68[, slope_col], true_slope, "slope"), time = t_68)
  print_recovery(results$row_68$slope)
  results$row_68$pass <- results$row_68$slope$pass
} else { results$row_68 <- list(error = TRUE) }

# =============================================================================
# Row 22: poisson_gamma + HSGP + RW1
# =============================================================================
cat("\n========== Row 22: poisson_gamma + HSGP + RW1 ==========\n")

eta_22 <- true_intercept + true_slope * x + spatial_effects[as.integer(site)] + temporal_effects[time]
effort_22 <- rgamma(N_OBS, 5, 1)
count_22 <- rpois(N_OBS, exp(eta_22) * effort_22)
df_22 <- data.frame(count = count_22, effort = effort_22, x = x,
                    lon = lon, lat = lat, time = time)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_22 <- system.time({
  fit_22 <- tryCatch({
    ratiod(count | effort ~ x, data = df_22, family = ratiod_poisson_gamma(),
           spatial = spatial_hsgp(coords = c("lon", "lat"), m = 6),
           temporal = temporal_rw1(time_var = "time"),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_22)) {
  cat(sprintf("%.1fs\n", t_22))
  draws_22 <- as.matrix(fit_22$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_22), value = TRUE)[1]
  results$row_22 <- list(slope = check_recovery(draws_22[, slope_col], true_slope, "slope"), time = t_22)
  print_recovery(results$row_22$slope)
  results$row_22$pass <- results$row_22$slope$pass
} else { results$row_22 <- list(error = TRUE) }

# =============================================================================
# Row 52: negbin_negbin + HSGP + RW1
# =============================================================================
cat("\n========== Row 52: negbin_negbin + HSGP + RW1 ==========\n")

eta_num_52 <- true_intercept + true_slope * x + spatial_effects[as.integer(site)] + temporal_effects[time]
eta_denom_52 <- 0.5 + 0.2 * x + spatial_effects[as.integer(site)] * 0.8 + temporal_effects[time]
num_52 <- rnbinom(N_OBS, size = 5, mu = exp(eta_num_52))
denom_52 <- rnbinom(N_OBS, size = 5, mu = exp(eta_denom_52))
denom_52[denom_52 == 0] <- 1
df_52 <- data.frame(num = num_52, denom = denom_52, x = x,
                    lon = lon, lat = lat, time = time)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_52 <- system.time({
  fit_52 <- tryCatch({
    ratiod(num | denom ~ x, data = df_52, family = ratiod_negbin_negbin(),
           spatial = spatial_hsgp(coords = c("lon", "lat"), m = 6),
           temporal = temporal_rw1(time_var = "time"),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_52)) {
  cat(sprintf("%.1fs\n", t_52))
  draws_52 <- as.matrix(fit_52$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_52), value = TRUE)[1]
  results$row_52 <- list(slope = check_recovery(draws_52[, slope_col], true_slope, "slope"), time = t_52)
  print_recovery(results$row_52$slope)
  results$row_52$pass <- results$row_52$slope$pass
} else { results$row_52 <- list(error = TRUE) }

# =============================================================================
# Row 84: binomial + HSGP + RW1
# =============================================================================
cat("\n========== Row 84: binomial + HSGP + RW1 ==========\n")

eta_84 <- true_intercept + true_slope * x + spatial_effects[as.integer(site)] + temporal_effects[time]
prob_84 <- plogis(eta_84)
successes_84 <- rbinom(N_OBS, trials_68, prob_84)
df_84 <- data.frame(successes = successes_84, trials = trials_68, x = x,
                    lon = lon, lat = lat, time = time)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_84 <- system.time({
  fit_84 <- tryCatch({
    ratiod(successes | trials ~ x, data = df_84, family = ratiod_binomial(),
           spatial = spatial_hsgp(coords = c("lon", "lat"), m = 6),
           temporal = temporal_rw1(time_var = "time"),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_84)) {
  cat(sprintf("%.1fs\n", t_84))
  draws_84 <- as.matrix(fit_84$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_84), value = TRUE)[1]
  results$row_84 <- list(slope = check_recovery(draws_84[, slope_col], true_slope, "slope"), time = t_84)
  print_recovery(results$row_84$slope)
  results$row_84$pass <- results$row_84$slope$pass
} else { results$row_84 <- list(error = TRUE) }

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - HSGP Simulation Validation\n")
cat("=======================================================\n\n")

for (rn in c("row_68", "row_22", "row_52", "row_84")) {
  r <- results[[rn]]
  row_num <- gsub("row_", "", rn)
  if (!is.null(r$error) && r$error) {
    cat(sprintf("Row %s: ERROR\n", row_num))
  } else {
    cat(sprintf("Row %s: %s (%.2f SD, %.1fs)\n", row_num,
                if(r$pass) "PASS" else "FAIL", r$slope$diff_sd, r$time))
  }
}

saveRDS(results, "benchmarks/results_sim_hsgp.rds")
cat("\nResults saved.\n")
