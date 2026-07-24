# Simulation-based validation for spatiotemporal models
# Rows 28-29, 58-59, 90-91: Knorr-Held Type I and IV

library(numdenom)
library(posterior)

set.seed(42)

N_OBS <- 150  # 15 sites × 10 times
N_ITER <- 1000
N_WARMUP <- 500
N_CHAINS <- 2
N_SITES <- 15
N_TIMES <- 10

cat("=======================================================\n")
cat("Spatiotemporal Models: Simulation-Based Validation\n")
cat("Rows 28-29, 58-59, 90-91 (Knorr-Held Type I & IV)\n")
cat("=======================================================\n")
cat(sprintf("N=%d (%d sites × %d times), iter=%d, chains=%d\n\n",
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

# Build adjacency matrix (chain graph)
adj_matrix <- matrix(0, N_SITES, N_SITES)
for (i in 1:(N_SITES - 1)) {
  adj_matrix[i, i + 1] <- 1
  adj_matrix[i + 1, i] <- 1
}

# Generate spatial effects (ICAR)
generate_icar <- function(n, sigma) {
  phi <- cumsum(rnorm(n, 0, sigma))
  phi - mean(phi)
}

# Generate temporal effects (RW1)
generate_rw1 <- function(n, sigma) {
  gamma <- cumsum(rnorm(n, 0, sigma))
  gamma - mean(gamma)
}

# Generate spatiotemporal interaction
generate_st_interaction <- function(n_sites, n_times, sigma, type = "I") {
  if (type == "I") {
    # Unstructured: independent for each site-time
    delta <- matrix(rnorm(n_sites * n_times, 0, sigma), n_sites, n_times)
  } else if (type == "IV") {
    # Kronecker: structured in both space and time
    phi_s <- generate_icar(n_sites, 1)
    gamma_t <- generate_rw1(n_times, 1)
    delta <- outer(phi_s, gamma_t) * sigma
  }
  delta - mean(delta)
}

results <- list()

# True parameters
true_intercept <- 1.0
true_slope <- 0.3
true_sigma_spatial <- 0.3
true_sigma_temporal <- 0.2
true_sigma_st <- 0.15

# Setup data structure
site <- factor(rep(1:N_SITES, each = N_TIMES))
time <- rep(1:N_TIMES, N_SITES)
x <- rnorm(N_OBS)

# Generate effects
spatial_effects <- generate_icar(N_SITES, true_sigma_spatial)
temporal_effects <- generate_rw1(N_TIMES, true_sigma_temporal)

# =============================================================================
# Row 28: poisson_gamma + ST-I
# =============================================================================
cat("\n========== Row 28: poisson_gamma + ST-I ==========\n")

st_effects_I <- generate_st_interaction(N_SITES, N_TIMES, true_sigma_st, "I")
eta_28 <- true_intercept + true_slope * x +
          spatial_effects[as.integer(site)] +
          temporal_effects[time] +
          st_effects_I[cbind(as.integer(site), time)]

effort_28 <- rgamma(N_OBS, 5, 1)
count_28 <- rpois(N_OBS, exp(eta_28) * effort_28)
df_28 <- data.frame(count = count_28, effort = effort_28, x = x, site = site, time = time)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_28 <- system.time({
  fit_28 <- tryCatch({
    tratio(count | effort ~ x, data = df_28, family = ratiod_poisson_gamma(),
           spatiotemporal = spatiotemporal(
             spatial = spatial_car(adj_matrix, group_var = "site"),
             temporal = temporal_rw1(time_var = "time"),
             type = "I"
           ),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_28)) {
  cat(sprintf("%.1fs\n", t_28))
  draws_28 <- as.matrix(fit_28$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_28), value = TRUE)[1]
  results$row_28 <- list(slope = check_recovery(draws_28[, slope_col], true_slope, "slope"), time = t_28)
  print_recovery(results$row_28$slope)
  results$row_28$pass <- results$row_28$slope$pass
} else { results$row_28 <- list(error = TRUE) }

# =============================================================================
# Row 29: poisson_gamma + ST-IV
# =============================================================================
cat("\n========== Row 29: poisson_gamma + ST-IV ==========\n")

st_effects_IV <- generate_st_interaction(N_SITES, N_TIMES, true_sigma_st, "IV")
eta_29 <- true_intercept + true_slope * x +
          spatial_effects[as.integer(site)] +
          temporal_effects[time] +
          st_effects_IV[cbind(as.integer(site), time)]

count_29 <- rpois(N_OBS, exp(eta_29) * effort_28)
df_29 <- data.frame(count = count_29, effort = effort_28, x = x, site = site, time = time)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_29 <- system.time({
  fit_29 <- tryCatch({
    tratio(count | effort ~ x, data = df_29, family = ratiod_poisson_gamma(),
           spatiotemporal = spatiotemporal(
             spatial = spatial_car(adj_matrix, group_var = "site"),
             temporal = temporal_rw1(time_var = "time"),
             type = "IV"
           ),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_29)) {
  cat(sprintf("%.1fs\n", t_29))
  draws_29 <- as.matrix(fit_29$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_29), value = TRUE)[1]
  results$row_29 <- list(slope = check_recovery(draws_29[, slope_col], true_slope, "slope"), time = t_29)
  print_recovery(results$row_29$slope)
  results$row_29$pass <- results$row_29$slope$pass
} else { results$row_29 <- list(error = TRUE) }

# =============================================================================
# Row 58: negbin_negbin + ST-I
# =============================================================================
cat("\n========== Row 58: negbin_negbin + ST-I ==========\n")

eta_num_58 <- true_intercept + true_slope * x + spatial_effects[as.integer(site)] +
              temporal_effects[time] + st_effects_I[cbind(as.integer(site), time)]
eta_denom_58 <- 0.5 + 0.2 * x + spatial_effects[as.integer(site)] * 0.8

num_58 <- rnbinom(N_OBS, size = 5, mu = exp(eta_num_58))
denom_58 <- rnbinom(N_OBS, size = 5, mu = exp(eta_denom_58))
denom_58[denom_58 == 0] <- 1
df_58 <- data.frame(num = num_58, denom = denom_58, x = x, site = site, time = time)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_58 <- system.time({
  fit_58 <- tryCatch({
    tratio(num | denom ~ x, data = df_58, family = ratiod_negbin_negbin(),
           spatiotemporal = spatiotemporal(
             spatial = spatial_car(adj_matrix, group_var = "site"),
             temporal = temporal_rw1(time_var = "time"),
             type = "I"
           ),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_58)) {
  cat(sprintf("%.1fs\n", t_58))
  draws_58 <- as.matrix(fit_58$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_58), value = TRUE)[1]
  results$row_58 <- list(slope = check_recovery(draws_58[, slope_col], true_slope, "slope"), time = t_58)
  print_recovery(results$row_58$slope)
  results$row_58$pass <- results$row_58$slope$pass
} else { results$row_58 <- list(error = TRUE) }

# =============================================================================
# Row 59: negbin_negbin + ST-IV
# =============================================================================
cat("\n========== Row 59: negbin_negbin + ST-IV ==========\n")

eta_num_59 <- true_intercept + true_slope * x + spatial_effects[as.integer(site)] +
              temporal_effects[time] + st_effects_IV[cbind(as.integer(site), time)]

num_59 <- rnbinom(N_OBS, size = 5, mu = exp(eta_num_59))
df_59 <- data.frame(num = num_59, denom = denom_58, x = x, site = site, time = time)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_59 <- system.time({
  fit_59 <- tryCatch({
    tratio(num | denom ~ x, data = df_59, family = ratiod_negbin_negbin(),
           spatiotemporal = spatiotemporal(
             spatial = spatial_car(adj_matrix, group_var = "site"),
             temporal = temporal_rw1(time_var = "time"),
             type = "IV"
           ),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_59)) {
  cat(sprintf("%.1fs\n", t_59))
  draws_59 <- as.matrix(fit_59$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_59), value = TRUE)[1]
  results$row_59 <- list(slope = check_recovery(draws_59[, slope_col], true_slope, "slope"), time = t_59)
  print_recovery(results$row_59$slope)
  results$row_59$pass <- results$row_59$slope$pass
} else { results$row_59 <- list(error = TRUE) }

# =============================================================================
# Row 90: binomial + ST-I
# =============================================================================
cat("\n========== Row 90: binomial + ST-I ==========\n")

eta_90 <- true_intercept + true_slope * x + spatial_effects[as.integer(site)] +
          temporal_effects[time] + st_effects_I[cbind(as.integer(site), time)]
prob_90 <- plogis(eta_90)
trials_90 <- sample(20:50, N_OBS, replace = TRUE)
successes_90 <- rbinom(N_OBS, trials_90, prob_90)
df_90 <- data.frame(successes = successes_90, trials = trials_90, x = x, site = site, time = time)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_90 <- system.time({
  fit_90 <- tryCatch({
    tratio(successes | trials ~ x, data = df_90, family = ratiod_binomial(),
           spatiotemporal = spatiotemporal(
             spatial = spatial_car(adj_matrix, group_var = "site"),
             temporal = temporal_rw1(time_var = "time"),
             type = "I"
           ),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_90)) {
  cat(sprintf("%.1fs\n", t_90))
  draws_90 <- as.matrix(fit_90$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_90), value = TRUE)[1]
  results$row_90 <- list(slope = check_recovery(draws_90[, slope_col], true_slope, "slope"), time = t_90)
  print_recovery(results$row_90$slope)
  results$row_90$pass <- results$row_90$slope$pass
} else { results$row_90 <- list(error = TRUE) }

# =============================================================================
# Row 91: binomial + ST-IV
# =============================================================================
cat("\n========== Row 91: binomial + ST-IV ==========\n")

eta_91 <- true_intercept + true_slope * x + spatial_effects[as.integer(site)] +
          temporal_effects[time] + st_effects_IV[cbind(as.integer(site), time)]
prob_91 <- plogis(eta_91)
successes_91 <- rbinom(N_OBS, trials_90, prob_91)
df_91 <- data.frame(successes = successes_91, trials = trials_90, x = x, site = site, time = time)

cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_91 <- system.time({
  fit_91 <- tryCatch({
    tratio(successes | trials ~ x, data = df_91, family = ratiod_binomial(),
           spatiotemporal = spatiotemporal(
             spatial = spatial_car(adj_matrix, group_var = "site"),
             temporal = temporal_rw1(time_var = "time"),
             type = "IV"
           ),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_91)) {
  cat(sprintf("%.1fs\n", t_91))
  draws_91 <- as.matrix(fit_91$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_91), value = TRUE)[1]
  results$row_91 <- list(slope = check_recovery(draws_91[, slope_col], true_slope, "slope"), time = t_91)
  print_recovery(results$row_91$slope)
  results$row_91$pass <- results$row_91$slope$pass
} else { results$row_91 <- list(error = TRUE) }

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - Spatiotemporal Simulation Validation\n")
cat("=======================================================\n\n")

for (rn in c("row_28", "row_29", "row_58", "row_59", "row_90", "row_91")) {
  r <- results[[rn]]
  row_num <- gsub("row_", "", rn)
  if (!is.null(r$error) && r$error) {
    cat(sprintf("Row %s: ERROR\n", row_num))
  } else {
    cat(sprintf("Row %s: %s (%.2f SD, %.1fs)\n", row_num,
                if(r$pass) "PASS" else "FAIL", r$slope$diff_sd, r$time))
  }
}

saveRDS(results, "benchmarks/results_sim_spatiotemporal.rds")
cat("\nResults saved.\n")
