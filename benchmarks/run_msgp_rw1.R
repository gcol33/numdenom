# MSGP+RW1 validation: rows 23, 53, 85
library(numdenom)
library(posterior)

set.seed(42)
N_MSGP <- 200
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1
N_TIMES <- 10

true_intercept <- 1.0
true_slope <- 0.3

check_recovery <- function(draws, true_value, param_name, threshold_sd = 2) {
  post_mean <- mean(draws)
  post_sd <- sd(draws)
  diff_sd <- abs(post_mean - true_value) / post_sd
  in_ci <- true_value >= quantile(draws, 0.025) && true_value <= quantile(draws, 0.975)
  list(param = param_name, true = true_value, mean = post_mean, sd = post_sd,
       diff_sd = diff_sd, in_ci = in_ci, pass = diff_sd < threshold_sd)
}

set.seed(123)
coord_x <- runif(N_MSGP, 0, 10)
coord_y <- runif(N_MSGP, 0, 10)
coords_mat <- cbind(coord_x, coord_y)
x <- rnorm(N_MSGP)

generate_msgp_effects <- function(coords, sigma_fine, sigma_coarse, ls_fine, ls_coarse) {
  n <- nrow(coords)
  dist_mat <- as.matrix(dist(coords))
  cov_fine <- sigma_fine^2 * exp(-dist_mat / ls_fine)
  diag(cov_fine) <- diag(cov_fine) + 1e-6
  L_fine <- chol(cov_fine)
  fine <- as.vector(t(L_fine) %*% rnorm(n))
  cov_coarse <- sigma_coarse^2 * exp(-dist_mat / ls_coarse)
  diag(cov_coarse) <- diag(cov_coarse) + 1e-6
  L_coarse <- chol(cov_coarse)
  coarse <- as.vector(t(L_coarse) %*% rnorm(n))
  effects <- fine + coarse
  effects - mean(effects)
}

spatial_effects <- generate_msgp_effects(coords_mat, 0.2, 0.3, 1.0, 4.0)

# Generate RW1 temporal effects
set.seed(456)
generate_rw1 <- function(n, sigma) {
  gamma <- cumsum(rnorm(n, 0, sigma))
  gamma - mean(gamma)
}
temporal_effects <- generate_rw1(N_TIMES, 0.15)
time_idx <- rep(1:N_TIMES, length.out = N_MSGP)

results <- list()

# Row 23: poisson_gamma + MSGP + RW1
cat("========== Row 23: poisson_gamma + MSGP + RW1 ==========\n")
eta_23 <- true_intercept + true_slope * x + spatial_effects + temporal_effects[time_idx]
effort_23 <- rgamma(N_MSGP, 5, 1)
count_23 <- rpois(N_MSGP, exp(eta_23) * effort_23)
df_23 <- data.frame(count = count_23, effort = effort_23, x = x,
                    coord_x = coord_x, coord_y = coord_y, time = time_idx)
cat("Fitting... ")
t_23 <- system.time({
  fit_23 <- tryCatch({
    ratiod(count | effort ~ x, data = df_23, family = ratiod_poisson_gamma(),
           spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
           temporal = temporal_rw1(time_var = "time"),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]
if (!is.null(fit_23)) {
  cat(sprintf("%.1fs\n", t_23))
  draws_23 <- as.matrix(fit_23$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_23), value = TRUE)[1]
  r <- check_recovery(draws_23[, slope_col], true_slope, "slope")
  cat(sprintf("  slope: true=%.3f, post=%.3f (SD=%.3f), %.2f SD => %s\n",
              r$true, r$mean, r$sd, r$diff_sd, if(r$pass) "PASS" else "FAIL"))
  results$row_23 <- list(slope = r, time = t_23, pass = r$pass)
} else { results$row_23 <- list(error = TRUE) }

# Row 53: negbin_negbin + MSGP + RW1
cat("\n========== Row 53: negbin_negbin + MSGP + RW1 ==========\n")
eta_num_53 <- true_intercept + true_slope * x + spatial_effects + temporal_effects[time_idx]
eta_denom_53 <- 0.5 + 0.2 * x + spatial_effects * 0.8 + temporal_effects[time_idx] * 0.6
num_53 <- rnbinom(N_MSGP, size = 5, mu = exp(eta_num_53))
denom_53 <- rnbinom(N_MSGP, size = 5, mu = exp(eta_denom_53))
denom_53[denom_53 == 0] <- 1
df_53 <- data.frame(num = num_53, denom = denom_53, x = x,
                    coord_x = coord_x, coord_y = coord_y, time = time_idx)
cat("Fitting... ")
t_53 <- system.time({
  fit_53 <- tryCatch({
    ratiod(num | denom ~ x, data = df_53, family = ratiod_negbin_negbin(),
           spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
           temporal = temporal_rw1(time_var = "time"),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]
if (!is.null(fit_53)) {
  cat(sprintf("%.1fs\n", t_53))
  draws_53 <- as.matrix(fit_53$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_53), value = TRUE)[1]
  r <- check_recovery(draws_53[, slope_col], true_slope, "slope")
  cat(sprintf("  slope: true=%.3f, post=%.3f (SD=%.3f), %.2f SD => %s\n",
              r$true, r$mean, r$sd, r$diff_sd, if(r$pass) "PASS" else "FAIL"))
  results$row_53 <- list(slope = r, time = t_53, pass = r$pass)
} else { results$row_53 <- list(error = TRUE) }

# Row 85: binomial + MSGP + RW1
cat("\n========== Row 85: binomial + MSGP + RW1 ==========\n")
eta_85 <- true_intercept + true_slope * x + spatial_effects + temporal_effects[time_idx]
prob_85 <- plogis(eta_85)
trials_85 <- sample(20:50, N_MSGP, replace = TRUE)
successes_85 <- rbinom(N_MSGP, trials_85, prob_85)
df_85 <- data.frame(successes = successes_85, trials = trials_85, x = x,
                    coord_x = coord_x, coord_y = coord_y, time = time_idx)
cat("Fitting... ")
t_85 <- system.time({
  fit_85 <- tryCatch({
    ratiod(successes | trials ~ x, data = df_85, family = ratiod_binomial(),
           spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
           temporal = temporal_rw1(time_var = "time"),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]
if (!is.null(fit_85)) {
  cat(sprintf("%.1fs\n", t_85))
  draws_85 <- as.matrix(fit_85$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_85), value = TRUE)[1]
  r <- check_recovery(draws_85[, slope_col], true_slope, "slope")
  cat(sprintf("  slope: true=%.3f, post=%.3f (SD=%.3f), %.2f SD => %s\n",
              r$true, r$mean, r$sd, r$diff_sd, if(r$pass) "PASS" else "FAIL"))
  results$row_85 <- list(slope = r, time = t_85, pass = r$pass)
} else { results$row_85 <- list(error = TRUE) }

cat("\n=== MSGP+RW1 SUMMARY ===\n")
for (rn in c("row_23", "row_53", "row_85")) {
  r <- results[[rn]]
  row_num <- gsub("row_", "", rn)
  if (!is.null(r$error) && r$error) {
    cat(sprintf("Row %s: ERROR\n", row_num))
  } else {
    cat(sprintf("Row %s: %s (%.2f SD, %.1fs)\n", row_num,
                if(r$pass) "PASS" else "FAIL", r$slope$diff_sd, r$time))
  }
}
saveRDS(results, "benchmarks/results_msgp_rw1.rds")
cat("\nDone. Results saved.\n")
