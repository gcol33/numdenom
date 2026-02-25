# Simulation-based validation for multiscale temporal (MS_t)
# Rows 15 (poisson_gamma) and 45 (negbin_negbin)
# Row 75 (binomial) already validated: 0.55 SD PASS

library(numdenom)
library(posterior)

set.seed(42)
N_OBS <- 200
N_ITER <- 1000
N_WARMUP <- 500
N_CHAINS <- 1
N_SITES <- 20
N_TIMES <- 20

true_intercept <- 0.5
true_slope <- 0.3

check_recovery <- function(draws, true_value, param_name, threshold_sd = 2) {
  post_mean <- mean(draws)
  post_sd <- sd(draws)
  diff_sd <- abs(post_mean - true_value) / post_sd
  in_ci <- true_value >= quantile(draws, 0.025) && true_value <= quantile(draws, 0.975)
  list(param = param_name, true = true_value, mean = post_mean, sd = post_sd,
       diff_sd = diff_sd, in_ci = in_ci, pass = diff_sd < threshold_sd)
}

# Setup shared data
site <- factor(rep(1:N_SITES, length.out = N_OBS))
time <- rep(1:N_TIMES, length.out = N_OBS)
x <- rnorm(N_OBS)
site_effects <- rnorm(N_SITES, 0, 0.3)

# Generate multiscale temporal effects
true_sigma_slow <- 0.3
true_sigma_fast <- 0.15
slow_trend <- cumsum(rnorm(N_TIMES, 0, true_sigma_slow / sqrt(N_TIMES)))
slow_trend <- slow_trend - mean(slow_trend)
fast_trend <- rnorm(N_TIMES, 0, true_sigma_fast)
fast_trend <- fast_trend - mean(fast_trend)
temporal_effects <- slow_trend + fast_trend

results <- list()

# Row 15: poisson_gamma + MS_t
cat("========== Row 15: poisson_gamma + MS_t ==========\n")
effort_15 <- rgamma(N_OBS, 5, 1)
eta_15 <- true_intercept + true_slope * x +
           site_effects[as.integer(site)] + temporal_effects[time]
count_15 <- rpois(N_OBS, exp(eta_15) * effort_15)
df_15 <- data.frame(count = count_15, effort = effort_15, x = x,
                    site = site, time = time)
cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_15 <- system.time({
  fit_15 <- tryCatch({
    ratiod(count | effort ~ x + (1 | site), data = df_15,
           family = ratiod_poisson_gamma(),
           temporal = temporal_multiscale(time_var = "time"),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]
if (!is.null(fit_15)) {
  cat(sprintf("%.1fs\n", t_15))
  draws_15 <- as.matrix(fit_15$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_15), value = TRUE)[1]
  r <- check_recovery(draws_15[, slope_col], true_slope, "slope")
  cat(sprintf("  slope: true=%.3f, post=%.3f (SD=%.3f), %.2f SD => %s\n",
              r$true, r$mean, r$sd, r$diff_sd, if(r$pass) "PASS" else "FAIL"))
  results$row_15 <- list(slope = r, time = t_15, pass = r$pass)
} else { results$row_15 <- list(error = TRUE) }

# Row 45: negbin_negbin + MS_t
cat("\n========== Row 45: negbin_negbin + MS_t ==========\n")
eta_num_45 <- true_intercept + true_slope * x +
               site_effects[as.integer(site)] + temporal_effects[time]
eta_denom_45 <- 0.5 + 0.2 * x
num_45 <- rnbinom(N_OBS, size = 5, mu = exp(eta_num_45))
denom_45 <- rnbinom(N_OBS, size = 5, mu = exp(eta_denom_45))
denom_45[denom_45 == 0] <- 1
df_45 <- data.frame(num = num_45, denom = denom_45, x = x,
                    site = site, time = time)
cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_45 <- system.time({
  fit_45 <- tryCatch({
    ratiod(num | denom ~ x + (1 | site), data = df_45,
           family = ratiod_negbin_negbin(),
           temporal = temporal_multiscale(time_var = "time"),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]
if (!is.null(fit_45)) {
  cat(sprintf("%.1fs\n", t_45))
  draws_45 <- as.matrix(fit_45$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_45), value = TRUE)[1]
  r <- check_recovery(draws_45[, slope_col], true_slope, "slope")
  cat(sprintf("  slope: true=%.3f, post=%.3f (SD=%.3f), %.2f SD => %s\n",
              r$true, r$mean, r$sd, r$diff_sd, if(r$pass) "PASS" else "FAIL"))
  results$row_45 <- list(slope = r, time = t_45, pass = r$pass)
} else { results$row_45 <- list(error = TRUE) }

cat("\n=== MS_t SUMMARY ===\n")
for (rn in c("row_15", "row_45")) {
  r <- results[[rn]]
  row_num <- gsub("row_", "", rn)
  if (!is.null(r$error) && r$error) {
    cat(sprintf("Row %s: ERROR\n", row_num))
  } else {
    cat(sprintf("Row %s: %s (%.2f SD, %.1fs)\n", row_num,
                if(r$pass) "PASS" else "FAIL", r$slope$diff_sd, r$time))
  }
}
saveRDS(results, "benchmarks/results_ms_temporal.rds")
cat("\nDone. Results saved.\n")
