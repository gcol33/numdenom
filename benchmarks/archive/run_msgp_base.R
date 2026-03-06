# MSGP base validation: rows 9, 39, 69
library(numdenom)
library(posterior)

set.seed(42)
N_MSGP <- 200
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1

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
results <- list()

# Row 9: poisson_gamma + MSGP
cat("========== Row 9: poisson_gamma + MSGP ==========\n")
eta_9 <- true_intercept + true_slope * x + spatial_effects
effort_9 <- rgamma(N_MSGP, 5, 1)
count_9 <- rpois(N_MSGP, exp(eta_9) * effort_9)
df_9 <- data.frame(count = count_9, effort = effort_9, x = x,
                   coord_x = coord_x, coord_y = coord_y)
cat("Fitting... ")
t_9 <- system.time({
  fit_9 <- tryCatch({
    ratiod(count | effort ~ x, data = df_9, family = ratiod_poisson_gamma(),
           spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]
if (!is.null(fit_9)) {
  cat(sprintf("%.1fs\n", t_9))
  draws_9 <- as.matrix(fit_9$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_9), value = TRUE)[1]
  r <- check_recovery(draws_9[, slope_col], true_slope, "slope")
  cat(sprintf("  slope: true=%.3f, post=%.3f (SD=%.3f), %.2f SD => %s\n",
              r$true, r$mean, r$sd, r$diff_sd, if(r$pass) "PASS" else "FAIL"))
  results$row_9 <- list(slope = r, time = t_9, pass = r$pass)
} else { results$row_9 <- list(error = TRUE) }

# Row 39: negbin_negbin + MSGP
cat("\n========== Row 39: negbin_negbin + MSGP ==========\n")
eta_num_39 <- true_intercept + true_slope * x + spatial_effects
eta_denom_39 <- 0.5 + 0.2 * x + spatial_effects * 0.8
num_39 <- rnbinom(N_MSGP, size = 5, mu = exp(eta_num_39))
denom_39 <- rnbinom(N_MSGP, size = 5, mu = exp(eta_denom_39))
denom_39[denom_39 == 0] <- 1
df_39 <- data.frame(num = num_39, denom = denom_39, x = x,
                    coord_x = coord_x, coord_y = coord_y)
cat("Fitting... ")
t_39 <- system.time({
  fit_39 <- tryCatch({
    ratiod(num | denom ~ x, data = df_39, family = ratiod_negbin_negbin(),
           spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]
if (!is.null(fit_39)) {
  cat(sprintf("%.1fs\n", t_39))
  draws_39 <- as.matrix(fit_39$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_39), value = TRUE)[1]
  r <- check_recovery(draws_39[, slope_col], true_slope, "slope")
  cat(sprintf("  slope: true=%.3f, post=%.3f (SD=%.3f), %.2f SD => %s\n",
              r$true, r$mean, r$sd, r$diff_sd, if(r$pass) "PASS" else "FAIL"))
  results$row_39 <- list(slope = r, time = t_39, pass = r$pass)
} else { results$row_39 <- list(error = TRUE) }

# Row 69: binomial + MSGP
cat("\n========== Row 69: binomial + MSGP ==========\n")
eta_69 <- true_intercept + true_slope * x + spatial_effects
prob_69 <- plogis(eta_69)
trials_69 <- sample(20:50, N_MSGP, replace = TRUE)
successes_69 <- rbinom(N_MSGP, trials_69, prob_69)
df_69 <- data.frame(successes = successes_69, trials = trials_69, x = x,
                    coord_x = coord_x, coord_y = coord_y)
cat("Fitting... ")
t_69 <- system.time({
  fit_69 <- tryCatch({
    ratiod(successes | trials ~ x, data = df_69, family = ratiod_binomial(),
           spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]
if (!is.null(fit_69)) {
  cat(sprintf("%.1fs\n", t_69))
  draws_69 <- as.matrix(fit_69$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_69), value = TRUE)[1]
  r <- check_recovery(draws_69[, slope_col], true_slope, "slope")
  cat(sprintf("  slope: true=%.3f, post=%.3f (SD=%.3f), %.2f SD => %s\n",
              r$true, r$mean, r$sd, r$diff_sd, if(r$pass) "PASS" else "FAIL"))
  results$row_69 <- list(slope = r, time = t_69, pass = r$pass)
} else { results$row_69 <- list(error = TRUE) }

cat("\n=== MSGP BASE SUMMARY ===\n")
for (rn in c("row_9", "row_39", "row_69")) {
  r <- results[[rn]]
  row_num <- gsub("row_", "", rn)
  if (!is.null(r$error) && r$error) {
    cat(sprintf("Row %s: ERROR\n", row_num))
  } else {
    cat(sprintf("Row %s: %s (%.2f SD, %.1fs)\n", row_num,
                if(r$pass) "PASS" else "FAIL", r$slope$diff_sd, r$time))
  }
}
saveRDS(results, "benchmarks/results_msgp_base.rds")
cat("\nDone. Results saved.\n")
