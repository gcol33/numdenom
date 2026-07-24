# Retry MSGP rows 9 (pg) and 69 (bin) with longer chains
# Previous run: 500 iter, both FAIL (4.33 SD, 6.13 SD)
# This run: 1000 iter, 500 warmup

library(numdenom)
library(posterior)

set.seed(42)
N_MSGP <- 200
N_ITER <- 1000
N_WARMUP <- 500
N_CHAINS <- 1
N_SITES <- 10

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

# Use SAME seed as the original run for comparable data
set.seed(123)
coord_x <- runif(N_MSGP, 0, 10)
coord_y <- runif(N_MSGP, 0, 10)
coords_mat <- cbind(coord_x, coord_y)
x <- rnorm(N_MSGP)
site <- factor(rep(1:N_SITES, length.out = N_MSGP))
site_effects <- rnorm(N_SITES, 0, 0.3)

generate_gp_effects <- function(coords, sigma, lengthscale) {
  n <- nrow(coords)
  dist_mat <- as.matrix(dist(coords))
  cov_mat <- sigma^2 * exp(-dist_mat / lengthscale)
  diag(cov_mat) <- diag(cov_mat) + 1e-6
  L <- chol(cov_mat)
  as.vector(t(L) %*% rnorm(n))
}

gp_fine <- generate_gp_effects(coords_mat, 0.2, 1.0)
gp_coarse <- generate_gp_effects(coords_mat, 0.3, 4.0)
spatial_effects <- gp_fine + gp_coarse

results <- list()

# Row 9: poisson_gamma + MSGP (retry with longer chains)
cat("========== Row 9 (RETRY): poisson_gamma + MSGP ==========\n")
cat(sprintf("N=%d, iter=%d (was 500), warmup=%d\n", N_MSGP, N_ITER, N_WARMUP))
effort_9 <- rgamma(N_MSGP, 5, 1)
eta_9 <- true_intercept + true_slope * x +
          site_effects[as.integer(site)] + spatial_effects
count_9 <- rpois(N_MSGP, exp(eta_9) * effort_9)
df_9 <- data.frame(count = count_9, effort = effort_9, x = x, site = site,
                   coord_x = coord_x, coord_y = coord_y)
cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_9 <- system.time({
  fit_9 <- tryCatch({
    tratio(count | effort ~ x + (1 | site), data = df_9,
           family = ratiod_poisson_gamma(),
           spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
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

# Row 69: binomial + MSGP (retry with longer chains)
cat("\n========== Row 69 (RETRY): binomial + MSGP ==========\n")
cat(sprintf("N=%d, iter=%d (was 500), warmup=%d\n", N_MSGP, N_ITER, N_WARMUP))
eta_69 <- true_intercept + true_slope * x +
           site_effects[as.integer(site)] + spatial_effects
prob_69 <- plogis(eta_69)
trials_69 <- sample(20:50, N_MSGP, replace = TRUE)
successes_69 <- rbinom(N_MSGP, trials_69, prob_69)
df_69 <- data.frame(successes = successes_69, trials = trials_69, x = x, site = site,
                    coord_x = coord_x, coord_y = coord_y)
cat(sprintf("True: slope=%.2f\n", true_slope))
cat("Fitting... ")
t_69 <- system.time({
  fit_69 <- tryCatch({
    tratio(successes | trials ~ x + (1 | site), data = df_69,
           family = ratiod_binomial(),
           spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
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

cat("\n=== MSGP RETRY SUMMARY ===\n")
for (rn in c("row_9", "row_69")) {
  r <- results[[rn]]
  row_num <- gsub("row_", "", rn)
  if (!is.null(r$error) && r$error) {
    cat(sprintf("Row %s: ERROR\n", row_num))
  } else {
    cat(sprintf("Row %s: %s (%.2f SD, %.1fs)\n", row_num,
                if(r$pass) "PASS" else "FAIL", r$slope$diff_sd, r$time))
  }
}
saveRDS(results, "benchmarks/results_msgp_retry.rds")
cat("\nDone. Results saved.\n")
