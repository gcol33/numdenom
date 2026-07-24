# SVC validation: rows 26, 56, 88
library(numdenom)
library(posterior)

set.seed(42)
N_SVC <- 100
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

set.seed(789)
coord_x <- runif(N_SVC, 0, 10)
coord_y <- runif(N_SVC, 0, 10)
coords_mat <- cbind(coord_x, coord_y)
x <- rnorm(N_SVC)

generate_gp_effects <- function(coords, sigma, lengthscale) {
  n <- nrow(coords)
  dist_mat <- as.matrix(dist(coords))
  cov_mat <- sigma^2 * exp(-dist_mat / lengthscale)
  diag(cov_mat) <- diag(cov_mat) + 1e-6
  L <- chol(cov_mat)
  as.vector(t(L) %*% rnorm(n))
}

svc_slope <- generate_gp_effects(coords_mat, 0.2, 3.0)
results <- list()

# Row 26: poisson_gamma + SVC
cat("========== Row 26: poisson_gamma + SVC ==========\n")
effort_26 <- rgamma(N_SVC, 5, 1)
eta_26 <- true_intercept + (true_slope + svc_slope) * x
count_26 <- rpois(N_SVC, exp(eta_26) * effort_26)
df_26 <- data.frame(count = count_26, effort = effort_26, x = x,
                    coord_x = coord_x, coord_y = coord_y)
cat(sprintf("True: mean_slope=%.2f (varies spatially)\n", true_slope))
cat("Fitting... ")
t_26 <- system.time({
  fit_26 <- tryCatch({
    tratio(count | effort ~ x, data = df_26, family = ratiod_poisson_gamma(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 1),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]
if (!is.null(fit_26)) {
  cat(sprintf("%.1fs\n", t_26))
  draws_26 <- as.matrix(fit_26$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_26), value = TRUE)[1]
  r <- check_recovery(draws_26[, slope_col], true_slope, "mean_slope")
  cat(sprintf("  mean_slope: true=%.3f, post=%.3f (SD=%.3f), %.2f SD => %s\n",
              r$true, r$mean, r$sd, r$diff_sd, if(r$pass) "PASS" else "FAIL"))
  results$row_26 <- list(slope = r, time = t_26, pass = r$pass)
} else { results$row_26 <- list(error = TRUE) }

# Row 56: negbin_negbin + SVC
cat("\n========== Row 56: negbin_negbin + SVC ==========\n")
eta_num_56 <- true_intercept + (true_slope + svc_slope) * x
eta_denom_56 <- 0.5 + 0.2 * x
num_56 <- rnbinom(N_SVC, size = 5, mu = exp(eta_num_56))
denom_56 <- rnbinom(N_SVC, size = 5, mu = exp(eta_denom_56))
denom_56[denom_56 == 0] <- 1
df_56 <- data.frame(num = num_56, denom = denom_56, x = x,
                    coord_x = coord_x, coord_y = coord_y)
cat(sprintf("True: mean_slope=%.2f (varies spatially)\n", true_slope))
cat("Fitting... ")
t_56 <- system.time({
  fit_56 <- tryCatch({
    tratio(num | denom ~ x, data = df_56, family = ratiod_negbin_negbin(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 1),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]
if (!is.null(fit_56)) {
  cat(sprintf("%.1fs\n", t_56))
  draws_56 <- as.matrix(fit_56$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_56), value = TRUE)[1]
  r <- check_recovery(draws_56[, slope_col], true_slope, "mean_slope")
  cat(sprintf("  mean_slope: true=%.3f, post=%.3f (SD=%.3f), %.2f SD => %s\n",
              r$true, r$mean, r$sd, r$diff_sd, if(r$pass) "PASS" else "FAIL"))
  results$row_56 <- list(slope = r, time = t_56, pass = r$pass)
} else { results$row_56 <- list(error = TRUE) }

# Row 88: binomial + SVC
cat("\n========== Row 88: binomial + SVC ==========\n")
eta_88 <- true_intercept + (true_slope + svc_slope) * x
prob_88 <- plogis(eta_88)
trials_88 <- sample(20:50, N_SVC, replace = TRUE)
successes_88 <- rbinom(N_SVC, trials_88, prob_88)
df_88 <- data.frame(successes = successes_88, trials = trials_88, x = x,
                    coord_x = coord_x, coord_y = coord_y)
cat(sprintf("True: mean_slope=%.2f (varies spatially)\n", true_slope))
cat("Fitting... ")
t_88 <- system.time({
  fit_88 <- tryCatch({
    tratio(successes | trials ~ x, data = df_88, family = ratiod_binomial(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 1),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]
if (!is.null(fit_88)) {
  cat(sprintf("%.1fs\n", t_88))
  draws_88 <- as.matrix(fit_88$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_88), value = TRUE)[1]
  r <- check_recovery(draws_88[, slope_col], true_slope, "mean_slope")
  cat(sprintf("  mean_slope: true=%.3f, post=%.3f (SD=%.3f), %.2f SD => %s\n",
              r$true, r$mean, r$sd, r$diff_sd, if(r$pass) "PASS" else "FAIL"))
  results$row_88 <- list(slope = r, time = t_88, pass = r$pass)
} else { results$row_88 <- list(error = TRUE) }

cat("\n=== SVC SUMMARY ===\n")
for (rn in c("row_26", "row_56", "row_88")) {
  r <- results[[rn]]
  row_num <- gsub("row_", "", rn)
  if (!is.null(r$error) && r$error) {
    cat(sprintf("Row %s: ERROR\n", row_num))
  } else {
    cat(sprintf("Row %s: %s (%.2f SD, %.1fs)\n", row_num,
                if(r$pass) "PASS" else "FAIL", r$slope$diff_sd, r$time))
  }
}
saveRDS(results, "benchmarks/results_svc.rds")
cat("\nDone. Results saved.\n")
