# Simulation-based validation for OI/ZOIB binomial models
# Rows 78, 79: One-inflated and Zero-one-inflated binomial

library(numdenom)
library(posterior)

set.seed(42)

N_OBS <- 200
N_ITER <- 1000
N_WARMUP <- 500
N_CHAINS <- 2
N_SITES <- 20

cat("=======================================================\n")
cat("OI/ZOIB Binomial: Simulation-Based Validation\n")
cat("Rows 78, 79\n")
cat("=======================================================\n")
cat(sprintf("N=%d, sites=%d, iter=%d, chains=%d\n\n", N_OBS, N_SITES, N_ITER, N_CHAINS))

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

# Common setup
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)
trials <- sample(20:50, N_OBS, replace = TRUE)

# True parameters
true_intercept <- 0.5
true_slope <- 0.3
site_effects <- rnorm(N_SITES, 0, 0.3)

# =============================================================================
# Row 78: binomial + OI (one-inflated)
# =============================================================================
cat("\n========== Row 78: binomial + OI ==========\n")

# One-inflation: some obs have prob = 1 (all successes)
true_oi_prob <- 0.15  # 15% are "all successes"

eta_78 <- true_intercept + true_slope * x + site_effects[as.integer(site)]
prob_78 <- plogis(eta_78)

# Generate with one-inflation
is_one_inflated <- rbinom(N_OBS, 1, true_oi_prob)
successes_78 <- ifelse(is_one_inflated == 1,
                       trials,  # All successes
                       rbinom(N_OBS, trials, prob_78))

df_78 <- data.frame(successes = successes_78, trials = trials, x = x, site = site)

cat(sprintf("True: slope=%.2f, OI_prob=%.2f\n", true_slope, true_oi_prob))
cat(sprintf("Proportion all-successes: %.2f\n", mean(successes_78 == trials)))
cat("Fitting... ")

t_78 <- system.time({
  fit_78 <- tryCatch({
    ratiod(successes | trials ~ x + (1 | site), data = df_78,
           family = ratiod_oibinomial(),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_78)) {
  cat(sprintf("%.1fs\n", t_78))
  draws_78 <- as.matrix(fit_78$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_78), value = TRUE)[1]

  results$row_78 <- list(
    slope = check_recovery(draws_78[, slope_col], true_slope, "slope"),
    time = t_78
  )
  print_recovery(results$row_78$slope)
  results$row_78$pass <- results$row_78$slope$pass
  cat(sprintf("Overall: %s (%.1fs)\n", if(results$row_78$pass) "PASS" else "FAIL", t_78))
} else {
  results$row_78 <- list(error = TRUE)
}

# =============================================================================
# Row 79: binomial + ZOIB (zero-and-one-inflated)
# =============================================================================
cat("\n========== Row 79: binomial + ZOIB ==========\n")

# Zero-one-inflation: some obs have prob = 0, some have prob = 1
true_zi_prob <- 0.10  # 10% are zeros
true_oi_prob_79 <- 0.10  # 10% are ones

eta_79 <- true_intercept + true_slope * x + site_effects[as.integer(site)]
prob_79 <- plogis(eta_79)

# Generate with zero-one-inflation
inflate_type <- sample(c("zero", "one", "neither"), N_OBS, replace = TRUE,
                       prob = c(true_zi_prob, true_oi_prob_79, 1 - true_zi_prob - true_oi_prob_79))
successes_79 <- ifelse(inflate_type == "zero", 0,
                ifelse(inflate_type == "one", trials,
                       rbinom(N_OBS, trials, prob_79)))

df_79 <- data.frame(successes = successes_79, trials = trials, x = x, site = site)

cat(sprintf("True: slope=%.2f, ZI_prob=%.2f, OI_prob=%.2f\n", true_slope, true_zi_prob, true_oi_prob_79))
cat(sprintf("Proportion zeros: %.2f, ones: %.2f\n",
            mean(successes_79 == 0), mean(successes_79 == trials)))
cat("Fitting... ")

t_79 <- system.time({
  fit_79 <- tryCatch({
    ratiod(successes | trials ~ x + (1 | site), data = df_79,
           family = ratiod_zoibinomial(),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})[["elapsed"]]

if (!is.null(fit_79)) {
  cat(sprintf("%.1fs\n", t_79))
  draws_79 <- as.matrix(fit_79$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_79), value = TRUE)[1]

  results$row_79 <- list(
    slope = check_recovery(draws_79[, slope_col], true_slope, "slope"),
    time = t_79
  )
  print_recovery(results$row_79$slope)
  results$row_79$pass <- results$row_79$slope$pass
  cat(sprintf("Overall: %s (%.1fs)\n", if(results$row_79$pass) "PASS" else "FAIL", t_79))
} else {
  results$row_79 <- list(error = TRUE)
}

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - OI/ZOIB Simulation Validation\n")
cat("=======================================================\n\n")

for (rn in c("row_78", "row_79")) {
  r <- results[[rn]]
  row_num <- gsub("row_", "", rn)
  if (!is.null(r$error) && r$error) {
    cat(sprintf("Row %s: ERROR\n", row_num))
  } else {
    cat(sprintf("Row %s: %s (%.2f SD, %.1fs)\n", row_num,
                if(r$pass) "PASS" else "FAIL", r$slope$diff_sd, r$time))
  }
}

saveRDS(results, "benchmarks/results_sim_oi_zoib.rds")
cat("\nResults saved.\n")
