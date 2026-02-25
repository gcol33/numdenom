# Validation of binomial spatiotemporal and multiscale temporal
# Rows 75, 90, 91: binomial + MS_t, ST-I, ST-IV
#
# Note: brms cannot express these exact structures, so we verify they run without errors
# and compare slopes to simpler models as sanity check

library(numdenom)
library(brms)
library(posterior)

set.seed(42)

N_OBS <- 200
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2
N_SITES <- 15
N_TIMES <- 10

cat("=======================================================\n")
cat("Binomial Spatiotemporal Validation: Rows 75, 90, 91\n")
cat("=======================================================\n\n")

compare_posteriors <- function(nd_draws, brms_draws, param_name, threshold_se = 2) {
  nd_mean <- mean(nd_draws)
  nd_sd <- sd(nd_draws)
  brms_mean <- mean(brms_draws)
  brms_sd <- sd(brms_draws)
  se_combined <- sqrt(nd_sd^2 / length(nd_draws) + brms_sd^2 / length(brms_draws))
  diff <- abs(nd_mean - brms_mean)
  ratio <- diff / se_combined
  pass <- ratio < threshold_se
  list(param = param_name, nd_mean = nd_mean, nd_sd = nd_sd,
       brms_mean = brms_mean, brms_sd = brms_sd, diff = diff, ratio = ratio, pass = pass)
}

# Setup data - ensure N_OBS is divisible by N_SITES
N_OBS <- N_SITES * (N_OBS %/% N_SITES)  # Make N_OBS divisible
site <- factor(rep(1:N_SITES, each = N_OBS / N_SITES))
time <- factor(rep(1:N_TIMES, length.out = N_OBS))
x <- rnorm(N_OBS)
trials <- sample(10:50, N_OBS, replace = TRUE)

# Build adjacency for ICAR
adj_matrix <- matrix(0, N_SITES, N_SITES)
for (i in 1:(N_SITES-1)) {
  adj_matrix[i, i+1] <- 1
  adj_matrix[i+1, i] <- 1
}

# Generate data
site_effects <- rnorm(N_SITES, 0, 0.3)
temporal_effects <- cumsum(rnorm(N_TIMES, 0, 0.15))
temporal_effects <- temporal_effects - mean(temporal_effects)

eta <- 0.5 + 0.3 * x + site_effects[as.integer(site)] + temporal_effects[as.integer(time)]
prob <- plogis(eta)
successes <- rbinom(N_OBS, trials, prob)

df <- data.frame(
  successes = successes,
  trials = trials,
  x = x,
  site = site,
  time = time
)

results <- list()

# =============================================================================
# Row 75: binomial + MS_t (Multiscale temporal) - SKIP if too slow
# =============================================================================
cat("\n========== Row 75: binomial + MS_t (Multiscale temporal) ==========\n")
cat("Note: MS_t is slow (~700s in benchmark). Testing with shorter chains...\n")

cat("Fitting numdenom (MS_t)... ")
t_nd_75 <- system.time({
  fit_nd_75 <- tryCatch({
    ratiod(
      successes | trials ~ x + (1 | site),
      data = df,
      family = ratiod_binomial(),
      temporal = temporal_multiscale("time", trend = "rw2", short_term = "ar1"),
      iter = 500, warmup = 250, chains = 1,
      verbose = FALSE
    )
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    NULL
  })
})[["elapsed"]]

if (!is.null(fit_nd_75)) {
  cat(sprintf("%.1fs\n", t_nd_75))
  cat("  Row 75 RUNS successfully - no brms equivalent for comparison\n")
  results$row_75 <- list(time_nd = t_nd_75, runs = TRUE)
} else {
  results$row_75 <- list(error = TRUE)
}

# =============================================================================
# Row 90: binomial + ICAR + RW1 + ST-I (Spatiotemporal Type I)
# =============================================================================
cat("\n========== Row 90: binomial + ST-I (Spatiotemporal Type I) ==========\n")

# Create spatial and temporal objects first
my_spatial <- spatial_car(adj_matrix, group_var = "site")
my_temporal <- temporal_rw1("time")

cat("Fitting numdenom (ST-I)... ")
t_nd_90 <- system.time({
  fit_nd_90 <- tryCatch({
    ratiod(
      successes | trials ~ x + (1 | site),
      data = df,
      family = ratiod_binomial(),
      spatiotemporal = spatiotemporal(spatial = my_spatial, temporal = my_temporal, type = "I"),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      verbose = FALSE
    )
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    NULL
  })
})[["elapsed"]]

if (!is.null(fit_nd_90)) {
  cat(sprintf("%.1fs\n", t_nd_90))

  # Fit simpler brms model for slope comparison
  cat("Fitting brms (simplified for slope comparison)... ")
  t_brms_90 <- system.time({
    fit_brms_90 <- brm(
      successes | trials(trials) ~ x + (1 | site) + (1 | time),
      data = df,
      family = binomial(),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      refresh = 0, silent = 2
    )
  })[["elapsed"]]
  cat(sprintf("%.1fs\n", t_brms_90))

  draws_nd <- as.matrix(fit_nd_90$draws)
  draws_brms <- as_draws_matrix(fit_brms_90)
  nd_slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_nd), value = TRUE)[1]

  result <- compare_posteriors(draws_nd[, nd_slope_col], draws_brms[, "b_x"], "slope (x)")
  cat(sprintf("  slope (x): nd=%.4f, brms=%.4f, diff=%.2f SE => %s\n",
              result$nd_mean, result$brms_mean, result$ratio, if(result$pass) "PASS" else "CHECK"))
  cat(sprintf("  Times: nd=%.1fs, brms=%.1fs\n", t_nd_90, t_brms_90))
  cat("  Note: Different model structure. Slopes compared as sanity check.\n")

  results$row_90 <- list(time_nd = t_nd_90, time_brms = t_brms_90, slope = result, runs = TRUE)
} else {
  results$row_90 <- list(error = TRUE)
}

# =============================================================================
# Row 91: binomial + ICAR + RW1 + ST-IV (Spatiotemporal Type IV)
# =============================================================================
cat("\n========== Row 91: binomial + ST-IV (Spatiotemporal Type IV) ==========\n")

cat("Fitting numdenom (ST-IV)... ")
t_nd_91 <- system.time({
  fit_nd_91 <- tryCatch({
    ratiod(
      successes | trials ~ x + (1 | site),
      data = df,
      family = ratiod_binomial(),
      spatiotemporal = spatiotemporal(
        spatial = spatial_car(adj_matrix, group_var = "site"),
        temporal = temporal_rw1("time"),
        type = "IV"
      ),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      verbose = FALSE
    )
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    NULL
  })
})[["elapsed"]]

if (!is.null(fit_nd_91)) {
  cat(sprintf("%.1fs\n", t_nd_91))

  draws_nd_91 <- as.matrix(fit_nd_91$draws)
  draws_brms_91 <- as_draws_matrix(fit_brms_90)  # reuse brms fit
  nd_slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_nd_91), value = TRUE)[1]

  result_91 <- compare_posteriors(draws_nd_91[, nd_slope_col], draws_brms_91[, "b_x"], "slope (x)")
  cat(sprintf("  slope (x): nd=%.4f, brms=%.4f, diff=%.2f SE => %s\n",
              result_91$nd_mean, result_91$brms_mean, result_91$ratio, if(result_91$pass) "PASS" else "CHECK"))
  cat(sprintf("  Time: %.1fs\n", t_nd_91))

  results$row_91 <- list(time_nd = t_nd_91, slope = result_91, runs = TRUE)
} else {
  results$row_91 <- list(error = TRUE)
}

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - Spatiotemporal Binomial Validation\n")
cat("=======================================================\n\n")

cat(sprintf("%-8s %-30s %10s %s\n", "Row", "Model", "numdenom", "Status"))
cat(paste(rep("-", 60), collapse = ""), "\n")

for (row_name in c("row_75", "row_90", "row_91")) {
  r <- results[[row_name]]
  row_num <- gsub("row_", "", row_name)
  model_name <- switch(row_num,
    "75" = "bin + MS_t",
    "90" = "bin + ST-I",
    "91" = "bin + ST-IV"
  )
  if (!is.null(r$error) && r$error) {
    cat(sprintf("%-8s %-30s %10s %s\n", row_num, model_name, "ERROR", "FAILED"))
  } else {
    cat(sprintf("%-8s %-30s %10.1fs %s\n", row_num, model_name, r$time_nd, "✓runs"))
  }
}

cat(paste(rep("-", 60), collapse = ""), "\n")

saveRDS(results, "benchmarks/results_spatiotemporal_bin.rds")
cat("\nResults saved to benchmarks/results_spatiotemporal_bin.rds\n")
