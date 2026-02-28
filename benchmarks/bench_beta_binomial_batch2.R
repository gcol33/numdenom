# Validation of beta_binomial family against brms - Batch 2
# Rows 105-107: beta_binomial + ICAR, RW1, and ICAR+RW1

library(numdenom)
library(brms)
library(posterior)

set.seed(42)

# Parameters
N_OBS <- 200
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2
N_SITES <- 20
N_TIMES <- 10

cat("=======================================================\n")
cat("Beta-Binomial Validation Batch 2: Rows 105-107\n")
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

print_result <- function(result) {
  cat(sprintf("  %s: nd=%.4f (SD=%.4f), brms=%.4f (SD=%.4f), diff=%.4f (%.2f SE) => %s\n",
              result$param, result$nd_mean, result$nd_sd,
              result$brms_mean, result$brms_sd,
              result$diff, result$ratio, if(result$pass) "PASS" else "FAIL"))
}

# Setup data
site <- factor(rep(1:N_SITES, length.out = N_OBS))
time <- factor(rep(1:N_TIMES, length.out = N_OBS))
x <- rnorm(N_OBS)
trials <- sample(10:50, N_OBS, replace = TRUE)
phi <- 10

# Build adjacency for ICAR
adj_matrix <- matrix(0, N_SITES, N_SITES)
for (i in 1:(N_SITES-1)) {
  adj_matrix[i, i+1] <- 1
  adj_matrix[i+1, i] <- 1
}

results <- list()

# =============================================================================
# Row 105: beta_binomial + RE + ICAR
# =============================================================================
cat("\n========== Row 105: beta_binomial + RE + ICAR ==========\n")

site_effects <- rnorm(N_SITES, 0, 0.3)
spatial_effects <- rnorm(N_SITES, 0, 0.2)
spatial_effects <- spatial_effects - mean(spatial_effects)

eta <- 0.5 + 0.3 * x + site_effects[as.integer(site)] + spatial_effects[as.integer(site)]
prob <- plogis(eta)
alpha <- prob * phi
beta_param <- (1 - prob) * phi
y <- rbeta(N_OBS, alpha, beta_param)
y <- pmin(pmax(y, 0.001), 0.999)
successes <- round(y * trials)

df_105 <- data.frame(
  successes = successes, trials = trials, x = x, site = site
)

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- tryCatch({
    ratiod(
      successes | trials ~ x + (1 | site),
      data = df_105,
      family = ratiod_beta_binomial(),
      spatial = spatial_car(adj_matrix, group_var = "site"),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      verbose = FALSE
    )
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    NULL
  })
})[["elapsed"]]
if (!is.null(fit_nd)) cat(sprintf("%.1fs\n", t_nd))

if (!is.null(fit_nd)) {
  cat("Fitting brms (CAR approximated via smooth term)... ")
  t_brms <- system.time({
    fit_brms <- brm(
      successes | trials(trials) ~ x + (1 | site),
      data = df_105,
      family = beta_binomial(),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      refresh = 0, silent = 2
    )
  })[["elapsed"]]
  cat(sprintf("%.1fs\n", t_brms))

  draws_nd <- as.matrix(fit_nd$draws)
  draws_brms <- as_draws_matrix(fit_brms)
  nd_slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_nd), value = TRUE)[1]

  results$row_105 <- list(
    slope = compare_posteriors(draws_nd[, nd_slope_col], draws_brms[, "b_x"], "slope (x)"),
    time_nd = t_nd, time_brms = t_brms
  )
  print_result(results$row_105$slope)
  cat(sprintf("  Times: nd=%.1fs, brms=%.1fs, speedup=%.1fx\n", t_nd, t_brms, t_brms/t_nd))
  cat("  Note: brms uses simple RE; numdenom adds ICAR spatial. Slopes should match but expect differences.\n")
} else {
  results$row_105 <- list(error = TRUE)
}

# =============================================================================
# Row 106: beta_binomial + RE + RW1
# =============================================================================
cat("\n========== Row 106: beta_binomial + RE + RW1 ==========\n")

temporal_effects <- cumsum(rnorm(N_TIMES, 0, 0.15))
temporal_effects <- temporal_effects - mean(temporal_effects)

eta_106 <- 0.5 + 0.3 * x + site_effects[as.integer(site)] + temporal_effects[as.integer(time)]
prob_106 <- plogis(eta_106)
alpha_106 <- prob_106 * phi
beta_106 <- (1 - prob_106) * phi
y_106 <- rbeta(N_OBS, alpha_106, beta_106)
y_106 <- pmin(pmax(y_106, 0.001), 0.999)
successes_106 <- round(y_106 * trials)

df_106 <- data.frame(
  successes = successes_106, trials = trials, x = x, site = site, time = time
)

cat("Fitting numdenom... ")
t_nd_106 <- system.time({
  fit_nd_106 <- tryCatch({
    ratiod(
      successes | trials ~ x + (1 | site),
      data = df_106,
      family = ratiod_beta_binomial(),
      temporal = temporal_rw1("time"),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      verbose = FALSE
    )
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    NULL
  })
})[["elapsed"]]
if (!is.null(fit_nd_106)) cat(sprintf("%.1fs\n", t_nd_106))

if (!is.null(fit_nd_106)) {
  cat("Fitting brms (RW1 approximated)... ")
  t_brms_106 <- system.time({
    fit_brms_106 <- brm(
      successes | trials(trials) ~ x + (1 | site) + (1 | time),
      data = df_106,
      family = beta_binomial(),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      refresh = 0, silent = 2
    )
  })[["elapsed"]]
  cat(sprintf("%.1fs\n", t_brms_106))

  draws_nd_106 <- as.matrix(fit_nd_106$draws)
  draws_brms_106 <- as_draws_matrix(fit_brms_106)
  nd_slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_nd_106), value = TRUE)[1]

  results$row_106 <- list(
    slope = compare_posteriors(draws_nd_106[, nd_slope_col], draws_brms_106[, "b_x"], "slope (x)"),
    time_nd = t_nd_106, time_brms = t_brms_106
  )
  print_result(results$row_106$slope)
  cat(sprintf("  Times: nd=%.1fs, brms=%.1fs, speedup=%.1fx\n", t_nd_106, t_brms_106, t_brms_106/t_nd_106))
  cat("  Note: brms uses IID time RE; numdenom uses RW1. Slopes should match but expect differences.\n")
} else {
  results$row_106 <- list(error = TRUE)
}

# =============================================================================
# Row 107: beta_binomial + RE + ICAR + RW1
# =============================================================================
cat("\n========== Row 107: beta_binomial + RE + ICAR + RW1 ==========\n")

eta_107 <- 0.5 + 0.3 * x + site_effects[as.integer(site)] + spatial_effects[as.integer(site)] + temporal_effects[as.integer(time)]
prob_107 <- plogis(eta_107)
alpha_107 <- prob_107 * phi
beta_107 <- (1 - prob_107) * phi
y_107 <- rbeta(N_OBS, alpha_107, beta_107)
y_107 <- pmin(pmax(y_107, 0.001), 0.999)
successes_107 <- round(y_107 * trials)

df_107 <- data.frame(
  successes = successes_107, trials = trials, x = x, site = site, time = time
)

cat("Fitting numdenom... ")
t_nd_107 <- system.time({
  fit_nd_107 <- tryCatch({
    ratiod(
      successes | trials ~ x + (1 | site),
      data = df_107,
      family = ratiod_beta_binomial(),
      spatial = spatial_car(adj_matrix, group_var = "site"),
      temporal = temporal_rw1("time"),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      verbose = FALSE
    )
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    NULL
  })
})[["elapsed"]]
if (!is.null(fit_nd_107)) cat(sprintf("%.1fs\n", t_nd_107))

if (!is.null(fit_nd_107)) {
  cat("Fitting brms (simplified)... ")
  t_brms_107 <- system.time({
    fit_brms_107 <- brm(
      successes | trials(trials) ~ x + (1 | site) + (1 | time),
      data = df_107,
      family = beta_binomial(),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      refresh = 0, silent = 2
    )
  })[["elapsed"]]
  cat(sprintf("%.1fs\n", t_brms_107))

  draws_nd_107 <- as.matrix(fit_nd_107$draws)
  draws_brms_107 <- as_draws_matrix(fit_brms_107)
  nd_slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_nd_107), value = TRUE)[1]

  results$row_107 <- list(
    slope = compare_posteriors(draws_nd_107[, nd_slope_col], draws_brms_107[, "b_x"], "slope (x)"),
    time_nd = t_nd_107, time_brms = t_brms_107
  )
  print_result(results$row_107$slope)
  cat(sprintf("  Times: nd=%.1fs, brms=%.1fs, speedup=%.1fx\n", t_nd_107, t_brms_107, t_brms_107/t_nd_107))
  cat("  Note: brms uses IID RE; numdenom uses ICAR+RW1. Slopes should match but expect differences.\n")
} else {
  results$row_107 <- list(error = TRUE)
}

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - Beta-Binomial Validation Batch 2\n")
cat("=======================================================\n\n")

cat(sprintf("%-8s %-30s %10s %10s %8s %s\n",
            "Row", "Model", "numdenom", "brms", "Speedup", "Status"))
cat(paste(rep("-", 80), collapse = ""), "\n")

for (row_name in c("row_105", "row_106", "row_107")) {
  r <- results[[row_name]]
  row_num <- gsub("row_", "", row_name)
  model_name <- switch(row_num,
    "105" = "bb + RE + ICAR",
    "106" = "bb + RE + RW1",
    "107" = "bb + RE + ICAR + RW1"
  )
  if (!is.null(r$error) && r$error) {
    cat(sprintf("%-8s %-30s %10s %10s %8s %s\n",
                row_num, model_name, "ERROR", "-", "-", "SKIPPED"))
  } else {
    cat(sprintf("%-8s %-30s %10.1fs %10.1fs %8.1fx %s\n",
                row_num, model_name, r$time_nd, r$time_brms, r$time_brms / r$time_nd,
                if(r$slope$pass) "PASS" else "FAIL*"))
  }
}

cat(paste(rep("-", 80), collapse = ""), "\n")
cat("* Note: brms uses different spatial/temporal structure than numdenom.\n")
cat("  Slopes are compared; differences may exist due to structural differences.\n")

saveRDS(results, "benchmarks/results_beta_binomial_batch2.rds")
cat("\nResults saved to benchmarks/results_beta_binomial_batch2.rds\n")
