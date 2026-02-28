# Validation of beta_binomial family against brms
# Rows 103-107: beta_binomial models
#
# beta_binomial family uses fixed trials, so brms comparison is VALID

library(numdenom)
library(brms)
library(posterior)

set.seed(42)

# Standard benchmark parameters
N_OBS <- 200
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2
N_SITES <- 20
N_TIMES <- 10

cat("=======================================================\n")
cat("Beta-Binomial Validation: Rows 103-107\n")
cat("=======================================================\n")
cat(sprintf("N=%d, iter=%d, warmup=%d, chains=%d\n\n", N_OBS, N_ITER, N_WARMUP, N_CHAINS))

# Helper function to compare posteriors
compare_posteriors <- function(nd_draws, brms_draws, param_name, threshold_se = 2) {
  nd_mean <- mean(nd_draws)
  nd_sd <- sd(nd_draws)
  brms_mean <- mean(brms_draws)
  brms_sd <- sd(brms_draws)

  se_combined <- sqrt(nd_sd^2 / length(nd_draws) + brms_sd^2 / length(brms_draws))
  diff <- abs(nd_mean - brms_mean)
  ratio <- diff / se_combined
  pass <- ratio < threshold_se

  list(
    param = param_name,
    nd_mean = nd_mean, nd_sd = nd_sd,
    brms_mean = brms_mean, brms_sd = brms_sd,
    diff = diff, ratio = ratio, pass = pass
  )
}

print_result <- function(result) {
  cat(sprintf("  %s: nd=%.4f (SD=%.4f), brms=%.4f (SD=%.4f), diff=%.4f (%.2f SE) => %s\n",
              result$param, result$nd_mean, result$nd_sd,
              result$brms_mean, result$brms_sd,
              result$diff, result$ratio, if(result$pass) "PASS" else "FAIL"))
}

# Setup data
site <- factor(rep(1:N_SITES, length.out = N_OBS))
time <- rep(1:N_TIMES, length.out = N_OBS)
x <- rnorm(N_OBS)
trials <- sample(10:50, N_OBS, replace = TRUE)

# Build adjacency matrix for ICAR
adj_list <- lapply(1:N_SITES, function(i) {
  neighbors <- c()
  if (i > 1) neighbors <- c(neighbors, i - 1)
  if (i < N_SITES) neighbors <- c(neighbors, i + 1)
  neighbors
})
adj_matrix <- matrix(0, N_SITES, N_SITES)
for (i in 1:N_SITES) {
  for (j in adj_list[[i]]) {
    adj_matrix[i, j] <- 1
  }
}

results <- list()

# =============================================================================
# Row 103: beta_binomial, no RE
# =============================================================================
cat("\n========== Row 103: beta_binomial (base) ==========\n")

eta <- 0.5 + 0.3 * x
prob <- plogis(eta)
phi <- 10  # concentration parameter
alpha <- prob * phi
beta_param <- (1 - prob) * phi
y <- rbeta(N_OBS, alpha, beta_param)
y <- pmin(pmax(y, 0.001), 0.999)  # Avoid exact 0 or 1
successes <- round(y * trials)

df_103 <- data.frame(
  successes = successes,
  trials = trials,
  x = x
)

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- tryCatch({
    ratiod(
      successes | trials ~ x,
      data = df_103,
      family = ratiod_beta_binomial(),
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
  cat("Fitting brms... ")
  t_brms <- system.time({
    fit_brms <- brm(
      successes | trials(trials) ~ x,
      data = df_103,
      family = beta_binomial(),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      refresh = 0, silent = 2
    )
  })[["elapsed"]]
  cat(sprintf("%.1fs\n", t_brms))

  draws_nd <- as.matrix(fit_nd$draws)
  draws_brms <- as_draws_matrix(fit_brms)

  nd_slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_nd), value = TRUE)[1]
  brms_slope_col <- "b_x"

  results$row_103 <- list(
    slope = compare_posteriors(
      draws_nd[, nd_slope_col],
      draws_brms[, brms_slope_col],
      "slope (x)"
    ),
    time_nd = t_nd,
    time_brms = t_brms
  )
  print_result(results$row_103$slope)
  cat(sprintf("  Times: nd=%.1fs, brms=%.1fs, speedup=%.1fx\n", t_nd, t_brms, t_brms/t_nd))
} else {
  results$row_103 <- list(error = TRUE)
  cat("  SKIPPED due to error\n")
}

# =============================================================================
# Row 104: beta_binomial + RE
# =============================================================================
cat("\n========== Row 104: beta_binomial + RE ==========\n")

site_effects <- rnorm(N_SITES, 0, 0.3)
eta_104 <- 0.5 + 0.3 * x + site_effects[as.integer(site)]
prob_104 <- plogis(eta_104)
alpha_104 <- prob_104 * phi
beta_104 <- (1 - prob_104) * phi
y_104 <- rbeta(N_OBS, alpha_104, beta_104)
y_104 <- pmin(pmax(y_104, 0.001), 0.999)
successes_104 <- round(y_104 * trials)

df_104 <- data.frame(
  successes = successes_104,
  trials = trials,
  x = x,
  site = site
)

cat("Fitting numdenom... ")
t_nd_104 <- system.time({
  fit_nd_104 <- tryCatch({
    ratiod(
      successes | trials ~ x + (1 | site),
      data = df_104,
      family = ratiod_beta_binomial(),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      verbose = FALSE
    )
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    NULL
  })
})[["elapsed"]]
if (!is.null(fit_nd_104)) cat(sprintf("%.1fs\n", t_nd_104))

if (!is.null(fit_nd_104)) {
  cat("Fitting brms... ")
  t_brms_104 <- system.time({
    fit_brms_104 <- brm(
      successes | trials(trials) ~ x + (1 | site),
      data = df_104,
      family = beta_binomial(),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      refresh = 0, silent = 2
    )
  })[["elapsed"]]
  cat(sprintf("%.1fs\n", t_brms_104))

  draws_nd_104 <- as.matrix(fit_nd_104$draws)
  draws_brms_104 <- as_draws_matrix(fit_brms_104)

  nd_slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_nd_104), value = TRUE)[1]

  results$row_104 <- list(
    slope = compare_posteriors(
      draws_nd_104[, nd_slope_col],
      draws_brms_104[, "b_x"],
      "slope (x)"
    ),
    time_nd = t_nd_104,
    time_brms = t_brms_104
  )
  print_result(results$row_104$slope)
  cat(sprintf("  Times: nd=%.1fs, brms=%.1fs, speedup=%.1fx\n", t_nd_104, t_brms_104, t_brms_104/t_nd_104))
} else {
  results$row_104 <- list(error = TRUE)
  cat("  SKIPPED due to error\n")
}

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - Beta-Binomial Validation\n")
cat("=======================================================\n\n")

cat(sprintf("%-8s %-30s %10s %10s %8s %s\n",
            "Row", "Model", "numdenom", "brms", "Speedup", "Status"))
cat(paste(rep("-", 80), collapse = ""), "\n")

for (row_name in names(results)) {
  r <- results[[row_name]]
  if (!is.null(r$error) && r$error) {
    cat(sprintf("%-8s %-30s %10s %10s %8s %s\n",
                gsub("row_", "", row_name), "beta_binomial", "ERROR", "-", "-", "SKIPPED"))
  } else {
    row_num <- gsub("row_", "", row_name)
    model_name <- switch(row_num,
      "103" = "bb (base)",
      "104" = "bb + RE",
      "105" = "bb + RE + ICAR",
      "106" = "bb + RE + RW1",
      "107" = "bb + RE + ICAR + RW1",
      "unknown"
    )
    cat(sprintf("%-8s %-30s %10.1fs %10.1fs %8.1fx %s\n",
                row_num, model_name, r$time_nd, r$time_brms, r$time_brms / r$time_nd,
                if(r$slope$pass) "PASS" else "FAIL"))
  }
}

cat(paste(rep("-", 80), collapse = ""), "\n")

saveRDS(results, "benchmarks/results_beta_binomial_validation.rds")
cat("\nResults saved to benchmarks/results_beta_binomial_validation.rds\n")

cat("\n\nUPDATE gradient_methods.md:\n")
for (row_name in names(results)) {
  r <- results[[row_name]]
  if (is.null(r$error) && r$slope$pass) {
    cat(sprintf("  Row %s: PASS - add '✓Stan' to Notes\n", gsub("row_", "", row_name)))
  }
}
