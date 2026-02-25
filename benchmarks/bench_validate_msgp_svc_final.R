# Final simulation-based validation for MSGP and SVC models (v3)
# MSGP: Rows 9, 23, 39, 53, 69, 85
# SVC:  Rows 26, 56, 88
#
# KEY INSIGHT: MSGP at N=500 with unique coords creates 1000 spatial params
# (2 × 500), causing HMC to fail (ESS=1, Rhat>2). Solution: use site-based
# data with duplicate coords. MSGP supports obs_to_loc mapping.
#
# Strategy:
#   MSGP: 30 sites × 10 obs/site = 300 total, 60 spatial params (tractable)
#   SVC:  100 obs with unique coords, terms=2 (spatially varying slope)
#         SVC doesn't support obs_to_loc, so we need unique coords.
#
# iter=2000, warmup=1000, chains=1

library(numdenom)
library(posterior)

set.seed(42)

# --- Parameters ---
N_SITES      <- 30     # MSGP: number of unique sites
OBS_PER_SITE <- 10     # MSGP: observations per site
N_MSGP       <- N_SITES * OBS_PER_SITE  # 300 total
N_SVC        <- 100    # SVC: unique coords per obs
N_ITER       <- 2000
N_WARMUP     <- 1000
N_CHAINS     <- 1
N_TIMES      <- 10     # temporal periods for MSGP+RW1

cat("================================================================\n")
cat("MSGP + SVC: Final Simulation-Based Validation (v3)\n")
cat("================================================================\n")
cat(sprintf("MSGP: %d sites x %d obs = %d total, 2x%d = %d spatial params\n",
            N_SITES, OBS_PER_SITE, N_MSGP, N_SITES, 2 * N_SITES))
cat(sprintf("SVC:  %d obs, unique coords, terms=2 (varying slope)\n", N_SVC))
cat(sprintf("iter=%d, warmup=%d, chains=%d\n", N_ITER, N_WARMUP, N_CHAINS))
cat(sprintf("Started: %s\n\n", Sys.time()))

# --- Helpers ---
check_recovery <- function(draws, true_value, param_name, threshold_sd = 2) {
  post_mean <- mean(draws)
  post_sd <- sd(draws)
  diff_sd <- abs(post_mean - true_value) / post_sd
  in_ci <- true_value >= quantile(draws, 0.025) && true_value <= quantile(draws, 0.975)
  list(param = param_name, true = true_value, mean = post_mean, sd = post_sd,
       diff_sd = diff_sd, in_ci = in_ci, pass = diff_sd < threshold_sd)
}

print_recovery <- function(r) {
  cat(sprintf("  %s: true=%.3f, post=%.3f (SD=%.3f), %.2f SD => %s [95%%CI: %s]\n",
              r$param, r$true, r$mean, r$sd, r$diff_sd,
              if (r$pass) "PASS" else "FAIL",
              if (r$in_ci) "contains true" else "MISSES true"))
}

get_diagnostics <- function(fit) {
  diag <- list(n_div = NA, min_ess = NA, max_rhat = NA)
  tryCatch({
    if (!is.null(fit$diagnostics)) {
      if (!is.null(fit$diagnostics$num_divergent))
        diag$n_div <- sum(fit$diagnostics$num_divergent)
    }
    draws_summary <- posterior::summarise_draws(fit$draws)
    if ("ess_bulk" %in% names(draws_summary))
      diag$min_ess <- min(draws_summary$ess_bulk, na.rm = TRUE)
    if ("rhat" %in% names(draws_summary))
      diag$max_rhat <- max(draws_summary$rhat, na.rm = TRUE)
  }, error = function(e) NULL)
  diag
}

print_diagnostics <- function(diag) {
  cat(sprintf("  Diagnostics: div=%s, min_ESS=%.0f, max_Rhat=%.3f\n",
              if (is.na(diag$n_div)) "?" else as.character(diag$n_div),
              if (is.na(diag$min_ess)) NA else diag$min_ess,
              if (is.na(diag$max_rhat)) NA else diag$max_rhat))
}

results <- list()

# True parameters
true_intercept <- 1.0
true_slope <- 0.3

# =============================================================================
# MSGP DATA GENERATION - SITE-BASED (duplicate coords per site)
# =============================================================================
# 30 unique site locations, each observed 10 times
set.seed(123)
site_x <- runif(N_SITES, 0, 10)
site_y <- runif(N_SITES, 0, 10)

# Each obs belongs to a site -> duplicate coordinates
site_id <- rep(1:N_SITES, each = OBS_PER_SITE)
coord_x_msgp <- site_x[site_id]
coord_y_msgp <- site_y[site_id]
coords_sites <- cbind(site_x, site_y)

x_msgp <- rnorm(N_MSGP)

# Generate MSGP effects at SITE level (30 values, repeated for obs within site)
generate_msgp_effects <- function(coords, sigma_fine, sigma_coarse, ls_fine, ls_coarse) {
  n <- nrow(coords)
  dist_mat <- as.matrix(dist(coords))
  # Fine scale
  cov_fine <- sigma_fine^2 * exp(-dist_mat / ls_fine)
  diag(cov_fine) <- diag(cov_fine) + 1e-6
  L_fine <- chol(cov_fine)
  fine <- as.vector(t(L_fine) %*% rnorm(n))
  # Coarse scale
  cov_coarse <- sigma_coarse^2 * exp(-dist_mat / ls_coarse)
  diag(cov_coarse) <- diag(cov_coarse) + 1e-6
  L_coarse <- chol(cov_coarse)
  coarse <- as.vector(t(L_coarse) %*% rnorm(n))
  effects <- fine + coarse
  effects - mean(effects)
}

# 30 site-level MSGP effects -> expanded to 300 obs via site_id
msgp_site_effects <- generate_msgp_effects(coords_sites, 0.3, 0.4, 1.5, 5.0)
spatial_effects_msgp <- msgp_site_effects[site_id]

# RW1 temporal effects for MSGP+RW1 rows
generate_rw1 <- function(n, sigma) {
  gamma <- cumsum(rnorm(n, 0, sigma))
  gamma - mean(gamma)
}
set.seed(456)
temporal_effects <- generate_rw1(N_TIMES, 0.15)
time_idx_msgp <- rep(1:N_TIMES, length.out = N_MSGP)

# Shared effort/trials
effort_msgp <- rgamma(N_MSGP, 5, 1)
trials_msgp <- sample(20:50, N_MSGP, replace = TRUE)

cat(sprintf("MSGP: %d sites, %d total obs, spatial SD=%.3f\n",
            N_SITES, N_MSGP, sd(spatial_effects_msgp)))

# =============================================================================
# SVC DATA GENERATION - UNIQUE COORDS (SVC doesn't support obs_to_loc)
# =============================================================================
set.seed(789)
coord_x_svc <- runif(N_SVC, 0, 10)
coord_y_svc <- runif(N_SVC, 0, 10)
x_svc <- rnorm(N_SVC)

generate_gp_effects <- function(coords, sigma, lengthscale) {
  n <- nrow(coords)
  dist_mat <- as.matrix(dist(coords))
  cov_mat <- sigma^2 * exp(-dist_mat / lengthscale)
  diag(cov_mat) <- diag(cov_mat) + 1e-6
  L <- chol(cov_mat)
  as.vector(t(L) %*% rnorm(n))
}

# Spatially varying slope: total slope at location s = 0.3 + w(s)
svc_slope <- generate_gp_effects(cbind(coord_x_svc, coord_y_svc), 0.15, 3.0)

effort_svc <- rgamma(N_SVC, 5, 1)
trials_svc <- sample(20:50, N_SVC, replace = TRUE)

cat(sprintf("SVC:  %d obs, unique coords, SVC slope SD=%.3f\n\n",
            N_SVC, sd(svc_slope)))

# =============================================================================
# Helper: fit and check
# =============================================================================
fit_and_check <- function(row_num, formula_call, description) {
  cat(sprintf("\n========== Row %d: %s ==========\n", row_num, description))
  cat(sprintf("True: slope=%.2f\n", true_slope))
  cat(sprintf("Fitting (started %s)... ", format(Sys.time(), "%H:%M:%S")))
  flush.console()

  elapsed <- system.time({
    fit <- tryCatch(formula_call(), error = function(e) {
      cat(sprintf("ERROR: %s\n", e$message))
      NULL
    })
  })[["elapsed"]]

  if (is.null(fit)) {
    cat("FAILED\n")
    return(list(row = row_num, error = TRUE, time = elapsed))
  }

  cat(sprintf("%.1fs (%.1f min)\n", elapsed, elapsed / 60))

  draws <- as.matrix(fit$draws)
  slope_col <- grep("^beta_num\\[2\\]$", colnames(draws), value = TRUE)[1]

  if (is.na(slope_col)) {
    cat("  WARNING: beta_num[2] not found in draws\n")
    cat("  Available columns:", paste(head(colnames(draws), 20), collapse = ", "), "\n")
    return(list(row = row_num, error = TRUE, time = elapsed, msg = "no beta_num[2]"))
  }

  slope_check <- check_recovery(draws[, slope_col], true_slope, "slope")
  print_recovery(slope_check)

  diag <- get_diagnostics(fit)
  print_diagnostics(diag)

  list(row = row_num, slope = slope_check, diag = diag, time = elapsed,
       pass = slope_check$pass, error = FALSE)
}

# =============================================================================
# MSGP MODELS - No temporal (Rows 9, 39, 69)
# =============================================================================

# Row 9: poisson_gamma + MSGP
eta_9 <- true_intercept + true_slope * x_msgp + spatial_effects_msgp
count_9 <- rpois(N_MSGP, exp(eta_9) * effort_msgp)
df_9 <- data.frame(count = count_9, effort = effort_msgp, x = x_msgp,
                   coord_x = coord_x_msgp, coord_y = coord_y_msgp)

results$row_9 <- fit_and_check(9, function() {
  ratiod(count | effort ~ x, data = df_9, family = ratiod_poisson_gamma(),
         spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
}, "poisson_gamma + MSGP (30 sites x 10 obs)")

# Save intermediate results after each model
saveRDS(results, "benchmarks/results_msgp_svc_final.rds")

# Row 39: negbin_negbin + MSGP
eta_num_39 <- true_intercept + true_slope * x_msgp + spatial_effects_msgp
eta_denom_39 <- 0.5 + 0.2 * x_msgp + spatial_effects_msgp * 0.8
num_39 <- rnbinom(N_MSGP, size = 5, mu = exp(eta_num_39))
denom_39 <- rnbinom(N_MSGP, size = 5, mu = exp(eta_denom_39))
denom_39[denom_39 == 0] <- 1
df_39 <- data.frame(num = num_39, denom = denom_39, x = x_msgp,
                    coord_x = coord_x_msgp, coord_y = coord_y_msgp)

results$row_39 <- fit_and_check(39, function() {
  ratiod(num | denom ~ x, data = df_39, family = ratiod_negbin_negbin(),
         spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
}, "negbin_negbin + MSGP (30 sites x 10 obs)")

saveRDS(results, "benchmarks/results_msgp_svc_final.rds")

# Row 69: binomial + MSGP
eta_69 <- true_intercept + true_slope * x_msgp + spatial_effects_msgp
prob_69 <- plogis(eta_69)
successes_69 <- rbinom(N_MSGP, trials_msgp, prob_69)
df_69 <- data.frame(successes = successes_69, trials = trials_msgp, x = x_msgp,
                    coord_x = coord_x_msgp, coord_y = coord_y_msgp)

results$row_69 <- fit_and_check(69, function() {
  ratiod(successes | trials ~ x, data = df_69, family = ratiod_binomial(),
         spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
}, "binomial + MSGP (30 sites x 10 obs)")

saveRDS(results, "benchmarks/results_msgp_svc_final.rds")

# =============================================================================
# MSGP + RW1 MODELS (Rows 23, 53, 85)
# =============================================================================

# Row 23: poisson_gamma + MSGP + RW1
eta_23 <- true_intercept + true_slope * x_msgp + spatial_effects_msgp + temporal_effects[time_idx_msgp]
count_23 <- rpois(N_MSGP, exp(eta_23) * effort_msgp)
df_23 <- data.frame(count = count_23, effort = effort_msgp, x = x_msgp,
                    coord_x = coord_x_msgp, coord_y = coord_y_msgp,
                    time = time_idx_msgp)

results$row_23 <- fit_and_check(23, function() {
  ratiod(count | effort ~ x, data = df_23, family = ratiod_poisson_gamma(),
         spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
         temporal = temporal_rw1(time_var = "time"),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
}, "poisson_gamma + MSGP + RW1 (30 sites x 10 obs)")

saveRDS(results, "benchmarks/results_msgp_svc_final.rds")

# Row 53: negbin_negbin + MSGP + RW1
eta_num_53 <- true_intercept + true_slope * x_msgp + spatial_effects_msgp + temporal_effects[time_idx_msgp]
eta_denom_53 <- 0.5 + 0.2 * x_msgp + spatial_effects_msgp * 0.8 + temporal_effects[time_idx_msgp] * 0.6
num_53 <- rnbinom(N_MSGP, size = 5, mu = exp(eta_num_53))
denom_53 <- rnbinom(N_MSGP, size = 5, mu = exp(eta_denom_53))
denom_53[denom_53 == 0] <- 1
df_53 <- data.frame(num = num_53, denom = denom_53, x = x_msgp,
                    coord_x = coord_x_msgp, coord_y = coord_y_msgp,
                    time = time_idx_msgp)

results$row_53 <- fit_and_check(53, function() {
  ratiod(num | denom ~ x, data = df_53, family = ratiod_negbin_negbin(),
         spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
         temporal = temporal_rw1(time_var = "time"),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
}, "negbin_negbin + MSGP + RW1 (30 sites x 10 obs)")

saveRDS(results, "benchmarks/results_msgp_svc_final.rds")

# Row 85: binomial + MSGP + RW1
eta_85 <- true_intercept + true_slope * x_msgp + spatial_effects_msgp + temporal_effects[time_idx_msgp]
prob_85 <- plogis(eta_85)
successes_85 <- rbinom(N_MSGP, trials_msgp, prob_85)
df_85 <- data.frame(successes = successes_85, trials = trials_msgp, x = x_msgp,
                    coord_x = coord_x_msgp, coord_y = coord_y_msgp,
                    time = time_idx_msgp)

results$row_85 <- fit_and_check(85, function() {
  ratiod(successes | trials ~ x, data = df_85, family = ratiod_binomial(),
         spatial = spatial_multiscale(coords = c("coord_x", "coord_y")),
         temporal = temporal_rw1(time_var = "time"),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
}, "binomial + MSGP + RW1 (30 sites x 10 obs)")

saveRDS(results, "benchmarks/results_msgp_svc_final.rds")

# =============================================================================
# MSGP INTERMEDIATE SUMMARY
# =============================================================================
cat("\n================================================================\n")
cat("MSGP INTERMEDIATE SUMMARY\n")
cat("================================================================\n")
msgp_rows <- c("row_9", "row_23", "row_39", "row_53", "row_69", "row_85")
for (rn in msgp_rows) {
  r <- results[[rn]]
  if (is.null(r)) { cat(sprintf("  %s: SKIPPED\n", rn)); next }
  if (r$error) { cat(sprintf("  %s: ERROR\n", rn)); next }
  cat(sprintf("  Row %d: %s (%.2f SD, %.1fs = %.1f min, div=%s)\n",
              r$row, if (r$pass) "PASS" else "FAIL", r$slope$diff_sd,
              r$time, r$time / 60,
              if (is.na(r$diag$n_div)) "?" else as.character(r$diag$n_div)))
}
cat(sprintf("MSGP done at %s\n\n", Sys.time()))

# =============================================================================
# SVC MODELS (Rows 26, 56, 88) - Spatially varying SLOPE (terms=2)
# =============================================================================
# DGP: eta = intercept + (slope + w(s)) * x, where w(s) ~ GP(0, K)
# Model: spatial_svc(terms = 2) → slope of x varies spatially
# Check: beta_num[2] recovers global mean slope (0.3)
#
# SVC does NOT support obs_to_loc → unique coords required
# N=100 gives 100 SVC weights → manageable for HMC

# Row 26: poisson_gamma + SVC
eta_26 <- true_intercept + (true_slope + svc_slope) * x_svc
count_26 <- rpois(N_SVC, exp(eta_26) * effort_svc)
df_26 <- data.frame(count = count_26, effort = effort_svc, x = x_svc,
                    coord_x = coord_x_svc, coord_y = coord_y_svc)

results$row_26 <- fit_and_check(26, function() {
  ratiod(count | effort ~ x, data = df_26, family = ratiod_poisson_gamma(),
         spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 2),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
}, "poisson_gamma + SVC (terms=2, varying slope, N=100)")

# Fallback to terms=1 if terms=2 errors
if (!is.null(results$row_26$error) && results$row_26$error) {
  cat("\n  >>> terms=2 failed. Trying terms=1 (varying intercept) with matching DGP...\n")
  svc_int <- generate_gp_effects(cbind(coord_x_svc, coord_y_svc), 0.25, 3.0)
  eta_26b <- (true_intercept + svc_int) + true_slope * x_svc
  count_26b <- rpois(N_SVC, exp(eta_26b) * effort_svc)
  df_26b <- data.frame(count = count_26b, effort = effort_svc, x = x_svc,
                       coord_x = coord_x_svc, coord_y = coord_y_svc)
  results$row_26 <- fit_and_check(26, function() {
    ratiod(count | effort ~ x, data = df_26b, family = ratiod_poisson_gamma(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 1),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, "poisson_gamma + SVC (terms=1, varying intercept FALLBACK)")
}
saveRDS(results, "benchmarks/results_msgp_svc_final.rds")

# Row 56: negbin_negbin + SVC
eta_num_56 <- true_intercept + (true_slope + svc_slope) * x_svc
eta_denom_56 <- 0.5 + 0.2 * x_svc
num_56 <- rnbinom(N_SVC, size = 5, mu = exp(eta_num_56))
denom_56 <- rnbinom(N_SVC, size = 5, mu = exp(eta_denom_56))
denom_56[denom_56 == 0] <- 1
df_56 <- data.frame(num = num_56, denom = denom_56, x = x_svc,
                    coord_x = coord_x_svc, coord_y = coord_y_svc)

results$row_56 <- fit_and_check(56, function() {
  ratiod(num | denom ~ x, data = df_56, family = ratiod_negbin_negbin(),
         spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 2),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
}, "negbin_negbin + SVC (terms=2, varying slope, N=100)")

if (!is.null(results$row_56$error) && results$row_56$error) {
  cat("\n  >>> terms=2 failed. Trying terms=1 (varying intercept) with matching DGP...\n")
  svc_int <- generate_gp_effects(cbind(coord_x_svc, coord_y_svc), 0.25, 3.0)
  eta_num_56b <- (true_intercept + svc_int) + true_slope * x_svc
  num_56b <- rnbinom(N_SVC, size = 5, mu = exp(eta_num_56b))
  df_56b <- data.frame(num = num_56b, denom = denom_56, x = x_svc,
                       coord_x = coord_x_svc, coord_y = coord_y_svc)
  results$row_56 <- fit_and_check(56, function() {
    ratiod(num | denom ~ x, data = df_56b, family = ratiod_negbin_negbin(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 1),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, "negbin_negbin + SVC (terms=1, varying intercept FALLBACK)")
}
saveRDS(results, "benchmarks/results_msgp_svc_final.rds")

# Row 88: binomial + SVC
eta_88 <- true_intercept + (true_slope + svc_slope) * x_svc
prob_88 <- plogis(eta_88)
successes_88 <- rbinom(N_SVC, trials_svc, prob_88)
df_88 <- data.frame(successes = successes_88, trials = trials_svc, x = x_svc,
                    coord_x = coord_x_svc, coord_y = coord_y_svc)

results$row_88 <- fit_and_check(88, function() {
  ratiod(successes | trials ~ x, data = df_88, family = ratiod_binomial(),
         spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 2),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
}, "binomial + SVC (terms=2, varying slope, N=100)")

if (!is.null(results$row_88$error) && results$row_88$error) {
  cat("\n  >>> terms=2 failed. Trying terms=1 (varying intercept) with matching DGP...\n")
  svc_int <- generate_gp_effects(cbind(coord_x_svc, coord_y_svc), 0.25, 3.0)
  eta_88b <- (true_intercept + svc_int) + true_slope * x_svc
  prob_88b <- plogis(eta_88b)
  successes_88b <- rbinom(N_SVC, trials_svc, prob_88b)
  df_88b <- data.frame(successes = successes_88b, trials = trials_svc, x = x_svc,
                       coord_x = coord_x_svc, coord_y = coord_y_svc)
  results$row_88 <- fit_and_check(88, function() {
    ratiod(successes | trials ~ x, data = df_88b, family = ratiod_binomial(),
           spatial = spatial_svc(coords = c("coord_x", "coord_y"), terms = 1),
           iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  }, "binomial + SVC (terms=1, varying intercept FALLBACK)")
}
saveRDS(results, "benchmarks/results_msgp_svc_final.rds")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
cat("\n================================================================\n")
cat("FINAL SUMMARY - MSGP + SVC Simulation Validation (v3)\n")
cat("================================================================\n\n")

all_rows <- c("row_9", "row_23", "row_39", "row_53", "row_69", "row_85",
              "row_26", "row_56", "row_88")

total_time <- 0
n_pass <- 0
n_fail <- 0
n_error <- 0

for (rn in all_rows) {
  r <- results[[rn]]
  if (is.null(r)) {
    cat(sprintf("Row %-3s: SKIPPED\n", gsub("row_", "", rn)))
    next
  }
  total_time <- total_time + r$time
  if (r$error) {
    n_error <- n_error + 1
    cat(sprintf("Row %-3s: ERROR (%.1fs)\n", gsub("row_", "", rn), r$time))
  } else if (r$pass) {
    n_pass <- n_pass + 1
    cat(sprintf("Row %-3s: PASS  (%.2f SD, %.1fs, div=%s, ESS=%.0f, Rhat=%.3f)\n",
                gsub("row_", "", rn), r$slope$diff_sd, r$time,
                if (is.na(r$diag$n_div)) "?" else as.character(r$diag$n_div),
                if (is.na(r$diag$min_ess)) NA else r$diag$min_ess,
                if (is.na(r$diag$max_rhat)) NA else r$diag$max_rhat))
  } else {
    n_fail <- n_fail + 1
    cat(sprintf("Row %-3s: FAIL  (%.2f SD, %.1fs, post=%.3f, true=%.3f, ESS=%.0f)\n",
                gsub("row_", "", rn), r$slope$diff_sd, r$time,
                r$slope$mean, r$slope$true,
                if (is.na(r$diag$min_ess)) NA else r$diag$min_ess))
  }
}

cat(sprintf("\n%d PASS / %d FAIL / %d ERROR out of %d rows\n",
            n_pass, n_fail, n_error, length(all_rows)))
cat(sprintf("Total time: %.0fs = %.1f min = %.1f hours\n",
            total_time, total_time / 60, total_time / 3600))
cat(sprintf("Finished: %s\n", Sys.time()))
cat("\nResults saved to benchmarks/results_msgp_svc_final.rds\n")
