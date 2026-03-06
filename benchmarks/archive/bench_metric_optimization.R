# Benchmark: Metric Optimization Validation
# Tests rows 14, 27, 38, 57 — the rows where numdenom lost to Stan
# due to blanket DENSE metric selection
#
# After the metric change: has_re, has_temporal, is_hsgp, is_temporal_gp,
# has_multiscale_temporal, has_tvc removed from needs_dense

library(numdenom)

set.seed(42)

cat("\n", strrep("=", 70), "\n")
cat("METRIC OPTIMIZATION BENCHMARK\n")
cat("Testing rows 14, 27, 38, 57 with verbose=TRUE\n")
cat(strrep("=", 70), "\n\n")

N_OBS <- 500
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1
N_SITES <- 50
N_TIMES <- 20

# Common data setup
site <- factor(rep(1:N_SITES, length.out = N_OBS))
time_idx <- rep(1:N_TIMES, length.out = N_OBS)
x <- rnorm(N_OBS)

results <- list()

# =============================================================================
# Row 27: poisson_gamma + RE + TVC
# =============================================================================
cat("\n", strrep("=", 60), "\n")
cat("ROW 27: poisson_gamma + RE + TVC\n")
cat(strrep("=", 60), "\n\n")

eta_num_27 <- 2.0 + 0.3 * x
eta_denom_27 <- 1.5 + 0.2 * x

df_27 <- data.frame(
  y = rpois(N_OBS, exp(eta_num_27)),
  effort = rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_denom_27)),
  x = x, site = site, time = factor(time_idx)
)

cat("Fitting numdenom (H gradient, verbose)...\n")
t_27 <- system.time({
  fit_27 <- ratiod(
    y | effort ~ x + (1 | site),
    data = df_27,
    family = ratiod_poisson_gamma(),
    temporal = temporal_tvc(time_var = "time", terms = "x"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    gradient_mode = "H",
    verbose = TRUE, refresh = 0
  )
})["elapsed"]

summ_27 <- summary(fit_27)
cat(sprintf("\n  Time: %.1fs (was 68.0s before optimization)\n", t_27))
cat(sprintf("  Divergent: %d\n", summ_27$diagnostics$divergent))
cat(sprintf("  beta_num intercept: %.3f (true: 2.0)\n", summ_27$fixed[1, "mean"]))

results$row_27 <- list(time = t_27,
                       divergent = summ_27$diagnostics$divergent,
                       beta_intercept = summ_27$fixed[1, "mean"])

# =============================================================================
# Row 57: negbin_negbin + RE + TVC
# =============================================================================
cat("\n", strrep("=", 60), "\n")
cat("ROW 57: negbin_negbin + RE + TVC\n")
cat(strrep("=", 60), "\n\n")

y_num_57 <- rnbinom(N_OBS, size = 5, mu = exp(eta_num_27))
y_denom_57 <- rnbinom(N_OBS, size = 5, mu = exp(eta_denom_27))
y_denom_57[y_denom_57 == 0] <- 1L

df_57 <- data.frame(
  y = y_num_57,
  effort = y_denom_57,
  x = x, site = site, time = factor(time_idx)
)

cat("Fitting numdenom (H gradient, verbose)...\n")
t_57 <- system.time({
  fit_57 <- ratiod(
    y | effort ~ x + (1 | site),
    data = df_57,
    family = ratiod_negbin_negbin(),
    temporal = temporal_tvc(time_var = "time", terms = "x"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    gradient_mode = "H",
    verbose = TRUE, refresh = 0
  )
})["elapsed"]

summ_57 <- summary(fit_57)
cat(sprintf("\n  Time: %.1fs (was 183.2s before optimization)\n", t_57))
cat(sprintf("  Divergent: %d\n", summ_57$diagnostics$divergent))
cat(sprintf("  beta_num intercept: %.3f (true: 2.0)\n", summ_57$fixed[1, "mean"]))

results$row_57 <- list(time = t_57,
                       divergent = summ_57$diagnostics$divergent,
                       beta_intercept = summ_57$fixed[1, "mean"])

# =============================================================================
# Row 38: negbin_negbin + RE + HSGP
# =============================================================================
cat("\n", strrep("=", 60), "\n")
cat("ROW 38: negbin_negbin + RE + HSGP\n")
cat(strrep("=", 60), "\n\n")

n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = seq(0, 1, length.out = n_side),
                    lat = seq(0, 1, length.out = n_side))[1:N_SITES, ]
site_int <- as.integer(site)
lon <- grid$lon[site_int]
lat <- grid$lat[site_int]

eta_num_38 <- 2.0 + 0.3 * x
eta_denom_38 <- 4.0 + 0.2 * x

y_num_38 <- rnbinom(N_OBS, mu = exp(eta_num_38), size = 5)
y_denom_38 <- rnbinom(N_OBS, mu = exp(eta_denom_38), size = 10)
y_denom_38[y_denom_38 == 0] <- 1L

df_38 <- data.frame(
  y = y_num_38,
  denom = y_denom_38,
  x = x, site = site, lon = lon, lat = lat
)

cat("Fitting numdenom (H gradient, verbose)...\n")
t_38 <- system.time({
  fit_38 <- ratiod(
    y | denom ~ x + (1 | site),
    data = df_38,
    family = ratiod_negbin_negbin(),
    spatial = spatial_hsgp(coords = c("lon", "lat"), m = 6),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    gradient_mode = "H",
    verbose = TRUE, refresh = 0
  )
})["elapsed"]

summ_38 <- summary(fit_38)
cat(sprintf("\n  Time: %.1fs (was 210.4s before optimization)\n", t_38))
cat(sprintf("  Divergent: %d\n", summ_38$diagnostics$divergent))
cat(sprintf("  beta_num intercept: %.3f (true: 2.0)\n", summ_38$fixed[1, "mean"]))

results$row_38 <- list(time = t_38,
                       divergent = summ_38$diagnostics$divergent,
                       beta_intercept = summ_38$fixed[1, "mean"])

# =============================================================================
# Row 14: poisson_gamma + RE + temporal GP
# =============================================================================
cat("\n", strrep("=", 60), "\n")
cat("ROW 14: poisson_gamma + RE + temporal GP\n")
cat(strrep("=", 60), "\n\n")

# Temporal GP effects
true_sigma_gp <- 0.5
true_phi_gp <- 2.0
dist_mat <- as.matrix(dist(1:N_TIMES))
K <- true_sigma_gp^2 * exp(-dist_mat / true_phi_gp)
gp_effects <- MASS::mvrnorm(1, rep(0, N_TIMES), K + diag(1e-6, N_TIMES))
gp_by_obs <- gp_effects[time_idx]

eta_num_14 <- 2.0 + 0.3 * x + gp_by_obs
eta_denom_14 <- 1.5 + 0.2 * x + gp_by_obs

df_14 <- data.frame(
  y = rpois(N_OBS, exp(eta_num_14)),
  effort = rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_denom_14)),
  x = x, site = site, time = time_idx
)
df_14$effort[df_14$effort < 0.01] <- 0.01

cat("Fitting numdenom (H gradient, verbose)...\n")
t_14 <- system.time({
  fit_14 <- ratiod(
    y | effort ~ x + (1 | site),
    data = df_14,
    family = ratiod_poisson_gamma(),
    temporal = temporal_gp(time_var = "time", cov = "exponential"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    gradient_mode = "H",
    verbose = TRUE, refresh = 0
  )
})["elapsed"]

summ_14 <- summary(fit_14)
cat(sprintf("\n  Time: %.1fs (was 84.6s before optimization)\n", t_14))
cat(sprintf("  Divergent: %d\n", summ_14$diagnostics$divergent))
cat(sprintf("  beta_num intercept: %.3f (true: 2.0)\n", summ_14$fixed[1, "mean"]))

results$row_14 <- list(time = t_14,
                       divergent = summ_14$diagnostics$divergent,
                       beta_intercept = summ_14$fixed[1, "mean"])

# =============================================================================
# Regression check: Row 2 (pg + RE, no spatial/temporal)
# =============================================================================
cat("\n", strrep("=", 60), "\n")
cat("REGRESSION CHECK: Row 2 (pg + RE only)\n")
cat(strrep("=", 60), "\n\n")

df_2 <- data.frame(
  y = rpois(N_OBS, exp(eta_num_27)),
  effort = rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_denom_27)),
  x = x, site = site
)

cat("Fitting numdenom (verbose)...\n")
t_2 <- system.time({
  fit_2 <- ratiod(
    y | effort ~ x + (1 | site),
    data = df_2,
    family = ratiod_poisson_gamma(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    gradient_mode = "H",
    verbose = TRUE, refresh = 0
  )
})["elapsed"]

summ_2 <- summary(fit_2)
cat(sprintf("\n  Time: %.1fs (was 1.25s, should stay ~1-2s)\n", t_2))
cat(sprintf("  Divergent: %d\n", summ_2$diagnostics$divergent))

results$row_2 <- list(time = t_2,
                      divergent = summ_2$diagnostics$divergent)

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n", strrep("=", 70), "\n")
cat("METRIC OPTIMIZATION RESULTS\n")
cat(strrep("=", 70), "\n\n")

cat(sprintf("%-6s %-30s %10s %10s %10s %8s\n",
            "Row", "Config", "Before", "After", "Speedup", "Diverg"))
cat(strrep("-", 80), "\n")

cat(sprintf("%-6s %-30s %10.1fs %10.1fs %10.1fx %8d\n",
            "27", "pg + RE + TVC", 68.0, results$row_27$time,
            68.0 / results$row_27$time, results$row_27$divergent))
cat(sprintf("%-6s %-30s %10.1fs %10.1fs %10.1fx %8d\n",
            "57", "nb + RE + TVC", 183.2, results$row_57$time,
            183.2 / results$row_57$time, results$row_57$divergent))
cat(sprintf("%-6s %-30s %10.1fs %10.1fs %10.1fx %8d\n",
            "38", "nb + RE + HSGP", 210.4, results$row_38$time,
            210.4 / results$row_38$time, results$row_38$divergent))
cat(sprintf("%-6s %-30s %10.1fs %10.1fs %10.1fx %8d\n",
            "14", "pg + RE + temporal GP", 84.6, results$row_14$time,
            84.6 / results$row_14$time, results$row_14$divergent))
cat(strrep("-", 80), "\n")
cat(sprintf("%-6s %-30s %10.1fs %10.1fs %10s %8d\n",
            "2", "pg + RE (regression check)", 1.25, results$row_2$time,
            "", results$row_2$divergent))

cat("\nDivergence threshold: <=5% of post-warmup iterations = ",
    floor(0.05 * (N_ITER - N_WARMUP)), " divergent transitions\n")

saveRDS(results, "benchmarks/results_metric_optimization.rds")
cat("\nResults saved to benchmarks/results_metric_optimization.rds\n")
