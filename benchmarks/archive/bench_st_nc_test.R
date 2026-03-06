#!/usr/bin/env Rscript
# Test ST Type IV NC reparameterization
# 1. Verify H vs N gradient agreement
# 2. Compare centered vs NC timing

library(numdenom)
set.seed(42)

# Small ST model setup
N_SITES <- 10
N_TIMES <- 5
N_OBS <- N_SITES * N_TIMES

site <- rep(1:N_SITES, each = N_TIMES)
time <- rep(1:N_TIMES, times = N_SITES)
x <- rnorm(N_OBS)
eta <- 0.5 + 0.3 * x + rnorm(N_OBS, 0, 0.3)
y <- rpois(N_OBS, exp(eta))
denom <- rpois(N_OBS, 10) + 1

# Adjacency for ICAR (ring lattice for simplicity)
adj <- Matrix::Matrix(0, N_SITES, N_SITES, sparse = TRUE)
for (i in 1:(N_SITES - 1)) adj[i, i + 1] <- adj[i + 1, i] <- 1
adj[1, N_SITES] <- adj[N_SITES, 1] <- 1

df <- data.frame(y = y, denom = denom, x = x, site = site, time = time)

cat("=== ST Type IV NC Reparameterization Test ===\n\n")
cat(sprintf("Setup: %d sites x %d times = %d obs, %d ST params\n\n",
            N_SITES, N_TIMES, N_OBS, N_SITES * N_TIMES))

# --- Test 1: H vs N gradient agreement with NC (default for Type IV) ---
cat("--- Test 1: Gradient Check (H vs N) ---\n")
tryCatch({
  fit_h <- ratiod(y | denom ~ x,
                  data = df, family = ratiod_poisson_gamma(),
                  spatial = spatial_car(adj, group_var = "site"),
                  temporal = temporal_rw1(time_var = "time"),
                  spatiotemporal = spatiotemporal(
                    spatial = spatial_car(adj, group_var = "site"),
                    temporal = temporal_rw1(time_var = "time"),
                    type = "IV"
                  ),
                  mode = "hmc", gradient_mode = "H",
                  iter = 50, warmup = 25, chains = 1, verbose = FALSE)
  cat("  H mode: OK\n")
  cat(sprintf("    Metric: %s, eps: %.4f, avg td: %.1f\n",
              if (!is.null(fit_h$diagnostics$metric)) fit_h$diagnostics$metric else "?",
              mean(fit_h$diagnostics$stepsize, na.rm = TRUE),
              mean(fit_h$diagnostics$treedepth, na.rm = TRUE)))
}, error = function(e) {
  cat(sprintf("  H mode FAILED: %s\n", conditionMessage(e)))
})

tryCatch({
  fit_n <- ratiod(y | denom ~ x,
                  data = df, family = ratiod_poisson_gamma(),
                  spatial = spatial_car(adj, group_var = "site"),
                  temporal = temporal_rw1(time_var = "time"),
                  spatiotemporal = spatiotemporal(
                    spatial = spatial_car(adj, group_var = "site"),
                    temporal = temporal_rw1(time_var = "time"),
                    type = "IV"
                  ),
                  mode = "hmc", gradient_mode = "N",
                  iter = 50, warmup = 25, chains = 1, verbose = FALSE)
  cat("  N mode: OK\n")
}, error = function(e) {
  cat(sprintf("  N mode FAILED: %s\n", conditionMessage(e)))
})

if (exists("fit_h") && exists("fit_n")) {
  # Compare posterior means
  h_means <- colMeans(fit_h$draws)
  n_means <- colMeans(fit_n$draws)
  common <- intersect(names(h_means), names(n_means))
  # Only compare structural params (not delta/z which are different parameterizations)
  struct_params <- common[!grepl("^st_delta|^st_z", common)]
  if (length(struct_params) > 0) {
    h_sub <- h_means[struct_params]
    n_sub <- n_means[struct_params]
    max_diff <- max(abs(h_sub - n_sub))
    cat(sprintf("  Max posterior diff (struct params): %.4f\n", max_diff))
  }
}

# --- Test 2: Timing comparison ---
cat("\n--- Test 2: Timing (500 iter, H mode) ---\n")

t_nc <- system.time({
  fit_nc <- ratiod(y | denom ~ x,
                   data = df, family = ratiod_poisson_gamma(),
                   spatial = spatial_car(adj, group_var = "site"),
                   temporal = temporal_rw1(time_var = "time"),
                   spatiotemporal = spatiotemporal(
                     spatial = spatial_car(adj, group_var = "site"),
                     temporal = temporal_rw1(time_var = "time"),
                     type = "IV"
                   ),
                   mode = "hmc", gradient_mode = "H",
                   iter = 500, warmup = 250, chains = 1, verbose = FALSE)
})["elapsed"]

cat(sprintf("  NC (default): %.1fs\n", t_nc))
cat(sprintf("    eps=%.4f, avg td=%.1f, div=%d\n",
            mean(fit_nc$diagnostics$stepsize, na.rm = TRUE),
            mean(fit_nc$diagnostics$treedepth, na.rm = TRUE),
            sum(fit_nc$diagnostics$divergent, na.rm = TRUE)))

cat("\nDone.\n")
