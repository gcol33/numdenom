#!/usr/bin/env Rscript
# Benchmark: Measure impact of fused log-posterior optimization
# Compares current (optimized) timings vs pre-optimization baselines in gradient_methods.md
#
# Optimizations applied:
# 1. Fused log-posterior into 9 specialized H gradient functions (eliminates duplicate O(N) pass)
# 2. Fixed find_reasonable_epsilon to use fused gradient+logpost call
#
# Expected improvements:
# - HSGP, TVC, temporal GP, latent, MSGP, SVC, GP: ~30-50% faster (eliminated duplicate obs loop)
# - Base models: minor improvement from find_reasonable_epsilon fix only

library(numdenom)

set.seed(42)

N_OBS <- 500
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1
N_SITES <- 50
N_TIMES <- 20

cat("=== Fused Log-Posterior Benchmark ===\n")
cat(sprintf("N=%d, iter=%d, warmup=%d, chains=%d\n\n", N_OBS, N_ITER, N_WARMUP, N_CHAINS))

# Helper: time a model fit
time_fit <- function(label, expr, n_runs = 1) {
  cat(sprintf("  %-40s ", label))
  times <- numeric(n_runs)
  for (r in seq_len(n_runs)) {
    gc(FALSE)
    t0 <- proc.time()["elapsed"]
    tryCatch({
      fit <- eval(expr)
      t1 <- proc.time()["elapsed"]
      times[r] <- t1 - t0
    }, error = function(e) {
      cat(sprintf("ERROR: %s\n", e$message))
      times[r] <<- NA
    })
  }
  avg <- mean(times, na.rm = TRUE)
  cat(sprintf("%6.1fs\n", avg))
  return(avg)
}

# Generate data once
site <- rep(1:N_SITES, length.out = N_OBS)
site2 <- rep(1:(N_SITES/2), length.out = N_OBS)
x <- rnorm(N_OBS)
time_idx <- rep(1:N_TIMES, length.out = N_OBS)

# Spatial adjacency for ICAR
W <- matrix(0, N_SITES, N_SITES)
for (i in 1:(N_SITES - 1)) W[i, i + 1] <- W[i + 1, i] <- 1

# Spatial coordinates for GP/HSGP
coords <- data.frame(
  lon = runif(N_SITES, 0, 10),
  lat = runif(N_SITES, 0, 10)
)

# --- Poisson-Gamma data ---
mu_num <- exp(1.0 + 0.3 * x)
mu_denom <- exp(2.0 + 0.1 * x)
y_num <- rpois(N_OBS, mu_num)
y_denom <- rgamma(N_OBS, shape = 5, rate = 5 / mu_denom)
y_denom[y_denom < 0.01] <- 0.01
df_pg <- data.frame(y_num = y_num, y_denom = y_denom, x = x, site = factor(site),
                    site2 = factor(site2), time = factor(time_idx),
                    lon = coords$lon[site], lat = coords$lat[site])

# --- NegBin-NegBin data ---
y_num_nb <- rnbinom(N_OBS, mu = exp(1.0 + 0.3 * x), size = 5)
y_denom_nb <- rnbinom(N_OBS, mu = exp(2.0 + 0.1 * x), size = 5)
y_denom_nb[y_denom_nb == 0] <- 1
df_nb <- data.frame(y_num = y_num_nb, y_denom = y_denom_nb, x = x, site = factor(site),
                    site2 = factor(site2), time = factor(time_idx),
                    lon = coords$lon[site], lat = coords$lat[site])

# --- Binomial data ---
n_trials <- sample(10:50, N_OBS, replace = TRUE)
p_true <- plogis(0.5 + 0.3 * x)
y_bin <- rbinom(N_OBS, n_trials, p_true)
df_bin <- data.frame(y_num = y_bin, y_denom = n_trials, x = x, site = factor(site),
                     site2 = factor(site2), time = factor(time_idx),
                     lon = coords$lon[site], lat = coords$lat[site])

results <- list()

# ============================================================
# GROUP 1: Base models (compute_gradient_analytical - already fused)
# Only find_reasonable_epsilon improvement expected
# ============================================================
cat("--- Group 1: Base models (already fused, find_reasonable_epsilon only) ---\n")

results[["Row 1: PG base"]] <- time_fit("Row 1: PG base (old: 9.1s)", quote(
  ratiod(y_num | y_denom ~ x, data = df_pg, family = ratiod_poisson_gamma(),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

results[["Row 31: NB base"]] <- time_fit("Row 31: NB base (old: 17.1s)", quote(
  ratiod(y_num | y_denom ~ x, data = df_nb, family = ratiod_negbin_negbin(),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

results[["Row 61: Bin base"]] <- time_fit("Row 61: Bin base (old: 12.7s)", quote(
  ratiod(y_num | y_denom ~ x, data = df_bin, family = ratiod_binomial(),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

# ============================================================
# GROUP 2: HSGP models (compute_gradient_hsgp - NOW fused)
# Should see ~30-50% improvement
# ============================================================
cat("\n--- Group 2: HSGP models (newly fused) ---\n")

results[["Row 8: PG+HSGP"]] <- time_fit("Row 8: PG+HSGP (old: 9.5s)", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_pg, family = ratiod_poisson_gamma(),
         spatial = spatial_hsgp(coords = ~ lon + lat, m = 5, group_var = "site"),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

results[["Row 38: NB+HSGP"]] <- time_fit("Row 38: NB+HSGP (old: 12.9s)", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_nb, family = ratiod_negbin_negbin(),
         spatial = spatial_hsgp(coords = ~ lon + lat, m = 5, group_var = "site"),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

results[["Row 68: Bin+HSGP"]] <- time_fit("Row 68: Bin+HSGP (old: 3.9s)", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_bin, family = ratiod_binomial(),
         spatial = spatial_hsgp(coords = ~ lon + lat, m = 5, group_var = "site"),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

# ============================================================
# GROUP 3: TVC models (compute_gradient_tvc_handcoded - NOW fused)
# Should see improvement
# ============================================================
cat("\n--- Group 3: TVC models (newly fused) ---\n")

results[["Row 27: PG+TVC"]] <- time_fit("Row 27: PG+TVC (old: 6.6s)", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_pg, family = ratiod_poisson_gamma(),
         temporal = temporal_tvc(time_var = "time", terms = 1),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

results[["Row 57: NB+TVC"]] <- time_fit("Row 57: NB+TVC (old: 15.0s)", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_nb, family = ratiod_negbin_negbin(),
         temporal = temporal_tvc(time_var = "time", terms = 1),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

results[["Row 89: Bin+TVC"]] <- time_fit("Row 89: Bin+TVC (old: 3.9s)", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_bin, family = ratiod_binomial(),
         temporal = temporal_tvc(time_var = "time", terms = 1),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

# ============================================================
# GROUP 4: Latent factor models (compute_gradient_latent_handcoded - NOW fused)
# Small N (fast), so improvement may be small
# ============================================================
cat("\n--- Group 4: Latent factor models (newly fused) ---\n")

# Latent needs small N to be fast
N_LAT <- 50
df_lat_bin <- data.frame(
  y_num = rbinom(N_LAT, 20, 0.5),
  y_denom = rep(20, N_LAT),
  x = rnorm(N_LAT),
  site = factor(rep(1:10, length.out = N_LAT))
)

results[["Row 92: Bin+Latent"]] <- time_fit("Row 92: Bin+Latent (old: 0.1s)", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_lat_bin, family = ratiod_binomial(),
         latent = latent_factor(n_factors = 1),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

# ============================================================
# GROUP 5: RE + temporal models (compute_gradient_analytical - already fused)
# Only find_reasonable_epsilon improvement expected
# ============================================================
cat("\n--- Group 5: RE + temporal (already fused) ---\n")

results[["Row 11: PG+RW1"]] <- time_fit("Row 11: PG+RW1 (old: 10.2s)", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_pg, family = ratiod_poisson_gamma(),
         temporal = temporal_rw1(time_var = "time"),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

results[["Row 41: NB+RW1"]] <- time_fit("Row 41: NB+RW1 (old: 8.2s)", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_nb, family = ratiod_negbin_negbin(),
         temporal = temporal_rw1(time_var = "time"),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

# ============================================================
# SUMMARY
# ============================================================
cat("\n\n=== SUMMARY ===\n")
cat(sprintf("%-35s %8s %8s %8s\n", "Model", "Old (s)", "New (s)", "Speedup"))
cat(paste0(rep("-", 75), collapse = ""), "\n")

old_times <- c(
  "Row 1: PG base" = 9.1,
  "Row 31: NB base" = 17.1,
  "Row 61: Bin base" = 12.7,
  "Row 8: PG+HSGP" = 9.5,
  "Row 38: NB+HSGP" = 12.9,
  "Row 68: Bin+HSGP" = 3.9,
  "Row 27: PG+TVC" = 6.6,
  "Row 57: NB+TVC" = 15.0,
  "Row 89: Bin+TVC" = 3.9,
  "Row 92: Bin+Latent" = 0.1,
  "Row 11: PG+RW1" = 10.2,
  "Row 41: NB+RW1" = 8.2
)

for (name in names(results)) {
  old <- old_times[name]
  new_time <- results[[name]]
  if (!is.na(new_time) && !is.na(old)) {
    speedup <- old / new_time
    cat(sprintf("%-35s %8.1f %8.1f %7.2fx\n", name, old, new_time, speedup))
  } else {
    cat(sprintf("%-35s %8.1f %8s %8s\n", name, old, ifelse(is.na(new_time), "ERROR", sprintf("%.1f", new_time)), "N/A"))
  }
}

cat("\nExpected patterns:\n")
cat("  - Group 1 (base): ~same (already fused, only epsilon fix)\n")
cat("  - Group 2 (HSGP): ~30-50% faster (fused obs loop)\n")
cat("  - Group 3 (TVC):  ~30-50% faster (fused obs loop)\n")
cat("  - Group 4 (Latent): minor (small N)\n")
cat("  - Group 5 (RW1):  ~same (already fused)\n")
