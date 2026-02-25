#!/usr/bin/env Rscript
# Benchmark: HSGP with fused log-posterior
# Old timings (gradient_methods.md, standard N=500, 500 iter):
#   Row 8:  PG+HSGP = 9.5s
#   Row 38: NB+HSGP = 12.9s
#   Row 68: Bin+HSGP = 3.9s

library(numdenom)
set.seed(42)

N_OBS <- 500
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1

cat("=== HSGP Fused Log-Posterior Benchmark ===\n")
cat(sprintf("N=%d, iter=%d, warmup=%d, chains=%d\n\n", N_OBS, N_ITER, N_WARMUP, N_CHAINS))

x <- rnorm(N_OBS)
coords <- data.frame(lon = runif(N_OBS, 0, 10), lat = runif(N_OBS, 0, 10))

# PG data
mu_num <- exp(1.0 + 0.3 * x)
mu_denom <- exp(2.0 + 0.1 * x)
y_num <- rpois(N_OBS, mu_num)
y_denom <- rgamma(N_OBS, shape = 5, rate = 5 / mu_denom)
y_denom[y_denom < 0.01] <- 0.01
df_pg <- data.frame(y_num = y_num, y_denom = y_denom, x = x,
                    lon = coords$lon, lat = coords$lat)

# NB data
y_num_nb <- rnbinom(N_OBS, mu = exp(1.0 + 0.3 * x), size = 5)
y_denom_nb <- rnbinom(N_OBS, mu = exp(2.0 + 0.1 * x), size = 5)
y_denom_nb[y_denom_nb == 0] <- 1
df_nb <- data.frame(y_num = y_num_nb, y_denom = y_denom_nb, x = x,
                    lon = coords$lon, lat = coords$lat)

# Binomial data
n_trials <- sample(10:50, N_OBS, replace = TRUE)
p_true <- plogis(0.5 + 0.3 * x)
y_bin <- rbinom(N_OBS, n_trials, p_true)
df_bin <- data.frame(y_num = y_bin, y_denom = n_trials, x = x,
                     lon = coords$lon, lat = coords$lat)

run_bench <- function(label, expr) {
  cat(sprintf("  %-45s ", label))
  gc(FALSE)
  t0 <- proc.time()["elapsed"]
  tryCatch({
    fit <- eval(expr)
    t1 <- proc.time()["elapsed"]
    elapsed <- t1 - t0
    cat(sprintf("%6.1fs\n", elapsed))
    return(elapsed)
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", substr(e$message, 1, 80)))
    return(NA)
  })
}

cat("--- HSGP models ---\n")

t8 <- run_bench("Row 8: PG+HSGP (old: 9.5s)", quote(
  ratiod(y_num | y_denom ~ x, data = df_pg, family = ratiod_poisson_gamma(),
         spatial = spatial_hsgp(coords = c("lon", "lat"), m = 6),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

t38 <- run_bench("Row 38: NB+HSGP (old: 12.9s)", quote(
  ratiod(y_num | y_denom ~ x, data = df_nb, family = ratiod_negbin_negbin(),
         spatial = spatial_hsgp(coords = c("lon", "lat"), m = 6),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

t68 <- run_bench("Row 68: Bin+HSGP (old: 3.9s)", quote(
  ratiod(y_num | y_denom ~ x, data = df_bin, family = ratiod_binomial(),
         spatial = spatial_hsgp(coords = c("lon", "lat"), m = 6),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

# Also test HSGP+RW1 combination
cat("\n--- HSGP+RW1 models ---\n")

time_idx <- rep(1:20, length.out = N_OBS)
df_pg$time <- factor(time_idx)
df_bin$time <- factor(time_idx)

t22 <- run_bench("Row 22: PG+HSGP+RW1 (old: 4.6s)", quote(
  ratiod(y_num | y_denom ~ x, data = df_pg, family = ratiod_poisson_gamma(),
         spatial = spatial_hsgp(coords = c("lon", "lat"), m = 6),
         temporal = temporal_rw1(time_var = "time"),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

t84 <- run_bench("Row 84: Bin+HSGP+RW1 (old: 3.6s)", quote(
  ratiod(y_num | y_denom ~ x, data = df_bin, family = ratiod_binomial(),
         spatial = spatial_hsgp(coords = c("lon", "lat"), m = 6),
         temporal = temporal_rw1(time_var = "time"),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

cat("\n=== SUMMARY ===\n")
cat(sprintf("%-35s %8s %8s %8s\n", "Model", "Old (s)", "New (s)", "Speedup"))
cat(paste0(rep("-", 65), collapse = ""), "\n")

old <- c(9.5, 12.9, 3.9, 4.6, 3.6)
new <- c(t8, t38, t68, t22, t84)
names_vec <- c("Row 8: PG+HSGP", "Row 38: NB+HSGP", "Row 68: Bin+HSGP",
               "Row 22: PG+HSGP+RW1", "Row 84: Bin+HSGP+RW1")

for (i in seq_along(old)) {
  if (!is.na(new[i])) {
    cat(sprintf("%-35s %8.1f %8.1f %7.2fx\n", names_vec[i], old[i], new[i], old[i] / new[i]))
  } else {
    cat(sprintf("%-35s %8.1f %8s %8s\n", names_vec[i], old[i], "ERROR", "N/A"))
  }
}
