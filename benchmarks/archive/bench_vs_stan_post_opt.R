#!/usr/bin/env Rscript
# Benchmark: Post-optimization numdenom NUTS vs Stan NUTS
# Same parameters as the previous Stan comparison: N=200, iter=1000, 1 chain
#
# Previous results (from gradient_methods.md):
#   Row 1 (PG base):    numdenom=2.8s, Stan=0.9s (3.3x slower)
#   Row 31 (NB base):   numdenom=5.2s, Stan=1.5s (3.5x slower)
#   Row 32 (NB+RE):     numdenom=6.6s, Stan=3.2s (2.0x slower)
#   Row 34 (NB+crossed): numdenom=17.3s, Stan=3.9s (4.4x slower)

library(numdenom)

set.seed(42)

# Match the previous comparison params
N_OBS <- 200
N_ITER <- 1000
N_WARMUP <- 500
N_CHAINS <- 1
N_SITES <- 30
N_SITES2 <- 15

cat("=== Post-Optimization: numdenom NUTS vs Stan ===\n")
cat(sprintf("N=%d, iter=%d, warmup=%d, chains=%d\n\n", N_OBS, N_ITER, N_WARMUP, N_CHAINS))

# Generate data
site <- rep(1:N_SITES, length.out = N_OBS)
site2 <- rep(1:N_SITES2, length.out = N_OBS)
x <- rnorm(N_OBS)

# --- PG data ---
mu_num_pg <- exp(1.0 + 0.3 * x)
mu_denom_pg <- exp(2.0 + 0.1 * x)
y_num_pg <- rpois(N_OBS, mu_num_pg)
y_denom_pg <- rgamma(N_OBS, shape = 5, rate = 5 / mu_denom_pg)
y_denom_pg[y_denom_pg < 0.01] <- 0.01
df_pg <- data.frame(y_num = y_num_pg, y_denom = y_denom_pg, x = x,
                    site = factor(site), site2 = factor(site2))

# --- NB data ---
y_num_nb <- rnbinom(N_OBS, mu = exp(1.0 + 0.3 * x), size = 5)
y_denom_nb <- rnbinom(N_OBS, mu = exp(2.0 + 0.1 * x), size = 5)
y_denom_nb[y_denom_nb == 0] <- 1
df_nb <- data.frame(y_num = y_num_nb, y_denom = y_denom_nb, x = x,
                    site = factor(site), site2 = factor(site2))

# --- Binomial data ---
n_trials <- sample(10:50, N_OBS, replace = TRUE)
p_true <- plogis(0.5 + 0.3 * x)
y_bin <- rbinom(N_OBS, n_trials, p_true)
df_bin <- data.frame(y_num = y_bin, y_denom = n_trials, x = x,
                     site = factor(site), site2 = factor(site2))

# HSGP data - needs coordinates
coords <- data.frame(lon = runif(N_OBS, 0, 10), lat = runif(N_OBS, 0, 10))
df_pg_hsgp <- cbind(df_pg, coords)
df_nb_hsgp <- cbind(df_nb, coords)
df_bin_hsgp <- cbind(df_bin, coords)

results <- list()

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
    cat(sprintf("ERROR: %s\n", substr(e$message, 1, 60)))
    return(NA)
  })
}

# ---- Simple models (Stan was 2-4x faster) ----
cat("--- Simple models (Stan was 2-4x faster) ---\n")

results[["PG base"]] <- run_bench("Row 1: PG base", quote(
  ratiod(y_num | y_denom ~ x, data = df_pg, family = ratiod_poisson_gamma(),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

results[["NB base"]] <- run_bench("Row 31: NB base", quote(
  ratiod(y_num | y_denom ~ x, data = df_nb, family = ratiod_negbin_negbin(),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

results[["NB+RE"]] <- run_bench("Row 32: NB+RE", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_nb, family = ratiod_negbin_negbin(),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

results[["NB+crossed"]] <- run_bench("Row 34: NB+crossed", quote(
  ratiod(y_num | y_denom ~ x + (1 | site) + (1 | site2), data = df_nb,
         family = ratiod_negbin_negbin(),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

results[["Bin base"]] <- run_bench("Row 61: Bin base", quote(
  ratiod(y_num | y_denom ~ x, data = df_bin, family = ratiod_binomial(),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

# ---- HSGP models (specialized gradient, newly fused) ----
cat("\n--- HSGP models (newly fused log-post) ---\n")

results[["PG+HSGP"]] <- run_bench("Row 8: PG+HSGP", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_pg_hsgp,
         family = ratiod_poisson_gamma(),
         spatial = spatial_hsgp(coords = ~ lon + lat, m = 5),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

results[["NB+HSGP"]] <- run_bench("Row 38: NB+HSGP", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_nb_hsgp,
         family = ratiod_negbin_negbin(),
         spatial = spatial_hsgp(coords = ~ lon + lat, m = 5),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

results[["Bin+HSGP"]] <- run_bench("Row 68: Bin+HSGP", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_bin_hsgp,
         family = ratiod_binomial(),
         spatial = spatial_hsgp(coords = ~ lon + lat, m = 5),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

# ---- Models where numdenom already beats Stan ----
cat("\n--- Models where numdenom was already faster ---\n")

# ICAR spatial
W_small <- matrix(0, N_SITES, N_SITES)
for (i in 1:(N_SITES - 1)) W_small[i, i + 1] <- W_small[i + 1, i] <- 1

results[["PG+ICAR"]] <- run_bench("Row 5: PG+RE+ICAR", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_pg,
         family = ratiod_poisson_gamma(),
         spatial = spatial_car(W = W_small, group_var = "site"),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

# RW1 temporal
time_idx <- rep(1:10, length.out = N_OBS)
df_pg$time <- factor(time_idx)
df_nb$time <- factor(time_idx)

results[["PG+RW1"]] <- run_bench("Row 11: PG+RE+RW1", quote(
  ratiod(y_num | y_denom ~ x + (1 | site), data = df_pg,
         family = ratiod_poisson_gamma(),
         temporal = temporal_rw1(time_var = "time"),
         iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, refresh = 0)
))

# ============================================================
# SUMMARY TABLE
# ============================================================
cat("\n\n=== SUMMARY: numdenom NUTS (post-optimization) vs Stan ===\n")
cat(sprintf("%-30s %10s %10s %10s %10s\n",
            "Model", "Old NUTS", "New NUTS", "Stan", "New/Stan"))
cat(paste0(rep("-", 75), collapse = ""), "\n")

old_nuts <- c(
  "PG base" = 2.8, "NB base" = 5.2, "NB+RE" = 6.6, "NB+crossed" = 17.3,
  "Bin base" = NA
)
stan_times <- c(
  "PG base" = 0.9, "NB base" = 1.5, "NB+RE" = 3.2, "NB+crossed" = 3.9,
  "Bin base" = NA
)

for (name in names(results)) {
  old <- if (name %in% names(old_nuts)) old_nuts[[name]] else NA
  stan <- if (name %in% names(stan_times)) stan_times[[name]] else NA
  new_time <- results[[name]]

  old_str <- if (!is.na(old)) sprintf("%8.1fs", old) else "     N/A"
  new_str <- if (!is.na(new_time)) sprintf("%8.1fs", new_time) else "   ERROR"
  stan_str <- if (!is.na(stan)) sprintf("%8.1fs", stan) else "     N/A"

  ratio_str <- if (!is.na(new_time) && !is.na(stan)) {
    r <- new_time / stan
    if (r < 1) sprintf("  %.1fx WIN", 1/r) else sprintf("  %.1fx slower", r)
  } else "     N/A"

  cat(sprintf("%-30s %10s %10s %10s %10s\n", name, old_str, new_str, stan_str, ratio_str))
}

# NUTS improvement
cat("\n--- NUTS improvement (old NUTS → new NUTS) ---\n")
for (name in intersect(names(old_nuts), names(results))) {
  if (!is.na(old_nuts[[name]]) && !is.na(results[[name]])) {
    speedup <- old_nuts[[name]] / results[[name]]
    cat(sprintf("  %-30s %.1fs → %.1fs  (%.1fx %s)\n",
                name, old_nuts[[name]], results[[name]],
                abs(speedup),
                if (speedup > 1) "faster" else "slower"))
  }
}
