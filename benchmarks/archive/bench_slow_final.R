# Final benchmark: slopes+ICAR and ST_IV models (after vectorization)
library(numdenom)
set.seed(42)

N <- 500L
n_sites <- 50L
n_times <- 20L
x <- rnorm(N)
site <- rep(1:n_sites, length.out = N)
time <- rep(1:n_times, length.out = N)

# Adjacency matrix (ring lattice)
W <- matrix(0, n_sites, n_sites)
for (i in 1:(n_sites - 1)) { W[i, i + 1] <- 1; W[i + 1, i] <- 1 }
W[1, n_sites] <- 1; W[n_sites, 1] <- 1

# Generate data for each family
y_pg <- rpois(N, exp(0.5 + 0.3 * x))
eff_pg <- rpois(N, exp(1.0)) + 1  # ensure no zeros

y_nb_num <- rpois(N, exp(0.5 + 0.3 * x))
y_nb_den <- rpois(N, exp(0.5)) + 1

y_bin <- rbinom(N, size = 10, prob = 0.3)
n_bin <- rep(10L, N)

df_pg <- data.frame(y = y_pg, effort = eff_pg, x = x, site = factor(site), time = time)
df_nb <- data.frame(y = y_nb_num, denom = y_nb_den, x = x, site = factor(site), time = time)
df_bin <- data.frame(y = y_bin, n = n_bin, x = x, site = factor(site), time = time)

run_bench <- function(desc, expr) {
  cat(sprintf("\n=== %s ===\n", desc))
  time <- system.time({
    fit <- tryCatch(expr, error = function(e) paste0("ERROR: ", e$message))
  })["elapsed"]
  if (is.character(fit)) {
    cat(sprintf("  FAILED: %s (%.1fs)\n", fit, time))
    return(data.frame(model = desc, time_s = time, status = "FAIL"))
  }
  cat(sprintf("  %.1fs\n", time))
  data.frame(model = desc, time_s = time, status = "OK")
}

results <- list()

# ============= slopes+ICAR =============
cat("\n========== SLOPES + ICAR ==========\n")

results[[length(results) + 1]] <- run_bench("PG+slopes+ICAR", {
  tratio(y | effort ~ (x | site), data = df_pg,
         spatial = spatial_car(W, group_var = "site"),
         family = ratiod_poisson_gamma(),
         control = list(iter = 500, warmup = 250, chains = 1, verbose = TRUE))
})

results[[length(results) + 1]] <- run_bench("NB+slopes+ICAR", {
  tratio(y | denom ~ (x | site), data = df_nb,
         spatial = spatial_car(W, group_var = "site"),
         family = ratiod_negbin_negbin(),
         control = list(iter = 500, warmup = 250, chains = 1, verbose = TRUE))
})

results[[length(results) + 1]] <- run_bench("Bin+slopes+ICAR", {
  tratio(y | n ~ (x | site), data = df_bin,
         spatial = spatial_car(W, group_var = "site"),
         family = ratiod_binomial(),
         control = list(iter = 500, warmup = 250, chains = 1, verbose = TRUE))
})

# ============= Spatiotemporal Type IV =============
cat("\n========== ST TYPE IV ==========\n")

results[[length(results) + 1]] <- run_bench("PG+ST_IV", {
  tratio(y | effort ~ x, data = df_pg,
         spatial = spatial_car(W, group_var = "site"),
         spatiotemporal = spatiotemporal(
           spatial = spatial_car(W, group_var = "site"),
           temporal = temporal_rw1("time"),
           type = "IV"),
         family = ratiod_poisson_gamma(),
         control = list(iter = 500, warmup = 250, chains = 1, verbose = TRUE))
})

results[[length(results) + 1]] <- run_bench("NB+ST_IV", {
  tratio(y | denom ~ x, data = df_nb,
         spatial = spatial_car(W, group_var = "site"),
         spatiotemporal = spatiotemporal(
           spatial = spatial_car(W, group_var = "site"),
           temporal = temporal_rw1("time"),
           type = "IV"),
         family = ratiod_negbin_negbin(),
         control = list(iter = 500, warmup = 250, chains = 1, verbose = TRUE))
})

results[[length(results) + 1]] <- run_bench("Bin+ST_IV", {
  tratio(y | n ~ x, data = df_bin,
         spatial = spatial_car(W, group_var = "site"),
         spatiotemporal = spatiotemporal(
           spatial = spatial_car(W, group_var = "site"),
           temporal = temporal_rw1("time"),
           type = "IV"),
         family = ratiod_binomial(),
         control = list(iter = 500, warmup = 250, chains = 1, verbose = TRUE))
})

# ============= Summary =============
cat("\n\n========== SUMMARY ==========\n")
df_results <- do.call(rbind, results)

# Previous results (before vectorization)
prev <- c(
  "PG+slopes+ICAR" = 50.8,
  "NB+slopes+ICAR" = 70.0,
  "Bin+slopes+ICAR" = 25.3,
  "PG+ST_IV" = 190.8,
  "NB+ST_IV" = 287.9,
  "Bin+ST_IV" = 80.7
)

stan <- c(
  "PG+slopes+ICAR" = 15.4,
  "NB+slopes+ICAR" = 46.6,
  "Bin+slopes+ICAR" = 14.5,
  "PG+ST_IV" = 82.5,
  "NB+ST_IV" = 133.5,
  "Bin+ST_IV" = 56.0
)

cat(sprintf("%-20s %8s %8s %8s %8s %8s\n", "Model", "Before", "After", "Stan", "Speedup", "vs Stan"))
cat(paste(rep("-", 72), collapse = ""), "\n")
for (i in 1:nrow(df_results)) {
  m <- df_results$model[i]
  after <- df_results$time_s[i]
  before <- prev[m]
  stan_t <- stan[m]
  speedup <- before / after
  vs_stan <- after / stan_t
  cat(sprintf("%-20s %7.1fs %7.1fs %7.1fs %7.1fx %7.1fx\n",
              m, before, after, stan_t, speedup, vs_stan))
}
