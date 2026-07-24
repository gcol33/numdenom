# Benchmark ALL missing models from gradient_methods.md
# Standard parameters: N=500, iter=500, warmup=250, chains=1

library(numdenom)
set.seed(123)

# Setup data
N <- 500
N_SITES <- 50
N_TIMES <- 20
x <- rnorm(N)
site <- factor(rep(1:N_SITES, length.out = N))
time <- factor(rep(1:N_TIMES, length.out = N))

# Spatial grid for GP models
n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_SITES, ]
site_int <- as.integer(site)
lon <- grid$lon[site_int]
lat <- grid$lat[site_int]

# Adjacency matrix for CAR
adj_mat <- matrix(0, N_SITES, N_SITES)
for (i in 1:N_SITES) {
  for (j in 1:N_SITES) {
    if (i != j) {
      dist <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
      if (dist <= 1.5) adj_mat[i, j] <- 1
    }
  }
}
spatial_site <- factor(rep(1:N_SITES, length.out = N))

# Generate data for each family
y_pg <- rpois(N, exp(2 + 0.3*x))
effort <- rgamma(N, 10, 1)
df_pg <- data.frame(y = y_pg, effort = effort, x = x, site = site, time = time,
                    lon = lon, lat = lat, spatial_site = spatial_site)

y_nb <- rnbinom(N, mu = exp(2 + 0.3*x), size = 5)
denom_nb <- rnbinom(N, mu = 100, size = 10)
denom_nb[denom_nb == 0] <- 1
df_nb <- data.frame(y = y_nb, denom = denom_nb, x = x, site = site, time = time,
                    lon = lon, lat = lat, spatial_site = spatial_site)

trials <- sample(10:50, N, replace = TRUE)
y_bin <- rbinom(N, trials, plogis(0.5 + 0.3*x))
df_bin <- data.frame(y = y_bin, trials = trials, x = x, site = site, time = time,
                     lon = lon, lat = lat, spatial_site = spatial_site)

# Gamma data
y_gamma_num <- rgamma(N, shape = 2, rate = 1/exp(1 + 0.3*x))
y_gamma_denom <- rgamma(N, shape = 5, rate = 1)
df_gamma <- data.frame(y = y_gamma_num, denom = y_gamma_denom, x = x, site = site,
                       time = time, lon = lon, lat = lat, spatial_site = spatial_site)

# Lognormal data
y_ln_num <- rlnorm(N, meanlog = 1 + 0.3*x, sdlog = 0.5)
y_ln_denom <- rlnorm(N, meanlog = 2, sdlog = 0.3)
df_ln <- data.frame(y = y_ln_num, denom = y_ln_denom, x = x, site = site,
                    time = time, lon = lon, lat = lat, spatial_site = spatial_site)

# Beta-binomial data (using binomial for now)
df_bb <- df_bin

# Benchmark parameters
ITER <- 500
WARMUP <- 250
CHAINS <- 1

results <- list()
errors <- list()

run_bench <- function(row, name, expr) {
  cat(sprintf("\n=== Row %d: %s ===\n", row, name))
  flush.console()
  tryCatch({
    time <- system.time(eval(expr))["elapsed"]
    cat(sprintf("  H: %.1fs\n", time))
    results[[as.character(row)]] <<- list(row = row, name = name, H = time)
  }, error = function(e) {
    cat(sprintf("  ERROR: %s\n", substr(conditionMessage(e), 1, 60)))
    errors[[as.character(row)]] <<- list(row = row, name = name, error = conditionMessage(e))
  })
}

# ============================================================
# NEGBIN FAMILY - GP/HSGP/MSGP/pCAR (rows 37-40)
# ============================================================

run_bench(37, "nb_gp", quote(
  tratio(y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin(),
         spatial = spatial_gp(lon = "lon", lat = "lat", level = "obs"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(38, "nb_hsgp", quote(
  tratio(y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin(),
         spatial = spatial_hsgp(lon = "lon", lat = "lat", level = "obs", m = 8),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(39, "nb_msgp", quote(
  tratio(y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin(),
         spatial = spatial_multiscale(lon = "lon", lat = "lat", level = "obs"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(40, "nb_pcar", quote(
  tratio(y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin(),
         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site", proper = TRUE),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

# ============================================================
# NEGBIN FAMILY - Temporal GP/MS (rows 44-45)
# ============================================================

run_bench(44, "nb_gp_t", quote(
  tratio(y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin(),
         temporal = temporal_gp("time"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(45, "nb_ms_t", quote(
  tratio(y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin(),
         temporal = temporal_multiscale("time"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

# ============================================================
# NEGBIN FAMILY - GP + Temporal (rows 51-53)
# ============================================================

run_bench(51, "nb_gp_rw1", quote(
  tratio(y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin(),
         spatial = spatial_gp(lon = "lon", lat = "lat", level = "obs"),
         temporal = temporal_rw1("time"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(52, "nb_hsgp_rw1", quote(
  tratio(y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin(),
         spatial = spatial_hsgp(lon = "lon", lat = "lat", level = "obs", m = 8),
         temporal = temporal_rw1("time"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(53, "nb_msgp_rw1", quote(
  tratio(y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin(),
         spatial = spatial_multiscale(lon = "lon", lat = "lat", level = "obs"),
         temporal = temporal_rw1("time"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

# ============================================================
# BINOMIAL FAMILY - GP/HSGP/MSGP/pCAR (rows 67-70)
# ============================================================

run_bench(67, "bin_gp", quote(
  tratio(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
         spatial = spatial_gp(lon = "lon", lat = "lat", level = "obs"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(68, "bin_hsgp", quote(
  tratio(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
         spatial = spatial_hsgp(lon = "lon", lat = "lat", level = "obs", m = 8),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(69, "bin_msgp", quote(
  tratio(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
         spatial = spatial_multiscale(lon = "lon", lat = "lat", level = "obs"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(70, "bin_pcar", quote(
  tratio(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site", proper = TRUE),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

# ============================================================
# BINOMIAL FAMILY - Temporal GP/MS (rows 74-75)
# ============================================================

run_bench(74, "bin_gp_t", quote(
  tratio(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
         temporal = temporal_gp("time"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(75, "bin_ms_t", quote(
  tratio(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
         temporal = temporal_multiscale("time"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

# ============================================================
# BINOMIAL FAMILY - GP + Temporal (rows 83-85)
# ============================================================

run_bench(83, "bin_gp_rw1", quote(
  tratio(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
         spatial = spatial_gp(lon = "lon", lat = "lat", level = "obs"),
         temporal = temporal_rw1("time"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(84, "bin_hsgp_rw1", quote(
  tratio(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
         spatial = spatial_hsgp(lon = "lon", lat = "lat", level = "obs", m = 8),
         temporal = temporal_rw1("time"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(85, "bin_msgp_rw1", quote(
  tratio(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
         spatial = spatial_multiscale(lon = "lon", lat = "lat", level = "obs"),
         temporal = temporal_rw1("time"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

# ============================================================
# GAMMA_GAMMA FAMILY (rows 93-97)
# ============================================================

run_bench(93, "gamma_base", quote(
  tratio(y | denom ~ x, data = df_gamma, family = ratiod_gamma_gamma(),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(94, "gamma_re", quote(
  tratio(y | denom ~ x + (1|site), data = df_gamma, family = ratiod_gamma_gamma(),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(95, "gamma_icar", quote(
  tratio(y | denom ~ x + (1|site), data = df_gamma, family = ratiod_gamma_gamma(),
         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(96, "gamma_rw1", quote(
  tratio(y | denom ~ x + (1|site), data = df_gamma, family = ratiod_gamma_gamma(),
         temporal = temporal_rw1("time"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(97, "gamma_icar_rw1", quote(
  tratio(y | denom ~ x + (1|site), data = df_gamma, family = ratiod_gamma_gamma(),
         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
         temporal = temporal_rw1("time"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

# ============================================================
# LOGNORMAL FAMILY (rows 98-102)
# ============================================================

run_bench(98, "ln_base", quote(
  tratio(y | denom ~ x, data = df_ln, family = ratiod_lognormal(),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(99, "ln_re", quote(
  tratio(y | denom ~ x + (1|site), data = df_ln, family = ratiod_lognormal(),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(100, "ln_icar", quote(
  tratio(y | denom ~ x + (1|site), data = df_ln, family = ratiod_lognormal(),
         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(101, "ln_rw1", quote(
  tratio(y | denom ~ x + (1|site), data = df_ln, family = ratiod_lognormal(),
         temporal = temporal_rw1("time"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(102, "ln_icar_rw1", quote(
  tratio(y | denom ~ x + (1|site), data = df_ln, family = ratiod_lognormal(),
         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
         temporal = temporal_rw1("time"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

# ============================================================
# BETA_BINOMIAL FAMILY (rows 103-107)
# ============================================================

run_bench(103, "bb_base", quote(
  tratio(y | trials ~ x, data = df_bb, family = ratiod_beta_binomial(),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(104, "bb_re", quote(
  tratio(y | trials ~ x + (1|site), data = df_bb, family = ratiod_beta_binomial(),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(105, "bb_icar", quote(
  tratio(y | trials ~ x + (1|site), data = df_bb, family = ratiod_beta_binomial(),
         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(106, "bb_rw1", quote(
  tratio(y | trials ~ x + (1|site), data = df_bb, family = ratiod_beta_binomial(),
         temporal = temporal_rw1("time"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(107, "bb_icar_rw1", quote(
  tratio(y | trials ~ x + (1|site), data = df_bb, family = ratiod_beta_binomial(),
         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
         temporal = temporal_rw1("time"),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

# ============================================================
# LATENT FACTOR MODELS (rows 30, 60, 92)
# ============================================================

run_bench(30, "pg_latent", quote(
  tratio(y | effort ~ x + (1|site), data = df_pg, family = ratiod_poisson_gamma(),
         latent = latent_factor(k = 2),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(60, "nb_latent", quote(
  tratio(y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin(),
         latent = latent_factor(k = 2),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

run_bench(92, "bin_latent", quote(
  tratio(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
         latent = latent_factor(k = 2),
         control = list(iter = ITER, warmup = WARMUP, chains = CHAINS, verbose = FALSE, gradient_mode = "H"))
))

# ============================================================
# Summary
# ============================================================
cat("\n\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("MISSING BENCHMARKS SUMMARY\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat(sprintf("%-6s %-20s %8s %s\n", "Row", "Name", "H(s)", "Status"))
cat(paste(rep("-", 60), collapse = ""), "\n")

for (row in sort(as.numeric(names(results)))) {
  r <- results[[as.character(row)]]
  cat(sprintf("%-6d %-20s %8.1f OK\n", r$row, r$name, r$H))
}

if (length(errors) > 0) {
  cat("\n=== ERRORS ===\n")
  for (row in sort(as.numeric(names(errors)))) {
    e <- errors[[as.character(row)]]
    cat(sprintf("Row %d (%s): %s\n", e$row, e$name, substr(e$error, 1, 50)))
  }
}

cat(sprintf("\nCompleted: %d | Errors: %d\n", length(results), length(errors)))

saveRDS(list(results = results, errors = errors), "benchmarks/results_missing_all.rds")
cat("Results saved to benchmarks/results_missing_all.rds\n")
