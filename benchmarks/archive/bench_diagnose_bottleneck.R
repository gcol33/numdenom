# =============================================================================
# Diagnostic benchmark: capture treedepth, epsilon, divergence stats for
# the 5 slow model categories (HSGP, BYM2, temporal GP, correlated slopes,
# Bin+AR1). Run after each optimization change to track improvement.
#
# Usage: Rscript benchmarks/bench_diagnose_bottleneck.R
# =============================================================================

suppressPackageStartupMessages(library(numdenom))

# --- Parameters ---
N_OBS     <- 500L
N_ITER    <- 500L
N_WARMUP  <- 250L
N_CHAINS  <- 1L
N_SITES   <- 50L
N_TIMES   <- 20L
SEED      <- 123L

set.seed(SEED)

# --- Data generation ---
site <- factor(rep(1:N_SITES, length.out = N_OBS))
time <- rep(1:N_TIMES, length.out = N_OBS)
x <- rnorm(N_OBS)
z <- rnorm(N_OBS)
site_int <- as.integer(site)

# Grid for spatial models
n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_SITES, ]
lon_site <- grid$lon[site_int]
lat_site <- grid$lat[site_int]

# Adjacency matrix
adj_mat <- matrix(0L, N_SITES, N_SITES)
for (i in 1:N_SITES) {
  for (j in 1:N_SITES) {
    if (i != j) {
      d <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
      if (d <= 1.5) adj_mat[i, j] <- 1L
    }
  }
}

# Response data
y_pg_num   <- rpois(N_OBS, exp(2 + 0.5 * x))
y_pg_denom <- rgamma(N_OBS, 10, 1)
y_bin_num  <- rbinom(N_OBS, 10, plogis(0.5 + 0.3 * x))
y_bin_denom <- rep(10L, N_OBS)

df <- data.frame(
  y_pg_num = y_pg_num, y_pg_denom = y_pg_denom,
  y_bin_num = y_bin_num, y_bin_denom = y_bin_denom,
  x = x, z = z, site = site, time = factor(time),
  lon = lon_site, lat = lat_site
)

# --- Helper: extract diagnostics from fit ---
extract_diag <- function(fit) {
  diag <- fit$diagnostics
  list(
    time        = fit$timing$total,
    epsilon     = mean(diag$epsilon),
    avg_td      = mean(diag$treedepth),
    max_td      = max(diag$treedepth),
    pct_maxd    = mean(diag$treedepth >= max(diag$treedepth)) * 100,
    n_div       = sum(diag$divergent),
    sampler     = fit$sampler_info$sampler,
    n_params    = length(fit$param_names)
  )
}

# --- Run each model and capture diagnostics ---
cat("=" , rep("=", 78), "\n", sep = "")
cat("BOTTLENECK DIAGNOSTIC BENCHMARK\n")
cat("N=", N_OBS, " iter=", N_ITER, " warmup=", N_WARMUP,
    " chains=", N_CHAINS, " sites=", N_SITES, " times=", N_TIMES, "\n", sep = "")
cat("=" , rep("=", 78), "\n\n", sep = "")

models <- list()

# 1. PG + HSGP (row 8-equivalent)
cat(">>> PG + HSGP ...\n")
tryCatch({
  t0 <- proc.time()
  fit <- tratio(
    y_pg_num | y_pg_denom ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    spatial = spatial_hsgp(coords = cbind(df$lon, df$lat), m = 6, c = 1.5),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H", verbose = TRUE)
  )
  models[["PG+HSGP"]] <- extract_diag(fit)
}, error = function(e) cat("  ERROR:", e$message, "\n"))

# 2. PG + BYM2 (row 6-equivalent)
cat("\n>>> PG + BYM2 ...\n")
tryCatch({
  fit <- tratio(
    y_pg_num | y_pg_denom ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    spatial = spatial_bym2(adj = adj_mat, group = site_int),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H", verbose = TRUE)
  )
  models[["PG+BYM2"]] <- extract_diag(fit)
}, error = function(e) cat("  ERROR:", e$message, "\n"))

# 3. PG + correlated slopes (row 33-equivalent)
cat("\n>>> PG + correlated slopes ...\n")
tryCatch({
  fit <- tratio(
    y_pg_num | y_pg_denom ~ x + (x | site),
    data = df,
    family = ratiod_poisson_gamma(),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H", verbose = TRUE)
  )
  models[["PG+slopes"]] <- extract_diag(fit)
}, error = function(e) cat("  ERROR:", e$message, "\n"))

# 4. PG + temporal_gp (row 14-equivalent)
cat("\n>>> PG + temporal_gp ...\n")
tryCatch({
  fit <- tratio(
    y_pg_num | y_pg_denom ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    temporal = temporal_gp(time = time, group = site_int),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H", verbose = TRUE)
  )
  models[["PG+GP_t"]] <- extract_diag(fit)
}, error = function(e) cat("  ERROR:", e$message, "\n"))

# 5. Bin + AR1 (row 73-equivalent)
cat("\n>>> Bin + AR1 ...\n")
tryCatch({
  fit <- tratio(
    y_bin_num | y_bin_denom ~ x,
    data = df,
    family = ratiod_binomial(),
    temporal = temporal_ar1(time = time, group = site_int),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H", verbose = TRUE)
  )
  models[["Bin+AR1"]] <- extract_diag(fit)
}, error = function(e) cat("  ERROR:", e$message, "\n"))

# 6. Reference: NB + AR1 (should be fast, ~3.5s)
cat("\n>>> NB + AR1 (reference) ...\n")
tryCatch({
  y_nb_num   <- rnbinom(N_OBS, mu = exp(2 + 0.5 * x), size = 5)
  y_nb_denom <- rnbinom(N_OBS, mu = exp(3 + 0.3 * x), size = 8)
  df$y_nb_num <- y_nb_num
  df$y_nb_denom <- y_nb_denom
  fit <- tratio(
    y_nb_num | y_nb_denom ~ x,
    data = df,
    family = ratiod_negbin_negbin(),
    temporal = temporal_ar1(time = time, group = site_int),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H", verbose = TRUE)
  )
  models[["NB+AR1"]] <- extract_diag(fit)
}, error = function(e) cat("  ERROR:", e$message, "\n"))

# --- Summary table ---
cat("\n\n", paste(rep("=", 80), collapse = ""), "\n")
cat(sprintf("%-15s %7s %8s %7s %7s %7s %5s %5s\n",
            "Model", "Time(s)", "Epsilon", "Avg_TD", "Max_TD", "%MaxD", "Div", "p"))
cat(paste(rep("-", 80), collapse = ""), "\n")
for (nm in names(models)) {
  m <- models[[nm]]
  cat(sprintf("%-15s %7.1f %8.5f %7.1f %7d %6.1f%% %5d %5d\n",
              nm, m$time, m$epsilon, m$avg_td, m$max_td,
              m$pct_maxd, m$n_div, m$n_params))
}
cat(paste(rep("=", 80), collapse = ""), "\n")
