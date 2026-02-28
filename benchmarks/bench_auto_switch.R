# =============================================================================
# Benchmark: Adaptive NUTS→HMC(L=10) switching
# Tests all major model configurations across 3 families
# =============================================================================

# Use installed package (must install first with devtools::install())
suppressPackageStartupMessages(library(numdenom))

N <- 500L
ITER <- 500L
WARMUP <- 250L
CHAINS <- 1L
S <- 50L
T <- 20L
SEED <- 42L

set.seed(SEED)

# --- Data generation ---
site <- factor(rep(1:S, each = N/S))
time_f <- factor(rep(1:T, length.out = N))
x <- rnorm(N)

# Adjacency matrix for spatial models
n_side <- ceiling(sqrt(S))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:S, ]
site_int <- as.integer(site)
adj_mat <- matrix(0L, S, S)
for (i in 1:S) for (j in 1:S) {
  d <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
  if (i != j && d <= 1.5) adj_mat[i, j] <- 1L
}

# PG data
df_pg <- data.frame(
  y = rpois(N, exp(2 + 0.5 * x)),
  denom = rgamma(N, 10, 1),
  x = x, site = site, time = time_f,
  lon = grid$lon[site_int], lat = grid$lat[site_int]
)

# NB data
y_nb_den <- rnbinom(N, mu = 100, size = 10)
y_nb_den[y_nb_den == 0] <- 1L
df_nb <- data.frame(
  y = rnbinom(N, mu = exp(2 + 0.3 * x), size = 5),
  denom = y_nb_den,
  x = x, site = site, time = time_f,
  lon = grid$lon[site_int], lat = grid$lat[site_int]
)

# Slopes data
eta_slopes <- 1.0 + 0.3 * x + rnorm(S, 0, 0.5)[site_int] +
  rnorm(S, 0, 0.2)[site_int] * x
df_nb_slopes <- data.frame(
  y = rnbinom(N, mu = exp(eta_slopes), size = 5),
  denom = y_nb_den,
  x = x, site = site, time = time_f,
  lon = grid$lon[site_int], lat = grid$lat[site_int]
)

# Binomial data
trials <- sample(10:50, N, replace = TRUE)
df_bin <- data.frame(
  y = rbinom(N, trials, plogis(0.5 + 0.3 * x)),
  trials = trials,
  x = x, site = site, time = time_f,
  lon = grid$lon[site_int], lat = grid$lat[site_int]
)

# --- Benchmark function ---
bm <- function(name, formula, data, family, spatial = NULL, temporal = NULL,
               zi = NULL, timeout = 600, ...) {
  cat(sprintf("%-40s", name))
  flush.console()

  result <- tryCatch({
    t <- system.time({
      fit <- ratiod(formula, data = data, family = family,
                    spatial = spatial, temporal = temporal, zi = zi,
                    mode = "hmc", iter = ITER, warmup = WARMUP,
                    chains = CHAINS, verbose = FALSE, ...)
    })
    td <- fit$diagnostics$treedepth
    maxd_pct <- round(sum(td >= 10) / length(td) * 100)
    div <- sum(fit$diagnostics$divergent)
    sampler <- fit$diagnostics$algorithm
    sprintf("%6.1fs | maxd=%3d%% | div=%3d | %s", t["elapsed"], maxd_pct, div, sampler)
  }, error = function(e) {
    sprintf("ERROR: %s", conditionMessage(e))
  })

  cat(result, "\n")
  flush.console()
}

cat("=============================================================================\n")
cat("Adaptive NUTS->HMC(L=10) Benchmark | N=", N, " iter=", ITER, " warmup=", WARMUP, "\n")
cat("=============================================================================\n\n")

cat("Model                                     Time  | maxd    | div | Sampler\n")
cat(strrep("-", 80), "\n")

# --- Poisson-Gamma family ---
cat("\n--- Poisson-Gamma ---\n")
bm("PG base",          y|denom ~ x, df_pg, ratiod_poisson_gamma())
bm("PG + RE",          y|denom ~ x + (1|site), df_pg, ratiod_poisson_gamma())
bm("PG + ICAR",        y|denom ~ x + (1|site), df_pg, ratiod_poisson_gamma(),
   spatial = spatial_car(adj_mat, group_var = "site"))
bm("PG + BYM2",        y|denom ~ x + (1|site), df_pg, ratiod_poisson_gamma(),
   spatial = spatial_bym2(adj_mat, group_var = "site"))
bm("PG + RW1",         y|denom ~ x + (1|site), df_pg, ratiod_poisson_gamma(),
   temporal = temporal_rw1(~time, group_var = "site"))
bm("PG + AR1",         y|denom ~ x + (1|site), df_pg, ratiod_poisson_gamma(),
   temporal = temporal_ar1(~time, group_var = "site"))

# --- NegBin-NegBin family ---
cat("\n--- NegBin-NegBin ---\n")
bm("NB base",          y|denom ~ x, df_nb, ratiod_negbin_negbin())
bm("NB + RE",          y|denom ~ x + (1|site), df_nb, ratiod_negbin_negbin())
bm("NB + slopes",      y|denom ~ x + (1+x|site), df_nb_slopes, ratiod_negbin_negbin())
bm("NB + ICAR",        y|denom ~ x + (1|site), df_nb, ratiod_negbin_negbin(),
   spatial = spatial_car(adj_mat, group_var = "site"))
bm("NB + BYM2",        y|denom ~ x + (1|site), df_nb, ratiod_negbin_negbin(),
   spatial = spatial_bym2(adj_mat, group_var = "site"))
bm("NB + RW1",         y|denom ~ x + (1|site), df_nb, ratiod_negbin_negbin(),
   temporal = temporal_rw1(~time, group_var = "site"))
bm("NB + AR1",         y|denom ~ x + (1|site), df_nb, ratiod_negbin_negbin(),
   temporal = temporal_ar1(~time, group_var = "site"))
bm("NB + ICAR + RW1",  y|denom ~ x + (1|site), df_nb, ratiod_negbin_negbin(),
   spatial = spatial_car(adj_mat, group_var = "site"),
   temporal = temporal_rw1(~time, group_var = "site"))
bm("NB + ZI",          y|denom ~ x + (1|site), df_nb, ratiod_zinegbin_negbin())
bm("NB + Hurdle",      y|denom ~ x + (1|site), df_nb, ratiod_hurdle_negbin_negbin())

# --- Binomial family ---
cat("\n--- Binomial ---\n")
bm("Bin base",         y|trials ~ x, df_bin, ratiod_binomial())
bm("Bin + RE",         y|trials ~ x + (1|site), df_bin, ratiod_binomial())
bm("Bin + slopes",     y|trials ~ x + (1+x|site), df_bin, ratiod_binomial())
bm("Bin + ICAR",       y|trials ~ x + (1|site), df_bin, ratiod_binomial(),
   spatial = spatial_car(adj_mat, group_var = "site"))
bm("Bin + BYM2",       y|trials ~ x + (1|site), df_bin, ratiod_binomial(),
   spatial = spatial_bym2(adj_mat, group_var = "site"))
bm("Bin + RW1",        y|trials ~ x + (1|site), df_bin, ratiod_binomial(),
   temporal = temporal_rw1(~time, group_var = "site"))
bm("Bin + AR1",        y|trials ~ x + (1|site), df_bin, ratiod_binomial(),
   temporal = temporal_ar1(~time, group_var = "site"))
bm("Bin + ZI",         y|trials ~ x + (1|site), df_bin, ratiod_zibinomial())
bm("Bin + OI",         y|trials ~ x + (1|site), df_bin, ratiod_oibinomial())

# --- Forced L=10 comparison for key models ---
cat("\n--- Forced L=10 comparison (key models) ---\n")
bm("NB + slopes (L=10)",  y|denom ~ x + (1+x|site), df_nb_slopes,
   ratiod_negbin_negbin(), L = 10)
bm("NB + BYM2 (L=10)",    y|denom ~ x + (1|site), df_nb,
   ratiod_negbin_negbin(), L = 10,
   spatial = spatial_bym2(adj_mat, group_var = "site"))
bm("NB + AR1 (L=10)",     y|denom ~ x + (1|site), df_nb,
   ratiod_negbin_negbin(), L = 10,
   temporal = temporal_ar1(~time, group_var = "site"))

cat("\n=== DONE ===\n")
