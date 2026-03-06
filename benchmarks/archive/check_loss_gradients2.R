# Deeper check for 3 LOSS models:
# 1. Does H vs N pass at init?
# 2. What's the epsilon, treedepth distribution?
# 3. Does A_r mode give different epsilon/treedepth?

library(numdenom)

N_OBS <- 500L
N_SITES <- 50L
N_TIMES <- 20L
SEED <- 123L
set.seed(SEED)

# Same data generation as bench_single_row.R
site <- factor(rep(1:N_SITES, length.out = N_OBS))
time <- rep(1:N_TIMES, length.out = N_OBS)
x <- rnorm(N_OBS)

n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_SITES, ]
site_int <- as.integer(site)

adj_mat <- matrix(0L, N_SITES, N_SITES)
for (i in 1:N_SITES) {
  for (j in 1:N_SITES) {
    if (i != j) {
      d <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
      if (d <= 1.5) adj_mat[i, j] <- 1L
    }
  }
}

# Use same data format as bench_single_row.R
y_pg_num   <- rpois(N_OBS, exp(2 + 0.5 * x))
y_pg_denom <- rgamma(N_OBS, 10, 1)
y_bin_num  <- rbinom(N_OBS, 20, 0.3)

df_pg <- data.frame(y = y_pg_num, denom = y_pg_denom, x = x,
                    site = site, time = factor(time), time_num = time,
                    lon = grid$lon[site_int], lat = grid$lat[site_int],
                    spatial_site = site)
df_bin <- data.frame(y = y_bin_num, trials = rep(20L, N_OBS), x = x,
                     site = site, time = factor(time), time_num = time,
                     lon = grid$lon[site_int], lat = grid$lat[site_int],
                     spatial_site = site)

bench_mode <- function(label, grad_mode, ...) {
  cat(sprintf("  %-5s: ", grad_mode))
  gc()
  t <- system.time({
    fit <- tryCatch(
      ratiod(..., mode = "hmc", iter = 500, warmup = 250,
             chains = 1, seed = SEED, gradient_mode = grad_mode, verbose = FALSE),
      error = function(e) { cat(sprintf("ERROR: %s\n", conditionMessage(e))); NULL },
      warning = function(w) {
        if (grepl("gradient mismatch|Falling back", conditionMessage(w))) {
          cat("FALLBACK! ")
        }
        suppressWarnings(
          ratiod(..., mode = "hmc", iter = 500, warmup = 250,
                 chains = 1, seed = SEED, gradient_mode = grad_mode, verbose = FALSE)
        )
      }
    )
  })["elapsed"]
  if (!is.null(fit)) {
    cat(sprintf("%.1fs\n", t))
  }
  invisible(t)
}

cat("=== Row 14: PG+GP_t ===\n")
t14_H <- bench_mode("PG+GP_t", "H",
  y | denom ~ x + (1|site), data = df_pg,
  family = ratiod_poisson_gamma(),
  temporal = temporal_gp("time_num"))
t14_A <- bench_mode("PG+GP_t", "A",
  y | denom ~ x + (1|site), data = df_pg,
  family = ratiod_poisson_gamma(),
  temporal = temporal_gp("time_num"))

cat("\n=== Row 74: Bin+GP_t ===\n")
t74_H <- bench_mode("Bin+GP_t", "H",
  y | trials ~ x + (1|site), data = df_bin,
  family = ratiod_binomial(),
  temporal = temporal_gp("time_num"))
t74_A <- bench_mode("Bin+GP_t", "A",
  y | trials ~ x + (1|site), data = df_bin,
  family = ratiod_binomial(),
  temporal = temporal_gp("time_num"))

cat("\n=== Row 91: Bin+ST_IV ===\n")
nd_sp <- spatial_car(adj_mat, level = "group", group_var = "spatial_site")
nd_tp <- temporal_rw1("time")
nd_st <- spatiotemporal(spatial = nd_sp, temporal = nd_tp, type = "IV")

t91_H <- bench_mode("Bin+ST_IV", "H",
  y | trials ~ x + (1|site), data = df_bin,
  family = ratiod_binomial(),
  spatial = nd_sp, temporal = nd_tp, spatiotemporal = nd_st)
t91_A <- bench_mode("Bin+ST_IV", "A",
  y | trials ~ x + (1|site), data = df_bin,
  family = ratiod_binomial(),
  spatial = nd_sp, temporal = nd_tp, spatiotemporal = nd_st)

cat("\n=== Summary (H vs A timing) ===\n")
cat(sprintf("  PG+GP_t:   H=%.1fs  A=%.1fs  (Stan 10.0s)\n", t14_H, t14_A))
cat(sprintf("  Bin+GP_t:  H=%.1fs  A=%.1fs  (Stan 3.3s)\n", t74_H, t74_A))
cat(sprintf("  Bin+ST_IV: H=%.1fs  A=%.1fs  (Stan 56.0s)\n", t91_H, t91_A))
cat("\nIf A is much faster than H, that indicates an H-mode gradient issue\n")
cat("(correct but causing poor NUTS geometry).\n")
