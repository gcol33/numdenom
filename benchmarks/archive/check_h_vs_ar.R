# Compare H vs A_r (arena reverse-mode autodiff) for 3 LOSS models
# A_r is O(N) like H, so timing comparison is fair.
# If A_r gets better epsilon/treedepth, H has a gradient issue.

library(numdenom)

N_OBS <- 500L
N_SITES <- 50L
N_TIMES <- 20L
SEED <- 123L
set.seed(SEED)

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

run_bench <- function(label, grad_mode, ...) {
  cat(sprintf("  %-5s: ", grad_mode))
  gc()
  t <- system.time({
    fit <- tryCatch(
      suppressWarnings(
        tratio(..., mode = "hmc",
               control = list(iter = 500, warmup = 250, chains = 1, seed = SEED, gradient_mode = grad_mode, verbose = FALSE))
      ),
      error = function(e) { cat(sprintf("ERROR: %s", conditionMessage(e))); NULL }
    )
  })["elapsed"]
  if (!is.null(fit)) {
    cat(sprintf("%.1fs", t))
    # Extract divergence count from fit
    if (!is.null(fit$diagnostics) && !is.null(fit$diagnostics$n_divergent)) {
      cat(sprintf("  (div=%d)", fit$diagnostics$n_divergent))
    }
    cat("\n")
  } else {
    cat("\n")
  }
  invisible(t)
}

cat("Comparing H vs A_r mode (both O(N)) for 3 LOSS models\n")
cat("======================================================\n")
cat("If A_r is significantly faster, H has a gradient quality issue.\n\n")

cat("=== Row 14: PG+GP_t ===\n")
t14_H <- run_bench("PG+GP_t", "H",
  y | denom ~ x + (1|site), data = df_pg,
  family = ratiod_poisson_gamma(),
  temporal = temporal_gp("time_num"))
t14_Ar <- run_bench("PG+GP_t", "A_r",
  y | denom ~ x + (1|site), data = df_pg,
  family = ratiod_poisson_gamma(),
  temporal = temporal_gp("time_num"))

cat("\n=== Row 74: Bin+GP_t ===\n")
t74_H <- run_bench("Bin+GP_t", "H",
  y | trials ~ x + (1|site), data = df_bin,
  family = ratiod_binomial(),
  temporal = temporal_gp("time_num"))
t74_Ar <- run_bench("Bin+GP_t", "A_r",
  y | trials ~ x + (1|site), data = df_bin,
  family = ratiod_binomial(),
  temporal = temporal_gp("time_num"))

cat("\n=== Row 91: Bin+ST_IV ===\n")
nd_sp <- spatial_car(adj_mat, level = "group", group_var = "spatial_site")
nd_tp <- temporal_rw1("time")
nd_st <- spatiotemporal(spatial = nd_sp, temporal = nd_tp, type = "IV")

t91_H <- run_bench("Bin+ST_IV", "H",
  y | trials ~ x + (1|site), data = df_bin,
  family = ratiod_binomial(),
  spatial = nd_sp, temporal = nd_tp, spatiotemporal = nd_st)
t91_Ar <- run_bench("Bin+ST_IV", "A_r",
  y | trials ~ x + (1|site), data = df_bin,
  family = ratiod_binomial(),
  spatial = nd_sp, temporal = nd_tp, spatiotemporal = nd_st)

cat("\n======================================================\n")
cat("Summary (H vs A_r):\n")
cat(sprintf("  PG+GP_t:   H=%.1fs  A_r=%.1fs  ratio=%.2fx  (Stan 10.0s)\n",
            t14_H, t14_Ar, t14_H / t14_Ar))
cat(sprintf("  Bin+GP_t:  H=%.1fs  A_r=%.1fs  ratio=%.2fx  (Stan 3.3s)\n",
            t74_H, t74_Ar, t74_H / t74_Ar))
cat(sprintf("  Bin+ST_IV: H=%.1fs  A_r=%.1fs  ratio=%.2fx  (Stan 56.0s)\n",
            t91_H, t91_Ar, t91_H / t91_Ar))
cat("\nRatio > 1.5 = H mode has gradient quality issue\n")
cat("Ratio ~ 1.0 = architectural (same NUTS geometry, H just faster per-eval)\n")
