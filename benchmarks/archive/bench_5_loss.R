# Benchmark the 5 remaining models slower than Stan
# Rows: 14 (PG+GP_t), 66 (Bin+BYM2), 68 (Bin+HSGP), 74 (Bin+GP_t), 91 (Bin+ST_IV)

library(numdenom)

N_OBS <- 500
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1
N_SITES <- 50
N_TIMES <- 20
SEED <- 42

set.seed(SEED)

# Generate spatial adjacency
coords <- cbind(runif(N_SITES), runif(N_SITES))
W <- matrix(0, N_SITES, N_SITES)
for (i in 1:(N_SITES-1)) for (j in (i+1):N_SITES) {
  if (sqrt(sum((coords[i,]-coords[j,])^2)) < 0.3) {
    W[i,j] <- W[j,i] <- 1
  }
}
for (i in which(rowSums(W) == 0)) {
  dists <- as.matrix(dist(coords))[i, ]
  dists[i] <- Inf
  nearest <- which.min(dists)
  W[i, nearest] <- W[nearest, i] <- 1
}

# Binomial data
df_bin <- data.frame(
  y = rbinom(N_OBS, size = 20, prob = 0.3),
  n_trials = rep(20L, N_OBS),
  x = rnorm(N_OBS),
  site = rep(1:N_SITES, each = N_OBS/N_SITES),
  time = rep(1:N_TIMES, times = N_OBS/N_TIMES),
  lon = coords[rep(1:N_SITES, each = N_OBS/N_SITES), 1],
  lat = coords[rep(1:N_SITES, each = N_OBS/N_SITES), 2]
)

# PG data
df_pg <- data.frame(
  y_num = rpois(N_OBS, 5),
  y_denom = rpois(N_OBS, 10) + 1,
  x = rnorm(N_OBS),
  site = rep(1:N_SITES, each = N_OBS/N_SITES),
  time = rep(1:N_TIMES, times = N_OBS/N_TIMES),
  lon = coords[rep(1:N_SITES, each = N_OBS/N_SITES), 1],
  lat = coords[rep(1:N_SITES, each = N_OBS/N_SITES), 2]
)

bench <- function(label, ...) {
  cat(sprintf("\n=== %s ===\n", label))
  gc()
  t <- system.time({
    fit <- tryCatch(
      tratio(..., mode = "hmc",
             control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, seed = SEED, verbose = FALSE)),
      error = function(e) { cat("ERROR:", conditionMessage(e), "\n"); NULL }
    )
  })["elapsed"]
  cat(sprintf("  Time: %.1fs\n", t))
  invisible(t)
}

cat("Benchmarking 5 remaining LOSS models (N=500, 500 iter, 1 chain)\n")
cat("================================================================\n")

# Row 68: Bin+HSGP
t68 <- bench("Row 68: Bin+HSGP",
  y | n_trials ~ x + (1|site), data = df_bin,
  family = ratiod_binomial(), gradient_mode = "H",
  spatial = spatial_hsgp(~ lon + lat))

# Row 66: Bin+BYM2
t66 <- bench("Row 66: Bin+BYM2",
  y | n_trials ~ x + (1|site), data = df_bin,
  family = ratiod_binomial(), gradient_mode = "H",
  spatial = spatial_bym2(W, group_var = "site"))

# Row 74: Bin+GP_t
t74 <- bench("Row 74: Bin+GP_t",
  y | n_trials ~ x + (1|site), data = df_bin,
  family = ratiod_binomial(), gradient_mode = "H",
  temporal = temporal_gp(time_var = "time", group_var = "site"))

# Row 14: PG+GP_t
t14 <- bench("Row 14: PG+GP_t",
  y_num | y_denom ~ x + (1|site), data = df_pg,
  family = ratiod_poisson_gamma(), gradient_mode = "H",
  temporal = temporal_gp(time_var = "time", group_var = "site"))

# Row 91: Bin+ST_IV
t91 <- bench("Row 91: Bin+ST_IV",
  y | n_trials ~ x + (1|site), data = df_bin,
  family = ratiod_binomial(), gradient_mode = "H",
  spatial = spatial_car(W, group_var = "site"),
  temporal = temporal_rw1(time_var = "time", group_var = "site"),
  spatiotemporal = spatiotemporal(type = "IV"))

cat("\n================================================================\n")
cat("Summary:\n")
cat(sprintf("  Row 68 Bin+HSGP:  %.1fs (was 6.6s, Stan 2.9s)\n", t68))
cat(sprintf("  Row 66 Bin+BYM2:  %.1fs (was 18.4s, Stan 12.4s)\n", t66))
cat(sprintf("  Row 74 Bin+GP_t:  %.1fs (was 9.7s, Stan 3.3s)\n", t74))
cat(sprintf("  Row 14 PG+GP_t:   %.1fs (was 11.9s, Stan 10.0s)\n", t14))
cat(sprintf("  Row 91 Bin+ST_IV: %.1fs (was 80.7s, Stan 56.0s)\n", t91))
