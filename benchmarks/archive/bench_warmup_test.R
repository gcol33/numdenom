# Test: does more warmup improve total time for parity models?
# Compare warmup=250 (default) vs warmup=500 (double)
library(numdenom)
args <- commandArgs(trailingOnly = TRUE)
model <- args[1]
seed <- as.integer(args[2])
warmup <- as.integer(args[3])
N <- 500L; N_s <- 50L; N_t <- 20L

set.seed(seed)
site <- factor(rep(1:N_s, length.out = N))
time_num <- rep(1:N_t, length.out = N)
x <- rnorm(N)
n_side <- ceiling(sqrt(N_s))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_s, ]
adj_mat <- matrix(0L, N_s, N_s)
for (ii in 1:N_s) for (jj in 1:N_s) {
  if (ii != jj && sqrt((grid$lon[ii]-grid$lon[jj])^2+(grid$lat[ii]-grid$lat[jj])^2) <= 1.5)
    adj_mat[ii, jj] <- 1L
}

iter <- warmup + 250L  # always 250 post-warmup samples

t <- system.time({
  if (model == "NB_HSGP") {
    coords <- grid[as.integer(site), ]
    df <- data.frame(y = rnbinom(N, mu = exp(2 + 0.3*x), size = 5),
                     denom = pmax(rnbinom(N, mu = 100, size = 10), 1L),
                     x = x, site = site, lon = coords$lon, lat = coords$lat)
    fit <- tratio(y | denom ~ x + (1|site), data = df, family = ratiod_negbin_negbin(),
                  spatial = spatial_hsgp(coords = c("lon", "lat")),
                  control = list(iter = iter, warmup = warmup, chains = 1, verbose = FALSE))
  } else if (model == "NB_ICAR") {
    df <- data.frame(y = rnbinom(N, mu = exp(2 + 0.3*x), size = 5),
                     denom = pmax(rnbinom(N, mu = 100, size = 10), 1L),
                     x = x, site = site, spatial_site = site)
    fit <- tratio(y | denom ~ x + (1|site), data = df, family = ratiod_negbin_negbin(),
                  spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                  control = list(iter = iter, warmup = warmup, chains = 1, verbose = FALSE))
  } else if (model == "Bin_slopes_ICAR") {
    trials <- sample(10:50, N, replace = TRUE)
    df <- data.frame(y = rbinom(N, trials, plogis(0.5 + 0.3*x)),
                     trials = trials, x = x, site = site, spatial_site = site)
    fit <- tratio(y | trials ~ x + (x|site), data = df, family = ratiod_binomial(),
                  spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                  control = list(iter = iter, warmup = warmup, chains = 1, verbose = FALSE))
  }
})[["elapsed"]]
cat(sprintf("%s w=%d seed=%d: %.1fs\n", model, warmup, seed, t))
