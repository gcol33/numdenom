library(numdenom)
args <- commandArgs(trailingOnly = TRUE)
model <- args[1]
seed <- as.integer(args[2])
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

t <- system.time({
  if (model == "NB_ICAR") {
    df <- data.frame(
      y = rnbinom(N, mu = exp(2 + 0.3*x), size = 5),
      denom = pmax(rnbinom(N, mu = 100, size = 10), 1L),
      x = x, site = site, spatial_site = site
    )
    fit <- ratiod(y | denom ~ x + (1|site), data = df, family = ratiod_negbin_negbin(),
                  spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                  iter = 500, warmup = 250, chains = 1, verbose = FALSE)
  } else if (model == "NB_HSGP") {
    coords <- grid[as.integer(site), ]
    df <- data.frame(
      y = rnbinom(N, mu = exp(2 + 0.3*x), size = 5),
      denom = pmax(rnbinom(N, mu = 100, size = 10), 1L),
      x = x, site = site, lon = coords$lon, lat = coords$lat
    )
    fit <- ratiod(y | denom ~ x + (1|site), data = df, family = ratiod_negbin_negbin(),
                  spatial = spatial_hsgp(coords = c("lon", "lat")),
                  iter = 500, warmup = 250, chains = 1, verbose = FALSE)
  } else if (model == "PG_GPt") {
    df <- data.frame(
      y = rpois(N, exp(2 + 0.5*x)),
      denom = rgamma(N, 10, 1),
      x = x, time_num = time_num
    )
    fit <- ratiod(y | denom ~ x, data = df, family = ratiod_poisson_gamma(),
                  temporal = temporal_gp("time_num"),
                  iter = 500, warmup = 250, chains = 1, verbose = FALSE)
  } else if (model == "Bin_GPt") {
    trials <- sample(10:50, N, replace = TRUE)
    df <- data.frame(
      y = rbinom(N, trials, plogis(0.5 + 0.3*x)),
      trials = trials, x = x, time_num = time_num
    )
    fit <- ratiod(y | trials ~ x, data = df, family = ratiod_binomial(),
                  temporal = temporal_gp("time_num"),
                  iter = 500, warmup = 250, chains = 1, verbose = FALSE)
  } else if (model == "PG_ST_IV") {
    df <- data.frame(
      y = rpois(N, exp(2 + 0.5*x)),
      denom = rgamma(N, 10, 1),
      x = x, site = site, spatial_site = site, time_num = time_num
    )
    sp <- spatial_car(adj_mat, level = "group", group_var = "spatial_site")
    tp <- temporal_rw1("time_num")
    fit <- ratiod(y | denom ~ x, data = df, family = ratiod_poisson_gamma(),
                  spatial = sp, temporal = tp,
                  spatiotemporal = spatiotemporal(spatial = sp, temporal = tp, type = "IV"),
                  iter = 500, warmup = 250, chains = 1, verbose = FALSE)
  } else if (model == "Bin_ST_IV") {
    trials <- sample(10:50, N, replace = TRUE)
    df <- data.frame(
      y = rbinom(N, trials, plogis(0.5 + 0.3*x)),
      trials = trials, x = x, site = site, spatial_site = site, time_num = time_num
    )
    sp <- spatial_car(adj_mat, level = "group", group_var = "spatial_site")
    tp <- temporal_rw1("time_num")
    fit <- ratiod(y | trials ~ x, data = df, family = ratiod_binomial(),
                  spatial = sp, temporal = tp,
                  spatiotemporal = spatiotemporal(spatial = sp, temporal = tp, type = "IV"),
                  iter = 500, warmup = 250, chains = 1, verbose = FALSE)
  }
})[["elapsed"]]
cat(sprintf("%s seed=%d: %.1fs\n", model, seed, t))
