# Quick benchmark: key rows affected by DIAG->DENSE recovery change
suppressPackageStartupMessages(library(numdenom))

N_OBS <- 500L; N_ITER <- 500L; N_WARMUP <- 250L; N_CHAINS <- 1L
N_SITES <- 50L; N_TIMES <- 20L; SEED <- 123L

set.seed(SEED)
site <- factor(rep(1:N_SITES, length.out = N_OBS))
time <- rep(1:N_TIMES, length.out = N_OBS)
time_factor <- factor(time)
x <- rnorm(N_OBS)

n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_SITES, ]
site_int <- as.integer(site)
lon_site <- grid$lon[site_int]
lat_site <- grid$lat[site_int]

adj_mat <- matrix(0L, N_SITES, N_SITES)
for (i in 1:N_SITES) for (j in 1:N_SITES) {
  if (i != j) {
    d <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
    if (d <= 1.5) adj_mat[i, j] <- 1L
  }
}

y_pg_num <- rpois(N_OBS, exp(2 + 0.5 * x))
y_pg_denom <- rgamma(N_OBS, 10, 1)
df_pg <- data.frame(y = y_pg_num, denom = y_pg_denom, x = x,
                    site = site, time = time_factor,
                    lon = lon_site, lat = lat_site, spatial_site = site)

y_nb_num <- rnbinom(N_OBS, mu = exp(2 + 0.3 * x), size = 5)
y_nb_denom <- rnbinom(N_OBS, mu = 100, size = 10)
y_nb_denom[y_nb_denom == 0] <- 1L
df_nb <- data.frame(y = y_nb_num, denom = y_nb_denom, x = x,
                    site = site, time = time_factor,
                    lon = lon_site, lat = lat_site, spatial_site = site)

trials <- sample(10:50, N_OBS, replace = TRUE)
y_bin <- rbinom(N_OBS, trials, plogis(0.5 + 0.3 * x))
df_bin <- data.frame(y = y_bin, trials = trials, x = x,
                     site = site, time = time_factor,
                     lon = lon_site, lat = lat_site, spatial_site = site)

run_bench <- function(label, ...) {
  args <- list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
               verbose = FALSE, gradient_mode = "H")
  args <- c(args, list(...))
  elapsed <- tryCatch({
    system.time(do.call(ratiod, args))["elapsed"]
  }, error = function(e) NA_real_)
  cat(sprintf("ROW|%-25s|%6.1f\n", label, elapsed))
  elapsed
}

results <- list()

# Core models with Stan comparison numbers
results[["R1"]]  <- run_bench("R1  PG base",
  formula = y | denom ~ x, data = df_pg, family = ratiod_poisson_gamma())
results[["R2"]]  <- run_bench("R2  PG+RE",
  formula = y | denom ~ x + (1|site), data = df_pg, family = ratiod_poisson_gamma())
results[["R5"]]  <- run_bench("R5  PG+ICAR",
  formula = y | denom ~ x + (1|site), data = df_pg, family = ratiod_poisson_gamma(),
  spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"))
results[["R13"]] <- run_bench("R13 PG+AR1",
  formula = y | denom ~ x + (1|site), data = df_pg, family = ratiod_poisson_gamma(),
  temporal = temporal_ar1("time"))
results[["R20"]] <- run_bench("R20 PG+ICAR+AR1",
  formula = y | denom ~ x + (1|site), data = df_pg, family = ratiod_poisson_gamma(),
  spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
  temporal = temporal_ar1("time"))
results[["R31"]] <- run_bench("R31 NB base",
  formula = y | denom ~ x, data = df_nb, family = ratiod_negbin_negbin())
results[["R32"]] <- run_bench("R32 NB+RE",
  formula = y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin())
results[["R35"]] <- run_bench("R35 NB+ICAR",
  formula = y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin(),
  spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"))
results[["R43"]] <- run_bench("R43 NB+AR1",
  formula = y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin(),
  temporal = temporal_ar1("time"))
results[["R50"]] <- run_bench("R50 NB+ICAR+AR1",
  formula = y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin(),
  spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
  temporal = temporal_ar1("time"))
results[["R61"]] <- run_bench("R61 Bin base",
  formula = y | trials ~ x, data = df_bin, family = ratiod_binomial())
results[["R62"]] <- run_bench("R62 Bin+RE",
  formula = y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial())
results[["R65"]] <- run_bench("R65 Bin+ICAR",
  formula = y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
  spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"))
results[["R73"]] <- run_bench("R73 Bin+AR1",
  formula = y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
  temporal = temporal_ar1("time"))
results[["R82"]] <- run_bench("R82 Bin+ICAR+AR1",
  formula = y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
  spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
  temporal = temporal_ar1("time"))
results[["R11"]] <- run_bench("R11 PG+RW1",
  formula = y | denom ~ x + (1|site), data = df_pg, family = ratiod_poisson_gamma(),
  temporal = temporal_rw1("time"))
results[["R41"]] <- run_bench("R41 NB+RW1",
  formula = y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin(),
  temporal = temporal_rw1("time"))
