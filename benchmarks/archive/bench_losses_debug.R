#!/usr/bin/env Rscript
# Diagnose LOSS models with verbose output
library(numdenom)
set.seed(123)

N <- 500L; N_s <- 50L; N_t <- 20L
site <- factor(rep(1:N_s, length.out = N))
time <- factor(rep(1:N_t, length.out = N))
time_num <- rep(1:N_t, length.out = N)
x <- rnorm(N)

n_side <- ceiling(sqrt(N_s))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_s, ]
site_int <- as.integer(site)
lon <- grid$lon[site_int]; lat <- grid$lat[site_int]

adj_mat <- matrix(0L, N_s, N_s)
for (i in 1:N_s) for (j in 1:N_s) {
  if (i != j && sqrt((grid$lon[i]-grid$lon[j])^2+(grid$lat[i]-grid$lat[j])^2) <= 1.5)
    adj_mat[i, j] <- 1L
}

# NB data
y_nb <- rnbinom(N, mu = exp(2 + 0.3*x), size = 5)
d_nb <- rnbinom(N, mu = 100, size = 10); d_nb[d_nb==0] <- 1L
df_nb <- data.frame(y=y_nb, denom=d_nb, x=x, site=site, time=time,
                    time_num=time_num, lon=lon, lat=lat, spatial_site=site)

# PG data
y_pg <- rpois(N, exp(2 + 0.5*x))
d_pg <- rgamma(N, 10, 1)
df_pg <- data.frame(y=y_pg, denom=d_pg, x=x, site=site, time=time,
                    time_num=time_num, lon=lon, lat=lat, spatial_site=site)

# Bin data
trials <- sample(10:50, N, replace=TRUE)
y_bin <- rbinom(N, trials, plogis(0.5 + 0.3*x))
df_bin <- data.frame(y=y_bin, trials=trials, x=x, site=site, time=time,
                     time_num=time_num, lon=lon, lat=lat, spatial_site=site)

run <- function(label, ...) {
  cat(sprintf("\n\n====== %s ======\n", label))
  t <- system.time(fit <- tratio(..., control = list(iter=500, warmup=250, chains=1, verbose=TRUE)))["elapsed"]
  cat(sprintf("\n  TIME: %.1fs\n", t))
  invisible(fit)
}

# 1. NB+ICAR (14.5s vs Stan 3.5s)
run("NB+ICAR", y | denom ~ x + (1|site), data=df_nb,
    family=ratiod_negbin_negbin(),
    spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"))

# 2. NB+HSGP (8.7s vs Stan 2.9s)
run("NB+HSGP", y | denom ~ x + (1|site), data=df_nb,
    family=ratiod_negbin_negbin(),
    spatial=spatial_hsgp(coords = ~ lon + lat))

# 3. PG+GP_t (22.1s vs Stan 10.0s)
run("PG+GP_t", y | denom ~ x + (1|site), data=df_pg,
    family=ratiod_poisson_gamma(),
    temporal=temporal_gp("time_num"))

# 4. Bin+GP_t (8.0s vs Stan 3.3s)
run("Bin+GP_t", y | trials ~ x + (1|site), data=df_bin,
    family=ratiod_binomial(),
    temporal=temporal_gp("time_num"))

# Also test with different metrics
cat("\n\n====== NB+ICAR metric=dense ======\n")
t <- system.time(tratio(y | denom ~ x + (1|site), data=df_nb,
    family=ratiod_negbin_negbin(),
    spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
    control = list(iter=500, warmup=250, chains=1, verbose=TRUE, metric="dense")))["elapsed"]
cat(sprintf("\n  TIME: %.1fs\n", t))

cat("\n\n====== NB+HSGP metric=block_diag ======\n")
t <- system.time(tratio(y | denom ~ x + (1|site), data=df_nb,
    family=ratiod_negbin_negbin(),
    spatial=spatial_hsgp(coords = ~ lon + lat),
    control = list(iter=500, warmup=250, chains=1, verbose=TRUE, metric="block_diag")))["elapsed"]
cat(sprintf("\n  TIME: %.1fs\n", t))

cat("\n\n====== PG+GP_t metric=block_diag ======\n")
t <- system.time(tratio(y | denom ~ x + (1|site), data=df_pg,
    family=ratiod_poisson_gamma(),
    temporal=temporal_gp("time_num"),
    control = list(iter=500, warmup=250, chains=1, verbose=TRUE, metric="block_diag")))["elapsed"]
cat(sprintf("\n  TIME: %.1fs\n", t))

cat("\n\n====== Bin+GP_t metric=block_diag ======\n")
t <- system.time(tratio(y | trials ~ x + (1|site), data=df_bin,
    family=ratiod_binomial(),
    temporal=temporal_gp("time_num"),
    control = list(iter=500, warmup=250, chains=1, verbose=TRUE, metric="block_diag")))["elapsed"]
cat(sprintf("\n  TIME: %.1fs\n", t))
