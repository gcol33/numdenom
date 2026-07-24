#!/usr/bin/env Rscript
# Fair comparison: numdenom models matching Stan's exact structure
library(numdenom)

set.seed(42)
N <- 500L; N_s <- 50L; N_t <- 20L
site <- factor(rep(1:N_s, length.out = N))
x <- rnorm(N)
time_num <- rep(1:N_t, length.out = N)

n_side <- ceiling(sqrt(N_s))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_s, ]
si <- as.integer(site)
lon <- grid$lon[si]; lat <- grid$lat[si]

adj_mat <- matrix(0L, N_s, N_s)
for (i in 1:N_s) for (j in 1:N_s) {
  if (i != j && sqrt((grid$lon[i] - grid$lon[j])^2 +
                      (grid$lat[i] - grid$lat[j])^2) <= 1.5)
    adj_mat[i, j] <- 1L
}

y_nb_num <- rnbinom(N, mu = exp(2 + 0.3 * x), size = 5)
y_nb_denom <- rnbinom(N, mu = 100, size = 10)
y_nb_denom[y_nb_denom == 0] <- 1L

y_pg_num <- rpois(N, exp(2 + 0.5 * x))
y_pg_denom <- rgamma(N, 10, 1)

trials <- sample(10:50, N, replace = TRUE)
y_bin <- rbinom(N, trials, plogis(0.5 + 0.3 * x))

stan_ref <- c("NB+ICAR" = 3.5, "NB+HSGP_noRE" = 2.9,
              "PG+GP_t_noRE" = 10.0, "Bin+GP_t_noRE" = 3.3)

cat("\n=== 1. NB+ICAR (FAIR: both have RE + ICAR) ===\n")
df1 <- data.frame(y = y_nb_num, denom = y_nb_denom, x = x,
                   site = site, spatial_site = site)
t1 <- system.time(fit1 <- tratio(
  y | denom ~ x + (1 | site), data = df1, family = ratiod_negbin_negbin(),
  spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
  control = list(iter = 500, warmup = 250, chains = 1, verbose = TRUE)
))["elapsed"]
cat(sprintf("  numdenom: %.1fs (%d params), Stan: %.1fs => %.1fx\n",
            t1, ncol(fit1$fit_raw$samples[[1]]), 3.5, 3.5 / t1))

cat("\n=== 2. NB+HSGP (FAIR: no RE, matching Stan) ===\n")
df2 <- data.frame(y = y_nb_num, denom = y_nb_denom, x = x,
                   lon = lon, lat = lat)
t2 <- system.time(fit2 <- tratio(
  y | denom ~ x, data = df2, family = ratiod_negbin_negbin(),
  spatial = spatial_hsgp(coords = ~ lon + lat),
  control = list(iter = 500, warmup = 250, chains = 1, verbose = TRUE)
))["elapsed"]
cat(sprintf("  numdenom: %.1fs (%d params), Stan: %.1fs => %.1fx\n",
            t2, ncol(fit2$fit_raw$samples[[1]]), 2.9, 2.9 / t2))

cat("\n=== 3. PG+GP_t (FAIR: no RE, matching Stan) ===\n")
df3 <- data.frame(y = y_pg_num, denom = y_pg_denom, x = x,
                   time_num = time_num)
t3 <- system.time(fit3 <- tratio(
  y | denom ~ x, data = df3, family = ratiod_poisson_gamma(),
  temporal = temporal_gp("time_num"),
  control = list(iter = 500, warmup = 250, chains = 1, verbose = TRUE)
))["elapsed"]
cat(sprintf("  numdenom: %.1fs (%d params), Stan: %.1fs => %.1fx\n",
            t3, ncol(fit3$fit_raw$samples[[1]]), 10.0, 10.0 / t3))

cat("\n=== 4. Bin+GP_t (FAIR: no RE, matching Stan) ===\n")
df4 <- data.frame(y = y_bin, trials = trials, x = x,
                   time_num = time_num)
t4 <- system.time(fit4 <- tratio(
  y | trials ~ x, data = df4, family = ratiod_binomial(),
  temporal = temporal_gp("time_num"),
  control = list(iter = 500, warmup = 250, chains = 1, verbose = TRUE)
))["elapsed"]
cat(sprintf("  numdenom: %.1fs (%d params), Stan: %.1fs => %.1fx\n",
            t4, ncol(fit4$fit_raw$samples[[1]]), 3.3, 3.3 / t4))

cat("\n========== FAIR COMPARISON SUMMARY ==========\n")
cat(sprintf("%-20s %8s %8s %8s\n", "Model", "numdenom", "Stan", "Ratio"))
cat(paste0(rep("-", 48), collapse = ""), "\n")
for (nm in c("NB+ICAR", "NB+HSGP", "PG+GP_t", "Bin+GP_t")) {
  nd <- switch(nm, "NB+ICAR" = t1, "NB+HSGP" = t2, "PG+GP_t" = t3, "Bin+GP_t" = t4)
  st <- switch(nm, "NB+ICAR" = 3.5, "NB+HSGP" = 2.9, "PG+GP_t" = 10.0, "Bin+GP_t" = 3.3)
  ratio <- nd / st
  v <- if (ratio <= 1) "WIN" else if (ratio <= 1.2) "~" else "LOSS"
  cat(sprintf("%-20s %7.1fs %7.1fs %7.1fx %s\n", nm, nd, st, 1/ratio, v))
}
