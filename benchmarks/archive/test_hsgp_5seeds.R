#!/usr/bin/env Rscript
# NB+HSGP fair comparison: 5 seeds, no RE, matching Stan structure
library(numdenom)

seeds <- c(42L, 123L, 456L, 789L, 2024L)
stan_ref <- 2.9  # Stan HSGP reference time

results <- data.frame(seed = integer(), time = numeric(), div = integer())

for (s in seeds) {
  set.seed(s)
  N <- 500L; N_s <- 50L
  x <- rnorm(N)
  n_side <- ceiling(sqrt(N_s))
  grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_s, ]
  si <- rep(1:N_s, length.out = N)
  lon <- grid$lon[si]; lat <- grid$lat[si]

  y <- rnbinom(N, mu = exp(2 + 0.3 * x), size = 5)
  d <- rnbinom(N, mu = 100, size = 10)
  d[d == 0] <- 1L

  df <- data.frame(y = y, denom = d, x = x, lon = lon, lat = lat)

  t_s <- system.time(fit <- tratio(
    y | denom ~ x, data = df, family = ratiod_negbin_negbin(),
    spatial = spatial_hsgp(coords = ~ lon + lat),
    control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE)
  ))["elapsed"]

  n_div <- fit$diagnostics$n_divergent
  cat(sprintf("  Seed %d: %.1fs (%d div)\n", s, t_s, n_div))
  results <- rbind(results, data.frame(seed = s, time = t_s, div = n_div))
}

med <- median(results$time)
cat(sprintf("\n=== NB+HSGP (no RE, m=6) ===\n"))
cat(sprintf("  Median: %.1fs (Stan: %.1fs) => %.1fx\n", med, stan_ref, stan_ref / med))
cat(sprintf("  Range: %.1f - %.1fs\n", min(results$time), max(results$time)))
cat(sprintf("  Divergences: %s\n", paste(results$div, collapse = ", ")))
if (med <= stan_ref) cat("  VERDICT: WIN\n") else if (med <= stan_ref * 1.2) cat("  VERDICT: PARITY\n") else cat("  VERDICT: LOSS\n")
