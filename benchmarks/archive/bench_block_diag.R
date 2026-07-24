#!/usr/bin/env Rscript
# Benchmark BLOCK_DIAG vs DIAG vs DENSE for models that previously needed DENSE
library(numdenom)

run_bench <- function(label, fit_fn, seeds = 1:3) {
  for (m in c("auto", "diag")) {
    times <- numeric(length(seeds))
    divs <- numeric(length(seeds))
    for (i in seq_along(seeds)) {
      set.seed(seeds[i])
      t <- system.time({ fit <- fit_fn(m) })["elapsed"]
      times[i] <- t
      d <- sum(unlist(lapply(fit$fit_raw$divergent, sum)))
      divs[i] <- d
    }
    cat(sprintf("  %-10s median=%.2fs  [%s]  divs=[%s]\n", m,
                median(times),
                paste(round(times, 2), collapse=", "),
                paste(divs, collapse=", ")))
  }
}

# --- HSGP ---
cat("=== PG+HSGP ===\n")
set.seed(42)
N <- 500; N_s <- 50
site <- rep(1:N_s, each = N / N_s)
coords <- cbind(runif(N_s), runif(N_s))
x <- rnorm(N)
y_num <- rpois(N, exp(1 + 0.3 * x))
y_denom <- rpois(N, exp(1))
df <- data.frame(y_num = y_num, y_denom = y_denom, x = x, site = factor(site))
df$lon <- coords[as.integer(site), 1]
df$lat <- coords[as.integer(site), 2]
run_bench("PG+HSGP", function(m) {
  tratio(y_num | y_denom ~ x, data = df, family = ratiod_poisson_gamma(),
         spatial = spatial_hsgp(coords = ~ lon + lat, m = 6),
         control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE, metric = m))
})

# --- BYM2 ---
cat("\n=== NB+BYM2 ===\n")
set.seed(42)
N_s <- 20; N_p <- 25; N <- N_s * N_p
site <- rep(1:N_s, each = N_p)
x <- rnorm(N)
y_num <- rpois(N, exp(1 + 0.3 * x))
y_denom <- rpois(N, exp(1))
df2 <- data.frame(y_num = y_num, y_denom = y_denom, x = x, site = factor(site))
ii <- c(); jj <- c()
for (i in 1:N_s) { j <- i %% N_s + 1; ii <- c(ii, i, j); jj <- c(jj, j, i) }
adj <- Matrix::sparseMatrix(i = ii, j = jj, x = 1, dims = c(N_s, N_s))
run_bench("NB+BYM2", function(m) {
  tratio(y_num | y_denom ~ x + (1 | site), data = df2,
         family = ratiod_negbin_negbin(),
         spatial = spatial_bym2(adj = adj, group_var = "site"),
         control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE, metric = m))
})

# --- GP spatial ---
cat("\n=== PG+GP ===\n")
set.seed(42)
N <- 200; N_s <- 30
site <- rep(1:N_s, length.out = N)
coords <- cbind(runif(N_s), runif(N_s))
x <- rnorm(N)
y_num <- rpois(N, exp(1 + 0.3 * x))
y_denom <- rpois(N, exp(1))
df3 <- data.frame(y_num = y_num, y_denom = y_denom, x = x, site = factor(site))
df3$lon <- coords[as.integer(site), 1]
df3$lat <- coords[as.integer(site), 2]
run_bench("PG+GP", function(m) {
  tratio(y_num | y_denom ~ x, data = df3, family = ratiod_poisson_gamma(),
         spatial = spatial_gp(coords = ~ lon + lat),
         control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE, metric = m))
})
