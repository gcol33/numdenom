# Compare DENSE vs DIAG metric for GP_t and slopes+ICAR
suppressPackageStartupMessages(library(numdenom))

set.seed(123)
# Use same data generation as bench_single_row.R
N <- 500L; NS <- 50L; NT <- 20L
site <- factor(rep(1:NS, length.out = N))
time <- rep(1:NT, length.out = N)
time_factor <- factor(time)
x <- rnorm(N); z <- rnorm(N)
n_side <- ceiling(sqrt(NS))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:NS, ]
si <- as.integer(site)
lon_site <- grid$lon[si]; lat_site <- grid$lat[si]
adj <- matrix(0L, NS, NS)
for (i in 1:NS) {
  for (j in 1:NS) {
    if (i != j && sum(abs(grid[i,] - grid[j,])) == 1) adj[i,j] <- 1L
  }
}

# NB data
eta <- 2 + 0.3*x + rnorm(NS, 0, 0.5)[si]
y <- rnbinom(N, mu = exp(eta), size = 5)
d <- rnbinom(N, mu = exp(1.5 + 0.1*x), size = 5) + 1L
df <- data.frame(y=y, d=d, x=x, z=z, site=site, time=time_factor,
                 time_num=as.numeric(time), spatial_site=site,
                 lon=lon_site, lat=lat_site)

cat("=== GP_t (NB, ~row 44) DENSE vs DIAG ===\n")
for (met in c("dense", "diag")) {
  t0 <- proc.time()["elapsed"]
  fit <- tryCatch(
    tratio(y | d ~ x + (1|site),
           data = df,
           family = ratiod_negbin_negbin(),
           temporal = temporal_gp("time_num"),
           mode = "hmc",
           control = list(gradient_mode = "H", metric = met, iter = 500L, warmup = 250L, chains = 1L, verbose = FALSE)),
    error = function(e) paste("ERROR:", e$message)
  )
  t1 <- proc.time()["elapsed"]
  if (is.character(fit)) {
    cat(sprintf("  metric=%s: %s\n", met, fit))
  } else {
    cat(sprintf("  metric=%s: %.1fs\n", met, t1-t0))
  }
}

cat("\n=== slopes+ICAR (NB, ~row 55) DENSE vs DIAG ===\n")
for (met in c("dense", "diag")) {
  t0 <- proc.time()["elapsed"]
  fit <- tryCatch(
    tratio(y | d ~ x + (1 + z | site),
           data = df,
           family = ratiod_negbin_negbin(),
           spatial = spatial_car(adj, level = "group", group_var = "spatial_site"),
           mode = "hmc",
           control = list(gradient_mode = "H", metric = met, iter = 500L, warmup = 250L, chains = 1L, verbose = FALSE)),
    error = function(e) paste("ERROR:", e$message)
  )
  t1 <- proc.time()["elapsed"]
  if (is.character(fit)) {
    cat(sprintf("  metric=%s: %s\n", met, fit))
  } else {
    cat(sprintf("  metric=%s: %.1fs\n", met, t1-t0))
  }
}
