library(numdenom)
N <- 500L; N_s <- 50L
set.seed(2)
site <- factor(rep(1:N_s, length.out = N))
x <- rnorm(N)
n_side <- ceiling(sqrt(N_s))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_s, ]
adj_mat <- matrix(0L, N_s, N_s)
for (ii in 1:N_s) for (jj in 1:N_s) {
  if (ii != jj && sqrt((grid$lon[ii]-grid$lon[jj])^2+(grid$lat[ii]-grid$lat[jj])^2) <= 1.5)
    adj_mat[ii, jj] <- 1L
}
df <- data.frame(
  y = rnbinom(N, mu = exp(2 + 0.3*x), size = 5),
  denom = pmax(rnbinom(N, mu = 100, size = 10), 1L),
  x = x, site = site, spatial_site = site
)
cat("=== Seed 2 verbose ===\n")
t <- system.time({
  fit <- ratiod(y | denom ~ x + (1|site), data = df, family = ratiod_negbin_negbin(),
                spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                iter = 500, warmup = 250, chains = 1, verbose = TRUE)
})[["elapsed"]]
cat(sprintf("\nTotal: %.1fs\n", t))
