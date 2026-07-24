library(numdenom)
N <- 500L; N_s <- 50L
set.seed(3)
site <- factor(rep(1:N_s, length.out = N))
x <- rnorm(N)
n_side <- ceiling(sqrt(N_s))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_s, ]
coords <- grid[as.integer(site), ]
df <- data.frame(
  y = rnbinom(N, mu = exp(2 + 0.3*x), size = 5),
  denom = pmax(rnbinom(N, mu = 100, size = 10), 1L),
  x = x, site = site, lon = coords$lon, lat = coords$lat
)
t <- system.time({
  fit <- tratio(y | denom ~ x + (1|site), data = df, family = ratiod_negbin_negbin(),
                spatial = spatial_hsgp(coords = c("lon", "lat")),
                control = list(iter = 500, warmup = 250, chains = 1, verbose = TRUE))
})[["elapsed"]]
cat(sprintf("\nTotal: %.1fs\n", t))
