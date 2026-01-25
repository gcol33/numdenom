# Row 7: poisson_gamma + GP
library(numdenom)
set.seed(123)

N <- 500
N_SITES <- 50
x <- rnorm(N)
site <- factor(rep(1:N_SITES, length.out = N))

n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_SITES, ]
site_int <- as.integer(site)
lon <- grid$lon[site_int]
lat <- grid$lat[site_int]

y <- rpois(N, exp(2 + 0.5*x))
effort <- rgamma(N, 10, 1)
df <- data.frame(y=y, effort=effort, x=x, site=site, lon=lon, lat=lat)

cat("Row 7: poisson_gamma + GP\n")
time_H <- system.time({
  fit <- ratiod(y | effort ~ x + (1 | site), data=df,
                family=ratiod_poisson_gamma(),
                spatial=spatial_gp(coords = ~ lon + lat),
                iter=500, warmup=250, chains=1, verbose=FALSE,
                gradient_mode="H")
})["elapsed"]
cat(sprintf("  H: %.1fs\n", time_H))
