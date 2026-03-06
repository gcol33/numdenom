# Benchmark remaining poisson_gamma rows with stale notes
library(numdenom)
set.seed(123)

N <- 500
N_SITES <- 50
N_TIMES <- 20
x <- rnorm(N)
site <- factor(rep(1:N_SITES, length.out = N))
time <- factor(rep(1:N_TIMES, length.out = N))

n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_SITES, ]
site_int <- as.integer(site)
lon <- grid$lon[site_int]
lat <- grid$lat[site_int]

# Adjacency matrix
adj_mat <- matrix(0, N_SITES, N_SITES)
for (i in 1:N_SITES) {
  for (j in 1:N_SITES) {
    if (i != j) {
      dist <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
      if (dist <= 1.5) adj_mat[i, j] <- 1
    }
  }
}

spatial_site <- factor(rep(1:N_SITES, length.out = N))

y <- rpois(N, exp(2 + 0.5*x))
effort <- rgamma(N, 10, 1)
df <- data.frame(y=y, effort=effort, x=x, site=site, time=time,
                 lon=lon, lat=lat, spatial_site=spatial_site)

bench <- function(name, row, ...) {
  cat(sprintf("Row %d: %s... ", row, name))
  flush.console()
  tryCatch({
    time <- system.time({
      fit <- ratiod(..., iter=500, warmup=250, chains=1, verbose=FALSE, gradient_mode="H")
    })["elapsed"]
    cat(sprintf("%.1fs\n", time))
    time
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", conditionMessage(e)))
    NA
  })
}

results <- list()

# Row 8: MSGP (skip - too slow, already verified works)
# Row 10: RW2
results[["10"]] <- bench("RW2", 10,
  y | effort ~ x + (1 | site), data=df, family=ratiod_poisson_gamma(),
  temporal=temporal_rw2("time"))

# Row 11: AR1
results[["11"]] <- bench("AR1", 11,
  y | effort ~ x + (1 | site), data=df, family=ratiod_poisson_gamma(),
  temporal=temporal_ar1("time"))

# Row 13: Hurdle
results[["13"]] <- bench("Hurdle", 13,
  y | effort ~ x + (1 | site), data=df, family=ratiod_hurdle_pois())

# Row 19: slopes + ICAR
results[["19"]] <- bench("slopes + ICAR", 19,
  y | effort ~ x + (x | site), data=df, family=ratiod_poisson_gamma(),
  spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"))

cat("\nDone!\n")
print(results)
