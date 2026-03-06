# Benchmark remaining negbin_negbin rows with stale notes
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

y <- rnbinom(N, mu=exp(2 + 0.3*x), size=5)
denom <- rnbinom(N, mu=100, size=10)
denom[denom == 0] <- 1
df <- data.frame(y=y, denom=denom, x=x, site=site, time=time,
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

# Row 23: slopes
results[["23"]] <- bench("slopes", 23,
  y | denom ~ x + (x | site), data=df, family=ratiod_negbin_negbin())

# Row 29: RW1
results[["29"]] <- bench("RW1", 29,
  y | denom ~ x + (1 | site), data=df, family=ratiod_negbin_negbin(),
  temporal=temporal_rw1("time"))

# Row 30: RW2
results[["30"]] <- bench("RW2", 30,
  y | denom ~ x + (1 | site), data=df, family=ratiod_negbin_negbin(),
  temporal=temporal_rw2("time"))

# Row 31: AR1
results[["31"]] <- bench("AR1", 31,
  y | denom ~ x + (1 | site), data=df, family=ratiod_negbin_negbin(),
  temporal=temporal_ar1("time"))

# Row 32: ZI
results[["32"]] <- bench("ZI", 32,
  y | denom ~ x + (1 | site), data=df, family=ratiod_zinegbin())

# Row 33: Hurdle
results[["33"]] <- bench("Hurdle", 33,
  y | denom ~ x + (1 | site), data=df, family=ratiod_hurdle_negbin())

# Row 39: slopes + ICAR
results[["39"]] <- bench("slopes + ICAR", 39,
  y | denom ~ x + (x | site), data=df, family=ratiod_negbin_negbin(),
  spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"))

cat("\nDone!\n")
print(results)
