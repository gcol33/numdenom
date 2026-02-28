# Benchmark remaining binomial rows with stale notes
# Standard parameters: N=500, iter=500, warmup=250, chains=1, N_SITES=50, N_TIMES=20
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

trials <- sample(10:50, N, replace=TRUE)
y <- rbinom(N, trials, plogis(0.5 + 0.3*x))
df <- data.frame(y=y, trials=trials, x=x, site=site, time=time,
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

# Row 43: slopes
results[["43"]] <- bench("slopes", 43,
  y | trials ~ x + (x | site), data=df, family=ratiod_binomial())

# Row 49: RW1
results[["49"]] <- bench("RW1", 49,
  y | trials ~ x + (1 | site), data=df, family=ratiod_binomial(),
  temporal=temporal_rw1("time"))

# Row 50: RW2
results[["50"]] <- bench("RW2", 50,
  y | trials ~ x + (1 | site), data=df, family=ratiod_binomial(),
  temporal=temporal_rw2("time"))

# Row 51: AR1
results[["51"]] <- bench("AR1", 51,
  y | trials ~ x + (1 | site), data=df, family=ratiod_binomial(),
  temporal=temporal_ar1("time"))

# Row 54: OI
results[["54"]] <- bench("OI", 54,
  y | trials ~ x + (1 | site), data=df, family=ratiod_oibinomial())

# Row 55: ZOIB
results[["55"]] <- bench("ZOIB", 55,
  y | trials ~ x + (1 | site), data=df, family=ratiod_zoibinomial())

# Row 61: slopes + ICAR
results[["61"]] <- bench("slopes + ICAR", 61,
  y | trials ~ x + (x | site), data=df, family=ratiod_binomial(),
  spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"))

cat("\nDone!\n")
print(results)
