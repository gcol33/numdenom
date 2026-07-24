# Benchmark spatial+temporal combinations
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

bench <- function(name, row, ...) {
  cat(sprintf("Row %d: %s... ", row, name))
  flush.console()
  tryCatch({
    time <- system.time({
      fit <- tratio(..., control = list(iter=500, warmup=250, chains=1, verbose=FALSE, gradient_mode="H"))
    })["elapsed"]
    cat(sprintf("%.1fs\n", time))
    time
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", conditionMessage(e)))
    NA
  })
}

results <- list()

# =============================================================================
# POISSON_GAMMA spatial+temporal (rows 14-18)
# =============================================================================
cat("=== POISSON_GAMMA ===\n")
y_pg <- rpois(N, exp(2 + 0.5*x))
effort <- rgamma(N, 10, 1)
df_pg <- data.frame(y=y_pg, effort=effort, x=x, site=site, time=time,
                    lon=lon, lat=lat, spatial_site=spatial_site)

# Row 14: ICAR + RW1
results[["14"]] <- bench("ICAR + RW1", 14,
  y | effort ~ x + (1 | site), data=df_pg, family=ratiod_poisson_gamma(),
  spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
  temporal=temporal_rw1("time"))

# Row 15: BYM2 + RW1
results[["15"]] <- bench("BYM2 + RW1", 15,
  y | effort ~ x + (1 | site), data=df_pg, family=ratiod_poisson_gamma(),
  spatial=spatial_bym2(adj_mat, level="group", group_var="spatial_site"),
  temporal=temporal_rw1("time"))

# Row 16: ICAR + AR1
results[["16"]] <- bench("ICAR + AR1", 16,
  y | effort ~ x + (1 | site), data=df_pg, family=ratiod_poisson_gamma(),
  spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
  temporal=temporal_ar1("time"))

# Row 18: ICAR + ZI
df_pg_zi <- df_pg
df_pg_zi$y[sample(N, 100)] <- 0
results[["18"]] <- bench("ICAR + ZI", 18,
  y | effort ~ x + (1 | site), data=df_pg_zi, family=ratiod_zipois(),
  spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"))

# =============================================================================
# NEGBIN_NEGBIN spatial+temporal (rows 34-38)
# =============================================================================
cat("\n=== NEGBIN_NEGBIN ===\n")
y_nb <- rnbinom(N, mu=exp(2 + 0.3*x), size=5)
denom <- rnbinom(N, mu=100, size=10)
denom[denom == 0] <- 1
df_nb <- data.frame(y=y_nb, denom=denom, x=x, site=site, time=time,
                    lon=lon, lat=lat, spatial_site=spatial_site)

# Row 34: ICAR + RW1
results[["34"]] <- bench("ICAR + RW1", 34,
  y | denom ~ x + (1 | site), data=df_nb, family=ratiod_negbin_negbin(),
  spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
  temporal=temporal_rw1("time"))

# Row 35: BYM2 + RW1
results[["35"]] <- bench("BYM2 + RW1", 35,
  y | denom ~ x + (1 | site), data=df_nb, family=ratiod_negbin_negbin(),
  spatial=spatial_bym2(adj_mat, level="group", group_var="spatial_site"),
  temporal=temporal_rw1("time"))

# Row 36: ICAR + AR1
results[["36"]] <- bench("ICAR + AR1", 36,
  y | denom ~ x + (1 | site), data=df_nb, family=ratiod_negbin_negbin(),
  spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
  temporal=temporal_ar1("time"))

# Row 38: ICAR + ZI
results[["38"]] <- bench("ICAR + ZI", 38,
  y | denom ~ x + (1 | site), data=df_nb, family=ratiod_zinegbin(),
  spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"))

# =============================================================================
# BINOMIAL spatial+temporal (rows 56-58)
# =============================================================================
cat("\n=== BINOMIAL ===\n")
trials <- sample(10:50, N, replace=TRUE)
y_bin <- rbinom(N, trials, plogis(0.5 + 0.3*x))
df_bin <- data.frame(y=y_bin, trials=trials, x=x, site=site, time=time,
                     lon=lon, lat=lat, spatial_site=spatial_site)

# Row 56: ICAR + RW1
results[["56"]] <- bench("ICAR + RW1", 56,
  y | trials ~ x + (1 | site), data=df_bin, family=ratiod_binomial(),
  spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
  temporal=temporal_rw1("time"))

# Row 57: BYM2 + RW1
results[["57"]] <- bench("BYM2 + RW1", 57,
  y | trials ~ x + (1 | site), data=df_bin, family=ratiod_binomial(),
  spatial=spatial_bym2(adj_mat, level="group", group_var="spatial_site"),
  temporal=temporal_rw1("time"))

# Row 58: ICAR + AR1
results[["58"]] <- bench("ICAR + AR1", 58,
  y | trials ~ x + (1 | site), data=df_bin, family=ratiod_binomial(),
  spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
  temporal=temporal_ar1("time"))

cat("\n=== DONE ===\n")
print(results)
