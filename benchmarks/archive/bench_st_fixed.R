# Benchmark the fixed spatiotemporal models (rows 28, 29, 58, 59, 90, 91)
# Standard parameters: N=500, iter=500, warmup=250, chains=1

library(numdenom)
set.seed(123)

N <- 500
N_SITES <- 50
N_TIMES <- 20
x <- rnorm(N)
site <- factor(rep(1:N_SITES, length.out = N))
time <- factor(rep(1:N_TIMES, length.out = N))

# Create spatial grid
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

# Data for each family
y_pg <- rpois(N, exp(2 + 0.5*x))
effort <- rgamma(N, 10, 1)
df_pg <- data.frame(y=y_pg, effort=effort, x=x, site=site, time=time,
                    lon=lon, lat=lat, spatial_site=spatial_site)

y_nb <- rnbinom(N, mu=exp(2 + 0.3*x), size=5)
denom_nb <- rnbinom(N, mu=100, size=10)
denom_nb[denom_nb == 0] <- 1
df_nb <- data.frame(y=y_nb, denom=denom_nb, x=x, site=site, time=time,
                    lon=lon, lat=lat, spatial_site=spatial_site)

trials <- sample(10:50, N, replace=TRUE)
y_bin <- rbinom(N, trials, plogis(0.5 + 0.3*x))
df_bin <- data.frame(y=y_bin, trials=trials, x=x, site=site, time=time,
                     lon=lon, lat=lat, spatial_site=spatial_site)

# Benchmark parameters
ITER <- 500
WARMUP <- 250
CHAINS <- 1

results <- list()
errors <- list()

# Row 28: poisson_gamma + ST Type I
cat("\n=== Row 28: pg_st1 ===\n")
cat("  H: ")
flush.console()
tryCatch({
  time <- system.time({
    fit <- ratiod(y | effort ~ x + (1 | site), data=df_pg, family=ratiod_poisson_gamma(),
                  spatiotemporal=spatiotemporal(
                    spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
                    temporal=temporal_rw1("time"),
                    type="I"),
                  iter=ITER, warmup=WARMUP, chains=CHAINS, verbose=FALSE, gradient_mode="H")
  })["elapsed"]
  cat(sprintf("%.1fs\n", time))
  results[["28"]] <- list(row=28, name="pg_st1", H=time)
}, error = function(e) {
  cat(sprintf("ERROR: %s\n", conditionMessage(e)))
  errors[["28"]] <- list(row=28, name="pg_st1", error=conditionMessage(e))
})

# Row 29: poisson_gamma + ST Type IV
cat("\n=== Row 29: pg_st4 ===\n")
cat("  H: ")
flush.console()
tryCatch({
  time <- system.time({
    fit <- ratiod(y | effort ~ x + (1 | site), data=df_pg, family=ratiod_poisson_gamma(),
                  spatiotemporal=spatiotemporal(
                    spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
                    temporal=temporal_rw1("time"),
                    type="IV"),
                  iter=ITER, warmup=WARMUP, chains=CHAINS, verbose=FALSE, gradient_mode="H")
  })["elapsed"]
  cat(sprintf("%.1fs\n", time))
  results[["29"]] <- list(row=29, name="pg_st4", H=time)
}, error = function(e) {
  cat(sprintf("ERROR: %s\n", conditionMessage(e)))
  errors[["29"]] <- list(row=29, name="pg_st4", error=conditionMessage(e))
})

# Row 58: negbin + ST Type I
cat("\n=== Row 58: nb_st1 ===\n")
cat("  H: ")
flush.console()
tryCatch({
  time <- system.time({
    fit <- ratiod(y | denom ~ x + (1 | site), data=df_nb, family=ratiod_negbin_negbin(),
                  spatiotemporal=spatiotemporal(
                    spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
                    temporal=temporal_rw1("time"),
                    type="I"),
                  iter=ITER, warmup=WARMUP, chains=CHAINS, verbose=FALSE, gradient_mode="H")
  })["elapsed"]
  cat(sprintf("%.1fs\n", time))
  results[["58"]] <- list(row=58, name="nb_st1", H=time)
}, error = function(e) {
  cat(sprintf("ERROR: %s\n", conditionMessage(e)))
  errors[["58"]] <- list(row=58, name="nb_st1", error=conditionMessage(e))
})

# Row 59: negbin + ST Type IV
cat("\n=== Row 59: nb_st4 ===\n")
cat("  H: ")
flush.console()
tryCatch({
  time <- system.time({
    fit <- ratiod(y | denom ~ x + (1 | site), data=df_nb, family=ratiod_negbin_negbin(),
                  spatiotemporal=spatiotemporal(
                    spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
                    temporal=temporal_rw1("time"),
                    type="IV"),
                  iter=ITER, warmup=WARMUP, chains=CHAINS, verbose=FALSE, gradient_mode="H")
  })["elapsed"]
  cat(sprintf("%.1fs\n", time))
  results[["59"]] <- list(row=59, name="nb_st4", H=time)
}, error = function(e) {
  cat(sprintf("ERROR: %s\n", conditionMessage(e)))
  errors[["59"]] <- list(row=59, name="nb_st4", error=conditionMessage(e))
})

# Row 90: binomial + ST Type I
cat("\n=== Row 90: bin_st1 ===\n")
cat("  H: ")
flush.console()
tryCatch({
  time <- system.time({
    fit <- ratiod(y | trials ~ x + (1 | site), data=df_bin, family=ratiod_binomial(),
                  spatiotemporal=spatiotemporal(
                    spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
                    temporal=temporal_rw1("time"),
                    type="I"),
                  iter=ITER, warmup=WARMUP, chains=CHAINS, verbose=FALSE, gradient_mode="H")
  })["elapsed"]
  cat(sprintf("%.1fs\n", time))
  results[["90"]] <- list(row=90, name="bin_st1", H=time)
}, error = function(e) {
  cat(sprintf("ERROR: %s\n", conditionMessage(e)))
  errors[["90"]] <- list(row=90, name="bin_st1", error=conditionMessage(e))
})

# Row 91: binomial + ST Type IV
cat("\n=== Row 91: bin_st4 ===\n")
cat("  H: ")
flush.console()
tryCatch({
  time <- system.time({
    fit <- ratiod(y | trials ~ x + (1 | site), data=df_bin, family=ratiod_binomial(),
                  spatiotemporal=spatiotemporal(
                    spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"),
                    temporal=temporal_rw1("time"),
                    type="IV"),
                  iter=ITER, warmup=WARMUP, chains=CHAINS, verbose=FALSE, gradient_mode="H")
  })["elapsed"]
  cat(sprintf("%.1fs\n", time))
  results[["91"]] <- list(row=91, name="bin_st4", H=time)
}, error = function(e) {
  cat(sprintf("ERROR: %s\n", conditionMessage(e)))
  errors[["91"]] <- list(row=91, name="bin_st4", error=conditionMessage(e))
})

# Print summary
cat("\n\n=== ST BENCHMARK RESULTS ===\n")
cat(sprintf("%-6s %-20s %8s %s\n", "Row", "Name", "H(s)", "Status"))
cat(paste(rep("-", 50), collapse=""), "\n")

for (row in c("28", "29", "58", "59", "90", "91")) {
  r <- results[[row]]
  e <- errors[[row]]
  if (!is.null(r)) {
    cat(sprintf("%-6s %-20s %8.1f OK\n", row, r$name, r$H))
  } else if (!is.null(e)) {
    cat(sprintf("%-6s %-20s %8s ERROR: %s\n", row, e$name, "-",
                substr(e$error, 1, 40)))
  }
}

if (length(errors) > 0) {
  cat("\n=== ERRORS ===\n")
  for (e in errors) {
    cat(sprintf("Row %d (%s): %s\n", e$row, e$name, e$error))
  }
}

# Save results
saveRDS(list(results=results, errors=errors), "benchmarks/results_st_fixed.rds")
cat("\nResults saved to benchmarks/results_st_fixed.rds\n")
