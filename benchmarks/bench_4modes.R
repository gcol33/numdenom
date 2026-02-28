# Benchmark representative models with all 4 gradient modes (N, A_t, A, H)
# Standard parameters: N=500, iter=500, warmup=250, chains=1
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

# Data for each family
y_pg <- rpois(N, exp(2 + 0.5*x))
effort <- rgamma(N, 10, 1)
df_pg <- data.frame(y=y_pg, effort=effort, x=x, site=site, time=time,
                    lon=lon, lat=lat, spatial_site=spatial_site)

y_nb <- rnbinom(N, mu=exp(2 + 0.3*x), size=5)
denom <- rnbinom(N, mu=100, size=10)
denom[denom == 0] <- 1
df_nb <- data.frame(y=y_nb, denom=denom, x=x, site=site, time=time,
                    lon=lon, lat=lat, spatial_site=spatial_site)

trials <- sample(10:50, N, replace=TRUE)
y_bin <- rbinom(N, trials, plogis(0.5 + 0.3*x))
df_bin <- data.frame(y=y_bin, trials=trials, x=x, site=site, time=time,
                     lon=lon, lat=lat, spatial_site=spatial_site)

bench_all_modes <- function(name, row, ...) {
  cat(sprintf("\n=== Row %d: %s ===\n", row, name))
  results <- list()
  for (mode in c("H", "A", "A_t", "N")) {
    cat(sprintf("  %s: ", mode))
    flush.console()
    tryCatch({
      time <- system.time({
        fit <- ratiod(..., iter=500, warmup=250, chains=1, verbose=FALSE, gradient_mode=mode)
      })["elapsed"]
      cat(sprintf("%.1fs\n", time))
      results[[mode]] <- time
    }, error = function(e) {
      cat(sprintf("ERROR: %s\n", conditionMessage(e)))
      results[[mode]] <- NA
    })
  }
  results
}

all_results <- list()

# Row 1: poisson_gamma base (no RE)
all_results[["1"]] <- bench_all_modes("pg_base", 1,
  y | effort ~ x, data=df_pg, family=ratiod_poisson_gamma())

# Row 2: poisson_gamma + RE
all_results[["2"]] <- bench_all_modes("pg_re", 2,
  y | effort ~ x + (1 | site), data=df_pg, family=ratiod_poisson_gamma())

# Row 5: poisson_gamma + ICAR
all_results[["5"]] <- bench_all_modes("pg_icar", 5,
  y | effort ~ x + (1 | site), data=df_pg, family=ratiod_poisson_gamma(),
  spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"))

# Row 9: poisson_gamma + RW1
all_results[["9"]] <- bench_all_modes("pg_rw1", 9,
  y | effort ~ x + (1 | site), data=df_pg, family=ratiod_poisson_gamma(),
  temporal=temporal_rw1("time"))

# Row 21: negbin_negbin base
all_results[["21"]] <- bench_all_modes("nb_base", 21,
  y | denom ~ x, data=df_nb, family=ratiod_negbin_negbin())

# Row 22: negbin_negbin + RE
all_results[["22"]] <- bench_all_modes("nb_re", 22,
  y | denom ~ x + (1 | site), data=df_nb, family=ratiod_negbin_negbin())

# Row 25: negbin_negbin + ICAR
all_results[["25"]] <- bench_all_modes("nb_icar", 25,
  y | denom ~ x + (1 | site), data=df_nb, family=ratiod_negbin_negbin(),
  spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"))

# Row 41: binomial base
all_results[["41"]] <- bench_all_modes("bin_base", 41,
  y | trials ~ x, data=df_bin, family=ratiod_binomial())

# Row 42: binomial + RE
all_results[["42"]] <- bench_all_modes("bin_re", 42,
  y | trials ~ x + (1 | site), data=df_bin, family=ratiod_binomial())

# Row 45: binomial + ICAR
all_results[["45"]] <- bench_all_modes("bin_icar", 45,
  y | trials ~ x + (1 | site), data=df_bin, family=ratiod_binomial(),
  spatial=spatial_car(adj_mat, level="group", group_var="spatial_site"))

# Print summary table
cat("\n\n=== SUMMARY TABLE ===\n")
cat(sprintf("%-20s %8s %8s %8s %8s\n", "Model", "H(s)", "A(s)", "A_t(s)", "N(s)"))
cat(paste(rep("-", 60), collapse=""), "\n")
for (row in names(all_results)) {
  r <- all_results[[row]]
  cat(sprintf("Row %-16s %8.1f %8.1f %8.1f %8.1f\n", row,
              ifelse(is.null(r$H), NA, r$H),
              ifelse(is.null(r$A), NA, r$A),
              ifelse(is.null(r$A_t), NA, r$A_t),
              ifelse(is.null(r$N), NA, r$N)))
}

saveRDS(all_results, "benchmarks/results_4modes.rds")
cat("\nResults saved to benchmarks/results_4modes.rds\n")
