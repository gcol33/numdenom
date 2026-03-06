# Benchmark rows 28-30 from gradient_methods.md
# Row 28: poisson_gamma + RE + ICAR + RW1 (ST-I, spatiotemporal Type I)
# Row 29: poisson_gamma + RE + ICAR + RW1 (ST-IV, spatiotemporal Type IV)
# Row 30: poisson_gamma + RE + latent factors

library(numdenom)

# Standard benchmark parameters
N_OBS <- 500
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1
N_SITES <- 50
N_TIMES <- 20

set.seed(123)

# Create spatial grid
coords <- expand.grid(x = 1:10, y = 1:5)
coords <- coords[1:N_SITES, ]

# Assign observations to sites and times
site_idx <- rep(1:N_SITES, length.out = N_OBS)
time_idx <- rep(1:N_TIMES, length.out = N_OBS)

# Create adjacency matrix based on grid neighbors
adj_mat <- matrix(0, N_SITES, N_SITES)
for (i in 1:N_SITES) {
  for (j in 1:N_SITES) {
    if (i != j) {
      dist <- sqrt((coords$x[i] - coords$x[j])^2 + (coords$y[i] - coords$y[j])^2)
      if (dist <= 1.5) adj_mat[i, j] <- 1
    }
  }
}

# Generate data
x <- rnorm(N_OBS)
site_effect <- rnorm(N_SITES, 0, 0.3)[site_idx]
time_effect <- cumsum(rnorm(N_TIMES, 0, 0.1))[time_idx]
eta <- 2 + 0.5 * x + site_effect + time_effect
y <- rpois(N_OBS, exp(eta))
effort <- rgamma(N_OBS, 10, 1)

df <- data.frame(
  y = y,
  effort = effort,
  x = x,
  site = factor(site_idx),
  time = factor(time_idx)
)

results <- list()
cat("\n", strrep("=", 60), "\n")
cat("Benchmarking poisson_gamma spatiotemporal and latent models\n")
cat(strrep("=", 60), "\n\n")

# Helper function
run_bench <- function(row_num, desc, ...) {
  cat(sprintf("Row %d: %s\n", row_num, desc))
  cat(strrep("-", 50), "\n")

  tryCatch({
    time_h <- system.time({
      fit <- ratiod(
        y | effort ~ x + (1 | site),
        data = df,
        family = ratiod_poisson_gamma(),
        iter = N_ITER,
        warmup = N_WARMUP,
        chains = N_CHAINS,
        gradient_mode = "H",
        refresh = 0,
        ...
      )
    })["elapsed"]

    cat(sprintf("  H time: %.1f s\n", time_h))
    list(row = row_num, desc = desc, H = round(time_h, 1), status = "OK")
  }, error = function(e) {
    cat(sprintf("  ERROR: %s\n", conditionMessage(e)))
    list(row = row_num, desc = desc, H = NA, status = "ERROR", error = conditionMessage(e))
  })
}

# Row 28: ST-I (spatiotemporal Type I - unstructured interaction)
cat("\n")
results[[1]] <- run_bench(
  28, "poisson_gamma + RE + ST-I",
  spatiotemporal = spatiotemporal(
    spatial = spatial_car(adjacency = adj_mat, level = "group", group_var = "site"),
    temporal = temporal_rw1(time_var = "time"),
    type = "I"
  )
)

# Row 29: ST-IV (spatiotemporal Type IV - full Kronecker)
cat("\n")
results[[2]] <- run_bench(
  29, "poisson_gamma + RE + ST-IV",
  spatiotemporal = spatiotemporal(
    spatial = spatial_car(adjacency = adj_mat, level = "group", group_var = "site"),
    temporal = temporal_rw1(time_var = "time"),
    type = "IV"
  )
)

# Row 30: Latent factors
cat("\n")
results[[3]] <- run_bench(
  30, "poisson_gamma + RE + latent",
  latent = latent_factor(n_factors = 2)
)

# Summary
cat("\n", strrep("=", 60), "\n")
cat("SUMMARY\n")
cat(strrep("=", 60), "\n\n")

summary_df <- do.call(rbind, lapply(results, function(r) {
  data.frame(
    Row = r$row,
    Description = r$desc,
    H_sec = ifelse(is.na(r$H), "-", sprintf("%.1f", r$H)),
    Status = r$status
  )
}))
print(summary_df, row.names = FALSE)

saveRDS(results, "benchmarks/results_rows_28_30.rds")
cat("\nResults saved to benchmarks/results_rows_28_30.rds\n")
