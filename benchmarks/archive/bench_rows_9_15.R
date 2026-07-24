# Benchmark rows 9-10, 14-15 from gradient_methods.md
# Row 9:  poisson_gamma + RE + MSGP
# Row 10: poisson_gamma + RE + pCAR
# Row 14: poisson_gamma + RE + GP_t (temporal GP)
# Row 15: poisson_gamma + RE + MS_t (multi-scale temporal)

library(numdenom)

# Standard benchmark parameters
N_OBS <- 500
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1
N_SITES <- 50
N_TIMES <- 20

set.seed(123)

# Create spatial grid for spatial models
coords <- expand.grid(x = 1:10, y = 1:5)
coords <- coords[1:N_SITES, ]
rownames(coords) <- 1:N_SITES

# Assign observations to sites
site_idx <- rep(1:N_SITES, length.out = N_OBS)

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

# Temporal structure
time_idx <- rep(1:N_TIMES, length.out = N_OBS)

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
  time = factor(time_idx),
  coord_x = coords$x[site_idx],
  coord_y = coords$y[site_idx]
)

results <- list()
cat("\n", strrep("=", 60), "\n")
cat("Benchmarking poisson_gamma spatial/temporal models\n")
cat(strrep("=", 60), "\n\n")

# Helper function to run benchmark
run_bench <- function(row_num, desc, ...) {
  cat(sprintf("Row %d: %s\n", row_num, desc))
  cat(strrep("-", 50), "\n")

  tryCatch({
    time_h <- system.time({
      fit <- tratio(
        y | effort ~ x + (1 | site),
        data = df,
        family = ratiod_poisson_gamma(),
        ...,
        control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H")
      )
    })["elapsed"]

    cat(sprintf("  H time: %.1f s\n", time_h))
    list(row = row_num, desc = desc, H = round(time_h, 1), status = "OK")
  }, error = function(e) {
    cat(sprintf("  ERROR: %s\n", conditionMessage(e)))
    list(row = row_num, desc = desc, H = NA, status = "ERROR", error = conditionMessage(e))
  })
}

# Row 9: MSGP (multi-scale GP)
# spatial_multiscale(coords, ...) - coords must be formula or character vector of length 2
cat("\n")
results[[1]] <- run_bench(
  9, "poisson_gamma + RE + MSGP",
  spatial = spatial_multiscale(
    coords = ~ coord_x + coord_y
  )
)

# Row 10: pCAR (proper CAR)
# spatial_car(adjacency, level, group_var, proper, shared)
cat("\n")
results[[2]] <- run_bench(
  10, "poisson_gamma + RE + pCAR",
  spatial = spatial_car(
    adjacency = adj_mat,
    level = "group",
    group_var = "site",
    proper = TRUE
  )
)

# Row 14: GP_t (temporal GP)
# temporal_gp(time_var, ...) - time_var is character
cat("\n")
results[[3]] <- run_bench(
  14, "poisson_gamma + RE + GP_t",
  temporal = temporal_gp(
    time_var = "time"
  )
)

# Row 15: MS_t (multi-scale temporal)
# temporal_multiscale(time_var, ...) - time_var is character
cat("\n")
results[[4]] <- run_bench(
  15, "poisson_gamma + RE + MS_t",
  temporal = temporal_multiscale(
    time_var = "time"
  )
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

# Save results
saveRDS(results, "benchmarks/results_rows_9_15.rds")
cat("\nResults saved to benchmarks/results_rows_9_15.rds\n")
