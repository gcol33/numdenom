# Benchmark rows 21-23, 26-27 from gradient_methods.md
# Row 21: poisson_gamma + RE + GP + RW1
# Row 22: poisson_gamma + RE + HSGP + RW1
# Row 23: poisson_gamma + RE + MSGP + RW1
# Row 26: poisson_gamma + RE + SVC
# Row 27: poisson_gamma + RE + TVC

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
rownames(coords) <- 1:N_SITES

# Assign observations to sites
site_idx <- rep(1:N_SITES, length.out = N_OBS)

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
cat("Benchmarking poisson_gamma spatial+temporal and SVC/TVC models\n")
cat(strrep("=", 60), "\n\n")

# Helper function to run benchmark
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

# Row 21: GP + RW1 (this will be slow due to GP)
cat("\n")
results[[1]] <- run_bench(
  21, "poisson_gamma + RE + GP + RW1",
  spatial = spatial_gp(coords = ~ coord_x + coord_y),
  temporal = temporal_rw1(time_var = "time")
)

# Row 22: HSGP + RW1
cat("\n")
results[[2]] <- run_bench(
  22, "poisson_gamma + RE + HSGP + RW1",
  spatial = spatial_hsgp(coords = ~ coord_x + coord_y),
  temporal = temporal_rw1(time_var = "time")
)

# Row 23: MSGP + RW1 (likely to crash based on row 9)
cat("\n")
results[[3]] <- run_bench(
  23, "poisson_gamma + RE + MSGP + RW1",
  spatial = spatial_multiscale(coords = ~ coord_x + coord_y),
  temporal = temporal_rw1(time_var = "time")
)

# Row 26: SVC (spatially varying coefficients)
cat("\n")
results[[4]] <- run_bench(
  26, "poisson_gamma + RE + SVC",
  spatial = spatial_svc(coords = ~ coord_x + coord_y, varying = ~ x)
)

# Row 27: TVC (time-varying coefficients)
cat("\n")
results[[5]] <- run_bench(
  27, "poisson_gamma + RE + TVC",
  temporal = temporal_tvc(time_var = "time", varying = ~ x)
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
saveRDS(results, "benchmarks/results_rows_21_27.rds")
cat("\nResults saved to benchmarks/results_rows_21_27.rds\n")
