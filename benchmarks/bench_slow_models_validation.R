# Benchmark slow O(N^3) and O(N^2) models for runs validation
# GP, MSGP, SVC rows

library(numdenom)

cat("======================================================================\n")
cat("VALIDATION BENCHMARK: Slow models (GP, MSGP, SVC)\n")
cat("======================================================================\n\n")

# Smaller test size for slow models
N_OBS <- 100
N_ITER <- 300
N_WARMUP <- 150
N_CHAINS <- 1
N_SITES <- 20
N_TIMES <- 10

set.seed(42)

# Helper to create coordinates
create_coords <- function(n) {
  data.frame(
    x = runif(n, 0, 10),
    y = runif(n, 0, 10)
  )
}

# Helper to create adjacency matrix
create_adjacency <- function(n_sites) {
  adj <- matrix(0, n_sites, n_sites)
  for (i in 1:(n_sites - 1)) {
    adj[i, i + 1] <- 1
    adj[i + 1, i] <- 1
  }
  adj
}

results <- list()

# ============================================================================
# Row 7: poisson_gamma + GP
# ============================================================================
cat("\n>>> Row 7: poisson_gamma + GP <<<\n")
tryCatch({
  coords7 <- create_coords(N_OBS)
  df7 <- data.frame(
    y_num = rpois(N_OBS, 10),
    y_denom = rgamma(N_OBS, 5, 1),
    x = rnorm(N_OBS),
    site = rep(1:N_SITES, length.out = N_OBS),
    coord_x = coords7$x,
    coord_y = coords7$y
  )

  t7 <- system.time({
    fit7 <- ratiod(
      y_num | y_denom ~ x + (1 | site),
      data = df7,
      family = ratiod_poisson_gamma(),
      spatial = spatial_gp(coords = c("coord_x", "coord_y")),
      iter = N_ITER,
      warmup = N_WARMUP,
      chains = N_CHAINS,
      gradient_mode = "H"
    )
  })["elapsed"]

  divs <- sum(fit7$diagnostics$divergent)
  cat(sprintf("Row 7: numdenom %.1fs\n  Divergences: %d\n", t7, divs))
  results$row7 <- list(time = t7, div = divs, status = "SUCCESS")
}, error = function(e) {
  cat(sprintf("Row 7: FAILED - %s\n", e$message))
  results$row7 <<- list(status = "FAILED", error = e$message)
})

# ============================================================================
# Row 37: negbin_negbin + GP
# ============================================================================
cat("\n>>> Row 37: negbin_negbin + GP <<<\n")
tryCatch({
  coords37 <- create_coords(N_OBS)
  df37 <- data.frame(
    y_num = rnbinom(N_OBS, size = 5, mu = 10),
    y_denom = rnbinom(N_OBS, size = 5, mu = 20),
    x = rnorm(N_OBS),
    site = rep(1:N_SITES, length.out = N_OBS),
    coord_x = coords37$x,
    coord_y = coords37$y
  )

  t37 <- system.time({
    fit37 <- ratiod(
      y_num | y_denom ~ x + (1 | site),
      data = df37,
      family = ratiod_negbin_negbin(),
      spatial = spatial_gp(coords = c("coord_x", "coord_y")),
      iter = N_ITER,
      warmup = N_WARMUP,
      chains = N_CHAINS,
      gradient_mode = "H"
    )
  })["elapsed"]

  divs <- sum(fit37$diagnostics$divergent)
  cat(sprintf("Row 37: numdenom %.1fs\n  Divergences: %d\n", t37, divs))
  results$row37 <- list(time = t37, div = divs, status = "SUCCESS")
}, error = function(e) {
  cat(sprintf("Row 37: FAILED - %s\n", e$message))
  results$row37 <<- list(status = "FAILED", error = e$message)
})

# ============================================================================
# Row 67: binomial + GP
# ============================================================================
cat("\n>>> Row 67: binomial + GP <<<\n")
tryCatch({
  coords67 <- create_coords(N_OBS)
  trials67 <- sample(10:30, N_OBS, replace = TRUE)
  df67 <- data.frame(
    y = rbinom(N_OBS, trials67, 0.5),
    trials = trials67,
    x = rnorm(N_OBS),
    site = rep(1:N_SITES, length.out = N_OBS),
    coord_x = coords67$x,
    coord_y = coords67$y
  )

  t67 <- system.time({
    fit67 <- ratiod(
      y | trials ~ x + (1 | site),
      data = df67,
      family = ratiod_binomial(),
      spatial = spatial_gp(coords = c("coord_x", "coord_y")),
      iter = N_ITER,
      warmup = N_WARMUP,
      chains = N_CHAINS,
      gradient_mode = "H"
    )
  })["elapsed"]

  divs <- sum(fit67$diagnostics$divergent)
  cat(sprintf("Row 67: numdenom %.1fs\n  Divergences: %d\n", t67, divs))
  results$row67 <- list(time = t67, div = divs, status = "SUCCESS")
}, error = function(e) {
  cat(sprintf("Row 67: FAILED - %s\n", e$message))
  results$row67 <<- list(status = "FAILED", error = e$message)
})

# ============================================================================
# Row 21: poisson_gamma + GP + RW1
# ============================================================================
cat("\n>>> Row 21: poisson_gamma + GP + RW1 <<<\n")
tryCatch({
  coords21 <- create_coords(N_OBS)
  df21 <- data.frame(
    y_num = rpois(N_OBS, 10),
    y_denom = rgamma(N_OBS, 5, 1),
    x = rnorm(N_OBS),
    site = rep(1:N_SITES, length.out = N_OBS),
    time = rep(1:N_TIMES, each = N_OBS / N_TIMES),
    coord_x = coords21$x,
    coord_y = coords21$y
  )

  t21 <- system.time({
    fit21 <- ratiod(
      y_num | y_denom ~ x + (1 | site),
      data = df21,
      family = ratiod_poisson_gamma(),
      spatial = spatial_gp(coords = c("coord_x", "coord_y")),
      temporal = temporal_rw1(time_var = "time"),
      iter = N_ITER,
      warmup = N_WARMUP,
      chains = N_CHAINS,
      gradient_mode = "H"
    )
  })["elapsed"]

  divs <- sum(fit21$diagnostics$divergent)
  cat(sprintf("Row 21: numdenom %.1fs\n  Divergences: %d\n", t21, divs))
  results$row21 <- list(time = t21, div = divs, status = "SUCCESS")
}, error = function(e) {
  cat(sprintf("Row 21: FAILED - %s\n", e$message))
  results$row21 <<- list(status = "FAILED", error = e$message)
})

# ============================================================================
# Row 51: negbin_negbin + GP + RW1
# ============================================================================
cat("\n>>> Row 51: negbin_negbin + GP + RW1 <<<\n")
tryCatch({
  coords51 <- create_coords(N_OBS)
  df51 <- data.frame(
    y_num = rnbinom(N_OBS, size = 5, mu = 10),
    y_denom = rnbinom(N_OBS, size = 5, mu = 20),
    x = rnorm(N_OBS),
    site = rep(1:N_SITES, length.out = N_OBS),
    time = rep(1:N_TIMES, each = N_OBS / N_TIMES),
    coord_x = coords51$x,
    coord_y = coords51$y
  )

  t51 <- system.time({
    fit51 <- ratiod(
      y_num | y_denom ~ x + (1 | site),
      data = df51,
      family = ratiod_negbin_negbin(),
      spatial = spatial_gp(coords = c("coord_x", "coord_y")),
      temporal = temporal_rw1(time_var = "time"),
      iter = N_ITER,
      warmup = N_WARMUP,
      chains = N_CHAINS,
      gradient_mode = "H"
    )
  })["elapsed"]

  divs <- sum(fit51$diagnostics$divergent)
  cat(sprintf("Row 51: numdenom %.1fs\n  Divergences: %d\n", t51, divs))
  results$row51 <- list(time = t51, div = divs, status = "SUCCESS")
}, error = function(e) {
  cat(sprintf("Row 51: FAILED - %s\n", e$message))
  results$row51 <<- list(status = "FAILED", error = e$message)
})

# ============================================================================
# Row 83: binomial + GP + RW1
# ============================================================================
cat("\n>>> Row 83: binomial + GP + RW1 <<<\n")
tryCatch({
  coords83 <- create_coords(N_OBS)
  trials83 <- sample(10:30, N_OBS, replace = TRUE)
  df83 <- data.frame(
    y = rbinom(N_OBS, trials83, 0.5),
    trials = trials83,
    x = rnorm(N_OBS),
    site = rep(1:N_SITES, length.out = N_OBS),
    time = rep(1:N_TIMES, each = N_OBS / N_TIMES),
    coord_x = coords83$x,
    coord_y = coords83$y
  )

  t83 <- system.time({
    fit83 <- ratiod(
      y | trials ~ x + (1 | site),
      data = df83,
      family = ratiod_binomial(),
      spatial = spatial_gp(coords = c("coord_x", "coord_y")),
      temporal = temporal_rw1(time_var = "time"),
      iter = N_ITER,
      warmup = N_WARMUP,
      chains = N_CHAINS,
      gradient_mode = "H"
    )
  })["elapsed"]

  divs <- sum(fit83$diagnostics$divergent)
  cat(sprintf("Row 83: numdenom %.1fs\n  Divergences: %d\n", t83, divs))
  results$row83 <- list(time = t83, div = divs, status = "SUCCESS")
}, error = function(e) {
  cat(sprintf("Row 83: FAILED - %s\n", e$message))
  results$row83 <<- list(status = "FAILED", error = e$message)
})

# ============================================================================
# MSGP models (9, 23, 69, 85) - skip for now, very slow
# ============================================================================
cat("\n>>> Skipping MSGP models (9, 23, 69, 85) - too slow <<<\n")

# ============================================================================
# SVC models (26, 56, 88) - skip for now, very slow
# ============================================================================
cat("\n>>> Skipping SVC models (26, 56, 88) - too slow <<<\n")

cat("\n======================================================================\n")
cat("VALIDATION COMPLETE\n")
cat("======================================================================\n")

# Print summary
cat("\nSummary:\n")
for (name in names(results)) {
  r <- results[[name]]
  if (r$status == "SUCCESS") {
    cat(sprintf("  %s: %.1fs, %d divergences\n", name, r$time, r$div))
  } else {
    cat(sprintf("  %s: FAILED\n", name))
  }
}
