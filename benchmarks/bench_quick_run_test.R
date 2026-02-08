# Quick run test for FIXED rows (39, 52, 53)
# Just verify they run without crashing

library(numdenom)
set.seed(20260208)

# Minimal parameters for quick test
N_OBS <- 50
N_ITER <- 100
N_WARMUP <- 50
N_CHAINS <- 1
N_SITES <- 10
N_TIMES <- 5

message(paste(rep("=", 60), collapse=""))
message("QUICK RUN TEST - FIXED ROWS")
message("Testing rows run without errors")
message(paste(rep("=", 60), collapse=""))

# Generate data with coordinates
df <- data.frame(
  site = rep(1:N_SITES, each = N_TIMES),
  time = rep(1:N_TIMES, N_SITES),
  x = rnorm(N_OBS),
  x_coord = runif(N_OBS),
  y_coord = runif(N_OBS)
)

# Generate response
eta <- 1.5 + 0.3 * df$x + rnorm(N_SITES, 0, 0.3)[df$site]
df$y_num <- rnbinom(N_OBS, size = 5, mu = exp(eta))
df$y_denom <- rnbinom(N_OBS, size = 5, mu = exp(eta + 0.5))
df$y_denom <- pmax(df$y_denom, 1)

# ===== Row 52: negbin_negbin + RE + HSGP + RW1 =====
# This should be fast because HSGP is 20-100x faster than GP
message("\n>>> Row 52: negbin_negbin + RE + HSGP + RW1 <<<")

tryCatch({
  time52 <- system.time({
    fit52 <- ratiod(
      y_num | y_denom ~ x + (1 | site),
      data = df,
      family = ratiod_negbin_negbin(),
      spatial = spatial_hsgp(coords = ~ x_coord + y_coord),
      temporal = temporal_rw1(time_var = "time"),
      iter = N_ITER,
      warmup = N_WARMUP,
      chains = N_CHAINS,
      gradient_mode = "H"
    )
  })["elapsed"]

  message(sprintf("Row 52: SUCCESS (%.1fs)", time52))
  message(sprintf("  Divergences: %d", sum(fit52$diagnostics$divergent)))
}, error = function(e) {
  message(sprintf("Row 52 ERROR: %s", e$message))
})

# ===== Row 39: negbin_negbin + RE + MSGP (quick test) =====
message("\n>>> Row 39: negbin_negbin + RE + MSGP <<<")

tryCatch({
  time39 <- system.time({
    fit39 <- ratiod(
      y_num | y_denom ~ x + (1 | site),
      data = df,
      family = ratiod_negbin_negbin(),
      spatial = spatial_multiscale(coords = ~ x_coord + y_coord),
      iter = N_ITER,
      warmup = N_WARMUP,
      chains = N_CHAINS,
      gradient_mode = "H"
    )
  })["elapsed"]

  message(sprintf("Row 39: SUCCESS (%.1fs)", time39))
  message(sprintf("  Divergences: %d", sum(fit39$diagnostics$divergent)))
}, error = function(e) {
  message(sprintf("Row 39 ERROR: %s", e$message))
})

# ===== Row 53: negbin_negbin + RE + MSGP + RW1 (quick test) =====
message("\n>>> Row 53: negbin_negbin + RE + MSGP + RW1 <<<")

tryCatch({
  time53 <- system.time({
    fit53 <- ratiod(
      y_num | y_denom ~ x + (1 | site),
      data = df,
      family = ratiod_negbin_negbin(),
      spatial = spatial_multiscale(coords = ~ x_coord + y_coord),
      temporal = temporal_rw1(time_var = "time"),
      iter = N_ITER,
      warmup = N_WARMUP,
      chains = N_CHAINS,
      gradient_mode = "H"
    )
  })["elapsed"]

  message(sprintf("Row 53: SUCCESS (%.1fs)", time53))
  message(sprintf("  Divergences: %d", sum(fit53$diagnostics$divergent)))
}, error = function(e) {
  message(sprintf("Row 53 ERROR: %s", e$message))
})

message(paste(rep("=", 60), collapse=""))
message("QUICK RUN TEST COMPLETE")
message(paste(rep("=", 60), collapse=""))
