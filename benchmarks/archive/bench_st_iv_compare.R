#!/usr/bin/env Rscript
# Compare ST Type IV: NC vs Centered, and H vs N gradient check
# Run after devtools::install()

library(numdenom)
set.seed(42)

# Standard benchmark parameters (smaller for quick test)
N_SITES <- 10
N_TIMES <- 5
N_OBS <- N_SITES * N_TIMES

site <- rep(1:N_SITES, each = N_TIMES)
time_var <- rep(1:N_TIMES, times = N_SITES)
x <- rnorm(N_OBS)
eta <- 0.5 + 0.3 * x + rnorm(N_OBS, 0, 0.3)
y <- rpois(N_OBS, exp(eta))
denom <- rpois(N_OBS, 10) + 1

adj <- Matrix::Matrix(0, N_SITES, N_SITES, sparse = TRUE)
for (i in 1:(N_SITES - 1)) adj[i, i + 1] <- adj[i + 1, i] <- 1
adj[1, N_SITES] <- adj[N_SITES, 1] <- 1

df <- data.frame(y = y, denom = denom, x = x, site = site, time = time_var)

cat(sprintf("=== ST Type IV: %d sites x %d times = %d params ===\n\n",
            N_SITES, N_TIMES, N_SITES * N_TIMES))

run_st <- function(gradient_mode, label, iter = 200, warmup = 100) {
  t <- system.time({
    fit <- tratio(y | denom ~ x,
                  data = df, family = ratiod_poisson_gamma(),
                  spatial = spatial_car(adj, group_var = "site"),
                  temporal = temporal_rw1(time_var = "time"),
                  spatiotemporal = spatiotemporal(
                    spatial = spatial_car(adj, group_var = "site"),
                    temporal = temporal_rw1(time_var = "time"),
                    type = "IV"
                  ),
                  mode = "hmc",
                  control = list(gradient_mode = gradient_mode, iter = iter, warmup = warmup, chains = 1, verbose = FALSE))
  })["elapsed"]

  div <- sum(fit$diagnostics$divergent, na.rm = TRUE)
  td <- mean(fit$diagnostics$treedepth, na.rm = TRUE)
  cat(sprintf("  %-20s %6.1fs  div=%d  td=%.1f\n", label, t, div, td))
  invisible(list(fit = fit, time = t))
}

# Test 1: H mode (NC is default for Type IV)
cat("--- H mode (NC default for Type IV) ---\n")
h_nc <- run_st("H", "H (NC)")

# Test 2: N mode (always correct gradients)
cat("--- N mode (reference) ---\n")
n_ref <- run_st("N", "N (reference)")

# Compare posterior means of structural params
h_means <- colMeans(h_nc$fit$draws)
n_means <- colMeans(n_ref$fit$draws)
beta_params <- grep("^beta", names(h_means), value = TRUE)
tau_params <- grep("^log_tau|^tau", names(h_means), value = TRUE)

cat("\n--- Posterior comparison (H vs N) ---\n")
cat("  Parameter            H-mean     N-mean     Diff\n")
for (p in c(beta_params, tau_params)) {
  if (p %in% names(n_means)) {
    cat(sprintf("  %-20s %8.3f   %8.3f   %8.3f\n",
                p, h_means[p], n_means[p], h_means[p] - n_means[p]))
  }
}

cat("\nDone.\n")
