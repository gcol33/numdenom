# Remaining PG/NB configurations needing joint Stan validation
# Rows: 22 (pg+HSGP+RW1), 24 (pg+ICAR+ZI), 54 (nb+ICAR+ZI)

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

# Standard parameters
N_OBS <- 200
N_SITES <- 20
N_TIMES <- 15
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2

cat("=======================================================\n")
cat("Remaining PG/NB Joint Stan Validations\n")
cat("=======================================================\n")
cat(sprintf("N=%d, iter=%d, warmup=%d, chains=%d\n", N_OBS, N_ITER, N_WARMUP, N_CHAINS))

results <- list()

# Helper function to compare posteriors
compare_posteriors <- function(nd_draws, stan_draws, param_name) {
  nd_mean <- mean(nd_draws)
  nd_sd <- sd(nd_draws)
  stan_mean <- mean(stan_draws)
  stan_sd <- sd(stan_draws)

  combined_se <- sqrt(nd_sd^2/length(nd_draws) + stan_sd^2/length(stan_draws))
  diff_ratio <- abs(nd_mean - stan_mean) / combined_se

  list(
    nd_mean = nd_mean,
    nd_sd = nd_sd,
    stan_mean = stan_mean,
    stan_sd = stan_sd,
    diff_se = diff_ratio,
    pass = diff_ratio < 2
  )
}

# =============================================================================
# Row 22: poisson_gamma + HSGP + RW1
# =============================================================================
cat("\n========== Row 22: poisson_gamma + HSGP + RW1 ==========\n")

# Generate coordinates and time structure
coords <- cbind(
  x = runif(N_OBS, 0, 10),
  y = runif(N_OBS, 0, 10)
)
time_idx <- rep(1:N_TIMES, length.out = N_OBS)
x <- rnorm(N_OBS)

# Generate spatial and temporal effects
spatial_effect <- sin(coords[,1] / 2) * 0.5 + cos(coords[,2] / 3) * 0.3
temporal_effect <- cumsum(rnorm(N_TIMES, 0, 0.2))
temporal_effect <- temporal_effect - mean(temporal_effect)

# Generate data
mu_num <- exp(2 + 0.3 * x + spatial_effect + temporal_effect[time_idx])
mu_denom <- rgamma(N_OBS, shape = 5, rate = 5 / exp(4 + spatial_effect + temporal_effect[time_idx]))
mu_denom[mu_denom < 0.01] <- 0.01
y_num <- rpois(N_OBS, mu_num)

df_22 <- data.frame(
  y = y_num,
  denom = mu_denom,
  x = x,
  time = time_idx,
  coord_x = coords[,1],
  coord_y = coords[,2]
)

cat("Fitting numdenom (HSGP+RW1)... ")
t_nd_22 <- system.time({
  fit_nd_22 <- tryCatch({
    ratiod(
      y | denom ~ x,
      data = df_22,
      family = ratiod_poisson_gamma(),
      spatial = spatial_hsgp(~ coord_x + coord_y, m = 8),
      temporal = temporal_rw1("time"),
      iter = N_ITER, warmup = N_WARMUP, chains = 1,
      verbose = FALSE
    )
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    NULL
  })
})[["elapsed"]]

if (!is.null(fit_nd_22)) {
  cat(sprintf("%.1fs, %d div\n", t_nd_22, sum(fit_nd_22$divergent)))
  results$row_22 <- list(time_nd = t_nd_22, runs = TRUE, divergent = sum(fit_nd_22$divergent))
} else {
  results$row_22 <- list(error = TRUE)
}

# =============================================================================
# Row 24: poisson_gamma + ICAR + ZI
# =============================================================================
cat("\n========== Row 24: poisson_gamma + ICAR + ZI ==========\n")

# Create adjacency matrix for sites
site_idx <- rep(1:N_SITES, length.out = N_OBS)
adj_matrix <- matrix(0, N_SITES, N_SITES)
for (i in 1:(N_SITES-1)) {
  adj_matrix[i, i+1] <- 1
  adj_matrix[i+1, i] <- 1
}
# Add some extra connections
adj_matrix[1, N_SITES] <- 1
adj_matrix[N_SITES, 1] <- 1

# Generate ICAR effect
site_effect <- cumsum(rnorm(N_SITES, 0, 0.3))
site_effect <- site_effect - mean(site_effect)

# Generate ZI data
zi_prob <- 0.2
mu_num_24 <- exp(2 + 0.3 * x + site_effect[site_idx])
mu_denom_24 <- rgamma(N_OBS, shape = 5, rate = 5 / exp(4 + site_effect[site_idx]))
mu_denom_24[mu_denom_24 < 0.01] <- 0.01

# Zero-inflate the numerator
y_num_24 <- rpois(N_OBS, mu_num_24)
y_num_24[runif(N_OBS) < zi_prob] <- 0

df_24 <- data.frame(
  y = y_num_24,
  denom = mu_denom_24,
  x = x,
  site = site_idx
)

cat("Fitting numdenom (ICAR+ZI)... ")
t_nd_24 <- system.time({
  fit_nd_24 <- tryCatch({
    ratiod(
      y | denom ~ x,
      data = df_24,
      family = ratiod_zipois(denom_family = "gamma"),
      spatial = spatial_car(adj_matrix, group_var = "site"),
      iter = N_ITER, warmup = N_WARMUP, chains = 1,
      verbose = FALSE
    )
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    NULL
  })
})[["elapsed"]]

if (!is.null(fit_nd_24)) {
  cat(sprintf("%.1fs, %d div\n", t_nd_24, sum(fit_nd_24$divergent)))
  results$row_24 <- list(time_nd = t_nd_24, runs = TRUE, divergent = sum(fit_nd_24$divergent))
} else {
  results$row_24 <- list(error = TRUE)
}

# =============================================================================
# Row 54: negbin_negbin + ICAR + ZI
# =============================================================================
cat("\n========== Row 54: negbin_negbin + ICAR + ZI ==========\n")

# Generate NB ZI data
mu_num_54 <- exp(2 + 0.3 * x + site_effect[site_idx])
mu_denom_54 <- exp(4 + site_effect[site_idx])

y_num_54 <- rnbinom(N_OBS, mu = mu_num_54, size = 5)
y_num_54[runif(N_OBS) < zi_prob] <- 0
y_denom_54 <- rnbinom(N_OBS, mu = mu_denom_54, size = 10)
y_denom_54[y_denom_54 == 0] <- 1

df_54 <- data.frame(
  y = y_num_54,
  denom = y_denom_54,
  x = x,
  site = site_idx
)

cat("Fitting numdenom (ICAR+ZI)... ")
t_nd_54 <- system.time({
  fit_nd_54 <- tryCatch({
    ratiod(
      y | denom ~ x,
      data = df_54,
      family = ratiod_zinegbin(denom_family = "negbin"),
      spatial = spatial_car(adj_matrix, group_var = "site"),
      iter = N_ITER, warmup = N_WARMUP, chains = 1,
      verbose = FALSE
    )
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    NULL
  })
})[["elapsed"]]

if (!is.null(fit_nd_54)) {
  cat(sprintf("%.1fs, %d div\n", t_nd_54, sum(fit_nd_54$divergent)))
  results$row_54 <- list(time_nd = t_nd_54, runs = TRUE, divergent = sum(fit_nd_54$divergent))
} else {
  results$row_54 <- list(error = TRUE)
}

# =============================================================================
# Summary
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - Remaining PG/NB Validations\n")
cat("=======================================================\n\n")

cat("Row      Model                     numdenom  Divergent  Status\n")
cat("--------------------------------------------------------------------\n")

if (!is.null(results$row_22)) {
  r <- results$row_22
  if (isTRUE(r$error)) {
    cat("22       pg + HSGP + RW1         ERROR\n")
  } else {
    cat(sprintf("22       pg + HSGP + RW1         %6.1fs   %5d      %s\n",
                r$time_nd, r$divergent, if(r$divergent == 0) "OK" else "CHECK"))
  }
}

if (!is.null(results$row_24)) {
  r <- results$row_24
  if (isTRUE(r$error)) {
    cat("24       pg + ICAR + ZI          ERROR\n")
  } else {
    cat(sprintf("24       pg + ICAR + ZI          %6.1fs   %5d      %s\n",
                r$time_nd, r$divergent, if(r$divergent == 0) "OK" else "CHECK"))
  }
}

if (!is.null(results$row_54)) {
  r <- results$row_54
  if (isTRUE(r$error)) {
    cat("54       nb + ICAR + ZI          ERROR\n")
  } else {
    cat(sprintf("54       nb + ICAR + ZI          %6.1fs   %5d      %s\n",
                r$time_nd, r$divergent, if(r$divergent == 0) "OK" else "CHECK"))
  }
}

cat("--------------------------------------------------------------------\n")

# Save results
saveRDS(results, "benchmarks/results_remaining_pgnb.rds")
cat("\nResults saved to benchmarks/results_remaining_pgnb.rds\n")
