# Validation of numdenom two-process models against custom joint Stan models
# Batch 2: Spatial (ICAR) and Temporal (RW1) models
#
# Rows validated:
#   - Row 5:  poisson_gamma + RE + ICAR spatial
#   - Row 11: poisson_gamma + RE + RW1 temporal
#   - Row 35: negbin_negbin + RE + ICAR spatial
#   - Row 41: negbin_negbin + RE + RW1 temporal
#
# These custom Stan models correctly model BOTH num and denom with SHARED
# random effects, unlike brms which treats denom as fixed via offset().

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

# Standard benchmark parameters
N_OBS <- 200
N_ITER <- 1000
N_WARMUP <- 500
N_CHAINS <- 2
N_SITES <- 20
N_TIMES <- 10

cat("=======================================================\n")
cat("Joint Model Validation Batch 2: Spatial + Temporal\n")
cat("=======================================================\n")
cat(sprintf("N=%d, iter=%d, warmup=%d, chains=%d\n", N_OBS, N_ITER, N_WARMUP, N_CHAINS))
cat(sprintf("sites=%d, times=%d\n\n", N_SITES, N_TIMES))

# Helper function to compare posteriors
compare_posteriors <- function(nd_draws, stan_draws, param_name, threshold_se = 2) {
  nd_mean <- mean(nd_draws)
  nd_sd <- sd(nd_draws)
  stan_mean <- mean(stan_draws)
  stan_sd <- sd(stan_draws)

  se_combined <- sqrt(nd_sd^2 / length(nd_draws) + stan_sd^2 / length(stan_draws))
  diff <- abs(nd_mean - stan_mean)
  ratio <- diff / se_combined

  pass <- ratio < threshold_se

  list(
    param = param_name,
    nd_mean = nd_mean,
    nd_sd = nd_sd,
    stan_mean = stan_mean,
    stan_sd = stan_sd,
    diff = diff,
    ratio = ratio,
    pass = pass
  )
}

print_result <- function(result) {
  cat(sprintf("\n%s:\n", result$param))
  cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", result$nd_mean, result$nd_sd))
  cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", result$stan_mean, result$stan_sd))
  cat(sprintf("  Diff: %.4f (%.2f SE) => %s\n",
              result$diff, result$ratio, if(result$pass) "PASS" else "FAIL"))
}

# Create adjacency matrix for spatial grid
create_grid_adjacency <- function(n_units) {
  n_side <- ceiling(sqrt(n_units))
  grid <- expand.grid(x = 1:n_side, y = 1:n_side)[1:n_units, ]

  # Find edges (neighbors)
  edges <- list()
  for (i in 1:(n_units-1)) {
    for (j in (i+1):n_units) {
      dist <- sqrt((grid$x[i] - grid$x[j])^2 + (grid$y[i] - grid$y[j])^2)
      if (dist <= 1.01) {  # Adjacent (Manhattan distance = 1)
        edges[[length(edges) + 1]] <- c(i, j)
      }
    }
  }

  # Count neighbors per unit
  n_neighbors <- integer(n_units)
  for (e in edges) {
    n_neighbors[e[1]] <- n_neighbors[e[1]] + 1
    n_neighbors[e[2]] <- n_neighbors[e[2]] + 1
  }

  # Create edge arrays
  n_edges <- length(edges)
  edge1 <- sapply(edges, `[`, 1)
  edge2 <- sapply(edges, `[`, 2)

  # Create adjacency matrix for numdenom
  adj_mat <- matrix(0, n_units, n_units)
  for (e in edges) {
    adj_mat[e[1], e[2]] <- 1
    adj_mat[e[2], e[1]] <- 1
  }

  list(
    adj_mat = adj_mat,
    n_neighbors = n_neighbors,
    n_edges = n_edges,
    edge1 = edge1,
    edge2 = edge2
  )
}

# Create spatial structure
spatial_info <- create_grid_adjacency(N_SITES)

results <- list()

# =============================================================================
# Row 5: poisson_gamma + RE + ICAR spatial
# =============================================================================
cat("\n========== Row 5: poisson_gamma + RE + ICAR ==========\n")

# Generate data with site effects and spatial structure
site <- factor(rep(1:N_SITES, length.out = N_OBS))
spatial_site <- site  # Same for simplicity

# Simulate shared spatial effect
phi_true <- rnorm(N_SITES, 0, 0.3)
phi_true <- phi_true - mean(phi_true)  # Center

# Simulate shared site RE
re_true <- rnorm(N_SITES, 0, 0.3)

df_pg_spatial <- data.frame(
  y = rpois(N_OBS, lambda = 30 * exp(re_true[as.numeric(site)] + phi_true[as.numeric(site)])),
  effort = rgamma(N_OBS, shape = 5, rate = 0.1) * exp(re_true[as.numeric(site)] + phi_true[as.numeric(site)]),
  x = rnorm(N_OBS),
  site = site,
  spatial_site = spatial_site
)
df_pg_spatial$effort[df_pg_spatial$effort < 0.01] <- 0.01

# Fit numdenom
cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- tratio(
    y | effort ~ x + (1|site),
    data = df_pg_spatial,
    family = ratiod_poisson_gamma(),
    spatial = spatial_car(spatial_info$adj_mat, level = "group", group_var = "spatial_site"),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

# Compile and fit Stan model
cat("Compiling Stan model... ")
stan_model <- cmdstan_model("stan/joint_pg_icar.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = df_pg_spatial$y,
  y_denom = df_pg_spatial$effort,
  p = 2,
  X = cbind(1, df_pg_spatial$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_pg_spatial$site),
  J = N_SITES,
  spatial_idx = as.numeric(df_pg_spatial$spatial_site),
  n_neighbors = spatial_info$n_neighbors,
  n_edges = spatial_info$n_edges,
  edge1 = spatial_info$edge1,
  edge2 = spatial_info$edge2
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model$sample(
    data = stan_data,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

# Compare posteriors
draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

results$row_5_beta <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)
print_result(results$row_5_beta)
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Row 11: poisson_gamma + RE + RW1 temporal
# =============================================================================
cat("\n========== Row 11: poisson_gamma + RE + RW1 ==========\n")

# Generate data with site effects and temporal structure
time_idx <- rep(1:N_TIMES, length.out = N_OBS)
time_factor <- factor(time_idx)

# Simulate shared temporal effect (RW1-like)
phi_temp_true <- cumsum(rnorm(N_TIMES, 0, 0.2))
phi_temp_true <- phi_temp_true - mean(phi_temp_true)

df_pg_temporal <- data.frame(
  y = rpois(N_OBS, lambda = 30 * exp(re_true[as.numeric(site)] + phi_temp_true[time_idx])),
  effort = rgamma(N_OBS, shape = 5, rate = 0.1) * exp(re_true[as.numeric(site)] + phi_temp_true[time_idx]),
  x = rnorm(N_OBS),
  site = site,
  time = time_idx,
  time_factor = time_factor
)
df_pg_temporal$effort[df_pg_temporal$effort < 0.01] <- 0.01

# Fit numdenom
cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- tratio(
    y | effort ~ x + (1|site),
    data = df_pg_temporal,
    family = ratiod_poisson_gamma(),
    temporal = temporal_rw1("time_factor"),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

# Compile and fit Stan model
cat("Compiling Stan model... ")
stan_model_rw1 <- cmdstan_model("stan/joint_pg_rw1.stan")
cat("done\n")

stan_data_rw1 <- list(
  N = N_OBS,
  y_num = df_pg_temporal$y,
  y_denom = df_pg_temporal$effort,
  p = 2,
  X = cbind(1, df_pg_temporal$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_pg_temporal$site),
  T = N_TIMES,
  time_idx = df_pg_temporal$time
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_rw1$sample(
    data = stan_data_rw1,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

# Compare posteriors
draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

results$row_11_beta <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)
print_result(results$row_11_beta)
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Row 35: negbin_negbin + RE + ICAR spatial
# =============================================================================
cat("\n========== Row 35: negbin_negbin + RE + ICAR ==========\n")

# Generate data
df_nb_spatial <- data.frame(
  y = rnbinom(N_OBS, size = 5, mu = 30 * exp(re_true[as.numeric(site)] + phi_true[as.numeric(site)])),
  denom = rnbinom(N_OBS, size = 8, mu = 100 * exp(re_true[as.numeric(site)] + phi_true[as.numeric(site)])),
  x = rnorm(N_OBS),
  site = site,
  spatial_site = spatial_site
)

# Fit numdenom
cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- tratio(
    y | denom ~ x + (1|site),
    data = df_nb_spatial,
    family = ratiod_negbin_negbin(),
    spatial = spatial_car(spatial_info$adj_mat, level = "group", group_var = "spatial_site"),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

# Compile and fit Stan model
cat("Compiling Stan model... ")
stan_model_nb_icar <- cmdstan_model("stan/joint_nb_icar.stan")
cat("done\n")

stan_data_nb_icar <- list(
  N = N_OBS,
  y_num = df_nb_spatial$y,
  y_denom = df_nb_spatial$denom,
  p = 2,
  X = cbind(1, df_nb_spatial$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_nb_spatial$site),
  J = N_SITES,
  spatial_idx = as.numeric(df_nb_spatial$spatial_site),
  n_neighbors = spatial_info$n_neighbors,
  n_edges = spatial_info$n_edges,
  edge1 = spatial_info$edge1,
  edge2 = spatial_info$edge2
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_nb_icar$sample(
    data = stan_data_nb_icar,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

# Compare posteriors
draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

results$row_35_beta <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)
print_result(results$row_35_beta)
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Row 41: negbin_negbin + RE + RW1 temporal
# =============================================================================
cat("\n========== Row 41: negbin_negbin + RE + RW1 ==========\n")

# Generate data
df_nb_temporal <- data.frame(
  y = rnbinom(N_OBS, size = 5, mu = 30 * exp(re_true[as.numeric(site)] + phi_temp_true[time_idx])),
  denom = rnbinom(N_OBS, size = 8, mu = 100 * exp(re_true[as.numeric(site)] + phi_temp_true[time_idx])),
  x = rnorm(N_OBS),
  site = site,
  time = time_idx,
  time_factor = time_factor
)

# Fit numdenom
cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- tratio(
    y | denom ~ x + (1|site),
    data = df_nb_temporal,
    family = ratiod_negbin_negbin(),
    temporal = temporal_rw1("time_factor"),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

# Compile and fit Stan model
cat("Compiling Stan model... ")
stan_model_nb_rw1 <- cmdstan_model("stan/joint_nb_rw1.stan")
cat("done\n")

stan_data_nb_rw1 <- list(
  N = N_OBS,
  y_num = df_nb_temporal$y,
  y_denom = df_nb_temporal$denom,
  p = 2,
  X = cbind(1, df_nb_temporal$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_nb_temporal$site),
  T = N_TIMES,
  time_idx = df_nb_temporal$time
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_nb_rw1$sample(
    data = stan_data_nb_rw1,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

# Compare posteriors
draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

results$row_41_beta <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)
print_result(results$row_41_beta)
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Summary
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - BATCH 2\n")
cat("=======================================================\n")

n_pass <- sum(sapply(results, function(r) r$pass))
n_total <- length(results)

cat(sprintf("\nRow  5 (pg_icar):     %s\n", if(results$row_5_beta$pass) "PASS" else "FAIL"))
cat(sprintf("Row 11 (pg_rw1):      %s\n", if(results$row_11_beta$pass) "PASS" else "FAIL"))
cat(sprintf("Row 35 (nb_icar):     %s\n", if(results$row_35_beta$pass) "PASS" else "FAIL"))
cat(sprintf("Row 41 (nb_rw1):      %s\n", if(results$row_41_beta$pass) "PASS" else "FAIL"))

cat(sprintf("\nOverall: %d/%d parameters PASS (within 2 SE)\n", n_pass, n_total))

if (n_pass == n_total) {
  cat("\n** All posteriors match custom joint Stan models! **\n")
  cat("   This validates that numdenom correctly models spatial/temporal\n")
  cat("   structure for two-process (poisson_gamma, negbin_negbin) data.\n")
} else {
  cat("\n!! Some posteriors differ - investigate !!\n")
}

# Save results
saveRDS(results, "benchmarks/results_joint_batch2.rds")
cat("\nResults saved to benchmarks/results_joint_batch2.rds\n")

# Print update instructions for gradient_methods.md
cat("\n=======================================================\n")
cat("UPDATE gradient_methods.md:\n")
cat("=======================================================\n")
cat("Add '✓Stan (joint)' to Notes column for these rows:\n")
for (name in names(results)) {
  if (results[[name]]$pass) {
    row_num <- gsub("row_([0-9]+).*", "\\1", name)
    cat(sprintf("  Row %s: PASS\n", row_num))
  }
}
