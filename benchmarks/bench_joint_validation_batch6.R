# Validation of numdenom two-process models against custom joint Stan models
# Batch 6: Spatial + Temporal combinations
#
# Rows validated:
#   - Row 18: poisson_gamma + RE + ICAR + RW1
#   - Row 20: poisson_gamma + RE + ICAR + AR1
#   - Row 48: negbin_negbin + RE + ICAR + RW1
#   - Row 50: negbin_negbin + RE + ICAR + AR1

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

# Standard benchmark parameters
N_OBS <- 200
N_ITER <- 1000
N_WARMUP <- 500
N_CHAINS <- 2
N_SITES <- 15
N_TIMES <- 8

cat("=======================================================\n")
cat("Joint Model Validation Batch 6: Spatial + Temporal\n")
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

# Generate shared effects
site <- factor(rep(1:N_SITES, length.out = N_OBS))
spatial_site <- site  # Same for simplicity
time_idx <- rep(1:N_TIMES, length.out = N_OBS)
time_factor <- factor(time_idx)

# Simulate shared RE
re_true <- rnorm(N_SITES, 0, 0.3)

# Simulate shared spatial effect
phi_spatial_true <- rnorm(N_SITES, 0, 0.3)
phi_spatial_true <- phi_spatial_true - mean(phi_spatial_true)

# Simulate shared temporal effect (RW1-like)
phi_temporal_true <- cumsum(rnorm(N_TIMES, 0, 0.15))
phi_temporal_true <- phi_temporal_true - mean(phi_temporal_true)

# Simulate AR1 temporal effect
rho_ar1_true <- 0.7
phi_ar1_true <- numeric(N_TIMES)
phi_ar1_true[1] <- rnorm(1, 0, 0.3)
for (t in 2:N_TIMES) {
  phi_ar1_true[t] <- rho_ar1_true * phi_ar1_true[t-1] + rnorm(1, 0, 0.15)
}
phi_ar1_true <- phi_ar1_true - mean(phi_ar1_true)

# =============================================================================
# Row 18: poisson_gamma + RE + ICAR + RW1
# =============================================================================
cat("\n========== Row 18: poisson_gamma + RE + ICAR + RW1 ==========\n")

eta_pg <- re_true[as.numeric(site)] + phi_spatial_true[as.numeric(site)] + phi_temporal_true[time_idx]

df_pg_icar_rw1 <- data.frame(
  y = rpois(N_OBS, lambda = 30 * exp(eta_pg)),
  effort = rgamma(N_OBS, shape = 5, rate = 0.1) * exp(eta_pg),
  x = rnorm(N_OBS),
  site = site,
  spatial_site = spatial_site,
  time = time_idx,
  time_factor = time_factor
)
df_pg_icar_rw1$effort[df_pg_icar_rw1$effort < 0.01] <- 0.01

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | effort ~ x + (1|site),
    data = df_pg_icar_rw1,
    family = ratiod_poisson_gamma(),
    spatial = spatial_car(spatial_info$adj_mat, level = "group", group_var = "spatial_site"),
    temporal = temporal_rw1("time_factor"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model_pg_icar_rw1 <- cmdstan_model("stan/joint_pg_icar_rw1.stan")
cat("done\n")

stan_data_pg_icar_rw1 <- list(
  N = N_OBS,
  y_num = df_pg_icar_rw1$y,
  y_denom = df_pg_icar_rw1$effort,
  p = 2,
  X = cbind(1, df_pg_icar_rw1$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_pg_icar_rw1$site),
  J = N_SITES,
  spatial_idx = as.numeric(df_pg_icar_rw1$spatial_site),
  n_neighbors = spatial_info$n_neighbors,
  n_edges = spatial_info$n_edges,
  edge1 = spatial_info$edge1,
  edge2 = spatial_info$edge2,
  T = N_TIMES,
  time_idx = df_pg_icar_rw1$time
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_pg_icar_rw1$sample(
    data = stan_data_pg_icar_rw1,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

results$row_18_beta <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)
print_result(results$row_18_beta)
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Row 20: poisson_gamma + RE + ICAR + AR1
# =============================================================================
cat("\n========== Row 20: poisson_gamma + RE + ICAR + AR1 ==========\n")

eta_pg_ar1 <- re_true[as.numeric(site)] + phi_spatial_true[as.numeric(site)] + phi_ar1_true[time_idx]

df_pg_icar_ar1 <- data.frame(
  y = rpois(N_OBS, lambda = 30 * exp(eta_pg_ar1)),
  effort = rgamma(N_OBS, shape = 5, rate = 0.1) * exp(eta_pg_ar1),
  x = rnorm(N_OBS),
  site = site,
  spatial_site = spatial_site,
  time = time_idx,
  time_factor = time_factor
)
df_pg_icar_ar1$effort[df_pg_icar_ar1$effort < 0.01] <- 0.01

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | effort ~ x + (1|site),
    data = df_pg_icar_ar1,
    family = ratiod_poisson_gamma(),
    spatial = spatial_car(spatial_info$adj_mat, level = "group", group_var = "spatial_site"),
    temporal = temporal_ar1("time_factor"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model_pg_icar_ar1 <- cmdstan_model("stan/joint_pg_icar_ar1.stan")
cat("done\n")

stan_data_pg_icar_ar1 <- list(
  N = N_OBS,
  y_num = df_pg_icar_ar1$y,
  y_denom = df_pg_icar_ar1$effort,
  p = 2,
  X = cbind(1, df_pg_icar_ar1$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_pg_icar_ar1$site),
  J = N_SITES,
  spatial_idx = as.numeric(df_pg_icar_ar1$spatial_site),
  n_neighbors = spatial_info$n_neighbors,
  n_edges = spatial_info$n_edges,
  edge1 = spatial_info$edge1,
  edge2 = spatial_info$edge2,
  T = N_TIMES,
  time_idx = df_pg_icar_ar1$time
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_pg_icar_ar1$sample(
    data = stan_data_pg_icar_ar1,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

results$row_20_beta <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)
print_result(results$row_20_beta)
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Row 48: negbin_negbin + RE + ICAR + RW1
# =============================================================================
cat("\n========== Row 48: negbin_negbin + RE + ICAR + RW1 ==========\n")

df_nb_icar_rw1 <- data.frame(
  y = rnbinom(N_OBS, size = 5, mu = 30 * exp(eta_pg)),
  denom = rnbinom(N_OBS, size = 8, mu = 100 * exp(eta_pg)),
  x = rnorm(N_OBS),
  site = site,
  spatial_site = spatial_site,
  time = time_idx,
  time_factor = time_factor
)

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x + (1|site),
    data = df_nb_icar_rw1,
    family = ratiod_negbin_negbin(),
    spatial = spatial_car(spatial_info$adj_mat, level = "group", group_var = "spatial_site"),
    temporal = temporal_rw1("time_factor"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model_nb_icar_rw1 <- cmdstan_model("stan/joint_nb_icar_rw1.stan")
cat("done\n")

stan_data_nb_icar_rw1 <- list(
  N = N_OBS,
  y_num = df_nb_icar_rw1$y,
  y_denom = df_nb_icar_rw1$denom,
  p = 2,
  X = cbind(1, df_nb_icar_rw1$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_nb_icar_rw1$site),
  J = N_SITES,
  spatial_idx = as.numeric(df_nb_icar_rw1$spatial_site),
  n_neighbors = spatial_info$n_neighbors,
  n_edges = spatial_info$n_edges,
  edge1 = spatial_info$edge1,
  edge2 = spatial_info$edge2,
  T = N_TIMES,
  time_idx = df_nb_icar_rw1$time
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_nb_icar_rw1$sample(
    data = stan_data_nb_icar_rw1,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

results$row_48_beta <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)
print_result(results$row_48_beta)
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Row 50: negbin_negbin + RE + ICAR + AR1
# =============================================================================
cat("\n========== Row 50: negbin_negbin + RE + ICAR + AR1 ==========\n")

df_nb_icar_ar1 <- data.frame(
  y = rnbinom(N_OBS, size = 5, mu = 30 * exp(eta_pg_ar1)),
  denom = rnbinom(N_OBS, size = 8, mu = 100 * exp(eta_pg_ar1)),
  x = rnorm(N_OBS),
  site = site,
  spatial_site = spatial_site,
  time = time_idx,
  time_factor = time_factor
)

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x + (1|site),
    data = df_nb_icar_ar1,
    family = ratiod_negbin_negbin(),
    spatial = spatial_car(spatial_info$adj_mat, level = "group", group_var = "spatial_site"),
    temporal = temporal_ar1("time_factor"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model_nb_icar_ar1 <- cmdstan_model("stan/joint_nb_icar_ar1.stan")
cat("done\n")

stan_data_nb_icar_ar1 <- list(
  N = N_OBS,
  y_num = df_nb_icar_ar1$y,
  y_denom = df_nb_icar_ar1$denom,
  p = 2,
  X = cbind(1, df_nb_icar_ar1$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_nb_icar_ar1$site),
  J = N_SITES,
  spatial_idx = as.numeric(df_nb_icar_ar1$spatial_site),
  n_neighbors = spatial_info$n_neighbors,
  n_edges = spatial_info$n_edges,
  edge1 = spatial_info$edge1,
  edge2 = spatial_info$edge2,
  T = N_TIMES,
  time_idx = df_nb_icar_ar1$time
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_nb_icar_ar1$sample(
    data = stan_data_nb_icar_ar1,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

results$row_50_beta <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)
print_result(results$row_50_beta)
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Summary
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - BATCH 6\n")
cat("=======================================================\n")

n_pass <- sum(sapply(results, function(r) r$pass))
n_total <- length(results)

cat(sprintf("\nRow 18 (pg_icar_rw1):   %s\n", if(results$row_18_beta$pass) "PASS" else "FAIL"))
cat(sprintf("Row 20 (pg_icar_ar1):   %s\n", if(results$row_20_beta$pass) "PASS" else "FAIL"))
cat(sprintf("Row 48 (nb_icar_rw1):   %s\n", if(results$row_48_beta$pass) "PASS" else "FAIL"))
cat(sprintf("Row 50 (nb_icar_ar1):   %s\n", if(results$row_50_beta$pass) "PASS" else "FAIL"))

cat(sprintf("\nOverall: %d/%d parameters PASS (within 2 SE)\n", n_pass, n_total))

if (n_pass == n_total) {
  cat("\n** All posteriors match custom joint Stan models! **\n")
} else {
  cat("\n!! Some posteriors differ - investigate !!\n")
}

# Save results
saveRDS(results, "results_joint_batch6.rds")
cat("\nResults saved to results_joint_batch6.rds\n")

# Print update instructions
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
