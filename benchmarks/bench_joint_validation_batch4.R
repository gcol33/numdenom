# Validation of numdenom two-process models against custom joint Stan models
# Batch 4: Proper CAR spatial and RW2 temporal models
#
# Rows validated:
#   - Row 10: poisson_gamma + RE + pCAR spatial
#   - Row 12: poisson_gamma + RE + RW2 temporal
#   - Row 40: negbin_negbin + RE + pCAR spatial
#   - Row 42: negbin_negbin + RE + RW2 temporal

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
N_TIMES <- 15  # Increased for RW2 (needs T >= 3)

cat("=======================================================\n")
cat("Joint Model Validation Batch 4: pCAR + RW2\n")
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

  edges <- list()
  for (i in 1:(n_units-1)) {
    for (j in (i+1):n_units) {
      dist <- sqrt((grid$x[i] - grid$x[j])^2 + (grid$y[i] - grid$y[j])^2)
      if (dist <= 1.01) {
        edges[[length(edges) + 1]] <- c(i, j)
      }
    }
  }

  n_neighbors <- integer(n_units)
  for (e in edges) {
    n_neighbors[e[1]] <- n_neighbors[e[1]] + 1
    n_neighbors[e[2]] <- n_neighbors[e[2]] + 1
  }

  n_edges <- length(edges)
  edge1 <- sapply(edges, `[`, 1)
  edge2 <- sapply(edges, `[`, 2)

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

spatial_info <- create_grid_adjacency(N_SITES)

results <- list()

# Generate shared effects for consistent data generation
site <- factor(rep(1:N_SITES, length.out = N_OBS))
spatial_site <- site

# Simulate shared spatial effect (for pCAR)
phi_spatial_true <- rnorm(N_SITES, 0, 0.3)
phi_spatial_true <- phi_spatial_true - mean(phi_spatial_true)

# Simulate shared site RE
re_true <- rnorm(N_SITES, 0, 0.3)

# Temporal indices
time_idx <- rep(1:N_TIMES, length.out = N_OBS)
time_factor <- factor(time_idx)

# Simulate RW2 temporal effect (smooth trend)
phi_temp_true <- numeric(N_TIMES)
phi_temp_true[1] <- rnorm(1, 0, 0.2)
phi_temp_true[2] <- phi_temp_true[1] + rnorm(1, 0, 0.15)
for (t in 3:N_TIMES) {
  # RW2: second differences are small
  phi_temp_true[t] <- 2*phi_temp_true[t-1] - phi_temp_true[t-2] + rnorm(1, 0, 0.1)
}
phi_temp_true <- phi_temp_true - mean(phi_temp_true)

# =============================================================================
# Row 10: poisson_gamma + RE + pCAR spatial
# =============================================================================
cat("\n========== Row 10: poisson_gamma + RE + pCAR ==========\n")

df_pg_pcar <- data.frame(
  y = rpois(N_OBS, lambda = 30 * exp(re_true[as.numeric(site)] + phi_spatial_true[as.numeric(site)])),
  effort = rgamma(N_OBS, shape = 5, rate = 0.1) * exp(re_true[as.numeric(site)] + phi_spatial_true[as.numeric(site)]),
  x = rnorm(N_OBS),
  site = site,
  spatial_site = spatial_site
)
df_pg_pcar$effort[df_pg_pcar$effort < 0.01] <- 0.01

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | effort ~ x + (1|site),
    data = df_pg_pcar,
    family = ratiod_poisson_gamma(),
    spatial = spatial_car(spatial_info$adj_mat, proper = TRUE, level = "group", group_var = "spatial_site"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model_pcar <- cmdstan_model("stan/joint_pg_pcar.stan")
cat("done\n")

stan_data_pcar <- list(
  N = N_OBS,
  y_num = df_pg_pcar$y,
  y_denom = df_pg_pcar$effort,
  p = 2,
  X = cbind(1, df_pg_pcar$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_pg_pcar$site),
  J = N_SITES,
  spatial_idx = as.numeric(df_pg_pcar$spatial_site),
  n_neighbors = spatial_info$n_neighbors,
  n_edges = spatial_info$n_edges,
  edge1 = spatial_info$edge1,
  edge2 = spatial_info$edge2
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_pcar$sample(
    data = stan_data_pcar,
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

results$row_10_beta <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)
print_result(results$row_10_beta)
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Row 12: poisson_gamma + RE + RW2 temporal
# =============================================================================
cat("\n========== Row 12: poisson_gamma + RE + RW2 ==========\n")

df_pg_rw2 <- data.frame(
  y = rpois(N_OBS, lambda = 30 * exp(re_true[as.numeric(site)] + phi_temp_true[time_idx])),
  effort = rgamma(N_OBS, shape = 5, rate = 0.1) * exp(re_true[as.numeric(site)] + phi_temp_true[time_idx]),
  x = rnorm(N_OBS),
  site = site,
  time = time_idx,
  time_factor = time_factor
)
df_pg_rw2$effort[df_pg_rw2$effort < 0.01] <- 0.01

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | effort ~ x + (1|site),
    data = df_pg_rw2,
    family = ratiod_poisson_gamma(),
    temporal = temporal_rw2("time_factor"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model_rw2 <- cmdstan_model("stan/joint_pg_rw2.stan")
cat("done\n")

stan_data_rw2 <- list(
  N = N_OBS,
  y_num = df_pg_rw2$y,
  y_denom = df_pg_rw2$effort,
  p = 2,
  X = cbind(1, df_pg_rw2$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_pg_rw2$site),
  T = N_TIMES,
  time_idx = df_pg_rw2$time
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_rw2$sample(
    data = stan_data_rw2,
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

results$row_12_beta <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)
print_result(results$row_12_beta)
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Row 40: negbin_negbin + RE + pCAR spatial
# =============================================================================
cat("\n========== Row 40: negbin_negbin + RE + pCAR ==========\n")

df_nb_pcar <- data.frame(
  y = rnbinom(N_OBS, size = 5, mu = 30 * exp(re_true[as.numeric(site)] + phi_spatial_true[as.numeric(site)])),
  denom = rnbinom(N_OBS, size = 8, mu = 100 * exp(re_true[as.numeric(site)] + phi_spatial_true[as.numeric(site)])),
  x = rnorm(N_OBS),
  site = site,
  spatial_site = spatial_site
)

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x + (1|site),
    data = df_nb_pcar,
    family = ratiod_negbin_negbin(),
    spatial = spatial_car(spatial_info$adj_mat, proper = TRUE, level = "group", group_var = "spatial_site"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model_nb_pcar <- cmdstan_model("stan/joint_nb_pcar.stan")
cat("done\n")

stan_data_nb_pcar <- list(
  N = N_OBS,
  y_num = df_nb_pcar$y,
  y_denom = df_nb_pcar$denom,
  p = 2,
  X = cbind(1, df_nb_pcar$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_nb_pcar$site),
  J = N_SITES,
  spatial_idx = as.numeric(df_nb_pcar$spatial_site),
  n_neighbors = spatial_info$n_neighbors,
  n_edges = spatial_info$n_edges,
  edge1 = spatial_info$edge1,
  edge2 = spatial_info$edge2
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_nb_pcar$sample(
    data = stan_data_nb_pcar,
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

results$row_40_beta <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)
print_result(results$row_40_beta)
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Row 42: negbin_negbin + RE + RW2 temporal
# =============================================================================
cat("\n========== Row 42: negbin_negbin + RE + RW2 ==========\n")

df_nb_rw2 <- data.frame(
  y = rnbinom(N_OBS, size = 5, mu = 30 * exp(re_true[as.numeric(site)] + phi_temp_true[time_idx])),
  denom = rnbinom(N_OBS, size = 8, mu = 100 * exp(re_true[as.numeric(site)] + phi_temp_true[time_idx])),
  x = rnorm(N_OBS),
  site = site,
  time = time_idx,
  time_factor = time_factor
)

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x + (1|site),
    data = df_nb_rw2,
    family = ratiod_negbin_negbin(),
    temporal = temporal_rw2("time_factor"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model_nb_rw2 <- cmdstan_model("stan/joint_nb_rw2.stan")
cat("done\n")

stan_data_nb_rw2 <- list(
  N = N_OBS,
  y_num = df_nb_rw2$y,
  y_denom = df_nb_rw2$denom,
  p = 2,
  X = cbind(1, df_nb_rw2$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_nb_rw2$site),
  T = N_TIMES,
  time_idx = df_nb_rw2$time
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_nb_rw2$sample(
    data = stan_data_nb_rw2,
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

results$row_42_beta <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)
print_result(results$row_42_beta)
cat(sprintf("  Speedup: %.1fx\n", t_stan / t_nd))

# =============================================================================
# Summary
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - BATCH 4\n")
cat("=======================================================\n")

n_pass <- sum(sapply(results, function(r) r$pass))
n_total <- length(results)

cat(sprintf("\nRow 10 (pg_pcar):     %s\n", if(results$row_10_beta$pass) "PASS" else "FAIL"))
cat(sprintf("Row 12 (pg_rw2):      %s\n", if(results$row_12_beta$pass) "PASS" else "FAIL"))
cat(sprintf("Row 40 (nb_pcar):     %s\n", if(results$row_40_beta$pass) "PASS" else "FAIL"))
cat(sprintf("Row 42 (nb_rw2):      %s\n", if(results$row_42_beta$pass) "PASS" else "FAIL"))

cat(sprintf("\nOverall: %d/%d parameters PASS (within 2 SE)\n", n_pass, n_total))

if (n_pass == n_total) {
  cat("\n** All posteriors match custom joint Stan models! **\n")
} else {
  cat("\n!! Some posteriors differ - investigate !!\n")
}

# Save results
saveRDS(results, "results_joint_batch4.rds")
cat("\nResults saved to results_joint_batch4.rds\n")

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
