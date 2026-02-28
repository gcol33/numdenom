# Validation of numdenom two-process models against custom joint Stan models
# Batch 9: Crossed RE, BYM2+RW1, Slopes+ICAR
#
# Rows validated:
#   - Row 4: poisson_gamma + crossed RE
#   - Row 34: negbin_negbin + crossed RE
#   - Row 19: poisson_gamma + BYM2 + RW1
#   - Row 49: negbin_negbin + BYM2 + RW1
#   - Row 25: poisson_gamma + slopes + ICAR
#   - Row 55: negbin_negbin + slopes + ICAR

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

# Standard benchmark parameters
N_OBS <- 200
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2
N_SITES <- 20
N_SITES2 <- 10
N_TIMES <- 10

cat("=======================================================\n")
cat("Joint Model Validation Batch 9: Crossed, BYM2+RW1, Slopes+ICAR\n")
cat("=======================================================\n")
cat(sprintf("N=%d, iter=%d, warmup=%d, chains=%d\n", N_OBS, N_ITER, N_WARMUP, N_CHAINS))
cat(sprintf("sites=%d, sites2=%d, times=%d\n\n", N_SITES, N_SITES2, N_TIMES))

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
  cat(sprintf("  %s: nd=%.4f (SD=%.4f), Stan=%.4f (SD=%.4f), diff=%.4f (%.2f SE) => %s\n",
              result$param, result$nd_mean, result$nd_sd,
              result$stan_mean, result$stan_sd,
              result$diff, result$ratio, if(result$pass) "PASS" else "FAIL"))
}

# =============================================================================
# DATA SETUP
# =============================================================================

site <- factor(rep(1:N_SITES, length.out = N_OBS))
site2 <- factor(sample(1:N_SITES2, N_OBS, replace = TRUE))
x <- rnorm(N_OBS)
time <- rep(1:N_TIMES, length.out = N_OBS)
time_factor <- factor(time)

# Spatial grid and adjacency
n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_SITES, ]
adj_mat <- matrix(0, N_SITES, N_SITES)
for (i in 1:N_SITES) {
  for (j in 1:N_SITES) {
    if (i != j) {
      dist <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
      if (dist <= 1.5) adj_mat[i, j] <- 1
    }
  }
}
spatial_site <- factor(rep(1:N_SITES, length.out = N_OBS))

# Build edge list from adjacency matrix
edges <- which(adj_mat == 1 & upper.tri(adj_mat), arr.ind = TRUE)
n_edges <- nrow(edges)
edge1 <- edges[, 1]
edge2 <- edges[, 2]
n_neighbors <- rowSums(adj_mat)

# BYM2 scale factor (geometric mean of eigenvalues)
Q <- diag(n_neighbors) - adj_mat
eig <- eigen(Q)$values
scale_factor <- exp(mean(log(eig[eig > 1e-10])))

results <- list()

# =============================================================================
# Row 4: poisson_gamma + crossed RE
# =============================================================================
cat("\n========== Row 4: poisson_gamma + crossed RE ==========\n")

# Generate data
eta_num <- 2 + 0.3 * x
eta_denom <- 4 + 0.2 * x
df_pg_crossed <- data.frame(
  y = rpois(N_OBS, exp(eta_num)),
  effort = rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_denom)),
  x = x,
  site = site,
  site2 = site2
)
df_pg_crossed$effort[df_pg_crossed$effort < 0.01] <- 0.01

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | effort ~ x + (1|site) + (1|site2),
    data = df_pg_crossed,
    family = ratiod_poisson_gamma(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_pg_crossed.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = df_pg_crossed$y,
  y_denom = df_pg_crossed$effort,
  p = 2,
  X = cbind(1, df_pg_crossed$x),
  n_groups1 = N_SITES,
  group_idx1 = as.integer(df_pg_crossed$site),
  n_groups2 = N_SITES2,
  group_idx2 = as.integer(df_pg_crossed$site2)
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

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

results$row_4 <- list(
  beta_num_2 = compare_posteriors(draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`, "beta_num[2]"),
  beta_denom_2 = compare_posteriors(draws_nd[, "beta_denom[2]"], draws_stan$`beta_denom[2]`, "beta_denom[2]"),
  time_nd = t_nd,
  time_stan = t_stan
)
print_result(results$row_4$beta_num_2)
print_result(results$row_4$beta_denom_2)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan/t_nd))

# =============================================================================
# Row 34: negbin_negbin + crossed RE
# =============================================================================
cat("\n========== Row 34: negbin_negbin + crossed RE ==========\n")

# Generate data
df_nb_crossed <- data.frame(
  y = rnbinom(N_OBS, mu = exp(eta_num), size = 5),
  denom = rnbinom(N_OBS, mu = exp(eta_denom), size = 10),
  x = x,
  site = site,
  site2 = site2
)
df_nb_crossed$denom[df_nb_crossed$denom == 0] <- 1

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x + (1|site) + (1|site2),
    data = df_nb_crossed,
    family = ratiod_negbin_negbin(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_nb_crossed.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = df_nb_crossed$y,
  y_denom = df_nb_crossed$denom,
  p = 2,
  X = cbind(1, df_nb_crossed$x),
  n_groups1 = N_SITES,
  group_idx1 = as.integer(df_nb_crossed$site),
  n_groups2 = N_SITES2,
  group_idx2 = as.integer(df_nb_crossed$site2)
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

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

results$row_34 <- list(
  beta_num_2 = compare_posteriors(draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`, "beta_num[2]"),
  beta_denom_2 = compare_posteriors(draws_nd[, "beta_denom[2]"], draws_stan$`beta_denom[2]`, "beta_denom[2]"),
  time_nd = t_nd,
  time_stan = t_stan
)
print_result(results$row_34$beta_num_2)
print_result(results$row_34$beta_denom_2)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan/t_nd))

# =============================================================================
# Row 19: poisson_gamma + BYM2 + RW1
# =============================================================================
cat("\n========== Row 19: poisson_gamma + BYM2 + RW1 ==========\n")

df_pg_bym2_rw1 <- data.frame(
  y = rpois(N_OBS, exp(eta_num)),
  effort = rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_denom)),
  x = x,
  site = site,
  spatial_site = spatial_site,
  time = time,
  time_factor = time_factor
)
df_pg_bym2_rw1$effort[df_pg_bym2_rw1$effort < 0.01] <- 0.01

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | effort ~ x + (1|site),
    data = df_pg_bym2_rw1,
    family = ratiod_poisson_gamma(),
    spatial = spatial_bym2(adj_mat, level = "group", group_var = "spatial_site"),
    temporal = temporal_rw1("time_factor"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_pg_bym2_rw1.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = df_pg_bym2_rw1$y,
  y_denom = df_pg_bym2_rw1$effort,
  p = 2,
  X = cbind(1, df_pg_bym2_rw1$x),
  n_groups = N_SITES,
  group_idx = as.integer(df_pg_bym2_rw1$site),
  J = N_SITES,
  spatial_idx = as.integer(df_pg_bym2_rw1$spatial_site),
  n_neighbors = as.array(n_neighbors),
  n_edges = n_edges,
  edge1 = as.array(edge1),
  edge2 = as.array(edge2),
  scale_factor = scale_factor,
  T = N_TIMES,
  time_idx = df_pg_bym2_rw1$time
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
    show_messages = FALSE,
    adapt_delta = 0.9
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

results$row_19 <- list(
  beta_num_2 = compare_posteriors(draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`, "beta_num[2]"),
  beta_denom_2 = compare_posteriors(draws_nd[, "beta_denom[2]"], draws_stan$`beta_denom[2]`, "beta_denom[2]"),
  time_nd = t_nd,
  time_stan = t_stan
)
print_result(results$row_19$beta_num_2)
print_result(results$row_19$beta_denom_2)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan/t_nd))

# =============================================================================
# Row 49: negbin_negbin + BYM2 + RW1
# =============================================================================
cat("\n========== Row 49: negbin_negbin + BYM2 + RW1 ==========\n")

df_nb_bym2_rw1 <- data.frame(
  y = rnbinom(N_OBS, mu = exp(eta_num), size = 5),
  denom = rnbinom(N_OBS, mu = exp(eta_denom), size = 10),
  x = x,
  site = site,
  spatial_site = spatial_site,
  time = time,
  time_factor = time_factor
)
df_nb_bym2_rw1$denom[df_nb_bym2_rw1$denom == 0] <- 1

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x + (1|site),
    data = df_nb_bym2_rw1,
    family = ratiod_negbin_negbin(),
    spatial = spatial_bym2(adj_mat, level = "group", group_var = "spatial_site"),
    temporal = temporal_rw1("time_factor"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_nb_bym2_rw1.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = df_nb_bym2_rw1$y,
  y_denom = df_nb_bym2_rw1$denom,
  p = 2,
  X = cbind(1, df_nb_bym2_rw1$x),
  n_groups = N_SITES,
  group_idx = as.integer(df_nb_bym2_rw1$site),
  J = N_SITES,
  spatial_idx = as.integer(df_nb_bym2_rw1$spatial_site),
  n_neighbors = as.array(n_neighbors),
  n_edges = n_edges,
  edge1 = as.array(edge1),
  edge2 = as.array(edge2),
  scale_factor = scale_factor,
  T = N_TIMES,
  time_idx = df_nb_bym2_rw1$time
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
    show_messages = FALSE,
    adapt_delta = 0.9
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

results$row_49 <- list(
  beta_num_2 = compare_posteriors(draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`, "beta_num[2]"),
  beta_denom_2 = compare_posteriors(draws_nd[, "beta_denom[2]"], draws_stan$`beta_denom[2]`, "beta_denom[2]"),
  time_nd = t_nd,
  time_stan = t_stan
)
print_result(results$row_49$beta_num_2)
print_result(results$row_49$beta_denom_2)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan/t_nd))

# =============================================================================
# Row 25: poisson_gamma + slopes + ICAR
# =============================================================================
cat("\n========== Row 25: poisson_gamma + slopes + ICAR ==========\n")

df_pg_slopes_icar <- data.frame(
  y = rpois(N_OBS, exp(eta_num)),
  effort = rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_denom)),
  x = x,
  site = site,
  spatial_site = spatial_site
)
df_pg_slopes_icar$effort[df_pg_slopes_icar$effort < 0.01] <- 0.01

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | effort ~ x + (1 + x|site),
    data = df_pg_slopes_icar,
    family = ratiod_poisson_gamma(),
    spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_pg_slopes_icar.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = df_pg_slopes_icar$y,
  y_denom = df_pg_slopes_icar$effort,
  p = 2,
  X = cbind(1, df_pg_slopes_icar$x),
  x_slope = df_pg_slopes_icar$x,
  n_groups = N_SITES,
  group_idx = as.integer(df_pg_slopes_icar$site),
  J = N_SITES,
  spatial_idx = as.integer(df_pg_slopes_icar$spatial_site),
  n_neighbors = as.array(n_neighbors),
  n_edges = n_edges,
  edge1 = as.array(edge1),
  edge2 = as.array(edge2)
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
    show_messages = FALSE,
    adapt_delta = 0.95
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

results$row_25 <- list(
  beta_num_2 = compare_posteriors(draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`, "beta_num[2]"),
  beta_denom_2 = compare_posteriors(draws_nd[, "beta_denom[2]"], draws_stan$`beta_denom[2]`, "beta_denom[2]"),
  time_nd = t_nd,
  time_stan = t_stan
)
print_result(results$row_25$beta_num_2)
print_result(results$row_25$beta_denom_2)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan/t_nd))

# =============================================================================
# Row 55: negbin_negbin + slopes + ICAR
# =============================================================================
cat("\n========== Row 55: negbin_negbin + slopes + ICAR ==========\n")

df_nb_slopes_icar <- data.frame(
  y = rnbinom(N_OBS, mu = exp(eta_num), size = 5),
  denom = rnbinom(N_OBS, mu = exp(eta_denom), size = 10),
  x = x,
  site = site,
  spatial_site = spatial_site
)
df_nb_slopes_icar$denom[df_nb_slopes_icar$denom == 0] <- 1

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x + (1 + x|site),
    data = df_nb_slopes_icar,
    family = ratiod_negbin_negbin(),
    spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_nb_slopes_icar.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = df_nb_slopes_icar$y,
  y_denom = df_nb_slopes_icar$denom,
  p = 2,
  X = cbind(1, df_nb_slopes_icar$x),
  x_slope = df_nb_slopes_icar$x,
  n_groups = N_SITES,
  group_idx = as.integer(df_nb_slopes_icar$site),
  J = N_SITES,
  spatial_idx = as.integer(df_nb_slopes_icar$spatial_site),
  n_neighbors = as.array(n_neighbors),
  n_edges = n_edges,
  edge1 = as.array(edge1),
  edge2 = as.array(edge2)
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
    show_messages = FALSE,
    adapt_delta = 0.95
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

results$row_55 <- list(
  beta_num_2 = compare_posteriors(draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`, "beta_num[2]"),
  beta_denom_2 = compare_posteriors(draws_nd[, "beta_denom[2]"], draws_stan$`beta_denom[2]`, "beta_denom[2]"),
  time_nd = t_nd,
  time_stan = t_stan
)
print_result(results$row_55$beta_num_2)
print_result(results$row_55$beta_denom_2)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan/t_nd))

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - BATCH 9\n")
cat("=======================================================\n\n")

cat(sprintf("%-8s %-30s %10s %10s %8s %s\n",
            "Row", "Model", "numdenom", "Stan", "Speedup", "Status"))
cat(paste(rep("-", 80), collapse = ""), "\n")

all_pass <- TRUE
for (row in names(results)) {
  r <- results[[row]]
  pass <- r$beta_num_2$pass && r$beta_denom_2$pass
  if (!pass) all_pass <- FALSE
  cat(sprintf("%-8s %-30s %10.1fs %10.1fs %8.1fx %s\n",
              row, paste0("beta_num[2], beta_denom[2]"),
              r$time_nd, r$time_stan, r$time_stan / r$time_nd,
              if(pass) "PASS" else "FAIL"))
}

cat(paste(rep("-", 80), collapse = ""), "\n")
cat(sprintf("\nOverall: %s\n", if(all_pass) "ALL PASS" else "SOME FAIL"))

# Save results
saveRDS(results, "benchmarks/results_joint_batch9.rds")
cat("\nResults saved to benchmarks/results_joint_batch9.rds\n")

# Print rows to update in gradient_methods.md
cat("\n\nUPDATE gradient_methods.md:\n")
cat("Add ✓Stan (joint) to Notes column for these rows:\n")
for (row in names(results)) {
  r <- results[[row]]
  pass <- r$beta_num_2$pass && r$beta_denom_2$pass
  if (pass) {
    row_num <- gsub("row_", "", row)
    cat(sprintf("  Row %s - PASS\n", row_num))
  }
}
