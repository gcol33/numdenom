# Debug script for rows 6 and 43 validation failures
# Run with more iterations to check if it's noise

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(12345)  # Different seed

N_OBS <- 200
N_SITES <- 20
N_TIMES <- 10
N_ITER <- 2000  # More iterations
N_WARMUP <- 1000
N_CHAINS <- 2

cat("=======================================================\n")
cat("Debug: Rows 6 and 43 with more iterations\n")
cat("=======================================================\n")
cat(sprintf("N=%d, iter=%d, warmup=%d, chains=%d\n\n", N_OBS, N_ITER, N_WARMUP, N_CHAINS))

# Helper function
compare_posteriors <- function(nd_draws, stan_draws, param_name, threshold_se = 2) {
  nd_mean <- mean(nd_draws)
  nd_sd <- sd(nd_draws)
  stan_mean <- mean(stan_draws)
  stan_sd <- sd(stan_draws)

  se_combined <- sqrt(nd_sd^2 / length(nd_draws) + stan_sd^2 / length(stan_draws))
  diff <- abs(nd_mean - stan_mean)
  ratio <- diff / se_combined

  list(
    param = param_name,
    nd_mean = nd_mean,
    nd_sd = nd_sd,
    stan_mean = stan_mean,
    stan_sd = stan_sd,
    diff = diff,
    ratio = ratio,
    pass = ratio < threshold_se
  )
}

# Create adjacency and BYM2 scale factor
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

  # Compute BYM2 scale factor using numdenom's exact method
  scale_factor <- numdenom:::compute_bym2_scale(adj_mat)

  cat(sprintf("BYM2 scale factor (numdenom): %.6f\n", scale_factor))

  list(
    adj_mat = adj_mat,
    n_neighbors = n_neighbors,
    n_edges = n_edges,
    edge1 = edge1,
    edge2 = edge2,
    scale_factor = scale_factor
  )
}

spatial_info <- create_grid_adjacency(N_SITES)

# Data
site <- factor(rep(1:N_SITES, length.out = N_OBS))
spatial_site <- site
re_true <- rnorm(N_SITES, 0, 0.3)
phi_true <- rnorm(N_SITES, 0, 0.3)
phi_true <- phi_true - mean(phi_true)

time_idx <- rep(1:N_TIMES, length.out = N_OBS)
time_factor <- factor(time_idx)
rho_true <- 0.7
phi_temp_true <- numeric(N_TIMES)
phi_temp_true[1] <- rnorm(1, 0, 0.3)
for (t in 2:N_TIMES) {
  phi_temp_true[t] <- rho_true * phi_temp_true[t-1] + rnorm(1, 0, 0.15)
}
phi_temp_true <- phi_temp_true - mean(phi_temp_true)

# =============================================================================
# Row 6: poisson_gamma + RE + BYM2
# =============================================================================
cat("\n========== Row 6: poisson_gamma + RE + BYM2 ==========\n")

df_pg_bym2 <- data.frame(
  y = rpois(N_OBS, lambda = 30 * exp(re_true[as.numeric(site)] + phi_true[as.numeric(site)])),
  effort = rgamma(N_OBS, shape = 5, rate = 0.1) * exp(re_true[as.numeric(site)] + phi_true[as.numeric(site)]),
  x = rnorm(N_OBS),
  site = site,
  spatial_site = spatial_site
)
df_pg_bym2$effort[df_pg_bym2$effort < 0.01] <- 0.01

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- tratio(
    y | effort ~ x + (1|site),
    data = df_pg_bym2,
    family = ratiod_poisson_gamma(),
    spatial = spatial_bym2(spatial_info$adj_mat, level = "group", group_var = "spatial_site"),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model_bym2 <- cmdstan_model("stan/joint_pg_bym2.stan")
cat("done\n")

stan_data_bym2 <- list(
  N = N_OBS,
  y_num = df_pg_bym2$y,
  y_denom = df_pg_bym2$effort,
  p = 2,
  X = cbind(1, df_pg_bym2$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_pg_bym2$site),
  J = N_SITES,
  spatial_idx = as.numeric(df_pg_bym2$spatial_site),
  n_neighbors = spatial_info$n_neighbors,
  n_edges = spatial_info$n_edges,
  edge1 = spatial_info$edge1,
  edge2 = spatial_info$edge2,
  scale_factor = spatial_info$scale_factor
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_bym2$sample(
    data = stan_data_bym2,
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

result_6 <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)

cat(sprintf("\n%s:\n", result_6$param))
cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", result_6$nd_mean, result_6$nd_sd))
cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", result_6$stan_mean, result_6$stan_sd))
cat(sprintf("  Diff: %.4f (%.2f SE) => %s\n",
            result_6$diff, result_6$ratio, if(result_6$pass) "PASS" else "FAIL"))

# =============================================================================
# Row 43: negbin_negbin + RE + AR1
# =============================================================================
cat("\n========== Row 43: negbin_negbin + RE + AR1 ==========\n")

df_nb_ar1 <- data.frame(
  y = rnbinom(N_OBS, size = 5, mu = 30 * exp(re_true[as.numeric(site)] + phi_temp_true[time_idx])),
  denom = rnbinom(N_OBS, size = 8, mu = 100 * exp(re_true[as.numeric(site)] + phi_temp_true[time_idx])),
  x = rnorm(N_OBS),
  site = site,
  time = time_idx,
  time_factor = time_factor
)

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- tratio(
    y | denom ~ x + (1|site),
    data = df_nb_ar1,
    family = ratiod_negbin_negbin(),
    temporal = temporal_ar1("time_factor"),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model_ar1 <- cmdstan_model("stan/joint_nb_ar1.stan")
cat("done\n")

stan_data_ar1 <- list(
  N = N_OBS,
  y_num = df_nb_ar1$y,
  y_denom = df_nb_ar1$denom,
  p = 2,
  X = cbind(1, df_nb_ar1$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df_nb_ar1$site),
  T = N_TIMES,
  time_idx = df_nb_ar1$time
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model_ar1$sample(
    data = stan_data_ar1,
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

result_43 <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2] (x slope)"
)

cat(sprintf("\n%s:\n", result_43$param))
cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", result_43$nd_mean, result_43$nd_sd))
cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", result_43$stan_mean, result_43$stan_sd))
cat(sprintf("  Diff: %.4f (%.2f SE) => %s\n",
            result_43$diff, result_43$ratio, if(result_43$pass) "PASS" else "FAIL"))

# =============================================================================
# Summary
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY\n")
cat("=======================================================\n")
cat(sprintf("Row  6 (pg_bym2):  %.2f SE => %s\n", result_6$ratio, if(result_6$pass) "PASS" else "FAIL"))
cat(sprintf("Row 43 (nb_ar1):   %.2f SE => %s\n", result_43$ratio, if(result_43$pass) "PASS" else "FAIL"))
