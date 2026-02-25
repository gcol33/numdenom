# Validation of numdenom lognormal family against custom joint Stan models
# Rows 98-102 in gradient_methods.md
#
# Row 98: lognormal (no RE)
# Row 99: lognormal + RE
# Row 100: lognormal + RE + ICAR
# Row 101: lognormal + RE + RW1
# Row 102: lognormal + RE + ICAR + RW1

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
N_TIMES <- 10

cat("=======================================================\n")
cat("Joint Model Validation: lognormal Family (Rows 98-102)\n")
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
  cat(sprintf("  %s: nd=%.4f (SD=%.4f), Stan=%.4f (SD=%.4f), diff=%.4f (%.2f SE) => %s\n",
              result$param, result$nd_mean, result$nd_sd,
              result$stan_mean, result$stan_sd,
              result$diff, result$ratio, if(result$pass) "PASS" else "FAIL"))
}

# =============================================================================
# DATA SETUP
# =============================================================================

site <- factor(rep(1:N_SITES, length.out = N_OBS))
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

# Generate data for lognormal family (must be positive)
# Base linear predictor (without random effects - those are added per model)
beta_num <- c(2, 0.3)     # intercept, slope
beta_denom <- c(3, 0.2)   # intercept, slope
sigma_num <- 0.5
sigma_denom <- 0.4

# True random effects for site
true_sigma_re <- 0.4
true_re <- rnorm(N_SITES, 0, true_sigma_re)

# True spatial effects (ICAR-like) - use simple spatial correlation
true_tau_spatial <- 2.0
true_phi_spatial <- rnorm(N_SITES, 0, 1/sqrt(true_tau_spatial))
true_phi_spatial <- true_phi_spatial - mean(true_phi_spatial)  # center

# True temporal effects (RW1)
true_tau_temporal <- 3.0
true_phi_temporal <- cumsum(rnorm(N_TIMES, 0, 1/sqrt(true_tau_temporal)))
true_phi_temporal <- true_phi_temporal - mean(true_phi_temporal)  # center

results <- list()

# =============================================================================
# Row 98: lognormal (no RE)
# =============================================================================
cat("\n========== Row 98: lognormal (no RE) ==========\n")

# No RE in DGP for base model
eta_num_base <- beta_num[1] + beta_num[2] * x
eta_denom_base <- beta_denom[1] + beta_denom[2] * x

df_ln_base <- data.frame(
  y = exp(rnorm(N_OBS, mean = eta_num_base, sd = sigma_num)),
  denom = exp(rnorm(N_OBS, mean = eta_denom_base, sd = sigma_denom)),
  x = x
)
# Ensure positive values
df_ln_base$y[df_ln_base$y <= 0] <- 0.01
df_ln_base$denom[df_ln_base$denom <= 0] <- 0.01

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x,
    data = df_ln_base,
    family = ratiod_lognormal(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_ln_base.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = df_ln_base$y,
  y_denom = df_ln_base$denom,
  p = 2,
  X = cbind(1, df_ln_base$x)
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

results$row_98 <- list(
  beta_num_2 = compare_posteriors(draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`, "beta_num[2]"),
  beta_denom_2 = compare_posteriors(draws_nd[, "beta_denom[2]"], draws_stan$`beta_denom[2]`, "beta_denom[2]"),
  time_nd = t_nd,
  time_stan = t_stan
)
print_result(results$row_98$beta_num_2)
print_result(results$row_98$beta_denom_2)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan/t_nd))

# =============================================================================
# Row 99: lognormal + RE
# =============================================================================
cat("\n========== Row 99: lognormal + RE ==========\n")

# Include true RE in DGP
eta_num_re <- beta_num[1] + beta_num[2] * x + true_re[as.integer(site)]
eta_denom_re <- beta_denom[1] + beta_denom[2] * x + true_re[as.integer(site)]

df_ln_re <- data.frame(
  y = exp(rnorm(N_OBS, mean = eta_num_re, sd = sigma_num)),
  denom = exp(rnorm(N_OBS, mean = eta_denom_re, sd = sigma_denom)),
  x = x,
  site = site
)
df_ln_re$y[df_ln_re$y <= 0] <- 0.01
df_ln_re$denom[df_ln_re$denom <= 0] <- 0.01

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x + (1|site),
    data = df_ln_re,
    family = ratiod_lognormal(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_ln_re.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = df_ln_re$y,
  y_denom = df_ln_re$denom,
  p = 2,
  X = cbind(1, df_ln_re$x),
  n_groups = N_SITES,
  group_idx = as.integer(df_ln_re$site)
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

results$row_99 <- list(
  beta_num_2 = compare_posteriors(draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`, "beta_num[2]"),
  beta_denom_2 = compare_posteriors(draws_nd[, "beta_denom[2]"], draws_stan$`beta_denom[2]`, "beta_denom[2]"),
  time_nd = t_nd,
  time_stan = t_stan
)
print_result(results$row_99$beta_num_2)
print_result(results$row_99$beta_denom_2)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan/t_nd))

# =============================================================================
# Row 100: lognormal + RE + ICAR
# =============================================================================
cat("\n========== Row 100: lognormal + RE + ICAR ==========\n")

# Include true RE + spatial in DGP
eta_num_icar <- beta_num[1] + beta_num[2] * x + true_re[as.integer(site)] + true_phi_spatial[as.integer(spatial_site)]
eta_denom_icar <- beta_denom[1] + beta_denom[2] * x + true_re[as.integer(site)] + true_phi_spatial[as.integer(spatial_site)]

df_ln_icar <- data.frame(
  y = exp(rnorm(N_OBS, mean = eta_num_icar, sd = sigma_num)),
  denom = exp(rnorm(N_OBS, mean = eta_denom_icar, sd = sigma_denom)),
  x = x,
  site = site,
  spatial_site = spatial_site
)
df_ln_icar$y[df_ln_icar$y <= 0] <- 0.01
df_ln_icar$denom[df_ln_icar$denom <= 0] <- 0.01

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x + (1|site),
    data = df_ln_icar,
    family = ratiod_lognormal(),
    spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_ln_icar.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = df_ln_icar$y,
  y_denom = df_ln_icar$denom,
  p = 2,
  X = cbind(1, df_ln_icar$x),
  n_groups = N_SITES,
  group_idx = as.integer(df_ln_icar$site),
  J = N_SITES,
  spatial_idx = as.integer(df_ln_icar$spatial_site),
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
    adapt_delta = 0.9
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

results$row_100 <- list(
  beta_num_2 = compare_posteriors(draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`, "beta_num[2]"),
  beta_denom_2 = compare_posteriors(draws_nd[, "beta_denom[2]"], draws_stan$`beta_denom[2]`, "beta_denom[2]"),
  time_nd = t_nd,
  time_stan = t_stan
)
print_result(results$row_100$beta_num_2)
print_result(results$row_100$beta_denom_2)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan/t_nd))

# =============================================================================
# Row 101: lognormal + RE + RW1
# =============================================================================
cat("\n========== Row 101: lognormal + RE + RW1 ==========\n")

# Include true RE + temporal in DGP
eta_num_rw1 <- beta_num[1] + beta_num[2] * x + true_re[as.integer(site)] + true_phi_temporal[time]
eta_denom_rw1 <- beta_denom[1] + beta_denom[2] * x + true_re[as.integer(site)] + true_phi_temporal[time]

df_ln_rw1 <- data.frame(
  y = exp(rnorm(N_OBS, mean = eta_num_rw1, sd = sigma_num)),
  denom = exp(rnorm(N_OBS, mean = eta_denom_rw1, sd = sigma_denom)),
  x = x,
  site = site,
  time = time,
  time_factor = time_factor
)
df_ln_rw1$y[df_ln_rw1$y <= 0] <- 0.01
df_ln_rw1$denom[df_ln_rw1$denom <= 0] <- 0.01

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x + (1|site),
    data = df_ln_rw1,
    family = ratiod_lognormal(),
    temporal = temporal_rw1("time_factor"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_ln_rw1.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = df_ln_rw1$y,
  y_denom = df_ln_rw1$denom,
  p = 2,
  X = cbind(1, df_ln_rw1$x),
  n_groups = N_SITES,
  group_idx = as.integer(df_ln_rw1$site),
  T = N_TIMES,
  time_idx = df_ln_rw1$time
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

results$row_101 <- list(
  beta_num_2 = compare_posteriors(draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`, "beta_num[2]"),
  beta_denom_2 = compare_posteriors(draws_nd[, "beta_denom[2]"], draws_stan$`beta_denom[2]`, "beta_denom[2]"),
  time_nd = t_nd,
  time_stan = t_stan
)
print_result(results$row_101$beta_num_2)
print_result(results$row_101$beta_denom_2)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan/t_nd))

# =============================================================================
# Row 102: lognormal + RE + ICAR + RW1
# =============================================================================
cat("\n========== Row 102: lognormal + RE + ICAR + RW1 ==========\n")

# Include true RE + spatial + temporal in DGP
eta_num_icar_rw1 <- beta_num[1] + beta_num[2] * x + true_re[as.integer(site)] + true_phi_spatial[as.integer(spatial_site)] + true_phi_temporal[time]
eta_denom_icar_rw1 <- beta_denom[1] + beta_denom[2] * x + true_re[as.integer(site)] + true_phi_spatial[as.integer(spatial_site)] + true_phi_temporal[time]

df_ln_icar_rw1 <- data.frame(
  y = exp(rnorm(N_OBS, mean = eta_num_icar_rw1, sd = sigma_num)),
  denom = exp(rnorm(N_OBS, mean = eta_denom_icar_rw1, sd = sigma_denom)),
  x = x,
  site = site,
  spatial_site = spatial_site,
  time = time,
  time_factor = time_factor
)
df_ln_icar_rw1$y[df_ln_icar_rw1$y <= 0] <- 0.01
df_ln_icar_rw1$denom[df_ln_icar_rw1$denom <= 0] <- 0.01

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x + (1|site),
    data = df_ln_icar_rw1,
    family = ratiod_lognormal(),
    spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
    temporal = temporal_rw1("time_factor"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_ln_icar_rw1.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = df_ln_icar_rw1$y,
  y_denom = df_ln_icar_rw1$denom,
  p = 2,
  X = cbind(1, df_ln_icar_rw1$x),
  n_groups = N_SITES,
  group_idx = as.integer(df_ln_icar_rw1$site),
  J = N_SITES,
  spatial_idx = as.integer(df_ln_icar_rw1$spatial_site),
  n_neighbors = as.array(n_neighbors),
  n_edges = n_edges,
  edge1 = as.array(edge1),
  edge2 = as.array(edge2),
  T = N_TIMES,
  time_idx = df_ln_icar_rw1$time
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

results$row_102 <- list(
  beta_num_2 = compare_posteriors(draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`, "beta_num[2]"),
  beta_denom_2 = compare_posteriors(draws_nd[, "beta_denom[2]"], draws_stan$`beta_denom[2]`, "beta_denom[2]"),
  time_nd = t_nd,
  time_stan = t_stan
)
print_result(results$row_102$beta_num_2)
print_result(results$row_102$beta_denom_2)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan/t_nd))

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - lognormal Family (Rows 98-102)\n")
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
saveRDS(results, "benchmarks/results_lognormal_validation.rds")
cat("\nResults saved to benchmarks/results_lognormal_validation.rds\n")

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
