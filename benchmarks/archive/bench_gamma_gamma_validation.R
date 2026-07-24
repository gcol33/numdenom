# Validation of numdenom gamma_gamma family against custom joint Stan models
# Rows 93-97 in gradient_methods.md
#
# Row 93: gamma_gamma (no RE)
# Row 94: gamma_gamma + RE
# Row 95: gamma_gamma + RE + ICAR
# Row 96: gamma_gamma + RE + RW1
# Row 97: gamma_gamma + RE + ICAR + RW1

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

# Standard benchmark parameters - REDUCED for gamma_gamma (it's slow)
N_OBS <- 150
N_ITER <- 1500
N_WARMUP <- 750
N_CHAINS <- 2
N_SITES <- 15
N_TIMES <- 8

cat("=======================================================\n")
cat("Joint Model Validation: gamma_gamma Family (Rows 93-97)\n")
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

# Generate data for gamma family (must be positive)
eta_num <- 2 + 0.3 * x
eta_denom <- 3 + 0.2 * x

# Shape parameters for gamma
shape_num <- 5
shape_denom <- 8

results <- list()

# =============================================================================
# Row 93: gamma_gamma (no RE)
# =============================================================================
cat("\n========== Row 93: gamma_gamma (no RE) ==========\n")

df_gg_base <- data.frame(
  y = rgamma(N_OBS, shape = shape_num, rate = shape_num / exp(eta_num)),
  denom = rgamma(N_OBS, shape = shape_denom, rate = shape_denom / exp(eta_denom)),
  x = x
)
# Ensure positive values
df_gg_base$y[df_gg_base$y <= 0] <- 0.01
df_gg_base$denom[df_gg_base$denom <= 0] <- 0.01

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- tratio(
    y | denom ~ x,
    data = df_gg_base,
    family = ratiod_gamma_gamma(),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_gg_base.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = df_gg_base$y,
  y_denom = df_gg_base$denom,
  p = 2,
  X = cbind(1, df_gg_base$x)
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

results$row_93 <- list(
  beta_num_2 = compare_posteriors(draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`, "beta_num[2]"),
  beta_denom_2 = compare_posteriors(draws_nd[, "beta_denom[2]"], draws_stan$`beta_denom[2]`, "beta_denom[2]"),
  time_nd = t_nd,
  time_stan = t_stan
)
print_result(results$row_93$beta_num_2)
print_result(results$row_93$beta_denom_2)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan/t_nd))

# =============================================================================
# Row 94: gamma_gamma + RE
# =============================================================================
cat("\n========== Row 94: gamma_gamma + RE ==========\n")

df_gg_re <- data.frame(
  y = rgamma(N_OBS, shape = shape_num, rate = shape_num / exp(eta_num)),
  denom = rgamma(N_OBS, shape = shape_denom, rate = shape_denom / exp(eta_denom)),
  x = x,
  site = site
)
df_gg_re$y[df_gg_re$y <= 0] <- 0.01
df_gg_re$denom[df_gg_re$denom <= 0] <- 0.01

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- tratio(
    y | denom ~ x + (1|site),
    data = df_gg_re,
    family = ratiod_gamma_gamma(),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_gg_re.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = df_gg_re$y,
  y_denom = df_gg_re$denom,
  p = 2,
  X = cbind(1, df_gg_re$x),
  n_groups = N_SITES,
  group_idx = as.integer(df_gg_re$site)
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

results$row_94 <- list(
  beta_num_2 = compare_posteriors(draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`, "beta_num[2]"),
  beta_denom_2 = compare_posteriors(draws_nd[, "beta_denom[2]"], draws_stan$`beta_denom[2]`, "beta_denom[2]"),
  time_nd = t_nd,
  time_stan = t_stan
)
print_result(results$row_94$beta_num_2)
print_result(results$row_94$beta_denom_2)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan/t_nd))

# =============================================================================
# Row 95: gamma_gamma + RE + ICAR
# =============================================================================
cat("\n========== Row 95: gamma_gamma + RE + ICAR ==========\n")

df_gg_icar <- data.frame(
  y = rgamma(N_OBS, shape = shape_num, rate = shape_num / exp(eta_num)),
  denom = rgamma(N_OBS, shape = shape_denom, rate = shape_denom / exp(eta_denom)),
  x = x,
  site = site,
  spatial_site = spatial_site
)
df_gg_icar$y[df_gg_icar$y <= 0] <- 0.01
df_gg_icar$denom[df_gg_icar$denom <= 0] <- 0.01

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- tratio(
    y | denom ~ x + (1|site),
    data = df_gg_icar,
    family = ratiod_gamma_gamma(),
    spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_gg_icar.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = df_gg_icar$y,
  y_denom = df_gg_icar$denom,
  p = 2,
  X = cbind(1, df_gg_icar$x),
  n_groups = N_SITES,
  group_idx = as.integer(df_gg_icar$site),
  J = N_SITES,
  spatial_idx = as.integer(df_gg_icar$spatial_site),
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

results$row_95 <- list(
  beta_num_2 = compare_posteriors(draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`, "beta_num[2]"),
  beta_denom_2 = compare_posteriors(draws_nd[, "beta_denom[2]"], draws_stan$`beta_denom[2]`, "beta_denom[2]"),
  time_nd = t_nd,
  time_stan = t_stan
)
print_result(results$row_95$beta_num_2)
print_result(results$row_95$beta_denom_2)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan/t_nd))

# =============================================================================
# Row 96: gamma_gamma + RE + RW1
# =============================================================================
cat("\n========== Row 96: gamma_gamma + RE + RW1 ==========\n")

df_gg_rw1 <- data.frame(
  y = rgamma(N_OBS, shape = shape_num, rate = shape_num / exp(eta_num)),
  denom = rgamma(N_OBS, shape = shape_denom, rate = shape_denom / exp(eta_denom)),
  x = x,
  site = site,
  time = time,
  time_factor = time_factor
)
df_gg_rw1$y[df_gg_rw1$y <= 0] <- 0.01
df_gg_rw1$denom[df_gg_rw1$denom <= 0] <- 0.01

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- tratio(
    y | denom ~ x + (1|site),
    data = df_gg_rw1,
    family = ratiod_gamma_gamma(),
    temporal = temporal_rw1("time_factor"),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_gg_rw1.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = df_gg_rw1$y,
  y_denom = df_gg_rw1$denom,
  p = 2,
  X = cbind(1, df_gg_rw1$x),
  n_groups = N_SITES,
  group_idx = as.integer(df_gg_rw1$site),
  T = N_TIMES,
  time_idx = df_gg_rw1$time
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

results$row_96 <- list(
  beta_num_2 = compare_posteriors(draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`, "beta_num[2]"),
  beta_denom_2 = compare_posteriors(draws_nd[, "beta_denom[2]"], draws_stan$`beta_denom[2]`, "beta_denom[2]"),
  time_nd = t_nd,
  time_stan = t_stan
)
print_result(results$row_96$beta_num_2)
print_result(results$row_96$beta_denom_2)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan/t_nd))

# =============================================================================
# Row 97: gamma_gamma + RE + ICAR + RW1
# =============================================================================
cat("\n========== Row 97: gamma_gamma + RE + ICAR + RW1 ==========\n")

df_gg_icar_rw1 <- data.frame(
  y = rgamma(N_OBS, shape = shape_num, rate = shape_num / exp(eta_num)),
  denom = rgamma(N_OBS, shape = shape_denom, rate = shape_denom / exp(eta_denom)),
  x = x,
  site = site,
  spatial_site = spatial_site,
  time = time,
  time_factor = time_factor
)
df_gg_icar_rw1$y[df_gg_icar_rw1$y <= 0] <- 0.01
df_gg_icar_rw1$denom[df_gg_icar_rw1$denom <= 0] <- 0.01

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- tratio(
    y | denom ~ x + (1|site),
    data = df_gg_icar_rw1,
    family = ratiod_gamma_gamma(),
    spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
    temporal = temporal_rw1("time_factor"),
    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_gg_icar_rw1.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = df_gg_icar_rw1$y,
  y_denom = df_gg_icar_rw1$denom,
  p = 2,
  X = cbind(1, df_gg_icar_rw1$x),
  n_groups = N_SITES,
  group_idx = as.integer(df_gg_icar_rw1$site),
  J = N_SITES,
  spatial_idx = as.integer(df_gg_icar_rw1$spatial_site),
  n_neighbors = as.array(n_neighbors),
  n_edges = n_edges,
  edge1 = as.array(edge1),
  edge2 = as.array(edge2),
  T = N_TIMES,
  time_idx = df_gg_icar_rw1$time
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

results$row_97 <- list(
  beta_num_2 = compare_posteriors(draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`, "beta_num[2]"),
  beta_denom_2 = compare_posteriors(draws_nd[, "beta_denom[2]"], draws_stan$`beta_denom[2]`, "beta_denom[2]"),
  time_nd = t_nd,
  time_stan = t_stan
)
print_result(results$row_97$beta_num_2)
print_result(results$row_97$beta_denom_2)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan/t_nd))

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - gamma_gamma Family (Rows 93-97)\n")
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
saveRDS(results, "benchmarks/results_gamma_gamma_validation.rds")
cat("\nResults saved to benchmarks/results_gamma_gamma_validation.rds\n")

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
