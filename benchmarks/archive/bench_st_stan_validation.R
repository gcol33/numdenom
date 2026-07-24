# Validation of numdenom spatiotemporal models against custom joint Stan models
# Rows validated:
#   - Row 28: poisson_gamma + RE + ICAR + RW1 + ST Type I
#   - Row 29: poisson_gamma + RE + ICAR + RW1 + ST Type IV
#   - Row 58: negbin_negbin + RE + ICAR + RW1 + ST Type I
#   - Row 59: negbin_negbin + RE + ICAR + RW1 + ST Type IV
#   - Row 90: binomial + RE + ICAR + RW1 + ST Type I
#   - Row 91: binomial + RE + ICAR + RW1 + ST Type IV
#
# Stan models: stan/joint_{pg,nb,binom}_st_{i,iv}.stan
# Pattern follows bench_joint_validation_batch2.R

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

# Standard benchmark parameters
N_SITES <- 15
N_TIMES <- 10
N_OBS <- N_SITES * N_TIMES  # 150
N_ITER <- 1000
N_WARMUP <- 500
N_CHAINS <- 2

cat("=======================================================\n")
cat("Joint Model Validation: Spatiotemporal (Knorr-Held)\n")
cat("Rows 28-29, 58-59, 90-91\n")
cat("=======================================================\n")
cat(sprintf("N=%d (%d sites x %d times), iter=%d, warmup=%d, chains=%d\n\n",
            N_OBS, N_SITES, N_TIMES, N_ITER, N_WARMUP, N_CHAINS))

# ---- Helpers ----

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
    nd_mean = nd_mean, nd_sd = nd_sd,
    stan_mean = stan_mean, stan_sd = stan_sd,
    diff = diff, ratio = ratio,
    pass = ratio < threshold_se
  )
}

print_result <- function(result) {
  cat(sprintf("\n%s:\n", result$param))
  cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", result$nd_mean, result$nd_sd))
  cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", result$stan_mean, result$stan_sd))
  cat(sprintf("  Diff: %.4f (%.2f SE) => %s\n",
              result$diff, result$ratio, if(result$pass) "PASS" else "FAIL"))
}

# ---- Build adjacency (chain graph for 15 sites) ----

create_grid_adjacency <- function(n_units) {
  n_side <- ceiling(sqrt(n_units))
  grid <- expand.grid(x = 1:n_side, y = 1:n_side)[1:n_units, ]

  edges <- list()
  for (i in 1:(n_units - 1)) {
    for (j in (i + 1):n_units) {
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

# ---- Shared data generation ----

# Observation structure: balanced panel (each site observed at each time)
site <- factor(rep(1:N_SITES, each = N_TIMES))
time_idx <- rep(1:N_TIMES, N_SITES)
x <- rnorm(N_OBS)

# True parameters
true_slope <- 0.3
true_sigma_re <- 0.3
true_sigma_st <- 0.15

# Shared random effects
re_true <- rnorm(N_SITES, 0, true_sigma_re)

# Shared ICAR spatial effects
phi_spatial <- cumsum(rnorm(N_SITES, 0, 0.3))
phi_spatial <- phi_spatial - mean(phi_spatial)

# Shared RW1 temporal effects
phi_temporal <- cumsum(rnorm(N_TIMES, 0, 0.2))
phi_temporal <- phi_temporal - mean(phi_temporal)

# Generate ST interaction Type I (IID)
st_effects_I <- matrix(rnorm(N_SITES * N_TIMES, 0, true_sigma_st), N_SITES, N_TIMES)
st_effects_I <- st_effects_I - mean(st_effects_I)

# Generate ST interaction Type IV (Kronecker: ICAR x RW1)
phi_s_kron <- cumsum(rnorm(N_SITES, 0, 1))
phi_s_kron <- phi_s_kron - mean(phi_s_kron)
gamma_t_kron <- cumsum(rnorm(N_TIMES, 0, 1))
gamma_t_kron <- gamma_t_kron - mean(gamma_t_kron)
st_effects_IV <- outer(phi_s_kron, gamma_t_kron) * true_sigma_st
st_effects_IV <- st_effects_IV - mean(st_effects_IV)

# Stan spatial data (shared across all Stan models)
stan_spatial <- list(
  J = N_SITES,
  n_neighbors = spatial_info$n_neighbors,
  n_edges = spatial_info$n_edges,
  edge1 = spatial_info$edge1,
  edge2 = spatial_info$edge2
)

results <- list()

# =============================================================================
# Row 28: poisson_gamma + RE + ICAR + RW1 + ST Type I
# =============================================================================
cat("\n========== Row 28: poisson_gamma + ST-I ==========\n")

# Shared effects enter BOTH num and denom processes (matching Stan model structure)
shared_I <- re_true[as.integer(site)] +
  phi_spatial[as.integer(site)] +
  phi_temporal[time_idx] +
  st_effects_I[cbind(as.integer(site), time_idx)]

eta_num_28 <- 1.0 + true_slope * x + shared_I
eta_denom_28 <- 0.5 + 0.1 * x + shared_I

count_28 <- rpois(N_OBS, exp(eta_num_28))
effort_28 <- rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_denom_28))
effort_28[effort_28 < 0.01] <- 0.01

df_28 <- data.frame(
  count = count_28, effort = effort_28, x = x,
  site = site, time = factor(time_idx)
)

# Fit numdenom
cat("Fitting numdenom... ")
t_nd_28 <- system.time({
  fit_nd_28 <- tryCatch({
    tratio(count | effort ~ x + (1|site), data = df_28,
           family = ratiod_poisson_gamma(),
           spatiotemporal = spatiotemporal(
             spatial = spatial_car(spatial_info$adj_mat, group_var = "site"),
             temporal = temporal_rw1(time_var = "time"),
             type = "I"
           ),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd_28))

# Compile and fit Stan
cat("Compiling Stan model... ")
stan_mod_pg_i <- cmdstan_model("stan/joint_pg_st_i.stan")
cat("done\n")

stan_data_28 <- c(list(
  N = N_OBS,
  y_num = as.integer(df_28$count),
  y_denom = df_28$effort,
  p = 2,
  X = cbind(1, df_28$x),
  n_groups = N_SITES,
  group_idx = as.integer(df_28$site),
  spatial_idx = as.integer(df_28$site),
  T = N_TIMES,
  time_idx = as.integer(df_28$time)
), stan_spatial)

cat("Fitting Stan model... ")
t_stan_28 <- system.time({
  fit_stan_28 <- stan_mod_pg_i$sample(
    data = stan_data_28,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan_28))

if (!is.null(fit_nd_28)) {
  draws_nd <- as.matrix(fit_nd_28$draws)
  draws_stan <- fit_stan_28$draws(format = "df")
  results$row_28 <- compare_posteriors(
    draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`,
    "Row 28 beta_num[2] (PG+ST-I)"
  )
  print_result(results$row_28)
  cat(sprintf("  Speedup: %.1fx\n", t_stan_28 / t_nd_28))
  results$row_28$t_nd <- t_nd_28
  results$row_28$t_stan <- t_stan_28
}

# =============================================================================
# Row 29: poisson_gamma + RE + ICAR + RW1 + ST Type IV
# =============================================================================
cat("\n========== Row 29: poisson_gamma + ST-IV ==========\n")

shared_IV <- re_true[as.integer(site)] +
  phi_spatial[as.integer(site)] +
  phi_temporal[time_idx] +
  st_effects_IV[cbind(as.integer(site), time_idx)]

eta_num_29 <- 1.0 + true_slope * x + shared_IV
eta_denom_29 <- 0.5 + 0.1 * x + shared_IV

count_29 <- rpois(N_OBS, exp(eta_num_29))
effort_29 <- rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_denom_29))
effort_29[effort_29 < 0.01] <- 0.01

df_29 <- data.frame(
  count = count_29, effort = effort_29, x = x,
  site = site, time = factor(time_idx)
)

cat("Fitting numdenom... ")
t_nd_29 <- system.time({
  fit_nd_29 <- tryCatch({
    tratio(count | effort ~ x + (1|site), data = df_29,
           family = ratiod_poisson_gamma(),
           spatiotemporal = spatiotemporal(
             spatial = spatial_car(spatial_info$adj_mat, group_var = "site"),
             temporal = temporal_rw1(time_var = "time"),
             type = "IV"
           ),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd_29))

cat("Compiling Stan model... ")
stan_mod_pg_iv <- cmdstan_model("stan/joint_pg_st_iv.stan")
cat("done\n")

stan_data_29 <- c(list(
  N = N_OBS,
  y_num = as.integer(df_29$count),
  y_denom = df_29$effort,
  p = 2,
  X = cbind(1, df_29$x),
  n_groups = N_SITES,
  group_idx = as.integer(df_29$site),
  spatial_idx = as.integer(df_29$site),
  T = N_TIMES,
  time_idx = as.integer(df_29$time)
), stan_spatial)

cat("Fitting Stan model... ")
t_stan_29 <- system.time({
  fit_stan_29 <- stan_mod_pg_iv$sample(
    data = stan_data_29,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan_29))

if (!is.null(fit_nd_29)) {
  draws_nd <- as.matrix(fit_nd_29$draws)
  draws_stan <- fit_stan_29$draws(format = "df")
  results$row_29 <- compare_posteriors(
    draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`,
    "Row 29 beta_num[2] (PG+ST-IV)"
  )
  print_result(results$row_29)
  cat(sprintf("  Speedup: %.1fx\n", t_stan_29 / t_nd_29))
  results$row_29$t_nd <- t_nd_29
  results$row_29$t_stan <- t_stan_29
}

# =============================================================================
# Row 58: negbin_negbin + RE + ICAR + RW1 + ST Type I
# =============================================================================
cat("\n========== Row 58: negbin_negbin + ST-I ==========\n")

# Full shared effects in both num and denom (matching Stan model)
eta_num_58 <- 1.0 + true_slope * x + shared_I
eta_denom_58 <- 0.5 + 0.1 * x + shared_I

num_58 <- rnbinom(N_OBS, size = 5, mu = exp(eta_num_58))
denom_58 <- rnbinom(N_OBS, size = 8, mu = exp(eta_denom_58))
denom_58[denom_58 == 0] <- 1L
df_58 <- data.frame(
  num = num_58, denom = denom_58, x = x,
  site = site, time = factor(time_idx)
)

cat("Fitting numdenom... ")
t_nd_58 <- system.time({
  fit_nd_58 <- tryCatch({
    tratio(num | denom ~ x + (1|site), data = df_58,
           family = ratiod_negbin_negbin(),
           spatiotemporal = spatiotemporal(
             spatial = spatial_car(spatial_info$adj_mat, group_var = "site"),
             temporal = temporal_rw1(time_var = "time"),
             type = "I"
           ),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd_58))

cat("Compiling Stan model... ")
stan_mod_nb_i <- cmdstan_model("stan/joint_nb_st_i.stan")
cat("done\n")

stan_data_58 <- c(list(
  N = N_OBS,
  y_num = as.integer(df_58$num),
  y_denom = as.integer(df_58$denom),
  p = 2,
  X = cbind(1, df_58$x),
  n_groups = N_SITES,
  group_idx = as.integer(df_58$site),
  spatial_idx = as.integer(df_58$site),
  T = N_TIMES,
  time_idx = as.integer(df_58$time)
), stan_spatial)

cat("Fitting Stan model... ")
t_stan_58 <- system.time({
  fit_stan_58 <- stan_mod_nb_i$sample(
    data = stan_data_58,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan_58))

if (!is.null(fit_nd_58)) {
  draws_nd <- as.matrix(fit_nd_58$draws)
  draws_stan <- fit_stan_58$draws(format = "df")
  results$row_58 <- compare_posteriors(
    draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`,
    "Row 58 beta_num[2] (NB+ST-I)"
  )
  print_result(results$row_58)
  cat(sprintf("  Speedup: %.1fx\n", t_stan_58 / t_nd_58))
  results$row_58$t_nd <- t_nd_58
  results$row_58$t_stan <- t_stan_58
}

# =============================================================================
# Row 59: negbin_negbin + RE + ICAR + RW1 + ST Type IV
# =============================================================================
cat("\n========== Row 59: negbin_negbin + ST-IV ==========\n")

# Full shared effects in both num and denom (matching Stan model)
eta_num_59 <- 1.0 + true_slope * x + shared_IV
eta_denom_59 <- 0.5 + 0.1 * x + shared_IV

num_59 <- rnbinom(N_OBS, size = 5, mu = exp(eta_num_59))
denom_59 <- rnbinom(N_OBS, size = 8, mu = exp(eta_denom_59))
denom_59[denom_59 == 0] <- 1L
df_59 <- data.frame(
  num = num_59, denom = denom_59, x = x,
  site = site, time = factor(time_idx)
)

cat("Fitting numdenom... ")
t_nd_59 <- system.time({
  fit_nd_59 <- tryCatch({
    tratio(num | denom ~ x + (1|site), data = df_59,
           family = ratiod_negbin_negbin(),
           spatiotemporal = spatiotemporal(
             spatial = spatial_car(spatial_info$adj_mat, group_var = "site"),
             temporal = temporal_rw1(time_var = "time"),
             type = "IV"
           ),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd_59))

cat("Compiling Stan model... ")
stan_mod_nb_iv <- cmdstan_model("stan/joint_nb_st_iv.stan")
cat("done\n")

stan_data_59 <- c(list(
  N = N_OBS,
  y_num = as.integer(df_59$num),
  y_denom = as.integer(df_59$denom),
  p = 2,
  X = cbind(1, df_59$x),
  n_groups = N_SITES,
  group_idx = as.integer(df_59$site),
  spatial_idx = as.integer(df_59$site),
  T = N_TIMES,
  time_idx = as.integer(df_59$time)
), stan_spatial)

cat("Fitting Stan model... ")
t_stan_59 <- system.time({
  fit_stan_59 <- stan_mod_nb_iv$sample(
    data = stan_data_59,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan_59))

if (!is.null(fit_nd_59)) {
  draws_nd <- as.matrix(fit_nd_59$draws)
  draws_stan <- fit_stan_59$draws(format = "df")
  results$row_59 <- compare_posteriors(
    draws_nd[, "beta_num[2]"], draws_stan$`beta_num[2]`,
    "Row 59 beta_num[2] (NB+ST-IV)"
  )
  print_result(results$row_59)
  cat(sprintf("  Speedup: %.1fx\n", t_stan_59 / t_nd_59))
  results$row_59$t_nd <- t_nd_59
  results$row_59$t_stan <- t_stan_59
}

# =============================================================================
# Row 90: binomial + RE + ICAR + RW1 + ST Type I
# =============================================================================
cat("\n========== Row 90: binomial + ST-I ==========\n")

eta_90 <- 1.0 + true_slope * x +
  re_true[as.integer(site)] +
  phi_spatial[as.integer(site)] +
  phi_temporal[time_idx] +
  st_effects_I[cbind(as.integer(site), time_idx)]
prob_90 <- plogis(eta_90)
trials_90 <- sample(20:50, N_OBS, replace = TRUE)
successes_90 <- rbinom(N_OBS, trials_90, prob_90)

df_90 <- data.frame(
  successes = successes_90, trials = trials_90, x = x,
  site = site, time = factor(time_idx)
)

cat("Fitting numdenom... ")
t_nd_90 <- system.time({
  fit_nd_90 <- tryCatch({
    tratio(successes | trials ~ x + (1|site), data = df_90,
           family = ratiod_binomial(),
           spatiotemporal = spatiotemporal(
             spatial = spatial_car(spatial_info$adj_mat, group_var = "site"),
             temporal = temporal_rw1(time_var = "time"),
             type = "I"
           ),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd_90))

cat("Compiling Stan model... ")
stan_mod_bin_i <- cmdstan_model("stan/joint_binom_st_i.stan")
cat("done\n")

stan_data_90 <- c(list(
  N = N_OBS,
  y = as.integer(df_90$successes),
  trials = as.integer(df_90$trials),
  p = 2,
  X = cbind(1, df_90$x),
  n_groups = N_SITES,
  group_idx = as.integer(df_90$site),
  spatial_idx = as.integer(df_90$site),
  T = N_TIMES,
  time_idx = as.integer(df_90$time)
), stan_spatial)

cat("Fitting Stan model... ")
t_stan_90 <- system.time({
  fit_stan_90 <- stan_mod_bin_i$sample(
    data = stan_data_90,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan_90))

if (!is.null(fit_nd_90)) {
  draws_nd <- as.matrix(fit_nd_90$draws)
  draws_stan <- fit_stan_90$draws(format = "df")
  # Binomial Stan model uses "beta" not "beta_num"
  results$row_90 <- compare_posteriors(
    draws_nd[, "beta_num[2]"], draws_stan$`beta[2]`,
    "Row 90 beta[2] (Bin+ST-I)"
  )
  print_result(results$row_90)
  cat(sprintf("  Speedup: %.1fx\n", t_stan_90 / t_nd_90))
  results$row_90$t_nd <- t_nd_90
  results$row_90$t_stan <- t_stan_90
}

# =============================================================================
# Row 91: binomial + RE + ICAR + RW1 + ST Type IV
# =============================================================================
cat("\n========== Row 91: binomial + ST-IV ==========\n")

eta_91 <- 1.0 + true_slope * x +
  re_true[as.integer(site)] +
  phi_spatial[as.integer(site)] +
  phi_temporal[time_idx] +
  st_effects_IV[cbind(as.integer(site), time_idx)]
prob_91 <- plogis(eta_91)
successes_91 <- rbinom(N_OBS, trials_90, prob_91)

df_91 <- data.frame(
  successes = successes_91, trials = trials_90, x = x,
  site = site, time = factor(time_idx)
)

cat("Fitting numdenom... ")
t_nd_91 <- system.time({
  fit_nd_91 <- tryCatch({
    tratio(successes | trials ~ x + (1|site), data = df_91,
           family = ratiod_binomial(),
           spatiotemporal = spatiotemporal(
             spatial = spatial_car(spatial_info$adj_mat, group_var = "site"),
             temporal = temporal_rw1(time_var = "time"),
             type = "IV"
           ),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE))
  }, error = function(e) { cat(sprintf("ERROR: %s\n", e$message)); NULL })
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd_91))

cat("Compiling Stan model... ")
stan_mod_bin_iv <- cmdstan_model("stan/joint_binom_st_iv.stan")
cat("done\n")

stan_data_91 <- c(list(
  N = N_OBS,
  y = as.integer(df_91$successes),
  trials = as.integer(df_91$trials),
  p = 2,
  X = cbind(1, df_91$x),
  n_groups = N_SITES,
  group_idx = as.integer(df_91$site),
  spatial_idx = as.integer(df_91$site),
  T = N_TIMES,
  time_idx = as.integer(df_91$time)
), stan_spatial)

cat("Fitting Stan model... ")
t_stan_91 <- system.time({
  fit_stan_91 <- stan_mod_bin_iv$sample(
    data = stan_data_91,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan_91))

if (!is.null(fit_nd_91)) {
  draws_nd <- as.matrix(fit_nd_91$draws)
  draws_stan <- fit_stan_91$draws(format = "df")
  # Binomial Stan model uses "beta" not "beta_num"
  results$row_91 <- compare_posteriors(
    draws_nd[, "beta_num[2]"], draws_stan$`beta[2]`,
    "Row 91 beta[2] (Bin+ST-IV)"
  )
  print_result(results$row_91)
  cat(sprintf("  Speedup: %.1fx\n", t_stan_91 / t_nd_91))
  results$row_91$t_nd <- t_nd_91
  results$row_91$t_stan <- t_stan_91
}

# =============================================================================
# Summary
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - Spatiotemporal Stan Validation\n")
cat("=======================================================\n")

row_names <- c("row_28", "row_29", "row_58", "row_59", "row_90", "row_91")
row_labels <- c(
  "28 (PG+ST-I)", "29 (PG+ST-IV)",
  "58 (NB+ST-I)", "59 (NB+ST-IV)",
  "90 (Bin+ST-I)", "91 (Bin+ST-IV)"
)

n_pass <- 0
n_total <- 0

for (i in seq_along(row_names)) {
  rn <- row_names[i]
  r <- results[[rn]]
  if (is.null(r)) {
    cat(sprintf("\nRow %s: ERROR (fit failed)\n", row_labels[i]))
  } else {
    n_total <- n_total + 1
    if (r$pass) n_pass <- n_pass + 1
    cat(sprintf("\nRow %s: %s (%.2f SE, nd=%.1fs, stan=%.1fs, speedup=%.1fx)\n",
                row_labels[i],
                if(r$pass) "PASS" else "FAIL",
                r$ratio,
                r$t_nd, r$t_stan, r$t_stan / r$t_nd))
  }
}

cat(sprintf("\nOverall: %d/%d parameters PASS (within 2 SE)\n", n_pass, n_total))

if (n_pass == n_total && n_total == 6) {
  cat("\n** All spatiotemporal posteriors match custom joint Stan models! **\n")
} else if (n_pass < n_total) {
  cat("\n!! Some posteriors differ - investigate !!\n")
}

# Save results
saveRDS(results, "results_st_stan_validation.rds")
cat("\nResults saved to benchmarks/results_st_stan_validation.rds\n")

# Print update instructions
cat("\n=======================================================\n")
cat("UPDATE gradient_methods.md:\n")
cat("=======================================================\n")
for (i in seq_along(row_names)) {
  rn <- row_names[i]
  r <- results[[rn]]
  if (!is.null(r) && r$pass) {
    cat(sprintf("  Row %s: ✓Stan (joint, %.1f SE)\n", row_labels[i], r$ratio))
  }
}
