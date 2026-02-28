# ICAR + ZI Validation - Rows 24, 54
# Validates poisson_gamma and negbin_negbin with ICAR spatial + Zero-Inflation

library(numdenom)
library(cmdstanr)
library(posterior)
library(Matrix)

set.seed(42)

# Standard benchmark parameters
N_OBS <- 200
N_ITER <- 1500
N_WARMUP <- 750
N_CHAINS <- 2
N_SITES <- 25

cat("=======================================================\n")
cat("ICAR + ZI Validation: Rows 24, 54\n")
cat("=======================================================\n\n")

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
    nd_mean = nd_mean, nd_sd = nd_sd,
    stan_mean = stan_mean, stan_sd = stan_sd,
    diff = diff, ratio = ratio, pass = pass
  )
}

print_result <- function(result) {
  cat(sprintf("  %s: nd=%.4f (SD=%.4f), Stan=%.4f (SD=%.4f), diff=%.4f (%.2f SE) => %s\n",
              result$param, result$nd_mean, result$nd_sd,
              result$stan_mean, result$stan_sd,
              result$diff, result$ratio, if(result$pass) "PASS" else "FAIL"))
}

# Create spatial adjacency structure (grid)
n_side <- 5  # 5x5 grid = 25 sites

# Build adjacency matrix (for numdenom)
adj_matrix <- matrix(0, N_SITES, N_SITES)
for (i in 1:N_SITES) {
  row_i <- (i - 1) %/% n_side + 1
  col_i <- (i - 1) %% n_side + 1

  if (row_i > 1) adj_matrix[i, i - n_side] <- 1  # above
  if (row_i < n_side) adj_matrix[i, i + n_side] <- 1  # below
  if (col_i > 1) adj_matrix[i, i - 1] <- 1  # left
  if (col_i < n_side) adj_matrix[i, i + 1] <- 1  # right
}

# Also create list form (for Stan edge list)
adj_list <- list()
for (i in 1:N_SITES) {
  adj_list[[i]] <- which(adj_matrix[i, ] == 1)
}

n_neighbors <- rowSums(adj_matrix)

# Build edge list for Stan
edges <- data.frame(from = integer(0), to = integer(0))
for (i in 1:N_SITES) {
  for (j in adj_list[[i]]) {
    if (i < j) edges <- rbind(edges, data.frame(from = i, to = j))
  }
}
n_edges <- nrow(edges)

# Generate shared spatial effects (ICAR)
Q <- matrix(0, N_SITES, N_SITES)
for (i in 1:N_SITES) {
  Q[i, i] <- n_neighbors[i]
  for (j in adj_list[[i]]) {
    Q[i, j] <- -1
  }
}
Q_reg <- Q + diag(0.001, N_SITES)  # Regularize
L <- chol(Q_reg)
spatial_effects <- backsolve(L, rnorm(N_SITES)) * 0.5
spatial_effects <- spatial_effects - mean(spatial_effects)  # Center

# Generate data
x <- rnorm(N_OBS)
site <- sample(1:N_SITES, N_OBS, replace = TRUE)
spatial_by_obs <- spatial_effects[site]

# True parameters
true_beta_num <- c(1.0, 0.3)
true_beta_denom <- c(0.8, 0.2)
true_zi_prob <- 0.2  # 20% structural zeros

eta_num <- true_beta_num[1] + true_beta_num[2] * x + spatial_by_obs
eta_denom <- true_beta_denom[1] + true_beta_denom[2] * x + spatial_by_obs

mu_num <- exp(eta_num)
mu_denom <- exp(eta_denom)

# =============================================================================
# Row 24: poisson_gamma + ICAR + ZI
# =============================================================================
cat("========== Row 24: poisson_gamma + ICAR + ZI ==========\n")

# Generate ZI-Poisson numerator and Gamma denominator
zi_indicator <- rbinom(N_OBS, 1, true_zi_prob)
y_num_pg <- ifelse(zi_indicator == 1, 0L, rpois(N_OBS, mu_num))
y_denom_pg <- rgamma(N_OBS, shape = 5, rate = 5 / mu_denom)

df_pg <- data.frame(
  y = y_num_pg,
  denom = y_denom_pg,
  x = x,
  site = as.factor(site)
)

cat(sprintf("Data: N=%d, sites=%d, zero rate=%.1f%%\n",
            N_OBS, N_SITES, 100 * mean(y_num_pg == 0)))

cat("Fitting numdenom... ")
t_nd_pg <- system.time({
  fit_nd_pg <- ratiod(
    y | denom ~ x, data = df_pg,
    family = ratiod_poisson_gamma(),
    spatial = spatial_car(adj = adj_matrix, group_var = "site"),
    zi = zi_poisson(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_nd_pg))

# Compile and run Stan model
cat("Compiling Stan model... ")
stan_model_pg <- cmdstan_model("benchmarks/stan/joint_pg_icar_zi.stan")
cat("done\n")

stan_data_pg <- list(
  N = N_OBS,
  y_num = y_num_pg,
  y_denom = y_denom_pg,
  p = 2,
  X = cbind(1, x),
  n_groups = N_SITES,
  group_idx = site,
  J = N_SITES,
  spatial_idx = site,
  n_neighbors = n_neighbors,
  n_edges = n_edges,
  edge1 = edges$from,
  edge2 = edges$to
)

cat("Fitting Stan model... ")
t_stan_pg <- system.time({
  fit_stan_pg <- stan_model_pg$sample(
    data = stan_data_pg,
    iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
    chains = N_CHAINS, parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE, adapt_delta = 0.95
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_stan_pg))

# Extract and compare draws
draws_nd_pg <- as.matrix(fit_nd_pg$draws)
draws_stan_pg <- fit_stan_pg$draws(format = "df")

results_pg <- list()

cat("\nComparing posteriors:\n")

# Beta num
results_pg$beta_num1 <- compare_posteriors(
  draws_nd_pg[, "beta_num[1]"], draws_stan_pg$`beta_num[1]`, "beta_num[1]"
)
print_result(results_pg$beta_num1)

results_pg$beta_num2 <- compare_posteriors(
  draws_nd_pg[, "beta_num[2]"], draws_stan_pg$`beta_num[2]`, "beta_num[2]"
)
print_result(results_pg$beta_num2)

# Beta denom
results_pg$beta_denom1 <- compare_posteriors(
  draws_nd_pg[, "beta_denom[1]"], draws_stan_pg$`beta_denom[1]`, "beta_denom[1]"
)
print_result(results_pg$beta_denom1)

results_pg$beta_denom2 <- compare_posteriors(
  draws_nd_pg[, "beta_denom[2]"], draws_stan_pg$`beta_denom[2]`, "beta_denom[2]"
)
print_result(results_pg$beta_denom2)

# Check for ZI parameter
zi_col <- grep("^logit_zi", colnames(draws_nd_pg), value = TRUE)[1]
if (!is.na(zi_col)) {
  results_pg$logit_zi <- compare_posteriors(
    draws_nd_pg[, zi_col], draws_stan_pg$logit_zi, "logit_zi"
  )
  print_result(results_pg$logit_zi)
}

# Count pass/fail
n_pass_pg <- sum(sapply(results_pg, function(r) r$pass))
n_total_pg <- length(results_pg)

cat(sprintf("\nRow 24 Result: %d/%d PASS (%.1f%%)\n", n_pass_pg, n_total_pg, 100 * n_pass_pg / n_total_pg))
cat(sprintf("Timing: numdenom=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd_pg, t_stan_pg, t_stan_pg / t_nd_pg))

# Check diagnostics
diag_pg <- fit_stan_pg$diagnostic_summary()
div_pct_pg <- 100 * sum(diag_pg$num_divergent) / (N_CHAINS * (N_ITER - N_WARMUP))
cat(sprintf("Stan divergences: %.1f%%\n", div_pct_pg))

# =============================================================================
# Row 54: negbin_negbin + ICAR + ZI
# =============================================================================
cat("\n========== Row 54: negbin_negbin + ICAR + ZI ==========\n")

# Generate ZI-NegBin numerator and NegBin denominator
y_num_nb <- ifelse(zi_indicator == 1, 0L, rnbinom(N_OBS, mu = mu_num, size = 5))
y_denom_nb <- rnbinom(N_OBS, mu = mu_denom, size = 5)
# Ensure denom > 0 (NegBin can produce 0)
y_denom_nb <- pmax(y_denom_nb, 1L)

df_nb <- data.frame(
  y = y_num_nb,
  denom = y_denom_nb,
  x = x,
  site = as.factor(site)
)

cat(sprintf("Data: N=%d, sites=%d, zero rate=%.1f%%\n",
            N_OBS, N_SITES, 100 * mean(y_num_nb == 0)))

cat("Fitting numdenom... ")
t_nd_nb <- system.time({
  fit_nd_nb <- ratiod(
    y | denom ~ x, data = df_nb,
    family = ratiod_negbin_negbin(),
    spatial = spatial_car(adj = adj_matrix, group_var = "site"),
    zi = zi_negbin(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_nd_nb))

# Compile and run Stan model
cat("Compiling Stan model... ")
stan_model_nb <- cmdstan_model("benchmarks/stan/joint_nb_icar_zi.stan")
cat("done\n")

stan_data_nb <- list(
  N = N_OBS,
  y_num = y_num_nb,
  y_denom = y_denom_nb,
  p = 2,
  X = cbind(1, x),
  n_groups = N_SITES,
  group_idx = site,
  J = N_SITES,
  spatial_idx = site,
  n_neighbors = n_neighbors,
  n_edges = n_edges,
  edge1 = edges$from,
  edge2 = edges$to
)

cat("Fitting Stan model... ")
t_stan_nb <- system.time({
  fit_stan_nb <- stan_model_nb$sample(
    data = stan_data_nb,
    iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
    chains = N_CHAINS, parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE, adapt_delta = 0.95
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_stan_nb))

# Extract and compare draws
draws_nd_nb <- as.matrix(fit_nd_nb$draws)
draws_stan_nb <- fit_stan_nb$draws(format = "df")

results_nb <- list()

cat("\nComparing posteriors:\n")

# Beta num
results_nb$beta_num1 <- compare_posteriors(
  draws_nd_nb[, "beta_num[1]"], draws_stan_nb$`beta_num[1]`, "beta_num[1]"
)
print_result(results_nb$beta_num1)

results_nb$beta_num2 <- compare_posteriors(
  draws_nd_nb[, "beta_num[2]"], draws_stan_nb$`beta_num[2]`, "beta_num[2]"
)
print_result(results_nb$beta_num2)

# Beta denom
results_nb$beta_denom1 <- compare_posteriors(
  draws_nd_nb[, "beta_denom[1]"], draws_stan_nb$`beta_denom[1]`, "beta_denom[1]"
)
print_result(results_nb$beta_denom1)

results_nb$beta_denom2 <- compare_posteriors(
  draws_nd_nb[, "beta_denom[2]"], draws_stan_nb$`beta_denom[2]`, "beta_denom[2]"
)
print_result(results_nb$beta_denom2)

# Dispersion parameters
phi_num_col <- grep("^(phi_num|log_phi_num)", colnames(draws_nd_nb), value = TRUE)[1]
if (!is.na(phi_num_col)) {
  if (grepl("^log_", phi_num_col)) {
    nd_phi_num <- exp(draws_nd_nb[, phi_num_col])
  } else {
    nd_phi_num <- draws_nd_nb[, phi_num_col]
  }
  results_nb$phi_num <- compare_posteriors(
    nd_phi_num, draws_stan_nb$phi_num, "phi_num"
  )
  print_result(results_nb$phi_num)
}

# Check for ZI parameter
zi_col_nb <- grep("^logit_zi", colnames(draws_nd_nb), value = TRUE)[1]
if (!is.na(zi_col_nb)) {
  results_nb$logit_zi <- compare_posteriors(
    draws_nd_nb[, zi_col_nb], draws_stan_nb$logit_zi, "logit_zi"
  )
  print_result(results_nb$logit_zi)
}

# Count pass/fail
n_pass_nb <- sum(sapply(results_nb, function(r) r$pass))
n_total_nb <- length(results_nb)

cat(sprintf("\nRow 54 Result: %d/%d PASS (%.1f%%)\n", n_pass_nb, n_total_nb, 100 * n_pass_nb / n_total_nb))
cat(sprintf("Timing: numdenom=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd_nb, t_stan_nb, t_stan_nb / t_nd_nb))

# Check diagnostics
diag_nb <- fit_stan_nb$diagnostic_summary()
div_pct_nb <- 100 * sum(diag_nb$num_divergent) / (N_CHAINS * (N_ITER - N_WARMUP))
cat(sprintf("Stan divergences: %.1f%%\n", div_pct_nb))

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - ICAR + ZI Validation\n")
cat("=======================================================\n\n")

all_pass_pg <- all(sapply(results_pg, function(r) r$pass))
all_pass_nb <- all(sapply(results_nb, function(r) r$pass))

cat(sprintf("Row 24 (pg+ICAR+ZI): %s (%d/%d params within 2 SE)\n",
            if(all_pass_pg) "PASS" else "PARTIAL", n_pass_pg, n_total_pg))
cat(sprintf("Row 54 (nb+ICAR+ZI): %s (%d/%d params within 2 SE)\n",
            if(all_pass_nb) "PASS" else "PARTIAL", n_pass_nb, n_total_nb))

cat(sprintf("\nTiming summary:\n"))
cat(sprintf("  Row 24: numdenom=%.1fs, Stan=%.1fs\n", t_nd_pg, t_stan_pg))
cat(sprintf("  Row 54: numdenom=%.1fs, Stan=%.1fs\n", t_nd_nb, t_stan_nb))

# Save results
saveRDS(list(
  pg = list(results = results_pg, time_nd = t_nd_pg, time_stan = t_stan_pg),
  nb = list(results = results_nb, time_nd = t_nd_nb, time_stan = t_stan_nb)
), "benchmarks/results_icar_zi.rds")

cat("\nResults saved to benchmarks/results_icar_zi.rds\n")

if (all_pass_pg) {
  cat("\nUPDATE gradient_methods.md:\n")
  cat("  Row 24: Change 'Stan model ready' to '\\u2713Stan (joint)'\n")
}
if (all_pass_nb) {
  cat("  Row 54: Change 'Stan model ready' to '\\u2713Stan (joint)'\n")
}
