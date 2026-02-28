# Debug script to understand AR1+ICAR failure
# Compare numdenom and Stan posteriors in detail

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

# Smaller test for debugging
N_OBS <- 100
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2
N_SITES <- 10
N_TIMES <- 5

cat("=== Debug AR1+ICAR (Row 50) ===\n")
cat(sprintf("N=%d, sites=%d, times=%d\n\n", N_OBS, N_SITES, N_TIMES))

# Setup
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

# Build edge list
edges <- which(adj_mat == 1 & upper.tri(adj_mat), arr.ind = TRUE)
n_edges <- nrow(edges)
edge1 <- edges[, 1]
edge2 <- edges[, 2]
n_neighbors <- rowSums(adj_mat)

# Generate data with known parameters
true_beta_num <- c(2, 0.3)
true_beta_denom <- c(4, 0.2)

eta_num <- true_beta_num[1] + true_beta_num[2] * x
eta_denom <- true_beta_denom[1] + true_beta_denom[2] * x

df <- data.frame(
  y = rnbinom(N_OBS, mu = exp(eta_num), size = 5),
  denom = rnbinom(N_OBS, mu = exp(eta_denom), size = 10),
  x = x,
  site = site,
  spatial_site = spatial_site,
  time = time,
  time_factor = time_factor
)
df$denom[df$denom == 0] <- 1

# Fit numdenom
cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x + (1|site),
    data = df,
    family = ratiod_negbin_negbin(),
    spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
    temporal = temporal_ar1("time_factor"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))

# Fit Stan
cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_nb_icar_ar1.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = df$y,
  y_denom = df$denom,
  p = 2,
  X = cbind(1, df$x),
  n_groups = N_SITES,
  group_idx = as.integer(df$site),
  J = N_SITES,
  spatial_idx = as.integer(df$spatial_site),
  n_neighbors = as.array(n_neighbors),
  n_edges = n_edges,
  edge1 = as.array(edge1),
  edge2 = as.array(edge2),
  T = N_TIMES,
  time_idx = df$time
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
cat(sprintf("%.1fs\n\n", t_stan))

# Compare all key parameters
draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

cat("=== Parameter Comparison ===\n\n")

params_to_compare <- list(
  list(nd = "beta_num[1]", stan = "beta_num[1]", name = "beta_num[1] (intercept)"),
  list(nd = "beta_num[2]", stan = "beta_num[2]", name = "beta_num[2] (slope)"),
  list(nd = "beta_denom[1]", stan = "beta_denom[1]", name = "beta_denom[1] (intercept)"),
  list(nd = "beta_denom[2]", stan = "beta_denom[2]", name = "beta_denom[2] (slope)")
)

for (p in params_to_compare) {
  nd_draws <- draws_nd[, p$nd]
  stan_draws <- draws_stan[[p$stan]]

  nd_mean <- mean(nd_draws)
  nd_sd <- sd(nd_draws)
  stan_mean <- mean(stan_draws)
  stan_sd <- sd(stan_draws)

  se_combined <- sqrt(nd_sd^2 / length(nd_draws) + stan_sd^2 / length(stan_draws))
  diff <- abs(nd_mean - stan_mean)
  ratio <- diff / se_combined

  cat(sprintf("%s:\n", p$name))
  cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", nd_mean, nd_sd))
  cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", stan_mean, stan_sd))
  cat(sprintf("  Diff: %.4f (%.2f SE) => %s\n\n", diff, ratio, if(ratio < 2) "PASS" else "FAIL"))
}

# Check temporal parameters
cat("=== Temporal Parameters ===\n\n")

# Check if numdenom has rho_ar1 or equivalent
nd_names <- colnames(draws_nd)
ar1_params_nd <- nd_names[grep("rho|ar1|temporal", nd_names, ignore.case = TRUE)]
cat("numdenom temporal params:", paste(ar1_params_nd, collapse = ", "), "\n")

# Stan has rho_ar1 and tau_temporal
if ("rho_ar1" %in% names(draws_stan)) {
  cat(sprintf("Stan rho_ar1: %.4f (SD=%.4f)\n", mean(draws_stan$rho_ar1), sd(draws_stan$rho_ar1)))
}
if ("tau_temporal" %in% names(draws_stan)) {
  cat(sprintf("Stan tau_temporal: %.4f (SD=%.4f)\n", mean(draws_stan$tau_temporal), sd(draws_stan$tau_temporal)))
}

# Check temporal effects
cat("\n=== Temporal Effects ===\n")
# Find temporal effect columns
nd_temporal <- nd_names[grep("^temporal\\[", nd_names)]
cat("numdenom temporal effects:", length(nd_temporal), "parameters\n")
if (length(nd_temporal) > 0) {
  cat("  First 5 names:", paste(head(nd_temporal, 5), collapse = ", "), "\n")
  for (i in 1:min(5, length(nd_temporal))) {
    cat(sprintf("  %s: %.4f (SD=%.4f)\n", nd_temporal[i], mean(draws_nd[, nd_temporal[i]]), sd(draws_nd[, nd_temporal[i]])))
  }
}

# Stan temporal effects
stan_temporal <- paste0("phi_temporal[", 1:N_TIMES, "]")
cat("\nStan temporal effects:", N_TIMES, "parameters\n")
for (i in 1:min(5, N_TIMES)) {
  if (stan_temporal[i] %in% names(draws_stan)) {
    cat(sprintf("  %s: %.4f (SD=%.4f)\n", stan_temporal[i], mean(draws_stan[[stan_temporal[i]]]), sd(draws_stan[[stan_temporal[i]]])))
  }
}

# Check spatial effects
cat("\n=== Spatial Effects ===\n")
nd_spatial <- nd_names[grep("^spatial\\[", nd_names)]
cat("numdenom spatial effects:", length(nd_spatial), "parameters\n")
if (length(nd_spatial) > 0) {
  cat("  First 5:", paste(head(nd_spatial, 5), collapse = ", "), "\n")
}

stan_spatial <- paste0("phi_spatial[", 1:N_SITES, "]")
cat("Stan spatial effects:", N_SITES, "parameters\n")

# Print summary
cat("\n=== Summary ===\n")
cat("Truth: beta_num[2] =", true_beta_num[2], ", beta_denom[2] =", true_beta_denom[2], "\n")
cat("numdenom: beta_num[2] =", mean(draws_nd[, "beta_num[2]"]), ", beta_denom[2] =", mean(draws_nd[, "beta_denom[2]"]), "\n")
cat("Stan: beta_num[2] =", mean(draws_stan$`beta_num[2]`), ", beta_denom[2] =", mean(draws_stan$`beta_denom[2]`), "\n")
