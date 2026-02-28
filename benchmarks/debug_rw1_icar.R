# Debug script to check ICAR+RW1 (which passed) for sum-to-zero constraints
library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

N_OBS <- 100
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2
N_SITES <- 10
N_TIMES <- 5

cat("=== Debug RW1+ICAR (passed validation) ===\n\n")

# Same setup as AR1 test
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)
time_factor <- factor(rep(1:N_TIMES, length.out = N_OBS))

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

edges <- which(adj_mat == 1 & upper.tri(adj_mat), arr.ind = TRUE)
n_edges <- nrow(edges)
edge1 <- edges[, 1]
edge2 <- edges[, 2]
n_neighbors <- rowSums(adj_mat)

eta_num <- 2 + 0.3 * x
eta_denom <- 4 + 0.2 * x
df <- data.frame(
  y = rnbinom(N_OBS, mu = exp(eta_num), size = 5),
  denom = rnbinom(N_OBS, mu = exp(eta_denom), size = 10),
  x = x, site = site, spatial_site = spatial_site,
  time = as.integer(time_factor), time_factor = time_factor
)
df$denom[df$denom == 0] <- 1

# Fit numdenom with RW1
cat("Fitting numdenom (RW1)...\n")
fit_nd <- ratiod(
  y | denom ~ x + (1|site), data = df, family = ratiod_negbin_negbin(),
  spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
  temporal = temporal_rw1("time_factor"),
  iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE
)
draws_nd <- as.matrix(fit_nd$draws)

# Fit Stan with RW1
cat("Fitting Stan (RW1)...\n")
stan_model <- cmdstan_model("benchmarks/stan/joint_nb_icar_rw1.stan")
stan_data <- list(
  N = N_OBS,
  y_num = df$y, y_denom = df$denom, p = 2, X = cbind(1, df$x),
  n_groups = N_SITES, group_idx = as.integer(df$site),
  J = N_SITES, spatial_idx = as.integer(df$spatial_site),
  n_neighbors = as.array(n_neighbors), n_edges = n_edges,
  edge1 = as.array(edge1), edge2 = as.array(edge2),
  T = N_TIMES, time_idx = df$time
)
fit_stan <- stan_model$sample(
  data = stan_data,
  iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
  chains = N_CHAINS, parallel_chains = N_CHAINS,
  refresh = 0, show_messages = FALSE
)
draws_stan <- fit_stan$draws(format = "df")

# Check sum-to-zero
cat("\n=== Sum-to-zero checks ===\n")
nd_spatial_sum <- rowMeans(sapply(1:N_SITES, function(i) draws_nd[, paste0("phi_spatial[", i, "]")]))
stan_spatial_sum <- rowMeans(sapply(1:N_SITES, function(i) draws_stan[[paste0("phi_spatial[", i, "]")]]))
cat(sprintf("Mean of phi_spatial sum: nd=%.4f (SD=%.4f), stan=%.4f (SD=%.4f)\n",
            mean(nd_spatial_sum), sd(nd_spatial_sum), mean(stan_spatial_sum), sd(stan_spatial_sum)))

nd_temporal_sum <- rowMeans(sapply(1:N_TIMES, function(i) draws_nd[, paste0("temporal[", i, "]")]))
stan_temporal_sum <- rowMeans(sapply(1:N_TIMES, function(i) draws_stan[[paste0("phi_temporal[", i, "]")]]))
cat(sprintf("Mean of phi_temporal sum: nd=%.4f (SD=%.4f), stan=%.4f (SD=%.4f)\n",
            mean(nd_temporal_sum), sd(nd_temporal_sum), mean(stan_temporal_sum), sd(stan_temporal_sum)))

# Compare beta_num[2]
nd_mean <- mean(draws_nd[, "beta_num[2]"])
stan_mean <- mean(draws_stan$`beta_num[2]`)
nd_sd <- sd(draws_nd[, "beta_num[2]"])
stan_sd <- sd(draws_stan$`beta_num[2]`)
se <- sqrt(nd_sd^2 / nrow(draws_nd) + stan_sd^2 / nrow(draws_stan))
diff <- abs(nd_mean - stan_mean)
cat(sprintf("\nbeta_num[2]: nd=%.4f (%.4f), stan=%.4f (%.4f), diff=%.4f (%.2f SE)\n",
            nd_mean, nd_sd, stan_mean, stan_sd, diff, diff/se))

# Check effective intercept
nd_re_mean <- rowMeans(sapply(1:N_SITES, function(i) draws_nd[, paste0("re[", i, "]")]))
stan_re_mean <- rowMeans(sapply(1:N_SITES, function(i) draws_stan[[paste0("re[", i, "]")]]))

nd_eff_int <- draws_nd[, "beta_num[1]"] + nd_re_mean + nd_spatial_sum + nd_temporal_sum
stan_eff_int <- draws_stan$`beta_num[1]` + stan_re_mean + stan_spatial_sum + stan_temporal_sum
cat(sprintf("\nEffective intercept: nd=%.4f (%.4f), stan=%.4f (%.4f)\n",
            mean(nd_eff_int), sd(nd_eff_int), mean(stan_eff_int), sd(stan_eff_int)))
