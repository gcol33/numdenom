# Debug GP centering - compare numdenom vs Stan GP effect means

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

N_OBS <- 200
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2
N_TIMES <- 10

# Setup data
x <- rnorm(N_OBS)
time <- rep(1:N_TIMES, length.out = N_OBS)
time_scaled <- (time - mean(time)) / sd(time)
time_unique_scaled <- (1:N_TIMES - mean(1:N_TIMES)) / sd(1:N_TIMES)

# True GP effect (for data generation)
true_sigma_gp <- 0.5
true_phi_gp <- 2.0
dist_mat <- as.matrix(dist(1:N_TIMES))
K <- true_sigma_gp^2 * exp(-dist_mat / true_phi_gp)
gp_effects_true <- MASS::mvrnorm(1, rep(0, N_TIMES), K + diag(1e-6, N_TIMES))
gp_by_obs <- gp_effects_true[time]

# Linear predictors
eta_num <- 2 + 0.3 * x + gp_by_obs
eta_denom <- 4 + 0.2 * x + gp_by_obs

# Generate negbin_negbin data
df <- data.frame(
  y = rnbinom(N_OBS, mu = exp(eta_num), size = 5),
  denom = rnbinom(N_OBS, mu = exp(eta_denom), size = 10),
  x = x, time = time
)
df$denom[df$denom == 0] <- 1

cat("True GP effects:\n")
print(gp_effects_true)
cat("\nMean of true GP effects:", mean(gp_effects_true), "\n")

cat("\nFitting numdenom...\n")
fit_nd <- ratiod(
  y | denom ~ x, data = df,
  family = ratiod_negbin_negbin(),
  temporal = temporal_gp(time_var = "time", cov = "exponential"),
  iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
  verbose = FALSE
)

cat("Fitting Stan...\n")
stan_model <- cmdstan_model("benchmarks/stan/temporal_gp_nb_joint.stan")

stan_data <- list(
  N = N_OBS,
  T = N_TIMES,
  y_num = df$y,
  y_denom = df$denom,
  x = df$x,
  time_idx = df$time,
  time_values = time_unique_scaled,
  sigma2_prior_U = 1.0,
  sigma2_prior_alpha = 0.01,
  phi_prior_lower = 0.01,
  phi_prior_upper = 10.0
)

fit_stan <- stan_model$sample(
  data = stan_data,
  iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
  chains = N_CHAINS, parallel_chains = N_CHAINS,
  refresh = 0, show_messages = FALSE, adapt_delta = 0.95
)

# Extract draws
draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

# Get GP effects
nd_names <- colnames(draws_nd)
gp_cols <- grep("^temporal_gp\\[", nd_names, value = TRUE)

cat("\n===== GP Effect Analysis =====\n\n")

# Compute mean of GP effects for each draw
nd_gp_matrix <- draws_nd[, gp_cols]
stan_gp_matrix <- as.matrix(draws_stan[, paste0("gp_effects[", 1:10, "]")])

nd_gp_rowmeans <- rowMeans(nd_gp_matrix)
stan_gp_rowmeans <- rowMeans(stan_gp_matrix)

cat("Mean of GP effects across time points (per draw, then averaged):\n")
cat("  numdenom:", mean(nd_gp_rowmeans), "(SD =", sd(nd_gp_rowmeans), ")\n")
cat("  Stan:    ", mean(stan_gp_rowmeans), "(SD =", sd(stan_gp_rowmeans), ")\n")

cat("\nPer time-point GP effect means:\n")
for (i in 1:10) {
  nd_gp <- draws_nd[, paste0("temporal_gp[", i, "]")]
  stan_gp <- draws_stan[[paste0("gp_effects[", i, "]")]]
  cat(sprintf("  Time %d: nd=%.4f (%.4f), stan=%.4f (%.4f)\n",
              i, mean(nd_gp), sd(nd_gp), mean(stan_gp), sd(stan_gp)))
}

cat("\n===== Intercepts + GP effect mean =====\n\n")
cat("If intercepts differ by GP mean, sum should match:\n")

nd_int_num <- mean(draws_nd[, "beta_num[1]"])
nd_int_denom <- mean(draws_nd[, "beta_denom[1]"])
stan_int_num <- mean(draws_stan$beta_num_0)
stan_int_denom <- mean(draws_stan$beta_denom_0)

cat(sprintf("  beta_num[1] + mean(GP):   nd=%.4f, stan=%.4f\n",
            nd_int_num + mean(nd_gp_rowmeans),
            stan_int_num + mean(stan_gp_rowmeans)))
cat(sprintf("  beta_denom[1] + mean(GP): nd=%.4f, stan=%.4f\n",
            nd_int_denom + mean(nd_gp_rowmeans),
            stan_int_denom + mean(stan_gp_rowmeans)))
