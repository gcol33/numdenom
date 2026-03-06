# Debug GP predictions - compare predictions between numdenom and Stan
# The key insight: intercept + mean(GP) should match, so predictions should match

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

cat("Fitting numdenom...\n")
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

# Compute fitted values (linear predictor) for a few observations
cat("\n===== Linear Predictor Comparison =====\n\n")
cat("Testing on observations 1, 50, 100, 150 (different time points)\n\n")

test_obs <- c(1, 50, 100, 150)

for (obs in test_obs) {
  xi <- df$x[obs]
  ti <- df$time[obs]

  # numdenom: beta[1] + beta[2]*x + GP[time]
  nd_beta1 <- draws_nd[, "beta_num[1]"]
  nd_beta2 <- draws_nd[, "beta_num[2]"]
  nd_gp <- draws_nd[, paste0("temporal_gp[", ti, "]")]
  nd_eta <- nd_beta1 + nd_beta2 * xi + nd_gp

  # Stan: beta_0 + beta_1*x + gp_effects[time]
  stan_beta0 <- draws_stan$beta_num_0
  stan_beta1 <- draws_stan$beta_num_1
  stan_gp <- draws_stan[[paste0("gp_effects[", ti, "]")]]
  stan_eta <- stan_beta0 + stan_beta1 * xi + stan_gp

  se <- sqrt(sd(nd_eta)^2/length(nd_eta) + sd(stan_eta)^2/length(stan_eta))
  diff <- abs(mean(nd_eta) - mean(stan_eta))
  ratio <- diff / se

  cat(sprintf("Obs %d (time=%d, x=%.2f):\n", obs, ti, xi))
  cat(sprintf("  numdenom eta: %.4f (SD=%.4f)\n", mean(nd_eta), sd(nd_eta)))
  cat(sprintf("  Stan eta:     %.4f (SD=%.4f)\n", mean(stan_eta), sd(stan_eta)))
  cat(sprintf("  Diff: %.4f (%.2f SE) => %s\n\n", diff, ratio, if(ratio < 2) "MATCH" else "DIFFER"))
}

cat("\n===== Ratio Predictions =====\n\n")

# Compute ratio = exp(eta_num - eta_denom) for same observations
for (obs in test_obs) {
  xi <- df$x[obs]
  ti <- df$time[obs]

  # numdenom
  nd_eta_num <- draws_nd[, "beta_num[1]"] + draws_nd[, "beta_num[2]"] * xi +
                draws_nd[, paste0("temporal_gp[", ti, "]")]
  nd_eta_denom <- draws_nd[, "beta_denom[1]"] + draws_nd[, "beta_denom[2]"] * xi +
                  draws_nd[, paste0("temporal_gp[", ti, "]")]
  nd_ratio <- exp(nd_eta_num - nd_eta_denom)

  # Stan
  stan_eta_num <- draws_stan$beta_num_0 + draws_stan$beta_num_1 * xi +
                  draws_stan[[paste0("gp_effects[", ti, "]")]]
  stan_eta_denom <- draws_stan$beta_denom_0 + draws_stan$beta_denom_1 * xi +
                    draws_stan[[paste0("gp_effects[", ti, "]")]]
  stan_ratio <- exp(stan_eta_num - stan_eta_denom)

  se <- sqrt(sd(nd_ratio)^2/length(nd_ratio) + sd(stan_ratio)^2/length(stan_ratio))
  diff <- abs(mean(nd_ratio) - mean(stan_ratio))
  ratio_se <- diff / se

  cat(sprintf("Obs %d (time=%d, x=%.2f):\n", obs, ti, xi))
  cat(sprintf("  numdenom ratio: %.4f (SD=%.4f)\n", mean(nd_ratio), sd(nd_ratio)))
  cat(sprintf("  Stan ratio:     %.4f (SD=%.4f)\n", mean(stan_ratio), sd(stan_ratio)))
  cat(sprintf("  Diff: %.4f (%.2f SE) => %s\n\n", diff, ratio_se, if(ratio_se < 2) "MATCH" else "DIFFER"))
}
