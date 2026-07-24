# Debug: Center GP effects and compare

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
fit_nd <- tratio(
  y | denom ~ x, data = df,
  family = ratiod_negbin_negbin(),
  temporal = temporal_gp(time_var = "time", cov = "exponential"),
  control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
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

# Center GP effects by subtracting mean
gp_cols <- paste0("temporal_gp[", 1:N_TIMES, "]")
stan_gp_cols <- paste0("gp_effects[", 1:N_TIMES, "]")

nd_gp <- draws_nd[, gp_cols]
stan_gp <- as.matrix(draws_stan[, stan_gp_cols])

nd_gp_mean <- rowMeans(nd_gp)
stan_gp_mean <- rowMeans(stan_gp)

nd_gp_centered <- nd_gp - nd_gp_mean
stan_gp_centered <- stan_gp - stan_gp_mean

cat("\n===== CENTERED GP EFFECTS COMPARISON =====\n\n")

for (t in 1:N_TIMES) {
  nd_t <- nd_gp_centered[, t]
  stan_t <- stan_gp_centered[, t]

  se <- sqrt(sd(nd_t)^2/length(nd_t) + sd(stan_t)^2/length(stan_t))
  diff <- abs(mean(nd_t) - mean(stan_t))
  ratio <- diff / se
  pass <- ratio < 2

  cat(sprintf("  Time %2d: nd=%7.4f (SD=%6.4f), stan=%7.4f (SD=%6.4f), diff=%6.4f (%5.2f SE) => %s\n",
              t, mean(nd_t), sd(nd_t), mean(stan_t), sd(stan_t), diff, ratio, if(pass) "PASS" else "FAIL"))
}

cat("\n===== CENTERED FITTED VALUES =====\n\n")

# Fitted values using centered GP
test_obs <- c(1, 50, 100, 150, 200)

for (obs in test_obs) {
  xi <- df$x[obs]
  ti <- df$time[obs]

  # numdenom: (intercept + GP_mean) + slope*x + (GP - GP_mean)
  nd_eta <- (draws_nd[, "beta_num[1]"] + nd_gp_mean) +
            draws_nd[, "beta_num[2]"] * xi +
            nd_gp_centered[, ti]

  # Stan: (intercept + GP_mean) + slope*x + (GP - GP_mean)
  stan_eta <- (draws_stan$beta_num_0 + stan_gp_mean) +
              draws_stan$beta_num_1 * xi +
              stan_gp_centered[, ti]

  se <- sqrt(sd(nd_eta)^2/length(nd_eta) + sd(stan_eta)^2/length(stan_eta))
  diff <- abs(mean(nd_eta) - mean(stan_eta))
  ratio <- diff / se
  pass <- ratio < 2

  cat(sprintf("  Obs %3d (t=%2d): nd=%7.4f (SD=%6.4f), stan=%7.4f (SD=%6.4f), diff=%6.4f (%5.2f SE) => %s\n",
              obs, ti, mean(nd_eta), sd(nd_eta), mean(stan_eta), sd(stan_eta), diff, ratio, if(pass) "PASS" else "FAIL"))
}
