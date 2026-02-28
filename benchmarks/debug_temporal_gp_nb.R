# Debug temporal GP negbin_negbin validation failure
# Detailed comparison of numdenom vs Stan posteriors

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

# Scale time for Stan (mean 0, sd 1)
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
df_nb_gp <- data.frame(
  y = rnbinom(N_OBS, mu = exp(eta_num), size = 5),
  denom = rnbinom(N_OBS, mu = exp(eta_denom), size = 10),
  x = x, time = time
)
df_nb_gp$denom[df_nb_gp$denom == 0] <- 1

cat("Fitting numdenom...\n")
fit_nd <- ratiod(
  y | denom ~ x, data = df_nb_gp,
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
  y_num = df_nb_gp$y,
  y_denom = df_nb_gp$denom,
  x = df_nb_gp$x,
  time_idx = df_nb_gp$time,
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

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

cat("\n===== PARAMETER COMPARISON =====\n\n")

# Print all numdenom parameter names
cat("All numdenom parameter names:\n")
nd_names <- colnames(draws_nd)
print(nd_names)

cat("\n\nAll Stan parameter names:\n")
print(names(draws_stan))

cat("\n\n===== Comparing key parameters =====\n\n")

# Helper function
compare_param <- function(nd_val, stan_val, name) {
  se <- sqrt(sd(nd_val)^2/length(nd_val) + sd(stan_val)^2/length(stan_val))
  diff <- abs(mean(nd_val) - mean(stan_val))
  ratio <- diff / se
  pass <- ratio < 2

  cat(sprintf("%s:\n", name))
  cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", mean(nd_val), sd(nd_val)))
  cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", mean(stan_val), sd(stan_val)))
  cat(sprintf("  diff=%.4f (%.2f SE) => %s\n\n", diff, ratio, if(pass) "PASS" else "FAIL"))
}

# Intercepts
nd_int <- draws_nd[, "beta_num[1]"]
compare_param(nd_int, draws_stan$beta_num_0, "beta_num[1] (intercept)")

nd_slope <- draws_nd[, "beta_num[2]"]
compare_param(nd_slope, draws_stan$beta_num_1, "beta_num[2] (slope)")

nd_int_d <- draws_nd[, "beta_denom[1]"]
compare_param(nd_int_d, draws_stan$beta_denom_0, "beta_denom[1] (intercept)")

nd_slope_d <- draws_nd[, "beta_denom[2]"]
compare_param(nd_slope_d, draws_stan$beta_denom_1, "beta_denom[2] (slope)")

# Dispersion
compare_param(draws_nd[, "phi_num"], draws_stan$phi_num, "phi_num")
compare_param(draws_nd[, "phi_denom"], draws_stan$phi_denom, "phi_denom")

# GP parameters - find the correct column names
sigma_cols <- grep("sigma.*gp|sigma2_gp", nd_names, value = TRUE)
phi_cols <- grep("phi.*gp|lengthscale", nd_names, value = TRUE)

cat("Sigma GP columns in numdenom:", paste(sigma_cols, collapse = ", "), "\n")
cat("Phi GP columns in numdenom:", paste(phi_cols, collapse = ", "), "\n\n")

if (length(sigma_cols) > 0) {
  compare_param(draws_nd[, sigma_cols[1]], draws_stan$sigma_gp, paste("GP sigma (", sigma_cols[1], ")"))
}

if (length(phi_cols) > 0) {
  compare_param(draws_nd[, phi_cols[1]], draws_stan$phi_gp, paste("GP phi (", phi_cols[1], ")"))
}

# GP effects - first few time points
cat("\n===== GP Effects (first 3 time points) =====\n\n")
gp_cols <- grep("^temporal_gp\\[", nd_names, value = TRUE)
cat("GP effect columns:", length(gp_cols), "\n")

for (i in 1:min(3, length(gp_cols))) {
  col_name <- paste0("gp_effects[", i, "]")
  if (col_name %in% names(draws_stan)) {
    compare_param(draws_nd[, gp_cols[i]], draws_stan[[col_name]], paste("GP effect", i))
  }
}
