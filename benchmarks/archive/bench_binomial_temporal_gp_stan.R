# Binomial + temporal GP validation against Stan (Row 74)
# Uses custom Stan model with matching GP implementation

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

N_OBS <- 200
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2
N_TIMES <- 10

cat("=======================================================\n")
cat("Binomial + Temporal GP Validation vs Stan (Row 74)\n")
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
  cat(sprintf("  %s: nd=%.4f (SD=%.4f), stan=%.4f (SD=%.4f), diff=%.4f (%.2f SE) => %s\n",
              result$param, result$nd_mean, result$nd_sd,
              result$stan_mean, result$stan_sd,
              result$diff, result$ratio, if(result$pass) "PASS" else "FAIL"))
}

# Setup data
x <- rnorm(N_OBS)
time <- rep(1:N_TIMES, length.out = N_OBS)
time_scaled <- (time - mean(time)) / sd(time)
time_unique_scaled <- (1:N_TIMES - mean(1:N_TIMES)) / sd(1:N_TIMES)

trials <- sample(10:50, N_OBS, replace = TRUE)

# True GP effect (for data generation)
true_sigma_gp <- 0.3
true_phi_gp <- 2.0
dist_mat <- as.matrix(dist(1:N_TIMES))
K <- true_sigma_gp^2 * exp(-dist_mat / true_phi_gp)
gp_effects_true <- MASS::mvrnorm(1, rep(0, N_TIMES), K + diag(1e-6, N_TIMES))
gp_by_obs <- gp_effects_true[time]

# Linear predictor for probability
eta <- 0.5 + 0.3 * x + gp_by_obs
prob <- plogis(eta)

# Generate binomial data
successes <- rbinom(N_OBS, trials, prob)

df <- data.frame(
  y = successes,
  trials = trials,
  x = x,
  time = time
)

cat("Fitting numdenom (binomial + temporal GP)...\n")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | trials ~ x, data = df,
    family = ratiod_binomial(),
    temporal = temporal_gp(time_var = "time", cov = "exponential"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})[[3]]
cat(sprintf("  Time: %.1fs\n", t_nd))

# Fit Stan model
cat("\nFitting Stan (binomial + temporal GP)...\n")
stan_model <- cmdstan_model("benchmarks/stan/temporal_gp_binomial.stan")

stan_data <- list(
  N = N_OBS,
  T = N_TIMES,
  y = df$y,
  trials = df$trials,
  x = df$x,
  time_idx = df$time,
  time_values = time_unique_scaled,
  sigma2_prior_U = 1.0,
  sigma2_prior_alpha = 0.01,
  phi_prior_lower = 0.01,
  phi_prior_upper = 10.0
)

t_stan <- system.time({
  fit_stan <- stan_model$sample(
    data = stan_data,
    iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
    chains = N_CHAINS, parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE, adapt_delta = 0.95
  )
})[[3]]
cat(sprintf("  Time: %.1fs\n", t_stan))

# Extract draws
draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

cat("\n===== PARAMETER COMPARISON =====\n\n")

# Print numdenom parameter names
cat("All numdenom parameter names:\n")
nd_names <- colnames(draws_nd)
print(nd_names)

cat("\n\nAll Stan parameter names:\n")
print(names(draws_stan))

cat("\n\n===== Comparing key parameters =====\n\n")

results <- list()

# Intercept - find the right column name in numdenom
# For binomial models, it should be beta_num[1] (intercept)
nd_int <- draws_nd[, "beta_num[1]"]
results[[1]] <- compare_posteriors(nd_int, draws_stan$beta_0, "beta[1] (intercept)")
print_result(results[[1]])

# Slope
nd_slope <- draws_nd[, "beta_num[2]"]
results[[2]] <- compare_posteriors(nd_slope, draws_stan$beta_1, "beta[2] (slope)")
print_result(results[[2]])

# GP parameters
sigma_cols <- grep("sigma.*temporal.*gp|sigma2.*temporal.*gp", nd_names, value = TRUE)
phi_cols <- grep("phi.*temporal.*gp|lengthscale.*temporal", nd_names, value = TRUE)

cat("\nSigma GP columns in numdenom:", paste(sigma_cols, collapse = ", "), "\n")
cat("Phi GP columns in numdenom:", paste(phi_cols, collapse = ", "), "\n\n")

if (length(sigma_cols) > 0) {
  # numdenom stores sigma2, Stan stores sigma - need to compare carefully
  nd_sigma2 <- draws_nd[, sigma_cols[1]]
  results[[3]] <- compare_posteriors(nd_sigma2, draws_stan$sigma2_gp, paste("GP sigma2 (", sigma_cols[1], ")"))
  print_result(results[[3]])
}

if (length(phi_cols) > 0) {
  results[[4]] <- compare_posteriors(draws_nd[, phi_cols[1]], draws_stan$phi_gp, paste("GP phi (", phi_cols[1], ")"))
  print_result(results[[4]])
}

# GP effects - first few time points
cat("\n===== GP Effects (first 3 time points) =====\n\n")
gp_cols <- grep("^temporal_gp\\[", nd_names, value = TRUE)
cat("GP effect columns:", length(gp_cols), "\n\n")

for (i in 1:min(3, length(gp_cols))) {
  col_name <- paste0("gp_effects[", i, "]")
  if (col_name %in% names(draws_stan)) {
    result <- compare_posteriors(draws_nd[, gp_cols[i]], draws_stan[[col_name]], paste("GP effect", i))
    print_result(result)
  }
}

cat(sprintf("\n\nTimes: numdenom=%.1fs, Stan=%.1fs\n", t_nd, t_stan))

# Overall pass/fail
n_pass <- sum(sapply(results, function(r) r$pass))
n_total <- length(results)

if (n_pass == n_total) {
  cat(sprintf("\n*** Row 74 PASSES validation (%d/%d parameters within 2 SE) ***\n", n_pass, n_total))
} else {
  cat(sprintf("\n*** Row 74 FAILS validation (%d/%d parameters within 2 SE) ***\n", n_pass, n_total))
}
