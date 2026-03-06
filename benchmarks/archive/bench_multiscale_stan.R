# Benchmark: Multiscale temporal - numdenom vs Stan
# Validates row 15 in gradient_methods.md

library(numdenom)
library(cmdstanr)

set.seed(42)

# Generate data
N_OBS <- 500
N_TIMES <- 50

time_idx <- rep(1:N_TIMES, length.out = N_OBS)
x <- rnorm(N_OBS)

# True effects
beta_num <- c(2.0, 0.3)
beta_denom <- c(1.5, 0.2)
true_trend <- cumsum(cumsum(rnorm(N_TIMES, 0, 0.05)))  # RW2-like
true_trend <- true_trend - mean(true_trend)  # Center
true_short <- arima.sim(list(ar = 0.7), n = N_TIMES, sd = 0.2)

# Linear predictors
X <- cbind(1, x)
eta_num <- X %*% beta_num + true_trend[time_idx] + true_short[time_idx]
eta_denom <- X %*% beta_denom + true_trend[time_idx] + true_short[time_idx]

# Generate observations
y_num <- rpois(N_OBS, exp(eta_num))
y_denom <- rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_denom))

df <- data.frame(
  y = y_num,
  effort = y_denom,
  x = x,
  time = factor(time_idx)
)

cat("\n", strrep("=", 60), "\n")
cat("Multiscale Temporal: numdenom vs Stan Validation\n")
cat(strrep("=", 60), "\n\n")

cat(sprintf("N = %d, T = %d\n\n", N_OBS, N_TIMES))

# =============================================================================
# numdenom fit
# =============================================================================
cat("Fitting numdenom model (H gradient)...\n")
t_numdenom <- system.time({
  fit_nd <- ratiod(
    y | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    iter = 1000,
    warmup = 500,
    chains = 1,
    gradient_mode = "H",
    temporal = temporal_multiscale(time_var = "time", seasonal = NULL),
    refresh = 100
  )
})["elapsed"]
cat(sprintf("  Time: %.1fs\n\n", t_numdenom))

# =============================================================================
# Stan fit
# =============================================================================
cat("Compiling Stan model...\n")
stan_file <- "benchmarks/stan/joint_pg_multiscale.stan"
mod <- cmdstan_model(stan_file)

stan_data <- list(
  N = N_OBS,
  y_num = y_num,
  y_denom = y_denom,
  p = 2,
  X = X,
  T = N_TIMES,
  time_idx = time_idx,
  seasonal_period = 0
)

cat("Fitting Stan model...\n")
t_stan <- system.time({
  fit_stan <- mod$sample(
    data = stan_data,
    chains = 1,
    iter_warmup = 500,
    iter_sampling = 500,
    refresh = 100,
    show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("  Time: %.1fs\n\n", t_stan))

# =============================================================================
# Compare posteriors
# =============================================================================
cat("Comparing posteriors:\n")
cat(strrep("-", 50), "\n")

# Extract Stan summaries
stan_summary <- fit_stan$summary()

# Beta_num[1] (intercept)
stan_beta1 <- stan_summary[stan_summary$variable == "beta_num[1]", ]
nd_beta1 <- summary(fit_nd)$fixed[1, ]  # First fixed effect

cat(sprintf("beta_num[1] (intercept):\n"))
cat(sprintf("  Stan:     %.3f (SE: %.3f)\n", stan_beta1$mean, stan_beta1$sd))
cat(sprintf("  numdenom: %.3f (SE: %.3f)\n", nd_beta1["mean"], nd_beta1["sd"]))
cat(sprintf("  True:     %.3f\n", beta_num[1]))

# Beta_num[2] (slope)
stan_beta2 <- stan_summary[stan_summary$variable == "beta_num[2]", ]
nd_beta2 <- summary(fit_nd)$fixed[2, ]

cat(sprintf("\nbeta_num[2] (slope):\n"))
cat(sprintf("  Stan:     %.3f (SE: %.3f)\n", stan_beta2$mean, stan_beta2$sd))
cat(sprintf("  numdenom: %.3f (SE: %.3f)\n", nd_beta2["mean"], nd_beta2["sd"]))
cat(sprintf("  True:     %.3f\n", beta_num[2]))

# Shape parameter
stan_shape <- stan_summary[stan_summary$variable == "shape", ]
nd_phi <- summary(fit_nd)$phi

cat(sprintf("\nshape (phi):\n"))
cat(sprintf("  Stan:     %.3f (SE: %.3f)\n", stan_shape$mean, stan_shape$sd))
cat(sprintf("  numdenom: %.3f (SE: %.3f)\n", nd_phi["mean"], nd_phi["sd"]))
cat(sprintf("  True:     5.0\n"))

# Sigma trend
stan_sigma_trend <- stan_summary[stan_summary$variable == "sigma_trend", ]
cat(sprintf("\nsigma_trend:\n"))
cat(sprintf("  Stan:     %.4f (SE: %.4f)\n", stan_sigma_trend$mean, stan_sigma_trend$sd))

# Rho short
stan_rho <- stan_summary[stan_summary$variable == "rho_short", ]
cat(sprintf("\nrho_short:\n"))
cat(sprintf("  Stan:     %.3f (SE: %.3f)\n", stan_rho$mean, stan_rho$sd))
cat(sprintf("  True:     0.7\n"))

# =============================================================================
# Summary
# =============================================================================
cat("\n", strrep("=", 60), "\n")
cat("Summary:\n")
cat(strrep("-", 50), "\n")
cat(sprintf("numdenom time: %.1fs\n", t_numdenom))
cat(sprintf("Stan time:     %.1fs\n", t_stan))
cat(sprintf("Speedup:       %.1fx\n", t_stan / t_numdenom))

# Check if within 2 SE
diff_beta1 <- abs(stan_beta1$mean - nd_beta1["mean"])
se_beta1 <- sqrt(stan_beta1$sd^2 + nd_beta1["sd"]^2)
within_2se_beta1 <- diff_beta1 < 2 * se_beta1

diff_beta2 <- abs(stan_beta2$mean - nd_beta2["mean"])
se_beta2 <- sqrt(stan_beta2$sd^2 + nd_beta2["sd"]^2)
within_2se_beta2 <- diff_beta2 < 2 * se_beta2

cat(sprintf("\nValidation (within 2 SE):\n"))
cat(sprintf("  beta_num[1]: %s\n", ifelse(within_2se_beta1, "PASS", "FAIL")))
cat(sprintf("  beta_num[2]: %s\n", ifelse(within_2se_beta2, "PASS", "FAIL")))

if (within_2se_beta1 && within_2se_beta2) {
  cat("\n*** VALIDATION PASSED ***\n")
} else {
  cat("\n*** VALIDATION FAILED ***\n")
}
