# Benchmark: TVC (Time-Varying Coefficients) - negbin_negbin family
# Validates row 57 in gradient_methods.md

library(numdenom)
library(cmdstanr)

set.seed(42)

# Generate data
N_OBS <- 200
N_TIMES <- 20

time_idx <- rep(1:N_TIMES, length.out = N_OBS)
x_tvc <- rnorm(N_OBS)  # Covariate with time-varying effect

# True effects
beta_num_intercept <- 2.0
beta_denom_intercept <- 1.5

# True TVC: coefficient varies over time (RW1-like)
true_beta_tvc <- cumsum(c(0.3, rnorm(N_TIMES - 1, 0, 0.1)))

# Linear predictors
eta_num <- beta_num_intercept + true_beta_tvc[time_idx] * x_tvc
eta_denom <- beta_denom_intercept + true_beta_tvc[time_idx] * x_tvc

# Generate observations - NegBin for both
size_num <- 5
size_denom <- 5
y_num <- rnbinom(N_OBS, size = size_num, mu = exp(eta_num))
y_denom <- rnbinom(N_OBS, size = size_denom, mu = exp(eta_denom))
# Ensure no zeros in denominator
y_denom[y_denom == 0] <- 1

df <- data.frame(
  y = y_num,
  effort = y_denom,
  x = x_tvc,
  time = factor(time_idx)
)

cat("\n", strrep("=", 60), "\n")
cat("TVC (negbin_negbin): numdenom vs Stan\n")
cat(strrep("=", 60), "\n\n")

cat(sprintf("N = %d, T = %d\n\n", N_OBS, N_TIMES))

# =============================================================================
# numdenom fit
# =============================================================================
cat("Fitting numdenom (H gradient)...\n")
t_numdenom <- system.time({
  fit_nd <- ratiod(
    y | effort ~ x,
    data = df,
    family = ratiod_negbin_negbin(),
    iter = 1000,
    warmup = 500,
    chains = 1,
    gradient_mode = "H",
    temporal = temporal_tvc(time_var = "time", terms = "x"),
    refresh = 0
  )
})["elapsed"]

nd_summ <- summary(fit_nd)
cat(sprintf("  Time: %.1fs, Divergent: %d\n\n", t_numdenom, nd_summ$diagnostics$divergent))

# =============================================================================
# Stan fit
# =============================================================================
cat("Compiling Stan model...\n")
stan_file <- "benchmarks/stan/joint_nb_tvc.stan"

# Check if Stan model exists
if (!file.exists(stan_file)) {
  cat("Stan model not found, skipping Stan comparison\n")
  cat("\n", strrep("=", 60), "\n")
  cat("RESULTS (numdenom only):\n")
  cat(strrep("-", 50), "\n")
  cat(sprintf("numdenom: %.1fs, %d divergent\n", t_numdenom, nd_summ$diagnostics$divergent))
  cat(sprintf("beta_num[1] (intercept): %.3f (true: %.2f)\n",
              nd_summ$fixed[1, "mean"], beta_num_intercept))
  cat(sprintf("beta_denom[1] (intercept): %.3f (true: %.2f)\n",
              nd_summ$fixed_denom[1, "mean"], beta_denom_intercept))
} else {
  mod <- cmdstan_model(stan_file, compile = FALSE)
  mod$compile(quiet = TRUE)

  X <- cbind(1, x_tvc)
  stan_data <- list(
    N = N_OBS,
    y_num = y_num,
    y_denom = as.integer(y_denom),
    p = 2,
    X = X,
    x_tvc = x_tvc,
    T = N_TIMES,
    time_idx = time_idx
  )

  cat("Fitting Stan...\n")
  t_stan <- system.time({
    fit_stan <- mod$sample(
      data = stan_data,
      chains = 1,
      iter_warmup = 500,
      iter_sampling = 500,
      refresh = 0,
      show_messages = FALSE
    )
  })["elapsed"]

  stan_summ <- fit_stan$summary()
  stan_diag <- fit_stan$diagnostic_summary()
  cat(sprintf("  Time: %.1fs, Divergent: %d\n\n", t_stan, sum(stan_diag$num_divergent)))

  # =============================================================================
  # Compare posteriors
  # =============================================================================
  cat(strrep("=", 60), "\n")
  cat("RESULTS:\n")
  cat(strrep("-", 50), "\n\n")

  cat(sprintf("numdenom: %.1fs, %d divergent\n", t_numdenom, nd_summ$diagnostics$divergent))
  cat(sprintf("Stan:     %.1fs, %d divergent\n", t_stan, sum(stan_diag$num_divergent)))
  cat(sprintf("Speedup:  %.1fx\n\n", t_stan / t_numdenom))

  # Fixed effects
  stan_b1_num <- stan_summ[stan_summ$variable == "beta_num[1]", ]
  cat(sprintf("beta_num[1]: numdenom=%.3f, Stan=%.3f, true=%.2f\n",
              nd_summ$fixed[1, "mean"], stan_b1_num$mean, beta_num_intercept))

  stan_b1_denom <- stan_summ[stan_summ$variable == "beta_denom[1]", ]
  cat(sprintf("beta_denom[1]: numdenom=%.3f, Stan=%.3f, true=%.2f\n",
              nd_summ$fixed_denom[1, "mean"], stan_b1_denom$mean, beta_denom_intercept))

  # Validation
  diff1 <- abs(nd_summ$fixed[1, "mean"] - stan_b1_num$mean)
  se1 <- sqrt(nd_summ$fixed[1, "sd"]^2 + stan_b1_num$sd^2)

  cat(sprintf("\nValidation (within 2 SE): %s\n",
              ifelse(diff1 < 2 * se1, "PASS", "FAIL")))
}
