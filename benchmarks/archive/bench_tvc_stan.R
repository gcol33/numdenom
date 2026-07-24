# Benchmark: TVC (Time-Varying Coefficients) - numdenom vs Stan
# Validates row 27 in gradient_methods.md

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

# Generate observations
y_num <- rpois(N_OBS, exp(eta_num))
y_denom <- rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_denom))

df <- data.frame(
  y = y_num,
  effort = y_denom,
  x = x_tvc,
  time = factor(time_idx)
)

cat("\n", strrep("=", 60), "\n")
cat("TVC (Time-Varying Coefficients): numdenom vs Stan\n")
cat(strrep("=", 60), "\n\n")

cat(sprintf("N = %d, T = %d\n\n", N_OBS, N_TIMES))

# =============================================================================
# numdenom fit
# =============================================================================
cat("Fitting numdenom (H gradient)...\n")
t_numdenom <- system.time({
  fit_nd <- tratio(
    y | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    temporal = temporal_tvc(time_var = "time", terms = "x"),
    control = list(iter = 1000, warmup = 500, chains = 1, gradient_mode = "H")
  )
})["elapsed"]

nd_summ <- summary(fit_nd)
cat(sprintf("  Time: %.1fs, Divergent: %d\n\n", t_numdenom, nd_summ$diagnostics$divergent))

# =============================================================================
# Stan fit
# =============================================================================
cat("Compiling Stan model...\n")
stan_file <- "benchmarks/stan/joint_pg_tvc.stan"
mod <- cmdstan_model(stan_file, compile = FALSE)
mod$compile(quiet = TRUE)

X <- cbind(1, x_tvc)  # Intercept + x as fixed effect
stan_data <- list(
  N = N_OBS,
  y_num = y_num,
  y_denom = y_denom,
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

# Fixed effects comparison
stan_b1_num <- stan_summ[stan_summ$variable == "beta_num[1]", ]
stan_b2_num <- stan_summ[stan_summ$variable == "beta_num[2]", ]
stan_b1_denom <- stan_summ[stan_summ$variable == "beta_denom[1]", ]
stan_b2_denom <- stan_summ[stan_summ$variable == "beta_denom[2]", ]
stan_shape <- stan_summ[stan_summ$variable == "shape", ]
stan_sigma_tvc <- stan_summ[stan_summ$variable == "sigma_tvc", ]

cat("Parameter        | True  | numdenom      | Stan\n")
cat(strrep("-", 60), "\n")
cat(sprintf("beta_num[1]      | %.2f  | %.3f (%.3f)  | %.3f (%.3f)\n",
            beta_num_intercept,
            nd_summ$fixed[1, "mean"], nd_summ$fixed[1, "sd"],
            stan_b1_num$mean, stan_b1_num$sd))
cat(sprintf("beta_num[2] (x)  | -     | %.3f (%.3f)  | %.3f (%.3f)\n",
            nd_summ$fixed[2, "mean"], nd_summ$fixed[2, "sd"],
            stan_b2_num$mean, stan_b2_num$sd))
cat(sprintf("beta_denom[1]    | %.2f  | %.3f (%.3f)  | %.3f (%.3f)\n",
            beta_denom_intercept,
            nd_summ$fixed_denom[1, "mean"], nd_summ$fixed_denom[1, "sd"],
            stan_b1_denom$mean, stan_b1_denom$sd))
cat(sprintf("beta_denom[2] (x)| -     | %.3f (%.3f)  | %.3f (%.3f)\n",
            nd_summ$fixed_denom[2, "mean"], nd_summ$fixed_denom[2, "sd"],
            stan_b2_denom$mean, stan_b2_denom$sd))
cat(sprintf("shape            | 5.00  | %.3f (%.3f)  | %.3f (%.3f)\n",
            nd_summ$phi["mean"], nd_summ$phi["sd"],
            stan_shape$mean, stan_shape$sd))
cat(sprintf("sigma_tvc        | 0.10  | -             | %.3f (%.3f)\n",
            stan_sigma_tvc$mean, stan_sigma_tvc$sd))

# Compare TVC coefficients (first few)
cat("\nTVC coefficients (first 5 time points):\n")
cat("Time | True  | Stan\n")
cat(strrep("-", 30), "\n")
for (t in 1:5) {
  stan_tvc <- stan_summ[stan_summ$variable == sprintf("beta_tvc[%d]", t), ]
  cat(sprintf("  %2d | %.3f | %.3f (%.3f)\n", t, true_beta_tvc[t], stan_tvc$mean, stan_tvc$sd))
}

# Validation
diff1 <- abs(nd_summ$fixed[1, "mean"] - stan_b1_num$mean)
se1 <- sqrt(nd_summ$fixed[1, "sd"]^2 + stan_b1_num$sd^2)

cat("\n", strrep("=", 60), "\n")
cat("Summary:\n")
cat(strrep("-", 50), "\n")
cat(sprintf("numdenom: %.1fs, %d divergent\n", t_numdenom, nd_summ$diagnostics$divergent))
cat(sprintf("Stan:     %.1fs, %d divergent\n", t_stan, sum(stan_diag$num_divergent)))
cat(sprintf("Speedup:  %.1fx (Stan/numdenom)\n", t_stan / t_numdenom))

cat(sprintf("\nValidation (within 2 SE): %s\n",
            ifelse(diff1 < 2 * se1, "PASS", "FAIL")))
