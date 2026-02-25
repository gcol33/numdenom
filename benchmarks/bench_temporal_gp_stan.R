# Validate temporal_gp against Stan/brms
# This validates that the posteriors are correct, not just that it runs

library(numdenom)
library(brms)
set.seed(123)

# Generate data with known temporal GP structure
N <- 100
n_times <- 10

# True parameters
true_beta <- c(0.5, -0.3)  # intercept, slope
true_sigma2_gp <- 0.5      # GP variance
true_phi_gp <- 2.0         # GP lengthscale

# Generate time values
time_vals <- rep(1:n_times, each = N/n_times)

# Generate GP covariance matrix (exponential)
time_unique <- 1:n_times
dist_mat <- as.matrix(dist(time_unique))
K <- true_sigma2_gp * exp(-dist_mat / true_phi_gp)

# Draw GP values
gp_effects <- MASS::mvrnorm(1, rep(0, n_times), K)
gp_by_obs <- gp_effects[time_vals]

# Generate data
x <- rnorm(N)
eta_num <- true_beta[1] + true_beta[2] * x + gp_by_obs
eta_denom <- 0  # Simple denominator

# Generate counts
y_num <- rnbinom(N, size = 5, mu = exp(eta_num) * 30)
y_denom <- rnbinom(N, size = 5, mu = exp(eta_denom) * 50)

df <- data.frame(
  num = y_num,
  denom = y_denom,
  x = x,
  time = time_vals
)

message("\n=== Data Summary ===")
message("N = ", N)
message("n_times = ", n_times)
message("True beta: ", paste(round(true_beta, 3), collapse = ", "))
message("True sigma2_gp: ", true_sigma2_gp)
message("True phi_gp: ", true_phi_gp)

# Fit with numdenom
message("\n=== Fitting numdenom temporal_gp ===")
t_numdenom <- system.time({
  fit_numdenom <- ratiod(
    num | denom ~ x,
    data = df,
    family = ratiod_negbin_negbin(),
    temporal = temporal_gp(time_var = "time", cov = "exponential"),
    iter = 1000,
    warmup = 500,
    chains = 2,
    verbose = FALSE
  )
})
message("Time: ", round(t_numdenom["elapsed"], 1), "s")

# Extract numdenom posteriors
draws_numdenom <- fit_numdenom$draws
beta_num_1 <- draws_numdenom[, "beta_num[1]"]
beta_num_2 <- draws_numdenom[, "beta_num[2]"]
sigma2_gp <- draws_numdenom[, "sigma2_temporal_gp"]
phi_gp <- draws_numdenom[, "phi_temporal_gp"]

message("\n=== numdenom posteriors ===")
message("beta_num[1]: ", round(mean(beta_num_1), 3), " (", round(sd(beta_num_1), 3), ")")
message("beta_num[2]: ", round(mean(beta_num_2), 3), " (", round(sd(beta_num_2), 3), ")")
message("sigma2_gp: ", round(mean(sigma2_gp), 3), " (", round(sd(sigma2_gp), 3), ")")
message("phi_gp: ", round(mean(phi_gp), 3), " (", round(sd(phi_gp), 3), ")")

# Fit with brms (GP temporal effect)
message("\n=== Fitting brms GP model ===")
# brms uses gp() for Gaussian process terms
# We model just the numerator with GP temporal effect
t_brms <- system.time({
  fit_brms <- brm(
    num ~ x + gp(time, cov = "exp_quad", k = n_times),
    data = df,
    family = negbinomial(),
    iter = 1000,
    warmup = 500,
    chains = 2,
    silent = 2,
    refresh = 0
  )
})
message("Time: ", round(t_brms["elapsed"], 1), "s")

# Extract brms posteriors
draws_brms <- as_draws_df(fit_brms)
beta_brms_1 <- draws_brms$b_Intercept
beta_brms_2 <- draws_brms$b_x
# brms uses different parameterization for GP
# sdgp = marginal SD, lscale = lengthscale
sdgp_brms <- draws_brms$sdgp_gptime
lscale_brms <- draws_brms$lscale_gptime

message("\n=== brms posteriors ===")
message("Intercept: ", round(mean(beta_brms_1), 3), " (", round(sd(beta_brms_1), 3), ")")
message("x: ", round(mean(beta_brms_2), 3), " (", round(sd(beta_brms_2), 3), ")")
message("sdgp (sqrt(sigma2)): ", round(mean(sdgp_brms), 3), " (", round(sd(sdgp_brms), 3), ")")
message("lscale (phi): ", round(mean(lscale_brms), 3), " (", round(sd(lscale_brms), 3), ")")

# Compare posteriors
message("\n=== Comparison ===")

# Note: numdenom models num|denom jointly, brms models only num
# So intercepts won't match exactly due to different model structures
# But slope coefficient and GP parameters should be comparable

# Compare x coefficient
diff_x <- abs(mean(beta_num_2) - mean(beta_brms_2))
se_x <- sqrt(sd(beta_num_2)^2 + sd(beta_brms_2)^2)
message("x coefficient:")
message("  numdenom: ", round(mean(beta_num_2), 3))
message("  brms:     ", round(mean(beta_brms_2), 3))
message("  diff:     ", round(diff_x, 3), " (", round(diff_x/se_x, 2), " pooled SE)")

# Compare GP variance (numdenom sigma2 vs brms sdgp^2)
sigma2_brms <- sdgp_brms^2
diff_sigma2 <- abs(mean(sigma2_gp) - mean(sigma2_brms))
se_sigma2 <- sqrt(sd(sigma2_gp)^2 + sd(sigma2_brms)^2)
message("\nGP variance (sigma2):")
message("  numdenom: ", round(mean(sigma2_gp), 3))
message("  brms:     ", round(mean(sigma2_brms), 3))
message("  diff:     ", round(diff_sigma2, 3), " (", round(diff_sigma2/se_sigma2, 2), " pooled SE)")

# Compare lengthscale
diff_phi <- abs(mean(phi_gp) - mean(lscale_brms))
se_phi <- sqrt(sd(phi_gp)^2 + sd(lscale_brms)^2)
message("\nGP lengthscale (phi):")
message("  numdenom: ", round(mean(phi_gp), 3))
message("  brms:     ", round(mean(lscale_brms), 3))
message("  diff:     ", round(diff_phi, 3), " (", round(diff_phi/se_phi, 2), " pooled SE)")

# Validation criteria: within 2 pooled SE
pass_x <- diff_x/se_x < 2
pass_sigma2 <- diff_sigma2/se_sigma2 < 2
pass_phi <- diff_phi/se_phi < 2

message("\n=== Validation Result ===")
message("x coefficient: ", if(pass_x) "PASS" else "FAIL")
message("GP variance:   ", if(pass_sigma2) "PASS" else "FAIL")
message("GP lengthscale: ", if(pass_phi) "PASS" else "FAIL")

if (all(c(pass_x, pass_sigma2, pass_phi))) {
  message("\nOVERALL: PASS - posteriors match Stan within 2 SE")
} else {
  message("\nOVERALL: FAIL - posteriors differ significantly from Stan")
}
