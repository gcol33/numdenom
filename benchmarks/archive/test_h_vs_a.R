# Compare H vs A/A_t for all three new families

library(numdenom)

set.seed(42)
N <- 200

compare_modes <- function(fit_A, fit_H, true_vals, family_name) {
  draws_A <- as.matrix(fit_A$draws)
  draws_H <- as.matrix(fit_H$draws)

  cat("\n=== ", family_name, " ===\n", sep = "")
  cat("Parameter         A_t              H                True\n")
  cat("----------------------------------------------------------------\n")

  for (p in names(true_vals)) {
    if (p %in% colnames(draws_A) && p %in% colnames(draws_H)) {
      mean_A <- mean(draws_A[, p])
      mean_H <- mean(draws_H[, p])
      true_v <- true_vals[[p]]
      diff <- abs(mean_A - mean_H)
      ok <- if (diff < 0.1) "OK" else "DIFF"
      cat(sprintf("%-16s  %.4f (+/- %.4f)   %.4f (+/- %.4f)   %.2f  %s\n",
                  p, mean_A, sd(draws_A[, p]), mean_H, sd(draws_H[, p]), true_v, ok))
    }
  }
}

# ==============================================================================
# GAMMA_GAMMA
# ==============================================================================
x <- rnorm(N)
y_num_gg <- rgamma(N, shape = 5, rate = 5 / exp(2 + 0.3 * x))
y_denom_gg <- rgamma(N, shape = 8, rate = 8 / exp(3 + 0.2 * x))
df_gg <- data.frame(y = y_num_gg, denom = y_denom_gg, x = x)

fit_A_gg <- tratio(y | denom ~ x, data = df_gg, family = ratiod_gamma_gamma(),
                   control = list(iter = 1000, warmup = 500, chains = 1, gradient_mode = "A_t", verbose = FALSE))
fit_H_gg <- tratio(y | denom ~ x, data = df_gg, family = ratiod_gamma_gamma(),
                   control = list(iter = 1000, warmup = 500, chains = 1, gradient_mode = "H", verbose = FALSE))

compare_modes(fit_A_gg, fit_H_gg,
              list("beta_num[1]" = 2, "beta_num[2]" = 0.3, "shape_num" = 5),
              "GAMMA_GAMMA")

# ==============================================================================
# LOGNORMAL
# ==============================================================================
y_num_ln <- rlnorm(N, meanlog = 2 + 0.3 * x, sdlog = 0.5)
y_denom_ln <- rlnorm(N, meanlog = 3 + 0.2 * x, sdlog = 0.4)
df_ln <- data.frame(y = y_num_ln, denom = y_denom_ln, x = x)

fit_A_ln <- tratio(y | denom ~ x, data = df_ln, family = ratiod_lognormal(),
                   control = list(iter = 1000, warmup = 500, chains = 1, gradient_mode = "A_t", verbose = FALSE))
fit_H_ln <- tratio(y | denom ~ x, data = df_ln, family = ratiod_lognormal(),
                   control = list(iter = 1000, warmup = 500, chains = 1, gradient_mode = "H", verbose = FALSE))

compare_modes(fit_A_ln, fit_H_ln,
              list("beta_num[1]" = 2, "beta_num[2]" = 0.3, "sigma_num" = 0.5),
              "LOGNORMAL")

# ==============================================================================
# BETA_BINOMIAL
# ==============================================================================
n_trials <- rep(50, N)
prob <- plogis(0.5 + 0.8 * x)
phi_bb <- 10
alpha_bb <- prob * phi_bb
beta_bb <- (1 - prob) * phi_bb
y_bb <- rbinom(N, n_trials, rbeta(N, alpha_bb, beta_bb))
df_bb <- data.frame(y = y_bb, n = n_trials, x = x)

fit_A_bb <- tratio(y | n ~ x, data = df_bb, family = ratiod_beta_binomial(),
                   control = list(iter = 1000, warmup = 500, chains = 1, gradient_mode = "A_t", verbose = FALSE))
fit_H_bb <- tratio(y | n ~ x, data = df_bb, family = ratiod_beta_binomial(),
                   control = list(iter = 1000, warmup = 500, chains = 1, gradient_mode = "H", verbose = FALSE))

compare_modes(fit_A_bb, fit_H_bb,
              list("beta_num[1]" = 0.5, "beta_num[2]" = 0.8, "phi_num" = 10),
              "BETA_BINOMIAL")

cat("\n=== DONE ===\n")
