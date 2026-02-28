# Quick diagnostic: Lognormal + RE - why does beta_denom fail?
# Focus on comparing numdenom vs Stan posteriors more carefully

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(12345)

# Small data for quick iteration
N <- 100
N_SITES <- 10

# Generate data WITH true random effects
site <- factor(rep(1:N_SITES, length.out = N))
x <- rnorm(N)

# True parameters
beta_num <- c(0.5, 0.3)
beta_denom <- c(-0.2, 0.2)
sigma_num <- 0.5
sigma_denom <- 0.6
true_sigma_re <- 0.5
true_re <- rnorm(N_SITES, 0, true_sigma_re)

# Generate data with shared RE
eta_num <- beta_num[1] + beta_num[2] * x + true_re[as.integer(site)]
eta_denom <- beta_denom[1] + beta_denom[2] * x + true_re[as.integer(site)]

y_num <- rlnorm(N, meanlog = eta_num, sdlog = sigma_num)
y_denom <- rlnorm(N, meanlog = eta_denom, sdlog = sigma_denom)

df <- data.frame(y_num, y_denom, x, site)

cat("True values:\n")
cat("  beta_num:", beta_num, "\n")
cat("  beta_denom:", beta_denom, "\n")
cat("  sigma_num:", sigma_num, "\n")
cat("  sigma_denom:", sigma_denom, "\n")
cat("  sigma_re:", true_sigma_re, "\n")
cat("  RE mean:", mean(true_re), "SD:", sd(true_re), "\n")

# ====================
# Fit numdenom
# ====================
cat("\n=== Fitting numdenom ===\n")
nd_fit <- ratiod(
  y_num | y_denom ~ x + (1 | site),
  data = df,
  family = ratiod_lognormal(),
  iter = 4000,
  warmup = 2000,
  chains = 2,
  cores = 2
)

# Extract posterior
nd_draws <- as_draws_matrix(nd_fit$draws)
nd_beta_num <- nd_draws[, "beta_num[2]"]
nd_beta_denom <- nd_draws[, "beta_denom[2]"]
nd_sigma_re <- nd_draws[, "sigma_re"]

cat("numdenom results:\n")
cat("  beta_num[2]:", mean(nd_beta_num), "SE:", sd(nd_beta_num), "\n")
cat("  beta_denom[2]:", mean(nd_beta_denom), "SE:", sd(nd_beta_denom), "\n")
cat("  sigma_re:", mean(nd_sigma_re), "SE:", sd(nd_sigma_re), "\n")

# ====================
# Fit Stan (non-centered)
# ====================
cat("\n=== Fitting Stan (non-centered) ===\n")
stan_model <- cmdstan_model("benchmarks/stan/joint_ln_re.stan")

stan_data <- list(
  N = N,
  y_num = y_num,
  y_denom = y_denom,
  p = 2,
  X = cbind(1, x),
  n_groups = N_SITES,
  group_idx = as.integer(site)
)

stan_fit <- stan_model$sample(
  data = stan_data,
  iter_sampling = 2000,
  iter_warmup = 1000,
  chains = 2,
  parallel_chains = 2,
  adapt_delta = 0.95,
  show_messages = FALSE
)

stan_draws <- stan_fit$draws(format = "matrix")
stan_beta_num <- stan_draws[, "beta_num[2]"]
stan_beta_denom <- stan_draws[, "beta_denom[2]"]
stan_sigma_re <- stan_draws[, "sigma_re"]

cat("Stan (non-centered) results:\n")
cat("  beta_num[2]:", mean(stan_beta_num), "SE:", sd(stan_beta_num), "\n")
cat("  beta_denom[2]:", mean(stan_beta_denom), "SE:", sd(stan_beta_denom), "\n")
cat("  sigma_re:", mean(stan_sigma_re), "SE:", sd(stan_sigma_re), "\n")

# ====================
# Comparison
# ====================
cat("\n=== Comparison ===\n")
diff_beta_num <- mean(nd_beta_num) - mean(stan_beta_num)
se_beta_num <- sqrt(var(nd_beta_num) + var(stan_beta_num))
cat("beta_num[2] diff:", diff_beta_num, "SE:", se_beta_num, "ratio:", abs(diff_beta_num)/se_beta_num, "\n")

diff_beta_denom <- mean(nd_beta_denom) - mean(stan_beta_denom)
se_beta_denom <- sqrt(var(nd_beta_denom) + var(stan_beta_denom))
cat("beta_denom[2] diff:", diff_beta_denom, "SE:", se_beta_denom, "ratio:", abs(diff_beta_denom)/se_beta_denom, "\n")

diff_sigma_re <- mean(nd_sigma_re) - mean(stan_sigma_re)
se_sigma_re <- sqrt(var(nd_sigma_re) + var(stan_sigma_re))
cat("sigma_re diff:", diff_sigma_re, "SE:", se_sigma_re, "ratio:", abs(diff_sigma_re)/se_sigma_re, "\n")

# Also look at true value recovery
cat("\n=== True value recovery ===\n")
cat("True beta_num[2]:", beta_num[2], "\n")
cat("  numdenom:", mean(nd_beta_num), "(error:", mean(nd_beta_num) - beta_num[2], ")\n")
cat("  Stan:", mean(stan_beta_num), "(error:", mean(stan_beta_num) - beta_num[2], ")\n")
cat("True beta_denom[2]:", beta_denom[2], "\n")
cat("  numdenom:", mean(nd_beta_denom), "(error:", mean(nd_beta_denom) - beta_denom[2], ")\n")
cat("  Stan:", mean(stan_beta_denom), "(error:", mean(stan_beta_denom) - beta_denom[2], ")\n")
