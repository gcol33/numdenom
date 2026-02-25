# Debug: Check if intercept + mean(RE) is the same
# The key insight: with shared RE, intercept and RE mean are confounded

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

N_OBS <- 200
N_ITER <- 4000
N_WARMUP <- 1500
N_CHAINS <- 4
N_SITES <- 20

cat("=== DEBUG: Intercept + mean(RE) comparison ===\n\n")

# Generate data with RE
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)

true_beta_num <- c(2, 0.5)
true_beta_denom <- c(3, 0.2)
true_phi_num <- 5.0
true_phi_denom <- 8.0
true_sigma_re <- 0.4

# Generate RE (SHARED between num and denom)
true_re <- rnorm(N_SITES, 0, true_sigma_re)

eta_num <- true_beta_num[1] + true_beta_num[2] * x
eta_denom <- true_beta_denom[1] + true_beta_denom[2] * x
for (i in 1:N_OBS) {
  g <- as.integer(site[i])
  eta_num[i] <- eta_num[i] + true_re[g]
  eta_denom[i] <- eta_denom[i] + true_re[g]  # SHARED
}

y_num <- rnbinom(N_OBS, size = true_phi_num, mu = exp(eta_num))
y_denom <- rnbinom(N_OBS, size = true_phi_denom, mu = exp(eta_denom))
y_denom[y_denom == 0] <- 1

df <- data.frame(y_num = y_num, y_denom = y_denom, x = x, site = site)

# Fit numdenom
cat("Fitting numdenom... ")
fit_nd <- ratiod(
  y_num | y_denom ~ x + (1|site),
  data = df,
  family = ratiod_negbin_negbin(),
  iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
  verbose = FALSE
)
cat("done\n")

# Stan model
stan_code <- "
data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;
  array[N] int<lower=0> y_denom;
  matrix[N, 2] X;
  int<lower=1> n_groups;
  array[N] int<lower=1,upper=n_groups> group_idx;
}
parameters {
  vector[2] beta_num;
  vector[2] beta_denom;
  real<lower=0> phi_num;
  real<lower=0> phi_denom;
  real<lower=0> sigma_re;
  vector[n_groups] re;
}
model {
  vector[N] eta_num;
  vector[N] eta_denom;

  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  phi_num ~ gamma(2, 0.1);
  phi_denom ~ gamma(2, 0.1);
  sigma_re ~ cauchy(0, 2.5);
  re ~ normal(0, sigma_re);

  eta_num = X * beta_num;
  eta_denom = X * beta_denom;
  for (n in 1:N) {
    eta_num[n] += re[group_idx[n]];
    eta_denom[n] += re[group_idx[n]];
  }

  y_num ~ neg_binomial_2_log(eta_num, phi_num);
  y_denom ~ neg_binomial_2_log(eta_denom, phi_denom);
}
generated quantities {
  real mean_re = mean(re);
  real effective_intercept_num = beta_num[1] + mean_re;
  real effective_intercept_denom = beta_denom[1] + mean_re;
}
"

writeLines(stan_code, "stan/joint_nb_intercept_gq.stan")
stan_mod <- cmdstan_model("stan/joint_nb_intercept_gq.stan")

stan_data <- list(
  N = N_OBS,
  y_num = df$y_num,
  y_denom = df$y_denom,
  X = cbind(1, df$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df$site)
)

cat("Fitting Stan... ")
fit_stan <- stan_mod$sample(
  data = stan_data,
  iter_sampling = N_ITER - N_WARMUP,
  iter_warmup = N_WARMUP,
  chains = N_CHAINS,
  parallel_chains = N_CHAINS,
  refresh = 0,
  show_messages = FALSE
)
cat("done\n\n")

# Extract draws
draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

# Get numdenom RE columns
nd_cols <- colnames(draws_nd)
re_cols_nd <- nd_cols[grep("^re\\[", nd_cols)]

# Compute mean RE for each draw in numdenom
nd_mean_re <- rowMeans(draws_nd[, re_cols_nd])
nd_eff_int_num <- draws_nd[, "beta_num[1]"] + nd_mean_re
nd_eff_int_denom <- draws_nd[, "beta_denom[1]"] + nd_mean_re

cat("=== Raw intercept comparison ===\n")
cat(sprintf("beta_num[1]:\n"))
cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", mean(draws_nd[, "beta_num[1]"]), sd(draws_nd[, "beta_num[1]"])))
cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", mean(draws_stan$`beta_num[1]`), sd(draws_stan$`beta_num[1]`)))

cat(sprintf("\nbeta_denom[1]:\n"))
cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", mean(draws_nd[, "beta_denom[1]"]), sd(draws_nd[, "beta_denom[1]"])))
cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", mean(draws_stan$`beta_denom[1]`), sd(draws_stan$`beta_denom[1]`)))

cat("\n=== mean(RE) comparison ===\n")
cat(sprintf("mean(RE):\n"))
cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", mean(nd_mean_re), sd(nd_mean_re)))
cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", mean(draws_stan$mean_re), sd(draws_stan$mean_re)))

cat("\n=== Effective intercept (beta[1] + mean(RE)) ===\n")
cat(sprintf("effective_intercept_num:\n"))
cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", mean(nd_eff_int_num), sd(nd_eff_int_num)))
cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", mean(draws_stan$effective_intercept_num), sd(draws_stan$effective_intercept_num)))

se_eff_num <- sqrt(sd(nd_eff_int_num)^2/length(nd_eff_int_num) + sd(draws_stan$effective_intercept_num)^2/length(draws_stan$effective_intercept_num))
diff_eff_num <- abs(mean(nd_eff_int_num) - mean(draws_stan$effective_intercept_num)) / se_eff_num
cat(sprintf("  Diff: %.2f SE => %s\n", diff_eff_num, if(diff_eff_num < 2) "PASS" else "FAIL"))

cat(sprintf("\neffective_intercept_denom:\n"))
cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", mean(nd_eff_int_denom), sd(nd_eff_int_denom)))
cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", mean(draws_stan$effective_intercept_denom), sd(draws_stan$effective_intercept_denom)))

se_eff_denom <- sqrt(sd(nd_eff_int_denom)^2/length(nd_eff_int_denom) + sd(draws_stan$effective_intercept_denom)^2/length(draws_stan$effective_intercept_denom))
diff_eff_denom <- abs(mean(nd_eff_int_denom) - mean(draws_stan$effective_intercept_denom)) / se_eff_denom
cat(sprintf("  Diff: %.2f SE => %s\n", diff_eff_denom, if(diff_eff_denom < 2) "PASS" else "FAIL"))
