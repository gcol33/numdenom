# Validation of numdenom negbin_negbin + random intercepts (no slopes) against Stan
# Row 2 in gradient_methods.md - simpler model to isolate the issue

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

# Smaller test
N_OBS <- 200
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 4
N_SITES <- 20

cat("=======================================================\n")
cat("Joint Model Validation: Row 2 (negbin_negbin + RE only)\n")
cat("=======================================================\n")

# Helper function to compare posteriors
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
    nd_mean = nd_mean,
    stan_mean = stan_mean,
    diff = diff,
    ratio = ratio,
    pass = pass
  )
}

print_result <- function(result) {
  cat(sprintf("\n%s:\n", result$param))
  cat(sprintf("  numdenom: %.4f\n", result$nd_mean))
  cat(sprintf("  Stan:     %.4f\n", result$stan_mean))
  cat(sprintf("  Diff: %.4f (%.2f SE) => %s\n",
              result$diff, result$ratio, if(result$pass) "PASS" else "FAIL"))
}

results <- list()

# Generate shared data
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)

# Generate data WITH known random effects (intercept only)
true_sigma_re <- 0.5
true_beta_num <- c(2, 0.5)
true_beta_denom <- c(3, 0.2)
true_phi_num <- 5.0
true_phi_denom <- 8.0

# Generate RE
true_re <- rnorm(N_SITES, 0, true_sigma_re)

# Compute linear predictors with SHARED RE
eta_num <- true_beta_num[1] + true_beta_num[2] * x
eta_denom <- true_beta_denom[1] + true_beta_denom[2] * x
for (i in 1:N_OBS) {
  g <- as.integer(site[i])
  eta_num[i] <- eta_num[i] + true_re[g]
  eta_denom[i] <- eta_denom[i] + true_re[g]
}

# Generate NegBin data
y_num <- rnbinom(N_OBS, size = true_phi_num, mu = exp(eta_num))
y_denom <- rnbinom(N_OBS, size = true_phi_denom, mu = exp(eta_denom))
y_denom[y_denom == 0] <- 1  # Same for both

df <- data.frame(y_num = y_num, y_denom = y_denom, x = x, site = site)

cat(sprintf("True: sigma_re=%.2f, phi_num=%.2f, phi_denom=%.2f\n",
            true_sigma_re, true_phi_num, true_phi_denom))

# Fit numdenom (intercept-only RE)
cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y_num | y_denom ~ x + (1|site),
    data = df,
    family = ratiod_negbin_negbin(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_nd))

# Simple Stan model for intercept-only RE
stan_code <- "
data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;
  array[N] int<lower=0> y_denom;
  int<lower=1> p;
  matrix[N, p] X;
  int<lower=1> n_groups;
  array[N] int<lower=1,upper=n_groups> group_idx;
}
parameters {
  vector[p] beta_num;
  vector[p] beta_denom;
  real<lower=0> phi_num;
  real<lower=0> phi_denom;
  real<lower=0> sigma_re;
  vector[n_groups] re;  // SHARED random intercepts
}
model {
  vector[N] eta_num;
  vector[N] eta_denom;

  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  for (n in 1:N) {
    eta_num[n] += re[group_idx[n]];
    eta_denom[n] += re[group_idx[n]];  // SHARED
  }

  // Priors matching numdenom defaults
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  phi_num ~ gamma(2, 0.1);
  phi_denom ~ gamma(2, 0.1);
  sigma_re ~ cauchy(0, 2.5);
  re ~ normal(0, sigma_re);

  // Likelihood
  y_num ~ neg_binomial_2_log(eta_num, phi_num);
  y_denom ~ neg_binomial_2_log(eta_denom, phi_denom);
}
"

# Write and compile Stan model
stan_file <- tempfile(fileext = ".stan")
writeLines(stan_code, stan_file)

cat("Compiling Stan model... ")
stan_model <- cmdstan_model(stan_file)
cat("done\n")

stan_data <- list(
  N = N_OBS,
  y_num = df$y_num,
  y_denom = df$y_denom,
  p = 2,
  X = cbind(1, df$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df$site)
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model$sample(
    data = stan_data,
    iter_sampling = N_ITER - N_WARMUP,
    iter_warmup = N_WARMUP,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    refresh = 0,
    show_messages = FALSE
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_stan))

# Extract draws
draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

cat("\n--- Parameter Comparisons ---\n")

# For non-centered, need to transform z to RE
nd_cols <- colnames(draws_nd)
nd_sigma_re <- draws_nd[, "sigma_re"]
re_cols_nd <- nd_cols[grep("^re\\[[0-9]+\\]$", nd_cols)]
n_groups <- length(re_cols_nd)
n_draws <- nrow(draws_nd)

# Transform z to RE: re = sigma_re * z (non-centered)
re_nd <- matrix(0, n_draws, n_groups)
for (g in seq_len(n_groups)) {
  z_g <- draws_nd[, re_cols_nd[g]]
  re_nd[, g] <- nd_sigma_re * z_g
}
nd_mean_re <- rowMeans(re_nd)

# Stan RE (centered)
stan_re_cols <- grep("^re\\[[0-9]+\\]$", names(draws_stan), value = TRUE)
stan_mean_re <- if (length(stan_re_cols) > 0) rowMeans(draws_stan[, stan_re_cols, drop=FALSE]) else 0

# Effective intercept
results$eff_int_num <- compare_posteriors(
  draws_nd[, "beta_num[1]"] + nd_mean_re,
  draws_stan$`beta_num[1]` + stan_mean_re,
  "effective_intercept_num"
)
print_result(results$eff_int_num)

results$eff_int_denom <- compare_posteriors(
  draws_nd[, "beta_denom[1]"] + nd_mean_re,
  draws_stan$`beta_denom[1]` + stan_mean_re,
  "effective_intercept_denom"
)
print_result(results$eff_int_denom)

results$beta_num2 <- compare_posteriors(
  draws_nd[, "beta_num[2]"],
  draws_stan$`beta_num[2]`,
  "beta_num[2]"
)
print_result(results$beta_num2)

results$beta_denom2 <- compare_posteriors(
  draws_nd[, "beta_denom[2]"],
  draws_stan$`beta_denom[2]`,
  "beta_denom[2]"
)
print_result(results$beta_denom2)

results$phi_num <- compare_posteriors(
  draws_nd[, "phi_num"],
  draws_stan$phi_num,
  "phi_num"
)
print_result(results$phi_num)

results$phi_denom <- compare_posteriors(
  draws_nd[, "phi_denom"],
  draws_stan$phi_denom,
  "phi_denom"
)
print_result(results$phi_denom)

results$sigma_re <- compare_posteriors(
  nd_sigma_re,
  draws_stan$sigma_re,
  "sigma_re"
)
print_result(results$sigma_re)

cat("\n--- Raw Values ---\n")
cat(sprintf("Raw beta_num[1]: nd=%.4f stan=%.4f\n",
            mean(draws_nd[, "beta_num[1]"]), mean(draws_stan$`beta_num[1]`)))
cat(sprintf("Mean RE (transformed): nd=%.4f stan=%.4f\n",
            mean(nd_mean_re), mean(stan_mean_re)))

cat(sprintf("\n  numdenom time: %.1fs\n", t_nd))
cat(sprintf("  Stan time: %.1fs\n", t_stan))

# Summary
cat("\n=======================================================\n")
n_pass <- sum(sapply(results, function(r) r$pass))
n_total <- length(results)
cat(sprintf("Overall: %d/%d parameters PASS (within 2 SE)\n", n_pass, n_total))
