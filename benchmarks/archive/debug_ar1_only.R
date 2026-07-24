# Debug: Test AR1 only (without spatial) to isolate the issue

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

N_OBS <- 200
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2
N_SITES <- 20
N_TIMES <- 10

cat("=== Debug AR1 Only (no spatial) ===\n\n")

site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)
time <- rep(1:N_TIMES, length.out = N_OBS)
time_factor <- factor(time)

eta_num <- 2 + 0.3 * x
eta_denom <- 4 + 0.2 * x

df <- data.frame(
  y = rnbinom(N_OBS, mu = exp(eta_num), size = 5),
  denom = rnbinom(N_OBS, mu = exp(eta_denom), size = 10),
  x = x, site = site, time = time, time_factor = time_factor
)
df$denom[df$denom == 0] <- 1

cat("Fitting numdenom (AR1 only)...\n")
fit_nd <- tratio(
  y | denom ~ x + (1|site), data = df, family = ratiod_negbin_negbin(),
  temporal = temporal_ar1("time_factor"),
  control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE)
)
draws_nd <- as.matrix(fit_nd$draws)

cat("Fitting Stan (AR1 only)...\n")
stan_code <- "
data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;
  array[N] int<lower=0> y_denom;
  int<lower=1> p;
  matrix[N, p] X;
  int<lower=1> n_groups;
  array[N] int<lower=1,upper=n_groups> group_idx;
  int<lower=2> T;
  array[N] int<lower=1,upper=T> time_idx;
}
parameters {
  vector[p] beta_num;
  vector[p] beta_denom;
  real<lower=0> phi_num;
  real<lower=0> phi_denom;
  vector[n_groups] z_re;
  real<lower=0> sigma_re;
  vector[T] phi_temporal;
  real<lower=0> tau_temporal;
  real<lower=-1,upper=1> rho_ar1;
}
transformed parameters {
  vector[n_groups] re = sigma_re * z_re;
}
model {
  vector[N] eta_num = X * beta_num;
  vector[N] eta_denom = X * beta_denom;

  for (n in 1:N) {
    eta_num[n] += re[group_idx[n]] + phi_temporal[time_idx[n]];
    eta_denom[n] += re[group_idx[n]] + phi_temporal[time_idx[n]];
  }

  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  phi_num ~ gamma(2, 0.1);
  phi_denom ~ gamma(2, 0.1);
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  tau_temporal ~ gamma(1, 0.01);
  target += log1m(square(rho_ar1));  // Jacobian

  {
    real var_stat = 1.0 / (tau_temporal * (1 - square(rho_ar1)));
    phi_temporal[1] ~ normal(0, sqrt(var_stat));
  }
  for (t in 2:T) {
    phi_temporal[t] ~ normal(rho_ar1 * phi_temporal[t-1], 1/sqrt(tau_temporal));
  }

  y_num ~ neg_binomial_2(exp(eta_num), phi_num);
  y_denom ~ neg_binomial_2(exp(eta_denom), phi_denom);
}
"
stan_file <- tempfile(fileext = ".stan")
writeLines(stan_code, stan_file)
stan_model <- cmdstan_model(stan_file)

stan_data <- list(
  N = N_OBS, y_num = df$y, y_denom = df$denom, p = 2, X = cbind(1, df$x),
  n_groups = N_SITES, group_idx = as.integer(df$site),
  T = N_TIMES, time_idx = df$time
)
fit_stan <- stan_model$sample(
  data = stan_data, iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
  chains = N_CHAINS, parallel_chains = N_CHAINS, refresh = 0, show_messages = FALSE
)
draws_stan <- fit_stan$draws(format = "df")

cat("\n=== Comparison ===\n")
for (p in c("beta_num[1]", "beta_num[2]", "beta_denom[1]", "beta_denom[2]")) {
  nd_vals <- draws_nd[, p]
  stan_vals <- draws_stan[[p]]
  nd_mean <- mean(nd_vals); nd_sd <- sd(nd_vals)
  stan_mean <- mean(stan_vals); stan_sd <- sd(stan_vals)
  se <- sqrt(nd_sd^2/length(nd_vals) + stan_sd^2/length(stan_vals))
  diff <- abs(nd_mean - stan_mean)
  cat(sprintf("%s: nd=%.4f (%.4f), stan=%.4f (%.4f), diff=%.2f SE => %s\n",
              p, nd_mean, nd_sd, stan_mean, stan_sd, diff/se,
              if(diff/se < 2) "PASS" else "FAIL"))
}

cat("\nTruth: beta_num[2]=0.3, beta_denom[2]=0.2\n")
