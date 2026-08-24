// Joint NegBin-NegBin model with SHARED random effects + RW2 temporal
// Matches numdenom ratiod_negbin_negbin() with (1|site) + temporal_rw2()
// Row 42 in gradient_methods.md
//
// KEY INSIGHT: Both RE and temporal effects are SHARED between num and denom
//
// RW2 prior: phi[t] - 2*phi[t-1] + phi[t-2] ~ N(0, 1/tau)
// p(phi|tau) propto tau^{(T-2)/2} exp(-0.5 * tau * sum((phi[t] - 2*phi[t-1] + phi[t-2])^2))

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;
  array[N] int<lower=0> y_denom;
  int<lower=1> p;
  matrix[N, p] X;

  // Random effects
  int<lower=1> n_groups;
  array[N] int<lower=1,upper=n_groups> group_idx;

  // Temporal structure
  int<lower=3> T;
  array[N] int<lower=1,upper=T> time_idx;
}

parameters {
  vector[p] beta_num;
  vector[p] beta_denom;
  real<lower=0> phi_num;
  real<lower=0> phi_denom;

  // SHARED random effects
  vector[n_groups] z_re;
  real<lower=0> sigma_re;

  // SHARED temporal effects
  vector[T] phi_temporal;
  real<lower=0> tau_temporal;
}

transformed parameters {
  vector[n_groups] re;
  re = sigma_re * z_re;
}

model {
  vector[N] eta_num;
  vector[N] eta_denom;
  vector[N] mu_num;
  vector[N] mu_denom;

  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  for (n in 1:N) {
    eta_num[n] += re[group_idx[n]] + phi_temporal[time_idx[n]];
    eta_denom[n] += re[group_idx[n]] + phi_temporal[time_idx[n]];
  }

  // Priors
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  phi_num ~ gamma(2, 0.1);
  phi_denom ~ gamma(2, 0.1);

  // RE priors
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  // Temporal priors (matching numdenom: tau ~ Gamma(1, 0.01))
  tau_temporal ~ gamma(1, 0.01);

  // RW2 prior: second differences
  for (t in 3:T) {
    target += normal_lpdf(phi_temporal[t] - 2*phi_temporal[t-1] + phi_temporal[t-2] | 0, 1/sqrt(tau_temporal));
  }

  // Soft sum-to-zero constraint
  sum(phi_temporal) ~ normal(0, 0.001 * T);

  // Means
  mu_num = exp(eta_num);
  mu_denom = exp(eta_denom);

  // Likelihoods
  y_num ~ neg_binomial_2(mu_num, phi_num);
  y_denom ~ neg_binomial_2(mu_denom, phi_denom);
}

generated quantities {
  vector[N] log_lik;

  for (n in 1:N) {
    real eta_num_n = X[n] * beta_num + re[group_idx[n]] + phi_temporal[time_idx[n]];
    real eta_denom_n = X[n] * beta_denom + re[group_idx[n]] + phi_temporal[time_idx[n]];
    real mu_num_n = exp(eta_num_n);
    real mu_denom_n = exp(eta_denom_n);

    log_lik[n] = neg_binomial_2_lpmf(y_num[n] | mu_num_n, phi_num)
               + neg_binomial_2_lpmf(y_denom[n] | mu_denom_n, phi_denom);
  }
}
