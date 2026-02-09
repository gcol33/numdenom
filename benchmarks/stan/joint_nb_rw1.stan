// Joint NegBin-NegBin model with SHARED random effects + RW1 temporal
// Matches numdenom ratiod_negbin_negbin() with (1|site) + temporal_rw1()
// Row 41 in gradient_methods.md
//
// KEY INSIGHT: Both RE and temporal effects are SHARED between num and denom
//
// RW1 prior: phi[t] - phi[t-1] ~ N(0, 1/tau)

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;         // Numerator counts (NegBin)
  array[N] int<lower=0> y_denom;       // Denominator counts (NegBin)
  int<lower=1> p;                      // Number of predictors
  matrix[N, p] X;                      // Design matrix (shared)

  // Random effects
  int<lower=1> n_groups;               // Number of RE groups (sites)
  array[N] int<lower=1,upper=n_groups> group_idx;  // Group assignment

  // Temporal structure
  int<lower=2> T;                      // Number of time points
  array[N] int<lower=1,upper=T> time_idx;  // Time point for each obs
}

parameters {
  vector[p] beta_num;                  // Numerator coefficients
  vector[p] beta_denom;                // Denominator coefficients
  real<lower=0> phi_num;               // NegBin overdispersion (numerator)
  real<lower=0> phi_denom;             // NegBin overdispersion (denominator)

  // SHARED random effects (non-centered)
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

  // Linear predictors with SHARED RE and SHARED temporal
  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  for (n in 1:N) {
    eta_num[n] += re[group_idx[n]] + phi_temporal[time_idx[n]];
    eta_denom[n] += re[group_idx[n]] + phi_temporal[time_idx[n]];
  }

  // Priors matching numdenom defaults
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  phi_num ~ gamma(2, 0.1);
  phi_denom ~ gamma(2, 0.1);

  // RE priors
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  // Temporal priors (matching numdenom: tau ~ Gamma(1, 0.01))
  tau_temporal ~ gamma(1, 0.01);

  // RW1 prior
  for (t in 2:T) {
    target += normal_lpdf(phi_temporal[t] | phi_temporal[t-1], 1/sqrt(tau_temporal));
  }
  sum(phi_temporal) ~ normal(0, 0.001 * T);

  // Means
  mu_num = exp(eta_num);
  mu_denom = exp(eta_denom);

  // Numerator: NegBin
  y_num ~ neg_binomial_2(mu_num, phi_num);

  // Denominator: NegBin
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
