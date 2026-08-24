// Joint Poisson-Gamma model with SHARED random effects + RW1 temporal
// Matches numdenom ratiod_poisson_gamma() with (1|site) + temporal_rw1()
// Row 11 in gradient_methods.md
//
// KEY INSIGHT: Both RE and temporal effects are SHARED between num and denom
//
// RW1 prior: phi[t] - phi[t-1] ~ N(0, 1/tau)
// p(phi|tau) propto tau^{(T-1)/2} exp(-0.5 * tau * sum((phi[t] - phi[t-1])^2))

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;         // Numerator counts (Poisson)
  vector<lower=0>[N] y_denom;          // Denominator (Gamma-distributed effort)
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
  real<lower=0> shape;                 // Gamma shape parameter

  // SHARED random effects (non-centered)
  vector[n_groups] z_re;               // Standard normal RE
  real<lower=0> sigma_re;              // RE standard deviation

  // SHARED temporal effects
  vector[T] phi_temporal;              // Temporal effects
  real<lower=0> tau_temporal;          // Temporal precision
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
  shape ~ gamma(2, 0.1);

  // RE priors
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  // Temporal priors (matching numdenom: tau ~ Gamma(1, 0.01))
  tau_temporal ~ gamma(1, 0.01);

  // RW1 prior: first differences
  // phi[1] gets implicit N(0, large variance) prior
  // phi[t] - phi[t-1] ~ N(0, 1/tau)
  for (t in 2:T) {
    target += normal_lpdf(phi_temporal[t] | phi_temporal[t-1], 1/sqrt(tau_temporal));
  }

  // Soft sum-to-zero constraint for identifiability
  sum(phi_temporal) ~ normal(0, 0.001 * T);

  // Means
  mu_num = exp(eta_num);
  mu_denom = exp(eta_denom);

  // Numerator: Poisson
  y_num ~ poisson(mu_num);

  // Denominator: Gamma
  for (n in 1:N) {
    if (y_denom[n] > 0) {
      target += gamma_lpdf(y_denom[n] | shape, shape / mu_denom[n]);
    }
  }
}

generated quantities {
  vector[N] log_lik;

  for (n in 1:N) {
    real eta_num_n = X[n] * beta_num + re[group_idx[n]] + phi_temporal[time_idx[n]];
    real eta_denom_n = X[n] * beta_denom + re[group_idx[n]] + phi_temporal[time_idx[n]];
    real mu_num_n = exp(eta_num_n);
    real mu_denom_n = exp(eta_denom_n);

    log_lik[n] = poisson_lpmf(y_num[n] | mu_num_n);
    if (y_denom[n] > 0) {
      log_lik[n] += gamma_lpdf(y_denom[n] | shape, shape / mu_denom_n);
    }
  }
}
