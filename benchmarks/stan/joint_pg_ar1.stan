// Joint Poisson-Gamma model with SHARED random effects + AR1 temporal
// Matches numdenom ratiod_poisson_gamma() with (1|site) + temporal_ar1()
// Row 13 in gradient_methods.md
//
// AR1 prior:
// phi[1] ~ N(0, 1/(tau*(1-rho^2)))  [stationary distribution]
// phi[t] | phi[t-1] ~ N(rho * phi[t-1], 1/tau)

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;
  vector<lower=0>[N] y_denom;
  int<lower=1> p;
  matrix[N, p] X;

  // Random effects
  int<lower=1> n_groups;
  array[N] int<lower=1,upper=n_groups> group_idx;

  // Temporal structure
  int<lower=2> T;
  array[N] int<lower=1,upper=T> time_idx;
}

parameters {
  vector[p] beta_num;
  vector[p] beta_denom;
  real<lower=0> shape;

  // SHARED random effects
  vector[n_groups] z_re;
  real<lower=0> sigma_re;

  // SHARED AR1 temporal
  vector[T] phi_temporal;
  real<lower=0> tau_temporal;
  real<lower=-1,upper=1> rho_ar1;
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

  // Priors
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  shape ~ gamma(2, 0.1);

  // RE priors
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  // AR1 temporal priors (matching numdenom HMC backend defaults)
  tau_temporal ~ gamma(1, 0.01);
  // Uniform prior on rho_ar1 via constraints

  // AR1 process
  // First time point: stationary distribution
  {
    real var_stationary = 1.0 / (tau_temporal * (1 - square(rho_ar1)));
    phi_temporal[1] ~ normal(0, sqrt(var_stationary));
  }

  // Subsequent time points
  for (t in 2:T) {
    phi_temporal[t] ~ normal(rho_ar1 * phi_temporal[t-1], 1/sqrt(tau_temporal));
  }

  // Means
  mu_num = exp(eta_num);
  mu_denom = exp(eta_denom);

  // Likelihoods
  y_num ~ poisson(mu_num);
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
