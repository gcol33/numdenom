// Joint Lognormal model with SHARED RE + ICAR + RW1
// Row 102 in gradient_methods.md

data {
  int<lower=1> N;
  vector<lower=0>[N] y_num;
  vector<lower=0>[N] y_denom;
  int<lower=1> p;
  matrix[N, p] X;
  int<lower=1> n_groups;
  array[N] int<lower=1,upper=n_groups> group_idx;
  int<lower=1> J;
  array[N] int<lower=1,upper=J> spatial_idx;
  array[J] int<lower=0> n_neighbors;
  int<lower=0> n_edges;
  array[n_edges] int<lower=1,upper=J> edge1;
  array[n_edges] int<lower=1,upper=J> edge2;
  int<lower=2> T;
  array[N] int<lower=1,upper=T> time_idx;
}

parameters {
  vector[p] beta_num;
  vector[p] beta_denom;
  real<lower=0> sigma_num;
  real<lower=0> sigma_denom;
  vector[n_groups] z_re;
  real<lower=0> sigma_re;
  vector[J] phi_spatial;
  real<lower=0> tau_spatial;
  vector[T] phi_temporal;
  real<lower=0> tau_temporal;
}

transformed parameters {
  vector[n_groups] re = sigma_re * z_re;
}

model {
  // Priors - must match numdenom
  // beta: normal(0, 10)
  // sigma (lognormal phi): gamma(2, 2) with mode=0.5, mean=1
  // sigma_re: half-cauchy(2.5)
  // tau_spatial, tau_temporal: gamma(1, 0.01)
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  sigma_num ~ gamma(2, 2);
  sigma_denom ~ gamma(2, 2);
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);
  tau_spatial ~ gamma(1, 0.01);
  tau_temporal ~ gamma(1, 0.01);

  // ICAR prior
  target += -0.5 * tau_spatial * dot_self(phi_spatial[edge1] - phi_spatial[edge2]);
  sum(phi_spatial) ~ normal(0, 0.001 * J);

  // RW1 prior
  for (t in 2:T) {
    phi_temporal[t] ~ normal(phi_temporal[t-1], 1/sqrt(tau_temporal));
  }
  sum(phi_temporal) ~ normal(0, 0.001 * T);

  // Likelihoods
  for (n in 1:N) {
    real eta_num = X[n] * beta_num + re[group_idx[n]] + phi_spatial[spatial_idx[n]] + phi_temporal[time_idx[n]];
    real eta_denom = X[n] * beta_denom + re[group_idx[n]] + phi_spatial[spatial_idx[n]] + phi_temporal[time_idx[n]];
    if (y_num[n] > 0) {
      target += lognormal_lpdf(y_num[n] | eta_num, sigma_num);
    }
    if (y_denom[n] > 0) {
      target += lognormal_lpdf(y_denom[n] | eta_denom, sigma_denom);
    }
  }
}
