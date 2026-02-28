
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
  vector[n_groups] re;  // SHARED RE (centered parameterization)
}
model {
  vector[N] eta_num;
  vector[N] eta_denom;

  // Priors
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  phi_num ~ gamma(2, 0.1);
  phi_denom ~ gamma(2, 0.1);
  sigma_re ~ cauchy(0, 2.5);

  // RE prior (centered)
  re ~ normal(0, sigma_re);

  // Linear predictors with SHARED RE
  eta_num = X * beta_num;
  eta_denom = X * beta_denom;
  for (n in 1:N) {
    eta_num[n] += re[group_idx[n]];
    eta_denom[n] += re[group_idx[n]];
  }

  // Likelihood
  y_num ~ neg_binomial_2_log(eta_num, phi_num);
  y_denom ~ neg_binomial_2_log(eta_denom, phi_denom);
}

