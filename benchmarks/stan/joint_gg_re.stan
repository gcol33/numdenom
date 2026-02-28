// Joint Gamma-Gamma model with SHARED random effects
// Matches numdenom ratiod_gamma_gamma() with (1|site)
// Row 94 in gradient_methods.md

data {
  int<lower=1> N;
  vector<lower=0>[N] y_num;
  vector<lower=0>[N] y_denom;
  int<lower=1> p;
  matrix[N, p] X;
  int<lower=1> n_groups;
  array[N] int<lower=1,upper=n_groups> group_idx;
}

parameters {
  vector[p] beta_num;
  vector[p] beta_denom;
  real<lower=0> shape_num;
  real<lower=0> shape_denom;
  vector[n_groups] z_re;
  real<lower=0> sigma_re;
}

transformed parameters {
  vector[n_groups] re = sigma_re * z_re;
}

model {
  vector[N] eta_num = X * beta_num;
  vector[N] eta_denom = X * beta_denom;

  for (n in 1:N) {
    eta_num[n] += re[group_idx[n]];
    eta_denom[n] += re[group_idx[n]];
  }

  vector[N] mu_num = exp(eta_num);
  vector[N] mu_denom = exp(eta_denom);

  // Priors - match numdenom: Gamma(2, 0.5) with mean=4, mode=2
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  shape_num ~ gamma(2, 0.5);
  shape_denom ~ gamma(2, 0.5);
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  // Likelihoods
  for (n in 1:N) {
    if (y_num[n] > 0) {
      target += gamma_lpdf(y_num[n] | shape_num, shape_num / mu_num[n]);
    }
    if (y_denom[n] > 0) {
      target += gamma_lpdf(y_denom[n] | shape_denom, shape_denom / mu_denom[n]);
    }
  }
}
