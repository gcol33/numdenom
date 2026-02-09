// Joint Lognormal-Lognormal model with SHARED random effects
// Matches numdenom ratiod_lognormal() with (1|site)
// Row 99 in gradient_methods.md

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
  real<lower=0> sigma_num;
  real<lower=0> sigma_denom;
  vector[n_groups] z_re;
  real<lower=0> sigma_re;
}

transformed parameters {
  vector[n_groups] re = sigma_re * z_re;
}

model {
  // Priors - matching numdenom HMC backend exactly
  // beta: normal(0, 10) - numdenom sigma_beta default
  // sigma (lognormal phi): gamma(2, 2) with mode=0.5, mean=1
  // sigma_re: half-cauchy(0, 2.5) - numdenom sigma_re_scale default
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  sigma_num ~ gamma(2, 2);
  sigma_denom ~ gamma(2, 2);
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);  // Half-Cauchy (truncated at 0 via <lower=0>)

  // Lognormal likelihoods with shared RE
  for (n in 1:N) {
    real eta_num = X[n] * beta_num + re[group_idx[n]];
    real eta_denom = X[n] * beta_denom + re[group_idx[n]];
    if (y_num[n] > 0) {
      target += lognormal_lpdf(y_num[n] | eta_num, sigma_num);
    }
    if (y_denom[n] > 0) {
      target += lognormal_lpdf(y_denom[n] | eta_denom, sigma_denom);
    }
  }
}
