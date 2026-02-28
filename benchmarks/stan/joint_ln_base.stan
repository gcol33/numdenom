// Joint Lognormal-Lognormal model - no random effects
// Matches numdenom ratiod_lognormal() base model
// Row 98 in gradient_methods.md

data {
  int<lower=1> N;
  vector<lower=0>[N] y_num;
  vector<lower=0>[N] y_denom;
  int<lower=1> p;
  matrix[N, p] X;
}

parameters {
  vector[p] beta_num;
  vector[p] beta_denom;
  real<lower=0> sigma_num;
  real<lower=0> sigma_denom;
}

model {
  // Priors - must match numdenom
  // beta: normal(0, 10)
  // sigma (lognormal phi): gamma(2, 2) with mode=0.5, mean=1
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  sigma_num ~ gamma(2, 2);
  sigma_denom ~ gamma(2, 2);

  // Lognormal likelihoods: log(y) ~ N(eta, sigma)
  for (n in 1:N) {
    if (y_num[n] > 0) {
      target += lognormal_lpdf(y_num[n] | X[n] * beta_num, sigma_num);
    }
    if (y_denom[n] > 0) {
      target += lognormal_lpdf(y_denom[n] | X[n] * beta_denom, sigma_denom);
    }
  }
}
