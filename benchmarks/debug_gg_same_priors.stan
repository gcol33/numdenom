
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
  real<lower=0> shape_num;
  real<lower=0> shape_denom;
}

model {
  vector[N] mu_num = exp(X * beta_num);
  vector[N] mu_denom = exp(X * beta_denom);

  // Use numdenom priors:
  // - beta ~ normal(0, 10)
  // - shape ~ gamma(2, 0.5)  <- numdenom default for gamma_gamma
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  shape_num ~ gamma(2, 0.5);
  shape_denom ~ gamma(2, 0.5);

  // Likelihoods (same as numdenom)
  for (n in 1:N) {
    if (y_num[n] > 0) {
      target += gamma_lpdf(y_num[n] | shape_num, shape_num / mu_num[n]);
    }
    if (y_denom[n] > 0) {
      target += gamma_lpdf(y_denom[n] | shape_denom, shape_denom / mu_denom[n]);
    }
  }
}

