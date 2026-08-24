
data {
  int<lower=1> N;
  vector<lower=0>[N] y_num;
  vector<lower=0>[N] y_denom;
}

parameters {
  real beta_num;
  real beta_denom;
  real<lower=0> shape_num;
  real<lower=0> shape_denom;
}

model {
  vector[N] mu_num = exp(rep_vector(beta_num, N));
  vector[N] mu_denom = exp(rep_vector(beta_denom, N));

  // Priors
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  shape_num ~ gamma(2, 0.5);
  shape_denom ~ gamma(2, 0.5);

  // Likelihoods
  y_num ~ gamma(shape_num, shape_num ./ mu_num);
  y_denom ~ gamma(shape_denom, shape_denom ./ mu_denom);
}

