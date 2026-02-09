
data {
  int<lower=1> N;
  vector<lower=0>[N] y_num;
  vector<lower=0>[N] y_denom;
  vector[N] x;
}
parameters {
  real beta_num_0;
  real beta_num_1;
  real beta_denom_0;
  real beta_denom_1;
  real<lower=0> sigma_num;
  real<lower=0> sigma_denom;
}
model {
  beta_num_0 ~ normal(0, 10);
  beta_num_1 ~ normal(0, 10);
  beta_denom_0 ~ normal(0, 10);
  beta_denom_1 ~ normal(0, 10);
  sigma_num ~ gamma(2, 2);
  sigma_denom ~ gamma(2, 2);
  
  for (n in 1:N) {
    y_num[n] ~ lognormal(beta_num_0 + beta_num_1 * x[n], sigma_num);
    y_denom[n] ~ lognormal(beta_denom_0 + beta_denom_1 * x[n], sigma_denom);
  }
}

