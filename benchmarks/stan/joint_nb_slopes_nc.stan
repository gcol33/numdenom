
data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;
  array[N] int<lower=0> y_denom;
  matrix[N, 2] X;
  int<lower=1> n_groups;
  array[N] int<lower=1,upper=n_groups> group_idx;
  vector[N] x_slope;
}
transformed data {
  real sigma_re_scale = 2.5;
}
parameters {
  vector[2] beta_num;
  vector[2] beta_denom;
  real<lower=0> phi_num;
  real<lower=0> phi_denom;
  real<lower=0> sigma_intercept;
  real<lower=0> sigma_slope;
  real<lower=-1, upper=1> L21;
  matrix[n_groups, 2] z;
}
transformed parameters {
  matrix[2, 2] L_Omega;
  matrix[n_groups, 2] re;
  L_Omega[1, 1] = 1.0;
  L_Omega[1, 2] = 0.0;
  L_Omega[2, 1] = L21;
  L_Omega[2, 2] = sqrt(1.0 - L21 * L21);
  for (g in 1:n_groups) {
    vector[2] Lz;
    Lz[1] = L_Omega[1, 1] * z[g, 1];
    Lz[2] = L_Omega[2, 1] * z[g, 1] + L_Omega[2, 2] * z[g, 2];
    re[g, 1] = sigma_intercept * Lz[1];
    re[g, 2] = sigma_slope * Lz[2];
  }
}
model {
  vector[N] eta_num;
  vector[N] eta_denom;
  eta_num = X * beta_num;
  eta_denom = X * beta_denom;
  for (n in 1:N) {
    int g = group_idx[n];
    real re_effect = re[g, 1] + re[g, 2] * x_slope[n];
    eta_num[n] += re_effect;
    eta_denom[n] += re_effect;
  }
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  phi_num ~ gamma(2, 0.1);
  phi_denom ~ gamma(2, 0.1);
  sigma_intercept ~ cauchy(0, sigma_re_scale);
  sigma_slope ~ cauchy(0, sigma_re_scale);
  target += 1.5 * log(1 - L21 * L21);
  to_vector(z) ~ std_normal();
  y_num ~ neg_binomial_2_log(eta_num, phi_num);
  y_denom ~ neg_binomial_2_log(eta_denom, phi_denom);
}
generated quantities {
  real mean_re_int = mean(re[,1]);
  real mean_re_slope = mean(re[,2]);
}

