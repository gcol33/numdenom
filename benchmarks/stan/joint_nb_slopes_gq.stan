
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
  matrix[n_groups, 2] re;
}
transformed parameters {
  matrix[2, 2] L_Omega;
  L_Omega[1, 1] = 1.0;
  L_Omega[1, 2] = 0.0;
  L_Omega[2, 1] = L21;
  L_Omega[2, 2] = sqrt(1.0 - L21 * L21);
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

  // MVN prior on RE (centered)
  for (g in 1:n_groups) {
    real y1 = re[g, 1] / sigma_intercept;
    real y2 = re[g, 2] / sigma_slope;
    real L22 = L_Omega[2, 2];
    real z1 = y1;
    real z2 = (y2 - L21 * z1) / L22;
    target += -0.5 * (z1 * z1 + z2 * z2);
  }
  target += -n_groups * (log(sigma_intercept) + log(sigma_slope) + log(L_Omega[2, 2]));

  y_num ~ neg_binomial_2_log(eta_num, phi_num);
  y_denom ~ neg_binomial_2_log(eta_denom, phi_denom);
}
generated quantities {
  real mean_re_int = mean(re[,1]);
  real mean_re_slope = mean(re[,2]);
  real eff_int_num = beta_num[1] + mean_re_int;
  real eff_int_denom = beta_denom[1] + mean_re_int;
  real eff_slope_num = beta_num[2] + mean_re_slope;
  real eff_slope_denom = beta_denom[2] + mean_re_slope;
}

