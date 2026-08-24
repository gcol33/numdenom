// Joint NegBin-NegBin model with SHARED correlated random slopes
// Matches numdenom ratiod_negbin_negbin() with (1 + x|site)
// Row 33 in gradient_methods.md
//
// KEY INSIGHT: Random slopes are SHARED between num and denom
// Uses LKJ(eta=2) prior on correlation matrix via Cholesky parameterization
// Matches numdenom's CENTERED parameterization exactly

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;        // Numerator counts (NegBin)
  array[N] int<lower=0> y_denom;      // Denominator counts (NegBin)
  int<lower=1> p;                     // Number of predictors
  matrix[N, p] X;                     // Design matrix
  int<lower=1> n_groups;              // Number of RE groups
  array[N] int<lower=1,upper=n_groups> group_idx;
  vector[N] x_slope;                  // Variable for random slope
}

transformed data {
  real sigma_re_scale = 2.5;  // Matches numdenom default
}

parameters {
  vector[p] beta_num;                 // Numerator coefficients
  vector[p] beta_denom;               // Denominator coefficients
  real<lower=0> phi_num;              // NegBin overdispersion (num)
  real<lower=0> phi_denom;            // NegBin overdispersion (denom)

  // Random effects standard deviations (half-Cauchy prior)
  real<lower=0> sigma_intercept;      // SD for random intercepts
  real<lower=0> sigma_slope;          // SD for random slopes

  // Cholesky factor off-diagonal element
  real<lower=-1, upper=1> L21;

  // Random effects (CENTERED parameterization matching numdenom)
  matrix[n_groups, 2] re;
}

transformed parameters {
  // Build 2x2 Cholesky factor L
  matrix[2, 2] L_Omega;
  L_Omega[1, 1] = 1.0;
  L_Omega[1, 2] = 0.0;
  L_Omega[2, 1] = L21;
  L_Omega[2, 2] = sqrt(1.0 - L21 * L21);
}

model {
  vector[N] eta_num;
  vector[N] eta_denom;

  // Linear predictors with SHARED random intercepts and slopes
  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  for (n in 1:N) {
    int g = group_idx[n];
    real re_effect = re[g, 1] + re[g, 2] * x_slope[n];
    eta_num[n] += re_effect;
    eta_denom[n] += re_effect;  // SHARED effect
  }

  // Priors matching numdenom defaults
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  phi_num ~ gamma(2, 0.1);
  phi_denom ~ gamma(2, 0.1);

  // Half-Cauchy(0, 2.5) priors on sigma
  sigma_intercept ~ cauchy(0, sigma_re_scale);
  sigma_slope ~ cauchy(0, sigma_re_scale);

  // LKJ(eta=2) prior on correlation matrix
  target += 1.5 * log(1 - L21 * L21);

  // MVN prior on RE (CENTERED parameterization)
  for (g in 1:n_groups) {
    real y1 = re[g, 1] / sigma_intercept;
    real y2 = re[g, 2] / sigma_slope;
    real L22 = L_Omega[2, 2];
    real z1 = y1;
    real z2 = (y2 - L21 * z1) / L22;
    target += -0.5 * (z1 * z1 + z2 * z2);
  }
  target += -n_groups * (log(sigma_intercept) + log(sigma_slope) + log(L_Omega[2, 2]));

  // Likelihood
  y_num ~ neg_binomial_2_log(eta_num, phi_num);
  y_denom ~ neg_binomial_2_log(eta_denom, phi_denom);
}

generated quantities {
  real rho = L21;  // Correlation coefficient

  vector[N] log_lik;
  for (n in 1:N) {
    int g = group_idx[n];
    real re_effect = re[g, 1] + re[g, 2] * x_slope[n];
    real eta_num_n = X[n] * beta_num + re_effect;
    real eta_denom_n = X[n] * beta_denom + re_effect;

    log_lik[n] = neg_binomial_2_log_lpmf(y_num[n] | eta_num_n, phi_num) +
                 neg_binomial_2_log_lpmf(y_denom[n] | eta_denom_n, phi_denom);
  }
}
