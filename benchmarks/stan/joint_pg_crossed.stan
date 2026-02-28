// Joint Poisson-Gamma model with CROSSED random effects
// Matches numdenom ratiod_poisson_gamma() with (1|site) + (1|site2)
// Row 4 in gradient_methods.md
//
// Two independent RE groups, both SHARED between num and denom

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;      // Numerator counts (Poisson)
  vector<lower=0>[N] y_denom;       // Denominator (Gamma-distributed effort)
  int<lower=1> p;                   // Number of predictors
  matrix[N, p] X;                   // Design matrix (shared)

  // First grouping factor
  int<lower=1> n_groups1;
  array[N] int<lower=1,upper=n_groups1> group_idx1;

  // Second grouping factor (crossed)
  int<lower=1> n_groups2;
  array[N] int<lower=1,upper=n_groups2> group_idx2;
}

parameters {
  vector[p] beta_num;               // Numerator coefficients
  vector[p] beta_denom;             // Denominator coefficients
  real<lower=0> shape;              // Gamma shape parameter

  // SHARED random effects for group 1 (non-centered)
  vector[n_groups1] z_re1;
  real<lower=0> sigma_re1;

  // SHARED random effects for group 2 (non-centered)
  vector[n_groups2] z_re2;
  real<lower=0> sigma_re2;
}

transformed parameters {
  vector[n_groups1] re1 = sigma_re1 * z_re1;
  vector[n_groups2] re2 = sigma_re2 * z_re2;
}

model {
  vector[N] eta_num;
  vector[N] eta_denom;

  // Linear predictors with SHARED crossed random effects
  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  for (n in 1:N) {
    eta_num[n] += re1[group_idx1[n]] + re2[group_idx2[n]];
    eta_denom[n] += re1[group_idx1[n]] + re2[group_idx2[n]];
  }

  // Priors
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  shape ~ gamma(2, 0.1);

  // RE priors
  z_re1 ~ std_normal();
  z_re2 ~ std_normal();
  sigma_re1 ~ cauchy(0, 2.5);
  sigma_re2 ~ cauchy(0, 2.5);

  // Likelihoods
  y_num ~ poisson(exp(eta_num));

  for (n in 1:N) {
    if (y_denom[n] > 0) {
      target += gamma_lpdf(y_denom[n] | shape, shape / exp(eta_denom[n]));
    }
  }
}

generated quantities {
  vector[N] log_lik;

  for (n in 1:N) {
    real eta_num_n = X[n] * beta_num + re1[group_idx1[n]] + re2[group_idx2[n]];
    real eta_denom_n = X[n] * beta_denom + re1[group_idx1[n]] + re2[group_idx2[n]];
    real mu_num_n = exp(eta_num_n);
    real mu_denom_n = exp(eta_denom_n);

    log_lik[n] = poisson_lpmf(y_num[n] | mu_num_n);
    if (y_denom[n] > 0) {
      log_lik[n] += gamma_lpdf(y_denom[n] | shape, shape / mu_denom_n);
    }
  }
}
