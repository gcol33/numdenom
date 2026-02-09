// Joint NegBin-NegBin model with CROSSED random effects
// Matches numdenom ratiod_negbin_negbin() with (1|site) + (1|site2)
// Row 34 in gradient_methods.md
//
// Two independent RE groups, both SHARED between num and denom

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;      // Numerator counts (NegBin)
  array[N] int<lower=0> y_denom;    // Denominator counts (NegBin)
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
  real<lower=0> phi_num;            // NegBin overdispersion for numerator
  real<lower=0> phi_denom;          // NegBin overdispersion for denominator

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
  phi_num ~ gamma(2, 0.1);
  phi_denom ~ gamma(2, 0.1);

  // RE priors
  z_re1 ~ std_normal();
  z_re2 ~ std_normal();
  sigma_re1 ~ cauchy(0, 2.5);
  sigma_re2 ~ cauchy(0, 2.5);

  // Likelihoods
  y_num ~ neg_binomial_2(exp(eta_num), phi_num);
  y_denom ~ neg_binomial_2(exp(eta_denom), phi_denom);
}

generated quantities {
  vector[N] log_lik;

  for (n in 1:N) {
    real eta_num_n = X[n] * beta_num + re1[group_idx1[n]] + re2[group_idx2[n]];
    real eta_denom_n = X[n] * beta_denom + re1[group_idx1[n]] + re2[group_idx2[n]];

    log_lik[n] = neg_binomial_2_lpmf(y_num[n] | exp(eta_num_n), phi_num) +
                 neg_binomial_2_lpmf(y_denom[n] | exp(eta_denom_n), phi_denom);
  }
}
