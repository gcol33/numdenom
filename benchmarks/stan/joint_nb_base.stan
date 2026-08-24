// Joint Negative Binomial - Negative Binomial model (no RE)
// Matches numdenom ratiod_negbin_negbin() exactly
// Row 31 in gradient_methods.md
//
// KEY DIFFERENCE from brms: Both num AND denom are modeled as random variables
// - Numerator: NegBin(mu_num, phi_num)
// - Denominator: NegBin(mu_denom, phi_denom)
//
// brms offset(log(denom)) treats denom as FIXED - fundamentally different model!

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;      // Numerator counts
  array[N] int<lower=0> y_denom;    // Denominator counts
  int<lower=1> p;                   // Number of predictors (same for num and denom)
  matrix[N, p] X;                   // Design matrix (shared)
}

parameters {
  vector[p] beta_num;               // Numerator coefficients
  vector[p] beta_denom;             // Denominator coefficients
  real<lower=0> phi_num;            // NegBin overdispersion for numerator
  real<lower=0> phi_denom;          // NegBin overdispersion for denominator
}

model {
  vector[N] eta_num;
  vector[N] eta_denom;
  vector[N] mu_num;
  vector[N] mu_denom;

  // Linear predictors (same X for both, different coefficients)
  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  // Priors matching numdenom defaults
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  phi_num ~ gamma(2, 0.1);
  phi_denom ~ gamma(2, 0.1);

  // Means
  mu_num = exp(eta_num);
  mu_denom = exp(eta_denom);

  // Likelihoods - BOTH num and denom are random
  y_num ~ neg_binomial_2(mu_num, phi_num);
  y_denom ~ neg_binomial_2(mu_denom, phi_denom);
}

generated quantities {
  vector[N] log_lik;
  vector[N] eta_num_out;
  vector[N] eta_denom_out;

  eta_num_out = X * beta_num;
  eta_denom_out = X * beta_denom;

  for (n in 1:N) {
    real mu_num_n = exp(eta_num_out[n]);
    real mu_denom_n = exp(eta_denom_out[n]);
    log_lik[n] = neg_binomial_2_lpmf(y_num[n] | mu_num_n, phi_num) +
                 neg_binomial_2_lpmf(y_denom[n] | mu_denom_n, phi_denom);
  }
}
