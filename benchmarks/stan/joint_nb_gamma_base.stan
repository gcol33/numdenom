// Joint NegBin-Gamma model (no RE)
// Matches numdenom ratiod_negbin_gamma() exactly
// Row 108 in gradient_methods.md
//
// Hybrid of NB-NB (numerator) and PG (denominator):
// - Numerator: NegBin(mu_num, phi_num) -- overdispersed counts
// - Denominator: Gamma(phi_denom, phi_denom / mu_denom) -- continuous effort
//
// Both num AND denom are modeled as random variables with independent linear predictors.
// The ratio is computed post-hoc as exp(eta_num - eta_denom).

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;      // Numerator counts (NegBin)
  vector<lower=0>[N] y_denom;       // Denominator (Gamma-distributed effort)
  int<lower=1> p;                   // Number of predictors (same for num and denom)
  matrix[N, p] X;                   // Design matrix (shared)
}

parameters {
  vector[p] beta_num;               // Numerator coefficients
  vector[p] beta_denom;             // Denominator coefficients
  real<lower=0> phi_num;            // NegBin overdispersion for numerator
  real<lower=0> phi_denom;          // Gamma shape parameter for denominator
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
  phi_num ~ gamma(2, 0.1);          // NB overdispersion prior
  phi_denom ~ gamma(2, 0.1);        // Gamma shape prior

  // Means
  mu_num = exp(eta_num);
  mu_denom = exp(eta_denom);

  // Numerator: Negative Binomial
  y_num ~ neg_binomial_2(mu_num, phi_num);

  // Denominator: Gamma (shape=phi_denom, rate=phi_denom/mu)
  // Parameterized so E[y_denom] = mu_denom, Var[y_denom] = mu_denom^2/phi_denom
  for (n in 1:N) {
    if (y_denom[n] > 0) {
      target += gamma_lpdf(y_denom[n] | phi_denom, phi_denom / mu_denom[n]);
    }
  }
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
    log_lik[n] = neg_binomial_2_lpmf(y_num[n] | mu_num_n, phi_num);
    if (y_denom[n] > 0) {
      log_lik[n] += gamma_lpdf(y_denom[n] | phi_denom, phi_denom / mu_denom_n);
    }
  }
}
