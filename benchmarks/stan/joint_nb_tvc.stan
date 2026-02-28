// Joint NegBin-NegBin model with Time-Varying Coefficients (TVC)
// Matches numdenom ratiod_negbin_negbin() with temporal_tvc()
// Row 57 in gradient_methods.md
//
// TVC: coefficient for covariate x varies over time via RW1 prior
// The TVC effect is SHARED between numerator and denominator

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;         // Numerator counts (NegBin)
  array[N] int<lower=0> y_denom;       // Denominator counts (NegBin)
  int<lower=1> p;                      // Number of fixed predictors
  matrix[N, p] X;                      // Design matrix for fixed effects
  vector[N] x_tvc;                     // Covariate with time-varying effect

  // Temporal structure
  int<lower=2> T;                      // Number of time points
  array[N] int<lower=1,upper=T> time_idx;  // Time point for each obs
}

parameters {
  vector[p] beta_num;                  // Numerator fixed effects
  vector[p] beta_denom;                // Denominator fixed effects
  real<lower=0> phi_num;               // Numerator overdispersion
  real<lower=0> phi_denom;             // Denominator overdispersion

  // Time-varying coefficient (RW1 prior)
  vector[T] beta_tvc;                  // TVC coefficient at each time point
  real<lower=0> sigma_tvc;             // TVC innovation SD
}

model {
  vector[N] eta_num;
  vector[N] eta_denom;

  // Linear predictors with TVC
  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  for (n in 1:N) {
    // Add time-varying coefficient effect (SHARED)
    real tvc_effect = beta_tvc[time_idx[n]] * x_tvc[n];
    eta_num[n] += tvc_effect;
    eta_denom[n] += tvc_effect;
  }

  // Priors
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  phi_num ~ gamma(2, 0.1);
  phi_denom ~ gamma(2, 0.1);

  // TVC prior: RW1
  sigma_tvc ~ exponential(1);
  beta_tvc[1] ~ normal(0, 2);  // Prior on first coefficient
  for (t in 2:T) {
    beta_tvc[t] ~ normal(beta_tvc[t-1], sigma_tvc);
  }

  // Likelihood
  // Numerator: NegBin
  for (n in 1:N) {
    y_num[n] ~ neg_binomial_2(exp(eta_num[n]), phi_num);
  }

  // Denominator: NegBin
  for (n in 1:N) {
    if (y_denom[n] > 0) {
      y_denom[n] ~ neg_binomial_2(exp(eta_denom[n]), phi_denom);
    }
  }
}

generated quantities {
  vector[N] log_lik;

  for (n in 1:N) {
    real tvc_effect = beta_tvc[time_idx[n]] * x_tvc[n];
    real eta_num_n = X[n] * beta_num + tvc_effect;
    real eta_denom_n = X[n] * beta_denom + tvc_effect;

    log_lik[n] = neg_binomial_2_lpmf(y_num[n] | exp(eta_num_n), phi_num);
    if (y_denom[n] > 0) {
      log_lik[n] += neg_binomial_2_lpmf(y_denom[n] | exp(eta_denom_n), phi_denom);
    }
  }
}
