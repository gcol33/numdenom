// Binomial model with Time-Varying Coefficients (TVC)
// Matches numdenom ratiod_binomial() with temporal_tvc()
// Row 89 in gradient_methods.md
//
// TVC: coefficient for covariate x varies over time via RW1 prior

data {
  int<lower=1> N;
  array[N] int<lower=0> y;             // Successes
  array[N] int<lower=1> n_trials;      // Number of trials
  int<lower=1> p;                      // Number of fixed predictors
  matrix[N, p] X;                      // Design matrix for fixed effects
  vector[N] x_tvc;                     // Covariate with time-varying effect

  // Temporal structure
  int<lower=2> T;                      // Number of time points
  array[N] int<lower=1,upper=T> time_idx;  // Time point for each obs
}

parameters {
  vector[p] beta;                      // Fixed effects

  // Time-varying coefficient (RW1 prior)
  vector[T] beta_tvc;                  // TVC coefficient at each time point
  real<lower=0> sigma_tvc;             // TVC innovation SD
}

model {
  vector[N] eta;

  // Linear predictor with TVC
  eta = X * beta;

  for (n in 1:N) {
    // Add time-varying coefficient effect
    eta[n] += beta_tvc[time_idx[n]] * x_tvc[n];
  }

  // Priors
  beta ~ normal(0, 10);

  // TVC prior: RW1
  sigma_tvc ~ exponential(1);
  beta_tvc[1] ~ normal(0, 2);  // Prior on first coefficient
  for (t in 2:T) {
    beta_tvc[t] ~ normal(beta_tvc[t-1], sigma_tvc);
  }

  // Likelihood: Binomial
  y ~ binomial_logit(n_trials, eta);
}

generated quantities {
  vector[N] log_lik;

  for (n in 1:N) {
    real tvc_effect = beta_tvc[time_idx[n]] * x_tvc[n];
    real eta_n = X[n] * beta + tvc_effect;
    log_lik[n] = binomial_logit_lpmf(y[n] | n_trials[n], eta_n);
  }
}
