// Joint Poisson-Gamma model with multiscale temporal decomposition
// Matches numdenom ratiod_poisson_gamma() with temporal_multiscale()
// Row 15 in gradient_methods.md
//
// Multiscale temporal: trend (RW2) + seasonal (cyclic RW1) + short-term (AR1)
// All components are SHARED between numerator and denominator

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;         // Numerator counts (Poisson)
  vector<lower=0>[N] y_denom;          // Denominator (Gamma-distributed effort)
  int<lower=1> p;                      // Number of predictors
  matrix[N, p] X;                      // Design matrix (shared)

  // Temporal structure
  int<lower=2> T;                      // Number of time points
  array[N] int<lower=1,upper=T> time_idx;  // Time point for each obs
  int<lower=0> seasonal_period;        // Seasonal period (0 = no seasonal)
}

parameters {
  vector[p] beta_num;                  // Numerator coefficients
  vector[p] beta_denom;                // Denominator coefficients
  real<lower=0> shape;                 // Gamma shape parameter

  // Trend component (RW2)
  vector[T] trend;
  real<lower=0> sigma_trend;

  // Seasonal component (cyclic RW1) - only if seasonal_period > 0
  vector[seasonal_period > 0 ? seasonal_period : 0] seasonal;
  real<lower=0> sigma_seasonal;

  // Short-term component (AR1)
  vector[T] short_term;
  real<lower=0> sigma_short;
  real<lower=-1,upper=1> rho_short;
}

model {
  vector[N] eta_num;
  vector[N] eta_denom;

  // Linear predictors
  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  // Add multiscale temporal effects (SHARED)
  for (n in 1:N) {
    int t = time_idx[n];
    real temporal_effect = trend[t] + short_term[t];
    if (seasonal_period > 0) {
      int s = ((t - 1) % seasonal_period) + 1;
      temporal_effect += seasonal[s];
    }
    eta_num[n] += temporal_effect;
    eta_denom[n] += temporal_effect;
  }

  // Priors matching numdenom defaults
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  shape ~ gamma(2, 0.1);

  // Trend: RW2 prior
  // Second differences: trend[t] - 2*trend[t-1] + trend[t-2] ~ N(0, sigma_trend^2)
  sigma_trend ~ exponential(1);
  for (t in 3:T) {
    target += normal_lpdf(trend[t] - 2*trend[t-1] + trend[t-2] | 0, sigma_trend);
  }
  // Soft sum-to-zero for identifiability
  sum(trend) ~ normal(0, 0.001 * T);

  // Seasonal: Cyclic RW1 prior
  if (seasonal_period > 0) {
    sigma_seasonal ~ exponential(1);
    for (s in 2:seasonal_period) {
      target += normal_lpdf(seasonal[s] - seasonal[s-1] | 0, sigma_seasonal);
    }
    // Cyclic: connect last to first
    target += normal_lpdf(seasonal[1] - seasonal[seasonal_period] | 0, sigma_seasonal);
    // Sum-to-zero constraint
    sum(seasonal) ~ normal(0, 0.001 * seasonal_period);
  }

  // Short-term: AR1 prior
  sigma_short ~ exponential(1);
  rho_short ~ uniform(-1, 1);
  // Stationary prior for first time point
  short_term[1] ~ normal(0, sigma_short / sqrt(1 - rho_short^2));
  for (t in 2:T) {
    short_term[t] ~ normal(rho_short * short_term[t-1], sigma_short);
  }

  // Likelihood
  // Numerator: Poisson
  y_num ~ poisson(exp(eta_num));

  // Denominator: Gamma
  {
    vector[N] mu_denom = exp(eta_denom);
    for (n in 1:N) {
      if (y_denom[n] > 0) {
        target += gamma_lpdf(y_denom[n] | shape, shape / mu_denom[n]);
      }
    }
  }
}

generated quantities {
  vector[N] log_lik;

  for (n in 1:N) {
    int t = time_idx[n];
    real temporal_effect = trend[t] + short_term[t];
    if (seasonal_period > 0) {
      int s = ((t - 1) % seasonal_period) + 1;
      temporal_effect += seasonal[s];
    }

    real eta_num_n = X[n] * beta_num + temporal_effect;
    real eta_denom_n = X[n] * beta_denom + temporal_effect;
    real mu_num_n = exp(eta_num_n);
    real mu_denom_n = exp(eta_denom_n);

    log_lik[n] = poisson_lpmf(y_num[n] | mu_num_n);
    if (y_denom[n] > 0) {
      log_lik[n] += gamma_lpdf(y_denom[n] | shape, shape / mu_denom_n);
    }
  }
}
