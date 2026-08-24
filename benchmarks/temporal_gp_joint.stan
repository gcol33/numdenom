// Joint num+denom model with temporal GP - matches numdenom exactly
// Both processes share the same GP temporal effect

data {
  int<lower=1> N;           // Number of observations
  int<lower=1> T;           // Number of unique time points
  array[N] int<lower=0> y_num;   // Numerator counts
  array[N] int<lower=0> y_denom; // Denominator counts
  vector[N] x;              // Covariate
  array[N] int<lower=1,upper=T> time_idx;  // Maps obs to time point
  vector[T] time_values;    // Unique time values
}

parameters {
  // Numerator parameters
  real beta_num_0;          // Intercept
  real beta_num_1;          // Slope for x
  real<lower=0> phi_num;    // NB dispersion

  // Denominator parameters
  real beta_denom_0;        // Intercept
  real<lower=0> phi_denom;  // NB dispersion

  // Shared GP parameters
  real<lower=0> sigma2_gp;  // GP variance
  real<lower=0> phi_gp;     // GP lengthscale
  vector[T] gp_raw;         // GP effects (raw)
}

transformed parameters {
  vector[T] gp_effects;

  // Build exponential covariance matrix and transform
  {
    matrix[T, T] K;
    matrix[T, T] L;

    for (i in 1:T) {
      for (j in 1:T) {
        real d = abs(time_values[i] - time_values[j]);
        K[i,j] = sigma2_gp * exp(-d / phi_gp);
      }
      K[i,i] = K[i,i] + 1e-8;  // Jitter
    }

    L = cholesky_decompose(K);
    gp_effects = L * gp_raw;
  }
}

model {
  vector[N] mu_num;
  vector[N] mu_denom;

  // Priors matching numdenom defaults
  beta_num_0 ~ normal(0, 10);
  beta_num_1 ~ normal(0, 5);
  beta_denom_0 ~ normal(0, 10);
  phi_num ~ gamma(1, 0.1);
  phi_denom ~ gamma(1, 0.1);

  // GP priors
  sigma2_gp ~ exponential(1);
  phi_gp ~ gamma(2, 0.5);
  gp_raw ~ std_normal();

  // Likelihood - both num and denom with SHARED GP effect
  for (n in 1:N) {
    mu_num[n] = exp(beta_num_0 + beta_num_1 * x[n] + gp_effects[time_idx[n]]);
    mu_denom[n] = exp(beta_denom_0 + gp_effects[time_idx[n]]);  // Shared GP
  }

  y_num ~ neg_binomial_2(mu_num, phi_num);
  y_denom ~ neg_binomial_2(mu_denom, phi_denom);
}
