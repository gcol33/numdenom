// Joint NegBin-NegBin model with temporal GP - matches numdenom exactly
// Both processes share the same GP temporal effect
// Uses IMPROVED parameterization for numerical stability

data {
  int<lower=1> N;           // Number of observations
  int<lower=1> T;           // Number of unique time points
  array[N] int<lower=0> y_num;   // Numerator counts
  array[N] int<lower=0> y_denom; // Denominator counts
  vector[N] x;              // Covariate
  array[N] int<lower=1,upper=T> time_idx;  // Maps obs to time point
  vector[T] time_values;    // Unique time values (SCALED - mean 0, sd 1)

  // PC prior parameters for sigma (default: P(sigma > 1) = 0.01)
  real<lower=0> sigma2_prior_U;      // Upper bound (default 1.0)
  real<lower=0,upper=1> sigma2_prior_alpha;  // Tail probability (default 0.01)

  // Uniform prior bounds for phi (lengthscale)
  real<lower=0> phi_prior_lower;     // Lower bound (default 0.01)
  real<lower=0> phi_prior_upper;     // Upper bound (default 10.0)
}

transformed data {
  // PC prior rate for sigma: rate = -log(alpha) / U
  real pc_rate = -log(sigma2_prior_alpha) / sigma2_prior_U;

  // Use reasonable bounds for lengthscale based on time range
  real time_range = max(time_values) - min(time_values);
  real phi_lower_bounded = fmax(phi_prior_lower, time_range / 20.0);  // At least 1/20 of range
  real phi_upper_bounded = fmin(phi_prior_upper, time_range * 2.0);   // At most 2x range
}

parameters {
  // Numerator parameters
  real beta_num_0;          // Intercept
  real beta_num_1;          // Slope for x
  real<lower=0> phi_num;    // NB dispersion

  // Denominator parameters
  real beta_denom_0;        // Intercept
  real beta_denom_1;        // Slope for x
  real<lower=0> phi_denom;  // NB dispersion

  // Shared GP parameters - BOUNDED for stability
  real<lower=0> sigma_gp;   // GP marginal standard deviation (not variance)
  real<lower=0.01, upper=10> phi_gp;  // GP lengthscale (bounded)
  vector[T] gp_raw;         // GP effects (standardized)
}

transformed parameters {
  vector[T] gp_effects;

  // Build exponential covariance matrix and transform
  // Using non-centered parameterization: gp_effects = L * gp_raw
  {
    matrix[T, T] K;
    matrix[T, T] L;
    real sigma2_gp = sigma_gp * sigma_gp;

    for (i in 1:T) {
      for (j in i:T) {
        real d = abs(time_values[i] - time_values[j]);
        real cov = sigma2_gp * exp(-d / phi_gp);
        K[i,j] = cov;
        K[j,i] = cov;
      }
      K[i,i] = K[i,i] + 1e-6;  // Jitter for numerical stability
    }

    L = cholesky_decompose(K);
    gp_effects = L * gp_raw;
  }
}

model {
  vector[N] mu_num;
  vector[N] mu_denom;

  // Fixed effect priors matching numdenom defaults
  beta_num_0 ~ normal(0, 10);
  beta_num_1 ~ normal(0, 5);
  beta_denom_0 ~ normal(0, 10);
  beta_denom_1 ~ normal(0, 5);

  // NB dispersion priors - MUST match numdenom defaults: Gamma(2, 0.1) with mode=10
  phi_num ~ gamma(2, 0.1);
  phi_denom ~ gamma(2, 0.1);

  // PC prior on sigma_gp (exponential prior approximating PC)
  sigma_gp ~ exponential(pc_rate);

  // Uniform prior on phi_gp (implicitly from bounded parameter)
  // No additional prior needed

  // Standard normal prior on raw GP effects
  gp_raw ~ std_normal();

  // Likelihood - vectorized
  for (n in 1:N) {
    mu_num[n] = exp(beta_num_0 + beta_num_1 * x[n] + gp_effects[time_idx[n]]);
    mu_denom[n] = exp(beta_denom_0 + beta_denom_1 * x[n] + gp_effects[time_idx[n]]);
  }

  y_num ~ neg_binomial_2(mu_num, phi_num);
  y_denom ~ neg_binomial_2(mu_denom, phi_denom);
}

generated quantities {
  // For compatibility with numdenom output naming
  real sigma2_gp = sigma_gp * sigma_gp;
  real sigma_temporal_gp = sigma_gp;
  real phi_temporal_gp = phi_gp;
}
