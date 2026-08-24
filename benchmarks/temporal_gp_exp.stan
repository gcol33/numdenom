// Temporal GP with exponential covariance (OU process)
// Matches numdenom's temporal_gp(cov = "exponential")

data {
  int<lower=1> N;           // Number of observations
  int<lower=1> T;           // Number of unique time points
  array[N] int<lower=0> y;  // Counts (numerator)
  vector[N] x;              // Covariate
  array[N] int<lower=1,upper=T> time_idx;  // Maps obs to time point
  vector[T] time_values;    // Unique time values
}

parameters {
  real beta0;               // Intercept
  real beta1;               // Slope for x
  real<lower=0> phi_nb;     // NB dispersion

  // GP parameters
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
      K[i,i] = K[i,i] + 1e-8;  // Jitter for numerical stability
    }

    L = cholesky_decompose(K);
    gp_effects = L * gp_raw;
  }
}

model {
  vector[N] mu;

  // Priors matching numdenom defaults
  beta0 ~ normal(0, 10);
  beta1 ~ normal(0, 5);
  phi_nb ~ gamma(1, 0.1);

  // GP priors (PC prior style)
  sigma2_gp ~ exponential(1);  // P(sigma > 1) ~ 0.37
  phi_gp ~ gamma(2, 0.5);      // Mode at 2, wide

  // Standard normal for GP raw values
  gp_raw ~ std_normal();

  // Likelihood
  for (n in 1:N) {
    mu[n] = exp(beta0 + beta1 * x[n] + gp_effects[time_idx[n]]);
  }

  y ~ neg_binomial_2(mu, phi_nb);
}
