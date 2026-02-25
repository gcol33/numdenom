// Joint Poisson-Gamma model with latent factors
// Row 30 in gradient_methods.md
// Latent factors capture unmeasured confounders

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;
  vector<lower=0>[N] y_denom;
  int<lower=1> p;
  matrix[N, p] X;

  // Latent factor configuration
  int<lower=1> K;  // Number of latent factors

  // Random effects
  int<lower=1> n_groups;
  array[N] int<lower=1,upper=n_groups> group_idx;
}

parameters {
  vector[p] beta_num;
  vector[p] beta_denom;
  real<lower=0> shape;

  // Random effects (non-centered)
  vector[n_groups] z_re;
  real<lower=0> sigma_re;

  // Latent factors
  matrix[N, K] z_factors;           // Latent factor scores (non-centered)
  vector<lower=0>[K] sigma_factors; // Factor standard deviations

  // Factor loadings (shared across num and denom)
  vector[K] lambda;
}

transformed parameters {
  vector[n_groups] re = sigma_re * z_re;

  // Compute latent factor effects
  vector[N] factor_effects = rep_vector(0.0, N);
  for (k in 1:K) {
    factor_effects += (sigma_factors[k] * col(z_factors, k)) * lambda[k];
  }
}

model {
  vector[N] eta_num;
  vector[N] eta_denom;

  // Linear predictors with shared RE and latent factors
  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  for (n in 1:N) {
    real shared_effect = re[group_idx[n]] + factor_effects[n];
    eta_num[n] += shared_effect;
    eta_denom[n] += shared_effect;
  }

  // Priors
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  shape ~ gamma(2, 0.1);

  // RE priors
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  // Latent factor priors
  to_vector(z_factors) ~ std_normal();
  sigma_factors ~ exponential(1);
  lambda ~ normal(0, 1);

  // Likelihoods
  vector[N] mu_num = exp(eta_num);
  vector[N] mu_denom = exp(eta_denom);

  y_num ~ poisson(mu_num);

  for (n in 1:N) {
    if (y_denom[n] > 0) {
      target += gamma_lpdf(y_denom[n] | shape, shape / mu_denom[n]);
    }
  }
}
