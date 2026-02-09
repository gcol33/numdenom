// Joint Poisson-Gamma model with SHARED correlated random slopes
// Matches numdenom ratiod_poisson_gamma() with (1 + x|site)
// Row 3 in gradient_methods.md
//
// KEY INSIGHT: Random slopes are SHARED between num and denom
// Uses LKJ(eta=2) prior on correlation matrix via Cholesky parameterization
// Matches numdenom's CENTERED parameterization exactly
//
// numdenom covariance: Sigma = diag(sigma) * L * L' * diag(sigma)
// where L is lower-triangular with L[i,i] = sqrt(1 - sum_{j<i} L[i,j]^2)
//
// CENTERED parameterization: re[g] ~ MVN(0, Sigma)
// (numdenom stores RE values directly, not as z-scores)

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;        // Numerator counts (Poisson)
  vector<lower=0>[N] y_denom;         // Denominator (Gamma-distributed effort)
  int<lower=1> p;                     // Number of fixed effect predictors
  matrix[N, p] X;                     // Fixed effect design matrix
  int<lower=1> n_groups;              // Number of RE groups (sites)
  array[N] int<lower=1,upper=n_groups> group_idx;  // Group assignment per obs
  vector[N] x_slope;                  // Variable for random slope (same as X[,2])
}

transformed data {
  real sigma_re_scale = 2.5;  // Matches numdenom default
}

parameters {
  vector[p] beta_num;                 // Numerator fixed effects
  vector[p] beta_denom;               // Denominator fixed effects
  real<lower=0> shape;                // Gamma shape parameter

  // Random effects standard deviations (half-Cauchy prior)
  real<lower=0> sigma_intercept;      // SD for random intercepts
  real<lower=0> sigma_slope;          // SD for random slopes

  // Cholesky factor off-diagonal element (constrained to valid range)
  // L = [[1, 0], [L21, sqrt(1-L21^2)]] for 2x2 case
  real<lower=-1, upper=1> L21;

  // Random effects (CENTERED parameterization matching numdenom)
  // re[g,1] = intercept, re[g,2] = slope (actual values, not z-scores)
  matrix[n_groups, 2] re;
}

transformed parameters {
  // Build 2x2 Cholesky factor L matching numdenom's parameterization
  matrix[2, 2] L_Omega;
  L_Omega[1, 1] = 1.0;
  L_Omega[1, 2] = 0.0;
  L_Omega[2, 1] = L21;
  L_Omega[2, 2] = sqrt(1.0 - L21 * L21);
}

model {
  vector[N] eta_num;
  vector[N] eta_denom;

  // Linear predictors with SHARED random intercepts and slopes
  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  for (n in 1:N) {
    int g = group_idx[n];
    real re_effect = re[g, 1] + re[g, 2] * x_slope[n];
    eta_num[n] += re_effect;
    eta_denom[n] += re_effect;  // SHARED effect
  }

  // Priors matching numdenom defaults
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  shape ~ gamma(2, 0.1);

  // Half-Cauchy(0, 2.5) priors on sigma (numdenom default)
  sigma_intercept ~ cauchy(0, sigma_re_scale);
  sigma_slope ~ cauchy(0, sigma_re_scale);

  // LKJ(eta=2) prior on correlation matrix
  // numdenom computes: (eta-1 + (k-i-1)/2) * 2 * log(L[i,i]) + (k-i) * log(L[i,i])
  // For k=2, i=1: (2-1 + 0) * 2 * log(L22) + 1 * log(L22) = 3 * log(L22)
  // where L22 = sqrt(1 - L21^2)
  // => 1.5 * log(1 - L21^2)
  target += 1.5 * log(1 - L21 * L21);

  // MVN prior on RE: re[g] ~ MVN(0, Sigma) where Sigma = D * L * L' * D
  // D = diag(sigma_intercept, sigma_slope)
  // This is the CENTERED parameterization
  // log p(re[g] | Sigma) = -0.5 * re[g]' * Sigma^{-1} * re[g] - log|Sigma|/2
  //
  // Using y = D^{-1} * re and z = L^{-1} * y:
  // log p = -0.5 * ||z||^2 - sum(log(sigma)) - sum(log(L[i,i]))
  for (g in 1:n_groups) {
    // Compute y = D^{-1} * re[g]
    real y1 = re[g, 1] / sigma_intercept;
    real y2 = re[g, 2] / sigma_slope;

    // Compute z = L^{-1} * y via forward substitution
    // L = [[1, 0], [L21, L22]]
    // z1 = y1 / 1 = y1
    // z2 = (y2 - L21 * z1) / L22
    real L22 = L_Omega[2, 2];
    real z1 = y1;
    real z2 = (y2 - L21 * z1) / L22;

    // Quadratic form: -0.5 * ||z||^2
    target += -0.5 * (z1 * z1 + z2 * z2);
  }
  // Log-determinant: -n_groups * (log(sigma_intercept) + log(sigma_slope) + log(L22))
  target += -n_groups * (log(sigma_intercept) + log(sigma_slope) + log(L_Omega[2, 2]));

  // Likelihood
  y_num ~ poisson_log(eta_num);

  for (n in 1:N) {
    if (y_denom[n] > 0) {
      real mu_denom_n = exp(eta_denom[n]);
      target += gamma_lpdf(y_denom[n] | shape, shape / mu_denom_n);
    }
  }
}

generated quantities {
  // Correlation coefficient (for comparison)
  real rho = L21;  // In 2x2 case, correlation = L21

  vector[N] log_lik;
  for (n in 1:N) {
    int g = group_idx[n];
    real re_effect = re[g, 1] + re[g, 2] * x_slope[n];
    real eta_num_n = X[n] * beta_num + re_effect;
    real eta_denom_n = X[n] * beta_denom + re_effect;
    real mu_num_n = exp(eta_num_n);
    real mu_denom_n = exp(eta_denom_n);

    log_lik[n] = poisson_lpmf(y_num[n] | mu_num_n);
    if (y_denom[n] > 0) {
      log_lik[n] += gamma_lpdf(y_denom[n] | shape, shape / mu_denom_n);
    }
  }
}
