// Joint Poisson-Gamma model (no RE)
// Matches numdenom ratiod_poisson_gamma() exactly
// Row 1 in gradient_methods.md
//
// KEY DIFFERENCE from brms: Both num AND denom are modeled as random variables
// - Numerator: Poisson(exp(eta_num))  -- rate is INDEPENDENT of effort
// - Denominator: Gamma(shape, shape / exp(eta_denom))
//
// numdenom parameterization: num and denom have INDEPENDENT linear predictors
// The ratio is computed post-hoc as exp(eta_num - eta_denom), NOT as count/effort
//
// brms offset(log(effort)) treats effort as FIXED - fundamentally different model!

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;      // Numerator counts (Poisson)
  vector<lower=0>[N] y_denom;       // Denominator (Gamma-distributed effort)
  int<lower=1> p;                   // Number of predictors (same for num and denom)
  matrix[N, p] X;                   // Design matrix (shared between num and denom)
}

parameters {
  vector[p] beta_num;               // Numerator coefficients
  vector[p] beta_denom;             // Denominator coefficients
  real<lower=0> shape;              // Gamma shape parameter for denominator
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
  // beta ~ N(0, sigma_beta^2) where sigma_beta = 10
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);

  // shape ~ Gamma(shape=2, rate=0.1)
  shape ~ gamma(2, 0.1);

  // Means - INDEPENDENTLY parameterized (numdenom does NOT multiply by effort!)
  mu_num = exp(eta_num);     // Poisson rate = exp(eta_num), NOT effort * exp(eta)
  mu_denom = exp(eta_denom); // Gamma mean = exp(eta_denom)

  // Numerator: Poisson
  y_num ~ poisson(mu_num);

  // Denominator: Gamma (shape=shape, rate=shape/mu)
  // Parameterized so E[y_denom] = mu_denom, Var[y_denom] = mu_denom^2/shape
  for (n in 1:N) {
    if (y_denom[n] > 0) {
      target += gamma_lpdf(y_denom[n] | shape, shape / mu_denom[n]);
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
    real mu_num_n = exp(eta_num_out[n]);  // NOT effort * exp(eta)
    real mu_denom_n = exp(eta_denom_out[n]);
    log_lik[n] = poisson_lpmf(y_num[n] | mu_num_n);
    if (y_denom[n] > 0) {
      log_lik[n] += gamma_lpdf(y_denom[n] | shape, shape / mu_denom_n);
    }
  }
}
