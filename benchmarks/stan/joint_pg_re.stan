// Joint Poisson-Gamma model with SHARED random effects
// Matches numdenom ratiod_poisson_gamma() with (1|site)
// Row 2 in gradient_methods.md
//
// KEY INSIGHT: The random effect is SHARED between num and denom
// This is the core philosophy of numdenom - shared latent structure by default
//
// numdenom parameterization:
// - Numerator: Poisson(exp(eta_num))  -- rate is INDEPENDENT of effort
// - Denominator: Gamma(shape, shape / exp(eta_denom))
// - Both get the SAME random effect added
//
// brms offset(log(effort)) + (1|site) puts RE only on numerator - different model!

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;      // Numerator counts (Poisson)
  vector<lower=0>[N] y_denom;       // Denominator (Gamma-distributed effort)
  int<lower=1> p;                   // Number of predictors (same for num and denom)
  matrix[N, p] X;                   // Design matrix (shared)
  int<lower=1> n_groups;            // Number of RE groups
  array[N] int<lower=1,upper=n_groups> group_idx;  // Group assignment
}

parameters {
  vector[p] beta_num;               // Numerator coefficients
  vector[p] beta_denom;             // Denominator coefficients
  real<lower=0> shape;              // Gamma shape parameter for denominator

  // SHARED random effects (non-centered parameterization)
  vector[n_groups] z_re;            // Standard normal RE
  real<lower=0> sigma_re;           // RE standard deviation
}

transformed parameters {
  vector[n_groups] re;              // Actual RE values

  // Non-centered parameterization: re = sigma_re * z_re
  re = sigma_re * z_re;
}

model {
  vector[N] eta_num;
  vector[N] eta_denom;
  vector[N] mu_num;
  vector[N] mu_denom;

  // Linear predictors with SHARED random effects
  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  for (n in 1:N) {
    eta_num[n] += re[group_idx[n]];    // RE added to numerator
    eta_denom[n] += re[group_idx[n]];  // SAME RE added to denominator
  }

  // Priors matching numdenom defaults
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  shape ~ gamma(2, 0.1);

  // RE priors (non-centered)
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  // Means - INDEPENDENTLY parameterized (numdenom does NOT multiply by effort!)
  mu_num = exp(eta_num);     // Poisson rate = exp(eta_num), NOT effort * exp(eta)
  mu_denom = exp(eta_denom); // Gamma mean = exp(eta_denom)

  // Numerator: Poisson
  y_num ~ poisson(mu_num);

  // Denominator: Gamma
  for (n in 1:N) {
    if (y_denom[n] > 0) {
      target += gamma_lpdf(y_denom[n] | shape, shape / mu_denom[n]);
    }
  }
}

generated quantities {
  vector[N] log_lik;

  for (n in 1:N) {
    real eta_num_n = X[n] * beta_num + re[group_idx[n]];
    real eta_denom_n = X[n] * beta_denom + re[group_idx[n]];
    real mu_num_n = exp(eta_num_n);  // NOT effort * exp(eta)
    real mu_denom_n = exp(eta_denom_n);

    log_lik[n] = poisson_lpmf(y_num[n] | mu_num_n);
    if (y_denom[n] > 0) {
      log_lik[n] += gamma_lpdf(y_denom[n] | shape, shape / mu_denom_n);
    }
  }
}
