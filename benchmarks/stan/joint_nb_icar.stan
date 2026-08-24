// Joint NegBin-NegBin model with SHARED random effects + ICAR spatial
// Matches numdenom ratiod_negbin_negbin() with (1|site) + spatial_car()
// Row 35 in gradient_methods.md
//
// KEY INSIGHT: Both RE and spatial effects are SHARED between num and denom
// This captures the reality that unmeasured factors affect BOTH counts.
//
// ICAR prior: phi ~ CAR(tau, W) where W is adjacency matrix

functions {
  real icar_lpdf(vector phi, int J, array[] int n_neighbors,
                 int n_edges, array[] int edge1, array[] int edge2, real tau) {
    real quad_form = 0;
    for (e in 1:n_edges) {
      quad_form += square(phi[edge1[e]] - phi[edge2[e]]);
    }
    return 0.5 * (J - 1) * log(tau) - 0.5 * tau * quad_form;
  }
}

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;         // Numerator counts (NegBin)
  array[N] int<lower=0> y_denom;       // Denominator counts (NegBin)
  int<lower=1> p;                      // Number of predictors
  matrix[N, p] X;                      // Design matrix (shared)

  // Random effects
  int<lower=1> n_groups;               // Number of RE groups (sites)
  array[N] int<lower=1,upper=n_groups> group_idx;  // Group assignment

  // Spatial structure
  int<lower=1> J;                      // Number of spatial units
  array[N] int<lower=1,upper=J> spatial_idx;       // Spatial unit for each obs
  array[J] int<lower=0> n_neighbors;   // Number of neighbors per unit
  int<lower=0> n_edges;                // Total number of edges
  array[n_edges] int<lower=1,upper=J> edge1;
  array[n_edges] int<lower=1,upper=J> edge2;
}

parameters {
  vector[p] beta_num;                  // Numerator coefficients
  vector[p] beta_denom;                // Denominator coefficients
  real<lower=0> phi_num;               // NegBin overdispersion (numerator)
  real<lower=0> phi_denom;             // NegBin overdispersion (denominator)

  // SHARED random effects (non-centered)
  vector[n_groups] z_re;
  real<lower=0> sigma_re;

  // SHARED spatial effects
  vector[J] phi_spatial;
  real<lower=0> tau_spatial;
}

transformed parameters {
  vector[n_groups] re;
  re = sigma_re * z_re;
}

model {
  vector[N] eta_num;
  vector[N] eta_denom;
  vector[N] mu_num;
  vector[N] mu_denom;

  // Linear predictors with SHARED RE and SHARED spatial
  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  for (n in 1:N) {
    eta_num[n] += re[group_idx[n]] + phi_spatial[spatial_idx[n]];
    eta_denom[n] += re[group_idx[n]] + phi_spatial[spatial_idx[n]];
  }

  // Priors matching numdenom defaults
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  phi_num ~ gamma(2, 0.1);    // Overdispersion prior
  phi_denom ~ gamma(2, 0.1);

  // RE priors
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  // Spatial priors (matching numdenom: tau ~ Gamma(1, 0.01))
  tau_spatial ~ gamma(1, 0.01);
  phi_spatial ~ icar(J, n_neighbors, n_edges, edge1, edge2, tau_spatial);
  sum(phi_spatial) ~ normal(0, 0.001 * J);

  // Means
  mu_num = exp(eta_num);
  mu_denom = exp(eta_denom);

  // Numerator: NegBin
  y_num ~ neg_binomial_2(mu_num, phi_num);

  // Denominator: NegBin
  y_denom ~ neg_binomial_2(mu_denom, phi_denom);
}

generated quantities {
  vector[N] log_lik;

  for (n in 1:N) {
    real eta_num_n = X[n] * beta_num + re[group_idx[n]] + phi_spatial[spatial_idx[n]];
    real eta_denom_n = X[n] * beta_denom + re[group_idx[n]] + phi_spatial[spatial_idx[n]];
    real mu_num_n = exp(eta_num_n);
    real mu_denom_n = exp(eta_denom_n);

    log_lik[n] = neg_binomial_2_lpmf(y_num[n] | mu_num_n, phi_num)
               + neg_binomial_2_lpmf(y_denom[n] | mu_denom_n, phi_denom);
  }
}
