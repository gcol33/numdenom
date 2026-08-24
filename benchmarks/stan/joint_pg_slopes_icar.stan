// Joint Poisson-Gamma model with SHARED correlated random slopes + ICAR spatial
// Matches numdenom ratiod_poisson_gamma() with (1 + x|site) + spatial_car()
// Row 25 in gradient_methods.md
//
// Correlated random slopes: (1|site) + (x|site) with correlation

functions {
  // ICAR log prior
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
  array[N] int<lower=0> y_num;         // Numerator counts (Poisson)
  vector<lower=0>[N] y_denom;          // Denominator (Gamma-distributed effort)
  int<lower=1> p;                      // Number of predictors
  matrix[N, p] X;                      // Design matrix (shared)
  vector[N] x_slope;                   // Covariate for random slope

  // Random effects groups
  int<lower=1> n_groups;
  array[N] int<lower=1,upper=n_groups> group_idx;

  // Spatial structure (ICAR)
  int<lower=1> J;
  array[N] int<lower=1,upper=J> spatial_idx;
  array[J] int<lower=0> n_neighbors;
  int<lower=0> n_edges;
  array[n_edges] int<lower=1,upper=J> edge1;
  array[n_edges] int<lower=1,upper=J> edge2;
}

parameters {
  vector[p] beta_num;
  vector[p] beta_denom;
  real<lower=0> shape;

  // SHARED correlated random slopes (Cholesky parameterization)
  matrix[2, n_groups] z_re;            // Standard normal [intercept, slope] x groups
  vector<lower=0>[2] sigma_re;         // SD for intercept and slope
  cholesky_factor_corr[2] L_re;        // Cholesky of correlation matrix

  // SHARED spatial effects
  vector[J] phi_spatial;
  real<lower=0> tau_spatial;
}

transformed parameters {
  matrix[n_groups, 2] re;              // [n_groups, 2] = [intercept, slope]

  // Non-centered: re = diag(sigma) * L * z
  re = (diag_pre_multiply(sigma_re, L_re) * z_re)';
}

model {
  vector[N] eta_num;
  vector[N] eta_denom;

  // Linear predictors
  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  for (n in 1:N) {
    int g = group_idx[n];
    int s = spatial_idx[n];
    real re_effect = re[g, 1] + re[g, 2] * x_slope[n];  // intercept + slope*x
    eta_num[n] += re_effect + phi_spatial[s];
    eta_denom[n] += re_effect + phi_spatial[s];
  }

  // Priors
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  shape ~ gamma(2, 0.1);

  // RE priors (correlated slopes)
  to_vector(z_re) ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);
  L_re ~ lkj_corr_cholesky(2);

  // Spatial priors (ICAR)
  tau_spatial ~ gamma(1, 0.01);
  phi_spatial ~ icar(J, n_neighbors, n_edges, edge1, edge2, tau_spatial);
  sum(phi_spatial) ~ normal(0, 0.001 * J);

  // Likelihoods
  y_num ~ poisson(exp(eta_num));

  for (n in 1:N) {
    if (y_denom[n] > 0) {
      target += gamma_lpdf(y_denom[n] | shape, shape / exp(eta_denom[n]));
    }
  }
}

generated quantities {
  corr_matrix[2] Omega_re;
  vector[N] log_lik;

  Omega_re = L_re * L_re';

  for (n in 1:N) {
    int g = group_idx[n];
    int s = spatial_idx[n];
    real re_effect = re[g, 1] + re[g, 2] * x_slope[n];
    real eta_num_n = X[n] * beta_num + re_effect + phi_spatial[s];
    real eta_denom_n = X[n] * beta_denom + re_effect + phi_spatial[s];

    log_lik[n] = poisson_lpmf(y_num[n] | exp(eta_num_n));
    if (y_denom[n] > 0) {
      log_lik[n] += gamma_lpdf(y_denom[n] | shape, shape / exp(eta_denom_n));
    }
  }
}
