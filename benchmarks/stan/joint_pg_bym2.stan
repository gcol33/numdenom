// Joint Poisson-Gamma model with SHARED random effects + BYM2 spatial
// Matches numdenom ratiod_poisson_gamma() with (1|site) + spatial_bym2()
// Row 6 in gradient_methods.md
//
// BYM2 parameterization (Riebler et al. 2016):
// spatial_effect = sigma * (sqrt(rho) * scaled_phi + sqrt(1-rho) * theta)
// where:
//   phi ~ ICAR (spatial structure)
//   theta ~ N(0, I) (unstructured)
//   rho ~ Beta(0.5, 0.5) (mixing proportion)
//   sigma ~ half-Cauchy (marginal SD)
//   scaled_phi = phi * scale_factor (for unit marginal variance)

functions {
  real icar_lpdf(vector phi, int J, array[] int n_neighbors,
                 int n_edges, array[] int edge1, array[] int edge2) {
    real quad_form = 0;
    for (e in 1:n_edges) {
      quad_form += square(phi[edge1[e]] - phi[edge2[e]]);
    }
    // No precision parameter - absorbed into sigma_bym2
    return -0.5 * quad_form;
  }
}

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;
  vector<lower=0>[N] y_denom;
  int<lower=1> p;
  matrix[N, p] X;

  // Random effects
  int<lower=1> n_groups;
  array[N] int<lower=1,upper=n_groups> group_idx;

  // Spatial structure (BYM2)
  int<lower=1> J;
  array[N] int<lower=1,upper=J> spatial_idx;
  array[J] int<lower=0> n_neighbors;
  int<lower=0> n_edges;
  array[n_edges] int<lower=1,upper=J> edge1;
  array[n_edges] int<lower=1,upper=J> edge2;
  real<lower=0> scale_factor;  // For unit marginal variance of ICAR
}

parameters {
  vector[p] beta_num;
  vector[p] beta_denom;
  real<lower=0> shape;

  // SHARED random effects
  vector[n_groups] z_re;
  real<lower=0> sigma_re;

  // SHARED BYM2 spatial
  vector[J] phi_raw;           // ICAR component (raw, sum-to-zero)
  vector[J] theta;             // Unstructured component
  real<lower=0> sigma_bym2;    // Marginal SD
  real<lower=0,upper=1> rho;   // Mixing proportion
}

transformed parameters {
  vector[n_groups] re;
  vector[J] spatial_effect;
  vector[J] phi_scaled;

  re = sigma_re * z_re;

  // Scale phi for unit marginal variance
  phi_scaled = phi_raw * scale_factor;

  // BYM2 combination
  for (s in 1:J) {
    spatial_effect[s] = sigma_bym2 * (sqrt(rho) * phi_scaled[s] + sqrt(1 - rho) * theta[s]);
  }
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
    eta_num[n] += re[group_idx[n]] + spatial_effect[spatial_idx[n]];
    eta_denom[n] += re[group_idx[n]] + spatial_effect[spatial_idx[n]];
  }

  // Priors
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  shape ~ gamma(2, 0.1);

  // RE priors
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  // BYM2 priors
  sigma_bym2 ~ cauchy(0, 2.5);  // Half-Cauchy on sigma
  rho ~ beta(0.5, 0.5);         // Jeffreys prior on mixing
  phi_raw ~ icar(J, n_neighbors, n_edges, edge1, edge2);
  // No sum-to-zero constraint to match numdenom
  theta ~ std_normal();

  // Means
  mu_num = exp(eta_num);
  mu_denom = exp(eta_denom);

  // Likelihoods
  y_num ~ poisson(mu_num);
  for (n in 1:N) {
    if (y_denom[n] > 0) {
      target += gamma_lpdf(y_denom[n] | shape, shape / mu_denom[n]);
    }
  }
}

generated quantities {
  vector[N] log_lik;

  for (n in 1:N) {
    real eta_num_n = X[n] * beta_num + re[group_idx[n]] + spatial_effect[spatial_idx[n]];
    real eta_denom_n = X[n] * beta_denom + re[group_idx[n]] + spatial_effect[spatial_idx[n]];
    real mu_num_n = exp(eta_num_n);
    real mu_denom_n = exp(eta_denom_n);

    log_lik[n] = poisson_lpmf(y_num[n] | mu_num_n);
    if (y_denom[n] > 0) {
      log_lik[n] += gamma_lpdf(y_denom[n] | shape, shape / mu_denom_n);
    }
  }
}
