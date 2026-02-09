// Joint NegBin-NegBin model with SHARED random effects + BYM2 spatial
// Matches numdenom ratiod_negbin_negbin() with (1|site) + spatial_bym2()
// Row 36 in gradient_methods.md
//
// BYM2 parameterization (Riebler et al. 2016):
// spatial_effect = sigma * (sqrt(rho) * scaled_phi + sqrt(1-rho) * theta)

functions {
  real icar_lpdf(vector phi, int J, array[] int n_neighbors,
                 int n_edges, array[] int edge1, array[] int edge2) {
    real quad_form = 0;
    for (e in 1:n_edges) {
      quad_form += square(phi[edge1[e]] - phi[edge2[e]]);
    }
    return -0.5 * quad_form;
  }
}

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;
  array[N] int<lower=0> y_denom;
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
  real<lower=0> scale_factor;
}

parameters {
  vector[p] beta_num;
  vector[p] beta_denom;
  real<lower=0> phi_num;
  real<lower=0> phi_denom;

  // SHARED random effects
  vector[n_groups] z_re;
  real<lower=0> sigma_re;

  // SHARED BYM2 spatial
  vector[J] phi_raw;
  vector[J] theta;
  real<lower=0> sigma_bym2;
  real<lower=0,upper=1> rho;
}

transformed parameters {
  vector[n_groups] re;
  vector[J] spatial_effect;
  vector[J] phi_scaled;

  re = sigma_re * z_re;
  phi_scaled = phi_raw * scale_factor;

  for (s in 1:J) {
    spatial_effect[s] = sigma_bym2 * (sqrt(rho) * phi_scaled[s] + sqrt(1 - rho) * theta[s]);
  }
}

model {
  vector[N] eta_num;
  vector[N] eta_denom;
  vector[N] mu_num;
  vector[N] mu_denom;

  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  for (n in 1:N) {
    eta_num[n] += re[group_idx[n]] + spatial_effect[spatial_idx[n]];
    eta_denom[n] += re[group_idx[n]] + spatial_effect[spatial_idx[n]];
  }

  // Priors
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  phi_num ~ gamma(2, 0.1);
  phi_denom ~ gamma(2, 0.1);

  // RE priors
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  // BYM2 priors
  sigma_bym2 ~ cauchy(0, 2.5);
  rho ~ beta(0.5, 0.5);
  phi_raw ~ icar(J, n_neighbors, n_edges, edge1, edge2);
  // No sum-to-zero constraint to match numdenom
  theta ~ std_normal();

  // Means
  mu_num = exp(eta_num);
  mu_denom = exp(eta_denom);

  // Likelihoods
  y_num ~ neg_binomial_2(mu_num, phi_num);
  y_denom ~ neg_binomial_2(mu_denom, phi_denom);
}

generated quantities {
  vector[N] log_lik;

  for (n in 1:N) {
    real eta_num_n = X[n] * beta_num + re[group_idx[n]] + spatial_effect[spatial_idx[n]];
    real eta_denom_n = X[n] * beta_denom + re[group_idx[n]] + spatial_effect[spatial_idx[n]];
    real mu_num_n = exp(eta_num_n);
    real mu_denom_n = exp(eta_denom_n);

    log_lik[n] = neg_binomial_2_lpmf(y_num[n] | mu_num_n, phi_num)
               + neg_binomial_2_lpmf(y_denom[n] | mu_denom_n, phi_denom);
  }
}
