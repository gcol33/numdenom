// Joint NegBin-NegBin model with SHARED random effects + proper CAR spatial
// Matches numdenom ratiod_negbin_negbin() with (1|site) + spatial_car(proper=TRUE)
// Row 40 in gradient_methods.md
//
// KEY INSIGHT: Both RE and spatial effects are SHARED between num and denom
//
// Proper CAR prior: phi ~ CAR(tau, rho, W)
// Q = D - rho * W where D is diagonal matrix of neighbor counts

functions {
  // Proper CAR log prior
  real proper_car_lpdf(vector phi, int J, array[] int n_neighbors,
                       int n_edges, array[] int edge1, array[] int edge2,
                       real tau, real rho) {
    real quad_form = 0;

    // Diagonal contribution
    for (i in 1:J) {
      quad_form += n_neighbors[i] * square(phi[i]);
    }

    // Off-diagonal contribution (pairwise)
    for (e in 1:n_edges) {
      quad_form -= 2 * rho * phi[edge1[e]] * phi[edge2[e]];
    }

    // Log-determinant approximation
    real avg_n_neighbors = sum(to_vector(n_neighbors)) * 1.0 / J;
    real log_det_approx = J * log(avg_n_neighbors - rho * avg_n_neighbors);

    return 0.5 * log_det_approx + 0.5 * J * log(tau) - 0.5 * tau * quad_form;
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

  // Spatial structure
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
  real<lower=0> phi_num;
  real<lower=0> phi_denom;

  // SHARED random effects
  vector[n_groups] z_re;
  real<lower=0> sigma_re;

  // SHARED proper CAR spatial
  vector[J] phi_spatial;
  real<lower=0> tau_spatial;
  real<lower=0,upper=1> rho_car;
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

  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  for (n in 1:N) {
    eta_num[n] += re[group_idx[n]] + phi_spatial[spatial_idx[n]];
    eta_denom[n] += re[group_idx[n]] + phi_spatial[spatial_idx[n]];
  }

  // Priors
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  phi_num ~ gamma(2, 0.1);
  phi_denom ~ gamma(2, 0.1);

  // RE priors
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  // Spatial priors
  tau_spatial ~ gamma(1, 0.01);
  rho_car ~ beta(1, 1);
  phi_spatial ~ proper_car(J, n_neighbors, n_edges, edge1, edge2, tau_spatial, rho_car);
  sum(phi_spatial) ~ normal(0, 0.001 * J);

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
    real eta_num_n = X[n] * beta_num + re[group_idx[n]] + phi_spatial[spatial_idx[n]];
    real eta_denom_n = X[n] * beta_denom + re[group_idx[n]] + phi_spatial[spatial_idx[n]];
    real mu_num_n = exp(eta_num_n);
    real mu_denom_n = exp(eta_denom_n);

    log_lik[n] = neg_binomial_2_lpmf(y_num[n] | mu_num_n, phi_num)
               + neg_binomial_2_lpmf(y_denom[n] | mu_denom_n, phi_denom);
  }
}
