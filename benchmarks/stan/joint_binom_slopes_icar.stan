// Joint Binomial model with SHARED correlated random slopes + ICAR spatial
// Row 87 in gradient_methods.md

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
  array[N] int<lower=0> y;
  array[N] int<lower=1> trials;
  int<lower=1> p;
  matrix[N, p] X;
  vector[N] x_slope;

  int<lower=1> n_groups;
  array[N] int<lower=1,upper=n_groups> group_idx;

  int<lower=1> J;
  array[N] int<lower=1,upper=J> spatial_idx;
  array[J] int<lower=0> n_neighbors;
  int<lower=0> n_edges;
  array[n_edges] int<lower=1,upper=J> edge1;
  array[n_edges] int<lower=1,upper=J> edge2;
}

parameters {
  vector[p] beta;

  matrix[2, n_groups] z_re;
  vector<lower=0>[2] sigma_re;
  cholesky_factor_corr[2] L_re;

  vector[J] phi_spatial;
  real<lower=0> tau_spatial;
}

transformed parameters {
  matrix[n_groups, 2] re;
  re = (diag_pre_multiply(sigma_re, L_re) * z_re)';
}

model {
  vector[N] eta;

  eta = X * beta;
  for (n in 1:N) {
    int g = group_idx[n];
    int s = spatial_idx[n];
    real re_effect = re[g, 1] + re[g, 2] * x_slope[n];
    eta[n] += re_effect + phi_spatial[s];
  }

  beta ~ normal(0, 10);

  to_vector(z_re) ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);
  L_re ~ lkj_corr_cholesky(2);

  tau_spatial ~ gamma(1, 0.01);
  phi_spatial ~ icar(J, n_neighbors, n_edges, edge1, edge2, tau_spatial);
  sum(phi_spatial) ~ normal(0, 0.001 * J);

  y ~ binomial_logit(trials, eta);
}
