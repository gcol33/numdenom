// Joint Binomial model with SHARED random effects + BYM2 spatial
// Row 66 in gradient_methods.md
// Binomial: y ~ Binomial(trials, inv_logit(eta))

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
  array[N] int<lower=0> y;
  array[N] int<lower=1> trials;
  int<lower=1> p;
  matrix[N, p] X;

  int<lower=1> n_groups;
  array[N] int<lower=1,upper=n_groups> group_idx;

  int<lower=1> J;
  array[N] int<lower=1,upper=J> spatial_idx;
  array[J] int<lower=0> n_neighbors;
  int<lower=0> n_edges;
  array[n_edges] int<lower=1,upper=J> edge1;
  array[n_edges] int<lower=1,upper=J> edge2;
  real<lower=0> scale_factor;
}

parameters {
  vector[p] beta;

  vector[n_groups] z_re;
  real<lower=0> sigma_re;

  vector[J] phi_raw;
  vector[J] theta;
  real<lower=0> sigma_bym2;
  real<lower=0,upper=1> rho;
}

transformed parameters {
  vector[n_groups] re = sigma_re * z_re;
  vector[J] spatial_effect;
  vector[J] phi_scaled = phi_raw * scale_factor;

  for (s in 1:J) {
    spatial_effect[s] = sigma_bym2 * (sqrt(rho) * phi_scaled[s] + sqrt(1 - rho) * theta[s]);
  }
}

model {
  vector[N] eta;

  eta = X * beta;
  for (n in 1:N) {
    eta[n] += re[group_idx[n]] + spatial_effect[spatial_idx[n]];
  }

  beta ~ normal(0, 10);
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  sigma_bym2 ~ cauchy(0, 2.5);
  rho ~ beta(0.5, 0.5);
  phi_raw ~ icar(J, n_neighbors, n_edges, edge1, edge2);
  theta ~ std_normal();

  y ~ binomial_logit(trials, eta);
}
