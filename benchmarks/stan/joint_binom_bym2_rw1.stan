// Joint Binomial model with SHARED RE + BYM2 spatial + RW1 temporal
// Row 81 in gradient_methods.md

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

  int<lower=1> n_groups;
  array[N] int<lower=1,upper=n_groups> group_idx;

  int<lower=1> J;
  array[N] int<lower=1,upper=J> spatial_idx;
  array[J] int<lower=0> n_neighbors;
  int<lower=0> n_edges;
  array[n_edges] int<lower=1,upper=J> edge1;
  array[n_edges] int<lower=1,upper=J> edge2;
  real<lower=0> scale_factor;

  int<lower=2> T;
  array[N] int<lower=1,upper=T> time_idx;
}

parameters {
  vector[p] beta;

  vector[n_groups] z_re;
  real<lower=0> sigma_re;

  vector[J] u_spatial;
  vector[J] v_spatial;
  real<lower=0> sigma_spatial;
  real<lower=0,upper=1> rho_spatial;

  vector[T] phi_temporal;
  real<lower=0> tau_temporal;
}

transformed parameters {
  vector[n_groups] re = sigma_re * z_re;
  vector[J] phi_spatial;

  phi_spatial = sigma_spatial * (sqrt(rho_spatial / scale_factor) * u_spatial +
                                  sqrt(1 - rho_spatial) * v_spatial);
}

model {
  vector[N] eta;

  eta = X * beta;
  for (n in 1:N) {
    eta[n] += re[group_idx[n]] + phi_spatial[spatial_idx[n]] + phi_temporal[time_idx[n]];
  }

  beta ~ normal(0, 10);
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  sigma_spatial ~ normal(0, 1);
  rho_spatial ~ beta(0.5, 0.5);
  v_spatial ~ std_normal();
  u_spatial ~ icar(J, n_neighbors, n_edges, edge1, edge2, 1.0);
  sum(u_spatial) ~ normal(0, 0.001 * J);

  tau_temporal ~ gamma(1, 0.01);
  for (t in 2:T) {
    target += normal_lpdf(phi_temporal[t] | phi_temporal[t-1], 1/sqrt(tau_temporal));
  }
  sum(phi_temporal) ~ normal(0, 0.001 * T);

  y ~ binomial_logit(trials, eta);
}
