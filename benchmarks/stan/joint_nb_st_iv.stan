// Joint NegBin-NegBin model with SHARED RE + ICAR + RW1 + Spatiotemporal Type IV
// Knorr-Held Type IV: Kronecker interaction (RW1 × ICAR)
// Row 59 in gradient_methods.md
//
// Model: eta = X*beta + re[g] + phi_s[s] + phi_t[t] + delta[s,t]
// where delta follows: ICAR over space for each t, RW1 over time for each s

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

  // Temporal structure
  int<lower=2> T;
  array[N] int<lower=1,upper=T> time_idx;
}

parameters {
  vector[p] beta_num;
  vector[p] beta_denom;
  real<lower=0> size_num;
  real<lower=0> size_denom;

  // SHARED random effects
  vector[n_groups] z_re;
  real<lower=0> sigma_re;

  // SHARED spatial effects (ICAR)
  vector[J] phi_spatial;
  real<lower=0> tau_spatial;

  // SHARED temporal effects (RW1)
  vector[T] phi_temporal;
  real<lower=0> tau_temporal;

  // SHARED spatiotemporal interaction (Type IV: Kronecker RW1 × ICAR)
  matrix[J, T] delta;
  real<lower=0> tau_delta;
}

transformed parameters {
  vector[n_groups] re = sigma_re * z_re;
}

model {
  vector[N] eta_num;
  vector[N] eta_denom;

  // Linear predictors with SHARED effects
  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  for (n in 1:N) {
    int s = spatial_idx[n];
    int t = time_idx[n];
    real shared_effect = re[group_idx[n]] + phi_spatial[s] + phi_temporal[t] + delta[s, t];
    eta_num[n] += shared_effect;
    eta_denom[n] += shared_effect;
  }

  // Priors
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  size_num ~ gamma(2, 0.1);
  size_denom ~ gamma(2, 0.1);

  // RE priors
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  // Spatial priors
  tau_spatial ~ gamma(1, 0.01);
  phi_spatial ~ icar(J, n_neighbors, n_edges, edge1, edge2, tau_spatial);
  sum(phi_spatial) ~ normal(0, 0.001 * J);

  // Temporal priors (RW1)
  tau_temporal ~ gamma(1, 0.01);
  for (t in 2:T) {
    target += normal_lpdf(phi_temporal[t] | phi_temporal[t-1], 1/sqrt(tau_temporal));
  }
  sum(phi_temporal) ~ normal(0, 0.001 * T);

  // Spatiotemporal interaction priors (Type IV: Kronecker RW1 × ICAR)
  tau_delta ~ gamma(1, 0.01);

  // ICAR structure over space for each time point
  for (t in 1:T) {
    delta[, t] ~ icar(J, n_neighbors, n_edges, edge1, edge2, tau_delta);
  }

  // RW1 structure over time for each spatial unit
  for (s in 1:J) {
    for (t in 2:T) {
      target += normal_lpdf(delta[s, t] | delta[s, t-1], 1/sqrt(tau_delta));
    }
  }

  // Sum-to-zero constraints
  for (t in 1:T) {
    sum(delta[, t]) ~ normal(0, 0.001 * J);
  }
  for (s in 1:J) {
    sum(delta[s, ]) ~ normal(0, 0.001 * T);
  }

  // Likelihoods
  vector[N] mu_num = exp(eta_num);
  vector[N] mu_denom = exp(eta_denom);

  y_num ~ neg_binomial_2(mu_num, size_num);
  y_denom ~ neg_binomial_2(mu_denom, size_denom);
}
