// Joint NegBin-NegBin model with SHARED RE + BYM2 spatial + RW1 temporal
// Matches numdenom ratiod_negbin_negbin() with (1|site) + spatial_bym2() + temporal_rw1()
// Row 49 in gradient_methods.md
//
// BYM2: phi_s = sigma_s * (sqrt(rho) * u + sqrt(1-rho) * v) where u ~ ICAR, v ~ N(0,1)

functions {
  // ICAR log prior - pairwise differences formulation
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

  // Spatial structure (BYM2)
  int<lower=1> J;                      // Number of spatial units
  array[N] int<lower=1,upper=J> spatial_idx;       // Spatial unit for each obs
  array[J] int<lower=0> n_neighbors;   // Number of neighbors per unit
  int<lower=0> n_edges;                // Total number of edges
  array[n_edges] int<lower=1,upper=J> edge1;       // First node of each edge
  array[n_edges] int<lower=1,upper=J> edge2;       // Second node of each edge
  real<lower=0> scale_factor;          // Scaling factor for BYM2

  // Temporal structure
  int<lower=2> T;                      // Number of time points
  array[N] int<lower=1,upper=T> time_idx;  // Time point for each obs
}

parameters {
  vector[p] beta_num;                  // Numerator coefficients
  vector[p] beta_denom;                // Denominator coefficients
  real<lower=0> phi_num;               // NegBin overdispersion for numerator
  real<lower=0> phi_denom;             // NegBin overdispersion for denominator

  // SHARED random effects (non-centered)
  vector[n_groups] z_re;               // Standard normal RE
  real<lower=0> sigma_re;              // RE standard deviation

  // SHARED BYM2 spatial effects
  vector[J] u_spatial;                 // Structured component (ICAR)
  vector[J] v_spatial;                 // Unstructured component
  real<lower=0> sigma_spatial;         // Spatial SD
  real<lower=0,upper=1> rho_spatial;   // Mixing proportion

  // SHARED temporal effects
  vector[T] phi_temporal;              // Temporal effects
  real<lower=0> tau_temporal;          // Temporal precision
}

transformed parameters {
  vector[n_groups] re;
  vector[J] phi_spatial;

  // RE
  re = sigma_re * z_re;

  // BYM2: phi = sigma * (sqrt(rho) * u + sqrt(1-rho) * v)
  phi_spatial = sigma_spatial * (sqrt(rho_spatial / scale_factor) * u_spatial +
                                  sqrt(1 - rho_spatial) * v_spatial);
}

model {
  vector[N] eta_num;
  vector[N] eta_denom;

  // Linear predictors with SHARED RE, spatial, and temporal
  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  for (n in 1:N) {
    eta_num[n] += re[group_idx[n]] + phi_spatial[spatial_idx[n]] + phi_temporal[time_idx[n]];
    eta_denom[n] += re[group_idx[n]] + phi_spatial[spatial_idx[n]] + phi_temporal[time_idx[n]];
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
  sigma_spatial ~ normal(0, 1);
  rho_spatial ~ beta(0.5, 0.5);
  v_spatial ~ std_normal();
  u_spatial ~ icar(J, n_neighbors, n_edges, edge1, edge2, 1.0);
  sum(u_spatial) ~ normal(0, 0.001 * J);

  // Temporal priors (RW1)
  tau_temporal ~ gamma(1, 0.01);
  for (t in 2:T) {
    target += normal_lpdf(phi_temporal[t] | phi_temporal[t-1], 1/sqrt(tau_temporal));
  }
  sum(phi_temporal) ~ normal(0, 0.001 * T);

  // Likelihoods
  y_num ~ neg_binomial_2(exp(eta_num), phi_num);
  y_denom ~ neg_binomial_2(exp(eta_denom), phi_denom);
}

generated quantities {
  vector[N] log_lik;

  for (n in 1:N) {
    real eta_num_n = X[n] * beta_num + re[group_idx[n]] + phi_spatial[spatial_idx[n]] + phi_temporal[time_idx[n]];
    real eta_denom_n = X[n] * beta_denom + re[group_idx[n]] + phi_spatial[spatial_idx[n]] + phi_temporal[time_idx[n]];

    log_lik[n] = neg_binomial_2_lpmf(y_num[n] | exp(eta_num_n), phi_num) +
                 neg_binomial_2_lpmf(y_denom[n] | exp(eta_denom_n), phi_denom);
  }
}
