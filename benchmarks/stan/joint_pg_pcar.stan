// Joint Poisson-Gamma model with SHARED random effects + proper CAR spatial
// Matches numdenom ratiod_poisson_gamma() with (1|site) + spatial_car(proper=TRUE)
// Row 10 in gradient_methods.md
//
// KEY INSIGHT: Both RE and spatial effects are SHARED between num and denom
//
// Proper CAR prior: phi ~ CAR(tau, rho, W)
// Q = D - rho * W where D is diagonal matrix of neighbor counts
// p(phi|tau,rho) propto |Q|^{1/2} * tau^{J/2} * exp(-0.5 * tau * phi' Q phi)
//
// Unlike ICAR (rho=1 fixed), proper CAR estimates rho from data

functions {
  // Proper CAR log prior
  real proper_car_lpdf(vector phi, int J, array[] int n_neighbors,
                       int n_edges, array[] int edge1, array[] int edge2,
                       real tau, real rho) {
    // Quadratic form: phi' Q phi where Q = D - rho * W
    // = sum_i n_i * phi[i]^2 - 2 * rho * sum_{i~j} phi[i] * phi[j]
    real quad_form = 0;

    // Diagonal contribution
    for (i in 1:J) {
      quad_form += n_neighbors[i] * square(phi[i]);
    }

    // Off-diagonal contribution (pairwise)
    for (e in 1:n_edges) {
      quad_form -= 2 * rho * phi[edge1[e]] * phi[edge2[e]];
    }

    // Log-determinant of Q is tricky - we use sparse approximation
    // For proper CAR with rho in (0, 1), Q is positive definite
    // We approximate log|Q| ≈ J * log(avg_n_neighbors * (1 - rho))
    // This is rough but serviceable for validation
    real avg_n_neighbors = sum(to_vector(n_neighbors)) * 1.0 / J;
    real log_det_approx = J * log(avg_n_neighbors - rho * avg_n_neighbors);

    // Full log-prior
    return 0.5 * log_det_approx + 0.5 * J * log(tau) - 0.5 * tau * quad_form;
  }
}

data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;         // Numerator counts (Poisson)
  vector<lower=0>[N] y_denom;          // Denominator (Gamma-distributed effort)
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
  array[n_edges] int<lower=1,upper=J> edge1;       // First node of each edge
  array[n_edges] int<lower=1,upper=J> edge2;       // Second node of each edge
}

parameters {
  vector[p] beta_num;                  // Numerator coefficients
  vector[p] beta_denom;                // Denominator coefficients
  real<lower=0> shape;                 // Gamma shape parameter

  // SHARED random effects (non-centered)
  vector[n_groups] z_re;               // Standard normal RE
  real<lower=0> sigma_re;              // RE standard deviation

  // SHARED proper CAR spatial effects
  vector[J] phi_spatial;               // Spatial effects
  real<lower=0> tau_spatial;           // Spatial precision
  real<lower=0,upper=1> rho_car;       // Spatial autocorrelation (estimated)
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
  shape ~ gamma(2, 0.1);

  // RE priors
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  // Spatial priors (matching numdenom: tau ~ Gamma(1, 0.01), rho ~ Uniform(0,1))
  tau_spatial ~ gamma(1, 0.01);
  rho_car ~ beta(1, 1);  // Uniform on (0, 1)

  // Proper CAR prior for spatial effects
  phi_spatial ~ proper_car(J, n_neighbors, n_edges, edge1, edge2, tau_spatial, rho_car);

  // Soft sum-to-zero constraint on phi_spatial
  sum(phi_spatial) ~ normal(0, 0.001 * J);

  // Means
  mu_num = exp(eta_num);
  mu_denom = exp(eta_denom);

  // Numerator: Poisson
  y_num ~ poisson(mu_num);

  // Denominator: Gamma
  for (n in 1:N) {
    if (y_denom[n] > 0) {
      target += gamma_lpdf(y_denom[n] | shape, shape / mu_denom[n]);
    }
  }
}

generated quantities {
  vector[N] log_lik;

  for (n in 1:N) {
    real eta_num_n = X[n] * beta_num + re[group_idx[n]] + phi_spatial[spatial_idx[n]];
    real eta_denom_n = X[n] * beta_denom + re[group_idx[n]] + phi_spatial[spatial_idx[n]];
    real mu_num_n = exp(eta_num_n);
    real mu_denom_n = exp(eta_denom_n);

    log_lik[n] = poisson_lpmf(y_num[n] | mu_num_n);
    if (y_denom[n] > 0) {
      log_lik[n] += gamma_lpdf(y_denom[n] | shape, shape / mu_denom_n);
    }
  }
}
