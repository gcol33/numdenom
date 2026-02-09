// Joint Poisson-Gamma model with HSGP spatial + RW1 temporal
// Row 22 in gradient_methods.md
// Uses Hilbert Space GP approximation (Riutort-Mayol et al. 2023)

functions {
  // 1D Laplacian eigenfunction
  real phi_1d(real x, int j, real L) {
    real norm = 1.0 / sqrt(L);
    return norm * sin(pi() * j * (x + L) / (2.0 * L));
  }

  // 1D eigenvalue
  real lambda_1d(int j, real L) {
    real tmp = pi() * j / (2.0 * L);
    return tmp * tmp;
  }

  // Spectral density for squared exponential kernel
  real spectral_density_se(real omega_sq, real sigma2, real lengthscale) {
    real ell = lengthscale;
    real ell2 = ell * ell;
    return sigma2 * sqrt(2.0 * pi()) * ell * exp(-0.5 * ell2 * omega_sq);
  }
}

data {
  int<lower=1> N;
  int<lower=1> M;                      // Basis functions per dimension
  array[N] int<lower=0> y_num;
  vector<lower=0>[N] y_denom;
  int<lower=1> p;
  matrix[N, p] X;
  matrix[N, 2] coords;

  // Random effects
  int<lower=1> n_groups;
  array[N] int<lower=1,upper=n_groups> group_idx;

  // Temporal structure
  int<lower=2> T;
  array[N] int<lower=1,upper=T> time_idx;

  // HSGP boundary factor
  real<lower=0> c;
}

transformed data {
  int M2 = M * M;

  // Coordinate ranges
  real x_min = min(coords[, 1]);
  real x_max = max(coords[, 1]);
  real y_min = min(coords[, 2]);
  real y_max = max(coords[, 2]);

  real x_range = x_max - x_min;
  real y_range = y_max - y_min;
  real x_center = (x_max + x_min) / 2.0;
  real y_center = (y_max + y_min) / 2.0;

  real L1 = fmax(c * x_range / 2.0, 0.1);
  real L2 = fmax(c * y_range / 2.0, 0.1);

  // Precompute eigenvalues
  array[M2] real eigenvalues;
  for (j1 in 1:M) {
    for (j2 in 1:M) {
      int idx = (j1 - 1) * M + j2;
      eigenvalues[idx] = lambda_1d(j1, L1) + lambda_1d(j2, L2);
    }
  }

  // Precompute basis matrix
  matrix[N, M2] phi_basis;
  for (i in 1:N) {
    real xi = coords[i, 1] - x_center;
    real yi = coords[i, 2] - y_center;
    for (j1 in 1:M) {
      real phi_x = phi_1d(xi, j1, L1);
      for (j2 in 1:M) {
        real phi_y = phi_1d(yi, j2, L2);
        int idx = (j1 - 1) * M + j2;
        phi_basis[i, idx] = phi_x * phi_y;
      }
    }
  }
}

parameters {
  vector[p] beta_num;
  vector[p] beta_denom;
  real<lower=0> shape;

  // SHARED random effects
  vector[n_groups] z_re;
  real<lower=0> sigma_re;

  // HSGP parameters
  real<lower=0> sigma_gp;
  real<lower=0.01> lengthscale;
  vector[M2] beta_hsgp;

  // SHARED temporal effects (RW1)
  vector[T] phi_temporal;
  real<lower=0> tau_temporal;
}

transformed parameters {
  vector[n_groups] re = sigma_re * z_re;
  real<lower=0> sigma2_gp = sigma_gp * sigma_gp;

  // HSGP spatial effects
  vector[N] gp_effects;
  {
    vector[M2] weighted_beta;
    for (j in 1:M2) {
      real S_j = spectral_density_se(eigenvalues[j], sigma2_gp, lengthscale);
      weighted_beta[j] = sqrt(fmax(S_j, 1e-10)) * beta_hsgp[j];
    }
    gp_effects = phi_basis * weighted_beta;
  }
}

model {
  vector[N] eta_num;
  vector[N] eta_denom;

  // Linear predictors with SHARED effects
  eta_num = X * beta_num;
  eta_denom = X * beta_denom;

  for (n in 1:N) {
    real shared_effect = re[group_idx[n]] + gp_effects[n] + phi_temporal[time_idx[n]];
    eta_num[n] += shared_effect;
    eta_denom[n] += shared_effect;
  }

  // Priors
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  shape ~ gamma(2, 0.1);

  // RE priors
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  // HSGP priors
  sigma_gp ~ exponential(2.3);  // PC prior: P(sigma > 1) ≈ 0.1
  lengthscale ~ lognormal(0, 1);
  beta_hsgp ~ std_normal();

  // Temporal priors (RW1)
  tau_temporal ~ gamma(1, 0.01);
  for (t in 2:T) {
    target += normal_lpdf(phi_temporal[t] | phi_temporal[t-1], 1/sqrt(tau_temporal));
  }
  sum(phi_temporal) ~ normal(0, 0.001 * T);

  // Likelihoods
  vector[N] mu_num = exp(eta_num);
  vector[N] mu_denom = exp(eta_denom);

  y_num ~ poisson(mu_num);

  for (n in 1:N) {
    if (y_denom[n] > 0) {
      target += gamma_lpdf(y_denom[n] | shape, shape / mu_denom[n]);
    }
  }
}
