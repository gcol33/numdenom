// Joint Poisson-Gamma model with HSGP spatial - matches numdenom exactly
// Numerator: Poisson, Denominator: Gamma
// Both processes share the same HSGP spatial effect
// Uses Hilbert Space GP approximation (Riutort-Mayol et al. 2023)

functions {
  // 1D Laplacian eigenfunction: phi_j(x) = 1/sqrt(L) * sin(pi*j*(x+L)/(2L))
  real phi_1d(real x, int j, real L) {
    real norm = 1.0 / sqrt(L);
    return norm * sin(pi() * j * (x + L) / (2.0 * L));
  }

  // 1D eigenvalue: lambda_j = (pi*j / (2*L))^2
  real lambda_1d(int j, real L) {
    real tmp = pi() * j / (2.0 * L);
    return tmp * tmp;
  }

  // Spectral density for squared exponential kernel
  // S(omega) = sigma^2 * sqrt(2*pi) * ell * exp(-0.5 * ell^2 * omega^2)
  real spectral_density_se(real omega_sq, real sigma2, real lengthscale) {
    real ell = lengthscale;
    real ell2 = ell * ell;
    return sigma2 * sqrt(2.0 * pi()) * ell * exp(-0.5 * ell2 * omega_sq);
  }
}

data {
  int<lower=1> N;           // Number of observations
  int<lower=1> M;           // Basis functions per dimension
  array[N] int<lower=0> y_num;   // Numerator counts (Poisson)
  vector<lower=0>[N] y_denom;    // Denominator values (Gamma)
  vector[N] x;              // Covariate
  matrix[N, 2] coords;      // Spatial coordinates (lon, lat)

  // Boundary factor
  real<lower=0> c;          // L = c * max_range (default 1.5)

  // PC prior parameters for sigma (default: P(sigma > 1) = 0.01)
  real<lower=0> sigma2_prior_U;
  real<lower=0,upper=1> sigma2_prior_alpha;

  // Uniform prior bounds for lengthscale
  real<lower=0> phi_prior_lower;
  real<lower=0> phi_prior_upper;
}

transformed data {
  int M2 = M * M;  // Total basis functions

  // Compute coordinate ranges and centers
  real x_min = min(coords[, 1]);
  real x_max = max(coords[, 1]);
  real y_min = min(coords[, 2]);
  real y_max = max(coords[, 2]);

  real x_range = x_max - x_min;
  real y_range = y_max - y_min;
  real x_center = (x_max + x_min) / 2.0;
  real y_center = (y_max + y_min) / 2.0;

  // Boundary factors
  real L1 = fmax(c * x_range / 2.0, 0.1);
  real L2 = fmax(c * y_range / 2.0, 0.1);

  // PC prior rate
  real pc_rate = -log(sigma2_prior_alpha) / sigma2_prior_U;

  // Precompute eigenvalues for 2D
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
  // Numerator (Poisson)
  real beta_num_0;
  real beta_num_1;

  // Denominator (Gamma)
  real beta_denom_0;
  real beta_denom_1;
  real<lower=0> phi_denom;  // Gamma shape

  // HSGP parameters - match numdenom priors
  real<lower=0> sigma_gp;  // Standard deviation (not variance)
  real<lower=0.01> lengthscale;  // Positive lengthscale (LogNormal prior in model block)
  vector[M2] beta_hsgp;  // Basis coefficients
}

transformed parameters {
  real<lower=0> sigma2_gp = sigma_gp * sigma_gp;

  // Compute HSGP spatial effects: f(x) = sum_j phi_j(x) * sqrt(S(lambda_j)) * beta_j
  vector[N] gp_effects;
  {
    vector[M2] weighted_beta;
    for (j in 1:M2) {
      real S_j = spectral_density_se(eigenvalues[j], sigma2_gp, lengthscale);
      // Add numerical safeguard
      weighted_beta[j] = sqrt(fmax(S_j, 1e-10)) * beta_hsgp[j];
    }
    gp_effects = phi_basis * weighted_beta;
  }
}

model {
  vector[N] lambda_num;
  vector[N] mu_denom;

  // Fixed effect priors - match numdenom default sigma_beta = 10
  beta_num_0 ~ normal(0, 10);
  beta_num_1 ~ normal(0, 10);
  beta_denom_0 ~ normal(0, 10);
  beta_denom_1 ~ normal(0, 10);

  // Gamma shape prior (weakly informative)
  phi_denom ~ gamma(2, 0.5);

  // PC prior on sigma_gp (exponential prior, simpler than full PC prior)
  sigma_gp ~ exponential(pc_rate);

  // LogNormal(0,1) prior on lengthscale - matches numdenom
  // This prior has median=1.0, 95% CI ~[0.14, 6.4]
  lengthscale ~ lognormal(0, 1);

  // Standard normal prior on basis coefficients
  beta_hsgp ~ std_normal();

  // Likelihood - vectorized for speed
  lambda_num = exp(beta_num_0 + beta_num_1 * x + gp_effects);
  mu_denom = exp(beta_denom_0 + beta_denom_1 * x + gp_effects);

  y_num ~ poisson(lambda_num);

  for (n in 1:N) {
    y_denom[n] ~ gamma(phi_denom, phi_denom / mu_denom[n]);
  }
}
