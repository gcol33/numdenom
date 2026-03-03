// Joint Binomial model with HSGP spatial
// Row 68 in gradient_methods.md

functions {
  real phi_1d(real x, int j, real L) {
    real norm = 1.0 / sqrt(L);
    return norm * sin(pi() * j * (x + L) / (2.0 * L));
  }

  real lambda_1d(int j, real L) {
    real tmp = pi() * j / (2.0 * L);
    return tmp * tmp;
  }

  real spectral_density_se(real omega_sq, real sigma2, real lengthscale) {
    real ell = lengthscale;
    real ell2 = ell * ell;
    return sigma2 * sqrt(2.0 * pi()) * ell * exp(-0.5 * ell2 * omega_sq);
  }
}

data {
  int<lower=1> N;
  int<lower=1> M;
  array[N] int<lower=0> y;
  array[N] int<lower=1> trials;
  vector[N] x;
  matrix[N, 2] coords;

  real<lower=0> c;
  real<lower=0> sigma2_prior_U;
  real<lower=0,upper=1> sigma2_prior_alpha;
  real<lower=0> phi_prior_lower;
  real<lower=0> phi_prior_upper;
}

transformed data {
  int M2 = M * M;

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

  real pc_rate = -log(sigma2_prior_alpha) / sigma2_prior_U;

  array[M2] real eigenvalues;
  for (j1 in 1:M) {
    for (j2 in 1:M) {
      int idx = (j1 - 1) * M + j2;
      eigenvalues[idx] = lambda_1d(j1, L1) + lambda_1d(j2, L2);
    }
  }

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
  real beta_0;
  real beta_1;

  real<lower=0> sigma_gp;
  real<lower=0.01> lengthscale;
  vector[M2] beta_hsgp;
}

transformed parameters {
  real<lower=0> sigma2_gp = sigma_gp * sigma_gp;

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
  vector[N] eta;

  beta_0 ~ normal(0, 10);
  beta_1 ~ normal(0, 10);
  sigma_gp ~ exponential(pc_rate);
  lengthscale ~ lognormal(0, 1);
  beta_hsgp ~ std_normal();

  eta = beta_0 + beta_1 * x + gp_effects;
  y ~ binomial_logit(trials, eta);
}
