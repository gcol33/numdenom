// quotr_poisson_gamma.stan
// Two-process ratio model: Poisson numerator, Gamma denominator
// Y_i ~ Poisson(mu_Y)
// E_i ~ Gamma(shape, shape / mu_E)
// Ratio = E[Y] / E[E] computed post hoc

functions {
  real pc_prior_sd_lpdf(real sigma, real U, real alpha) {
    real lambda = -log(alpha) / U;
    return exponential_lpdf(sigma | lambda);
  }
}

data {
  // Dimensions
  int<lower=1> N;                         // Number of observations
  int<lower=1> K_num;                     // Number of numerator fixed effects
  int<lower=1> K_denom;                   // Number of denominator fixed effects

  // Responses
  array[N] int<lower=0> y_num;            // Count numerator (e.g., catch)
  vector<lower=0>[N] y_denom;             // Continuous positive denominator (e.g., effort)

  // Design matrices
  matrix[N, K_num] X_num;
  matrix[N, K_denom] X_denom;

  // Random effects structure
  int<lower=0> n_re_groups;
  int<lower=0> n_re_total;
  array[N, max(n_re_groups, 1)] int<lower=0, upper=max(n_re_total, 1)> re_idx;
  array[max(n_re_groups, 1)] int<lower=0> re_group_size;
  array[max(n_re_groups, 1)] int<lower=0> re_group_start;
  array[max(n_re_groups, 1)] int<lower=0, upper=1> re_shared;

  // Spatial structure (optional)
  int<lower=0, upper=1> use_spatial;
  int<lower=0> n_spatial;
  int<lower=0> n_edges;
  array[use_spatial ? n_edges : 0] int<lower=1> node1;
  array[use_spatial ? n_edges : 0] int<lower=1> node2;
  array[use_spatial ? N : 0] int<lower=1> spatial_idx;

  // Priors
  real<lower=0> prior_sigma_U;
  real<lower=0, upper=1> prior_sigma_alpha;
  real<lower=0> prior_shape_U;
  real<lower=0, upper=1> prior_shape_alpha;
}

transformed data {
  real lambda_sigma = -log(prior_sigma_alpha) / prior_sigma_U;
  real lambda_shape = -log(prior_shape_alpha) / prior_shape_U;
}

parameters {
  // Fixed effects
  vector[K_num] beta_num;
  vector[K_denom] beta_denom;

  // Shared random effects (non-centered)
  vector[n_re_total] z_shared;
  vector<lower=0>[n_re_groups] sigma_shared;

  // Process-specific random effects
  vector[n_re_total] z_num;
  vector[n_re_total] z_denom;
  vector<lower=0>[n_re_groups] sigma_num;
  vector<lower=0>[n_re_groups] sigma_denom;

  // Gamma shape parameter for denominator
  real<lower=0> shape_denom;

  // Spatial effects (if used)
  vector[use_spatial ? n_spatial : 0] spatial_raw;
  real<lower=0> sigma_spatial[use_spatial ? 1 : 0];
}

transformed parameters {
  // Linear predictors
  vector[N] eta_num = X_num * beta_num;
  vector[N] eta_denom = X_denom * beta_denom;

  // Add random effects
  if (n_re_groups > 0) {
    for (g in 1:n_re_groups) {
      int start = re_group_start[g];
      int size = re_group_size[g];

      if (re_shared[g] == 1) {
        // Shared random effect
        vector[size] re_g = sigma_shared[g] * z_shared[start:(start + size - 1)];
        for (n in 1:N) {
          if (re_idx[n, g] > 0) {
            int idx = re_idx[n, g] - start + 1;
            eta_num[n] += re_g[idx];
            eta_denom[n] += re_g[idx];
          }
        }
      } else {
        // Separate random effects
        vector[size] re_num_g = sigma_num[g] * z_num[start:(start + size - 1)];
        vector[size] re_denom_g = sigma_denom[g] * z_denom[start:(start + size - 1)];
        for (n in 1:N) {
          if (re_idx[n, g] > 0) {
            int idx = re_idx[n, g] - start + 1;
            eta_num[n] += re_num_g[idx];
            eta_denom[n] += re_denom_g[idx];
          }
        }
      }
    }
  }

  // Add spatial effects (shared)
  if (use_spatial) {
    for (n in 1:N) {
      eta_num[n] += sigma_spatial[1] * spatial_raw[spatial_idx[n]];
      eta_denom[n] += sigma_spatial[1] * spatial_raw[spatial_idx[n]];
    }
  }
}

model {
  // Priors: Fixed effects
  beta_num ~ normal(0, 2.5);
  beta_denom ~ normal(0, 2.5);

  // Priors: Random effect SDs
  for (g in 1:n_re_groups) {
    if (re_shared[g] == 1) {
      sigma_shared[g] ~ exponential(lambda_sigma);
    } else {
      sigma_num[g] ~ exponential(lambda_sigma);
      sigma_denom[g] ~ exponential(lambda_sigma);
    }
  }

  z_shared ~ std_normal();
  z_num ~ std_normal();
  z_denom ~ std_normal();

  // Prior: Gamma shape (PC prior favoring exponential)
  shape_denom ~ exponential(lambda_shape);

  // Spatial prior
  if (use_spatial) {
    sigma_spatial[1] ~ exponential(lambda_sigma);
    target += -0.5 * dot_self(spatial_raw[node1] - spatial_raw[node2]);
    sum(spatial_raw) ~ normal(0, 0.001 * n_spatial);
  }

  // Likelihood: Poisson for numerator (counts)
  y_num ~ poisson_log(eta_num);

  // Likelihood: Gamma for denominator (continuous positive)
  // Gamma parameterized as shape, rate where E[Y] = shape/rate = exp(eta)
  // So rate = shape / exp(eta) = shape * exp(-eta)
  {
    vector[N] rate_denom = shape_denom * exp(-eta_denom);
    y_denom ~ gamma(shape_denom, rate_denom);
  }
}

generated quantities {
  // Ratios: E[Y_num] / E[Y_denom] = exp(eta_num) / exp(eta_denom)
  vector[N] ratio;
  vector[N] log_ratio;

  // Expected values
  vector[N] mu_num = exp(eta_num);
  vector[N] mu_denom = exp(eta_denom);

  // Posterior predictive
  array[N] int y_num_rep;
  vector[N] y_denom_rep;

  // Log-likelihood for LOO/WAIC
  vector[N] log_lik_num;
  vector[N] log_lik_denom;
  vector[N] log_lik;

  // Group-level ratio effects
  array[n_re_groups > 0 ? n_re_groups : 1] vector[n_re_groups > 0 ? max(re_group_size) : 1] ratio_group;

  for (n in 1:N) {
    // Ratio of expected values
    log_ratio[n] = eta_num[n] - eta_denom[n];
    ratio[n] = exp(log_ratio[n]);

    // Posterior predictive
    y_num_rep[n] = poisson_log_rng(eta_num[n]);
    y_denom_rep[n] = gamma_rng(shape_denom, shape_denom * exp(-eta_denom[n]));

    // Log-likelihood
    log_lik_num[n] = poisson_log_lpmf(y_num[n] | eta_num[n]);
    log_lik_denom[n] = gamma_lpdf(y_denom[n] | shape_denom, shape_denom * exp(-eta_denom[n]));
    log_lik[n] = log_lik_num[n] + log_lik_denom[n];
  }

  // Group-level ratio effects
  if (n_re_groups > 0) {
    for (g in 1:n_re_groups) {
      int start = re_group_start[g];
      int size = re_group_size[g];

      for (j in 1:size) {
        if (re_shared[g] == 1) {
          // Shared effect cancels in ratio
          ratio_group[g][j] = 1.0;
        } else {
          real re_diff = sigma_num[g] * z_num[start + j - 1] -
                         sigma_denom[g] * z_denom[start + j - 1];
          ratio_group[g][j] = exp(re_diff);
        }
      }
    }
  }
}
