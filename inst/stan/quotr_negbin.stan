// quotr_negbin.stan
// Two-process ratio model: NegBin numerator, NegBin denominator
// Shared random effects enter both linear predictors

functions {
  // PC prior density for standard deviation
  // P(sigma > U) = alpha => exponential on sigma with rate -log(alpha)/U
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
  array[N] int<lower=0> y_num;            // Numerator counts
  array[N] int<lower=0> y_denom;          // Denominator counts

  // Design matrices
  matrix[N, K_num] X_num;                 // Numerator fixed effects
  matrix[N, K_denom] X_denom;             // Denominator fixed effects

  // Random effects structure
  int<lower=0> n_re_groups;               // Number of grouping factors
  int<lower=0> n_re_total;                // Total number of RE levels across all groups

  // For each observation, indices into the RE vectors
  // Packed format: re_idx[n, g] gives the level for observation n in group g
  array[N, n_re_groups] int<lower=0, upper=n_re_total> re_idx;

  // Group sizes (for building RE vectors)
  array[n_re_groups] int<lower=0> re_group_size;
  array[n_re_groups] int<lower=0> re_group_start;  // Start index in flattened RE vector

  // Shared vs separate random effects
  // 0 = separate, 1 = shared
  array[n_re_groups] int<lower=0, upper=1> re_shared;

  // Spatial structure (optional)
  int<lower=0, upper=1> use_spatial;
  int<lower=0> n_spatial;                 // Number of spatial units

  // CAR adjacency structure (if use_spatial)
  int<lower=0> n_edges;                   // Number of neighbor pairs
  array[use_spatial ? n_edges : 0] int<lower=1> node1;
  array[use_spatial ? n_edges : 0] int<lower=1> node2;

  // Spatial index for observations
  array[use_spatial ? N : 0] int<lower=1> spatial_idx;

  // Priors (PC prior parameters)
  real<lower=0> prior_sigma_U;            // Upper bound for P(sigma > U) = alpha
  real<lower=0, upper=1> prior_sigma_alpha;
  real<lower=0> prior_phi_U;              // For overdispersion
  real<lower=0, upper=1> prior_phi_alpha;
}

transformed data {
  // Precompute PC prior rate
  real lambda_sigma = -log(prior_sigma_alpha) / prior_sigma_U;
  real lambda_phi = -log(prior_phi_alpha) / prior_phi_U;
}

parameters {
  // Fixed effects
  vector[K_num] beta_num;
  vector[K_denom] beta_denom;

  // Shared random effects (non-centered)
  vector[n_re_total] z_shared;
  vector<lower=0>[n_re_groups] sigma_shared;

  // Process-specific random effects (for non-shared groups)
  vector[n_re_total] z_num;
  vector[n_re_total] z_denom;
  vector<lower=0>[n_re_groups] sigma_num;
  vector<lower=0>[n_re_groups] sigma_denom;

  // Overdispersion parameters
  real<lower=0> phi_num;
  real<lower=0> phi_denom;

  // Spatial effects (if used)
  vector[use_spatial ? n_spatial : 0] spatial_raw;
  array[use_spatial ? 1 : 0] real<lower=0> sigma_spatial;
}

transformed parameters {
  // Linear predictors
  vector[N] eta_num = X_num * beta_num;
  vector[N] eta_denom = X_denom * beta_denom;

  // Add random effects
  for (g in 1:n_re_groups) {
    int start = re_group_start[g];
    int size = re_group_size[g];

    if (re_shared[g] == 1) {
      // Shared random effect: same effect enters both predictors
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

  // Add spatial effects (shared between processes)
  if (use_spatial) {
    for (n in 1:N) {
      eta_num[n] += sigma_spatial[1] * spatial_raw[spatial_idx[n]];
      eta_denom[n] += sigma_spatial[1] * spatial_raw[spatial_idx[n]];
    }
  }
}

model {
  // Priors: Fixed effects (weakly informative)
  beta_num ~ normal(0, 2.5);
  beta_denom ~ normal(0, 2.5);

  // Priors: Random effect SDs (PC priors)
  for (g in 1:n_re_groups) {
    if (re_shared[g] == 1) {
      sigma_shared[g] ~ exponential(lambda_sigma);
    } else {
      sigma_num[g] ~ exponential(lambda_sigma);
      sigma_denom[g] ~ exponential(lambda_sigma);
    }
  }

  // Non-centered parameterization
  z_shared ~ std_normal();
  z_num ~ std_normal();
  z_denom ~ std_normal();

  // Priors: Overdispersion (PC prior favoring Poisson limit)
  phi_num ~ exponential(lambda_phi);
  phi_denom ~ exponential(lambda_phi);

  // Spatial prior (ICAR)
  if (use_spatial) {
    sigma_spatial[1] ~ exponential(lambda_sigma);
    // Sum-to-zero soft constraint + pairwise differences
    target += -0.5 * dot_self(spatial_raw[node1] - spatial_raw[node2]);
    sum(spatial_raw) ~ normal(0, 0.001 * n_spatial);
  }

  // Likelihoods
  y_num ~ neg_binomial_2_log(eta_num, phi_num);
  y_denom ~ neg_binomial_2_log(eta_denom, phi_denom);
}

generated quantities {
  // THE KEY: Ratios computed post hoc with full uncertainty propagation
  vector[N] ratio;
  vector[N] log_ratio;

  // Group-level ratio effects (for each shared RE group)
  // These represent multiplicative effects on the ratio
  array[n_re_groups] vector[n_re_groups > 0 ? max(re_group_size) : 0] ratio_group;

  // Posterior predictive draws
  array[N] int y_num_rep;
  array[N] int y_denom_rep;

  // Log-likelihood for LOO/WAIC
  vector[N] log_lik_num;
  vector[N] log_lik_denom;
  vector[N] log_lik;

  // Observation-level ratios and log-likelihood
  for (n in 1:N) {
    // Ratio of expected values: E[Y_num] / E[Y_denom]
    log_ratio[n] = eta_num[n] - eta_denom[n];
    ratio[n] = exp(log_ratio[n]);

    // Posterior predictive
    y_num_rep[n] = neg_binomial_2_log_rng(eta_num[n], phi_num);
    y_denom_rep[n] = neg_binomial_2_log_rng(eta_denom[n], phi_denom);

    // Log-likelihood
    log_lik_num[n] = neg_binomial_2_log_lpmf(y_num[n] | eta_num[n], phi_num);
    log_lik_denom[n] = neg_binomial_2_log_lpmf(y_denom[n] | eta_denom[n], phi_denom);
    log_lik[n] = log_lik_num[n] + log_lik_denom[n];
  }

  // Group-level ratio effects
  // For shared RE, this shows how each group deviates from overall ratio
  for (g in 1:n_re_groups) {
    int start = re_group_start[g];
    int size = re_group_size[g];

    for (j in 1:size) {
      if (re_shared[g] == 1) {
        // Shared effect: enters both, so cancels in ratio
        // But we can report the shared effect magnitude
        ratio_group[g][j] = 1.0;  // No net effect on ratio from shared RE
      } else {
        // Separate effects: difference affects ratio
        real re_diff = sigma_num[g] * z_num[start + j - 1] -
                       sigma_denom[g] * z_denom[start + j - 1];
        ratio_group[g][j] = exp(re_diff);
      }
    }
  }
}
