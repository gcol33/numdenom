// quotr_binomial.stan
// Trial-based ratio model: Binomial numerator with fixed or modelled denominator
// Y_i ~ Binomial(N_i, p_i)
// logit(p_i) = X_i * beta + random effects + spatial

functions {
  real pc_prior_sd_lpdf(real sigma, real U, real alpha) {
    real lambda = -log(alpha) / U;
    return exponential_lpdf(sigma | lambda);
  }
}

data {
  // Dimensions
  int<lower=1> N;                         // Number of observations
  int<lower=1> K;                         // Number of fixed effects

  // Responses
  array[N] int<lower=0> y_num;            // Successes (numerator)
  array[N] int<lower=0> y_denom;          // Trials (denominator) - treated as known

  // Design matrix
  matrix[N, K] X;

  // Random effects structure
  int<lower=0> n_re_groups;
  int<lower=0> n_re_total;
  array[N, max(n_re_groups, 1)] int<lower=0, upper=max(n_re_total, 1)> re_idx;
  array[max(n_re_groups, 1)] int<lower=0> re_group_size;
  array[max(n_re_groups, 1)] int<lower=0> re_group_start;

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
  real<lower=0> prior_beta_sd;
}

transformed data {
  real lambda_sigma = -log(prior_sigma_alpha) / prior_sigma_U;
}

parameters {
  // Fixed effects
  vector[K] beta;

  // Random effects (non-centered)
  vector[n_re_total] z_re;
  vector<lower=0>[n_re_groups] sigma_re;

  // Spatial effects (if used)
  vector[use_spatial ? n_spatial : 0] spatial_raw;
  array[use_spatial ? 1 : 0] real<lower=0> sigma_spatial;
}

transformed parameters {
  // Linear predictor
  vector[N] eta = X * beta;

  // Add random effects
  if (n_re_groups > 0) {
    for (g in 1:n_re_groups) {
      int start = re_group_start[g];
      int size = re_group_size[g];
      vector[size] re_g = sigma_re[g] * z_re[start:(start + size - 1)];

      for (n in 1:N) {
        if (re_idx[n, g] > 0) {
          int idx = re_idx[n, g] - start + 1;
          eta[n] += re_g[idx];
        }
      }
    }
  }

  // Add spatial effects
  if (use_spatial) {
    for (n in 1:N) {
      eta[n] += sigma_spatial[1] * spatial_raw[spatial_idx[n]];
    }
  }
}

model {
  // Priors: Fixed effects
  beta ~ normal(0, prior_beta_sd);

  // Priors: Random effect SDs
  for (g in 1:n_re_groups) {
    sigma_re[g] ~ exponential(lambda_sigma);
  }
  z_re ~ std_normal();

  // Spatial prior (ICAR)
  if (use_spatial) {
    sigma_spatial[1] ~ exponential(lambda_sigma);
    target += -0.5 * dot_self(spatial_raw[node1] - spatial_raw[node2]);
    sum(spatial_raw) ~ normal(0, 0.001 * n_spatial);
  }

  // Likelihood: Binomial
  y_num ~ binomial_logit(y_denom, eta);
}

generated quantities {
  // Probabilities (the "ratio" for binomial is just the probability)
  vector[N] prob = inv_logit(eta);
  vector[N] log_odds = eta;

  // Group-level probabilities
  array[n_re_groups > 0 ? n_re_groups : 1] vector[n_re_groups > 0 ? max(re_group_size) : 1] prob_group;

  // Posterior predictive
  array[N] int y_num_rep;

  // Log-likelihood for LOO/WAIC
  vector[N] log_lik;

  for (n in 1:N) {
    y_num_rep[n] = binomial_rng(y_denom[n], prob[n]);
    log_lik[n] = binomial_logit_lpmf(y_num[n] | y_denom[n], eta[n]);
  }

  // Group-level probability effects
  if (n_re_groups > 0) {
    for (g in 1:n_re_groups) {
      int start = re_group_start[g];
      int size = re_group_size[g];

      for (j in 1:size) {
        real re_effect = sigma_re[g] * z_re[start + j - 1];
        // Probability at group mean (intercept + group effect)
        prob_group[g][j] = inv_logit(beta[1] + re_effect);
      }
    }
  }
}
