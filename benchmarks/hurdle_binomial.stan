// Hurdle binomial model
// Zero process: Bernoulli for P(Y > 0)
// Count process: Truncated binomial for Y | Y > 0

data {
  int<lower=0> N;              // number of observations
  array[N] int<lower=0> y;     // successes
  array[N] int<lower=1> trials; // trials
  matrix[N, 2] X;              // design matrix (intercept + x)
  int<lower=1> N_sites;        // number of sites
  array[N] int<lower=1> site;  // site index
}

parameters {
  vector[2] beta;              // fixed effects for count
  real beta_hu;                // hurdle intercept (logit P(Y>0))
  vector[N_sites] z_site;      // standardized site effects
  real<lower=0> sigma_site;    // site SD
}

transformed parameters {
  vector[N_sites] u_site = z_site * sigma_site;
}

model {
  // Priors
  beta ~ normal(0, 2);
  beta_hu ~ normal(0, 2);
  z_site ~ std_normal();
  sigma_site ~ exponential(1);

  // Likelihood
  for (n in 1:N) {
    real eta = X[n] * beta + u_site[site[n]];
    real p = inv_logit(eta);
    real theta = inv_logit(beta_hu);  // P(Y > 0)

    if (y[n] == 0) {
      // Zero from hurdle
      target += log1m(theta);
    } else {
      // Positive from truncated binomial
      // P(Y=y|Y>0) = P(Y=y) / P(Y>0) = P(Y=y) / (1 - P(Y=0))
      // P(Y=0) = (1-p)^n
      real log_p_zero = trials[n] * log1m(p);
      target += log(theta) + binomial_lpmf(y[n] | trials[n], p) - log1m_exp(log_p_zero);
    }
  }
}
