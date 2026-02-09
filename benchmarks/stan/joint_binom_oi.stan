// One-Inflated Binomial model with SHARED random effects
// OI = excess probability of y = n (all trials successful)
// Row 78 in gradient_methods.md
//
// P(Y = n | n trials) = pi + (1-pi) * Binomial(n | n, p)
// P(Y = y | y < n)    = (1-pi) * Binomial(y | n, p)
//
// pi is modeled via logit link: logit(pi) = gamma_0

data {
  int<lower=1> N;
  array[N] int<lower=0> y;        // Successes
  array[N] int<lower=1> trials;   // Number of trials
  int<lower=1> p;                 // Number of predictors
  matrix[N, p] X;                 // Design matrix

  // Random effects
  int<lower=1> n_groups;
  array[N] int<lower=1,upper=n_groups> group_idx;
}

parameters {
  vector[p] beta;                 // Coefficients

  // SHARED random effects (non-centered)
  vector[n_groups] z_re;
  real<lower=0> sigma_re;

  // One-inflation parameter
  real logit_oi;                  // logit(pi) for one-inflation
}

transformed parameters {
  vector[n_groups] re = sigma_re * z_re;
}

model {
  vector[N] eta;
  vector[N] prob;

  // Linear predictor with SHARED RE
  eta = X * beta;

  for (n in 1:N) {
    eta[n] += re[group_idx[n]];
  }

  // Priors
  beta ~ normal(0, 10);

  // RE priors
  z_re ~ std_normal();
  sigma_re ~ cauchy(0, 2.5);

  // OI prior (logit scale)
  logit_oi ~ normal(0, 1.5);

  // Probability on natural scale
  prob = inv_logit(eta);

  // One-Inflated Binomial likelihood
  {
    real oi_prob = inv_logit(logit_oi);
    for (n in 1:N) {
      if (y[n] == trials[n]) {
        // y = n (all trials successful) = mixture of one-inflation and binomial
        // log(pi + (1-pi)*p^n)
        target += log_sum_exp(
          log(oi_prob),
          log1m(oi_prob) + binomial_lpmf(y[n] | trials[n], prob[n])
        );
      } else {
        // y < n: just binomial with (1-pi) weight
        target += log1m(oi_prob) + binomial_lpmf(y[n] | trials[n], prob[n]);
      }
    }
  }
}

generated quantities {
  vector[N] log_lik;
  real oi_prob = inv_logit(logit_oi);

  for (n in 1:N) {
    real eta_n = X[n] * beta + re[group_idx[n]];
    real prob_n = inv_logit(eta_n);

    if (y[n] == trials[n]) {
      log_lik[n] = log_sum_exp(
        log(oi_prob),
        log1m(oi_prob) + binomial_lpmf(y[n] | trials[n], prob_n)
      );
    } else {
      log_lik[n] = log1m(oi_prob) + binomial_lpmf(y[n] | trials[n], prob_n);
    }
  }
}
