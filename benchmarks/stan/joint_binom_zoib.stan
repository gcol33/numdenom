// Zero-One Inflated Binomial model with SHARED random effects
// ZOIB = excess probability at both y=0 and y=n
// Row 79 in gradient_methods.md
//
// Three-component mixture:
// P(Y = 0)     = pi0 + (1-pi0-pi1) * Binomial(0 | n, p)
// P(Y = n)     = pi1 + (1-pi0-pi1) * Binomial(n | n, p)
// P(Y = y)     = (1-pi0-pi1) * Binomial(y | n, p), for 0 < y < n
//
// pi0, pi1 modeled via softmax on 3 logits (zero, one, neither)

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

  // Zero-One inflation parameters
  // Using stick-breaking: pi0 = inv_logit(alpha0), pi1 = (1-pi0) * inv_logit(alpha1)
  real alpha0;                    // logit for zero-inflation
  real alpha1;                    // logit for one-inflation (conditional on not zero)
}

transformed parameters {
  vector[n_groups] re = sigma_re * z_re;
  real<lower=0,upper=1> pi0 = inv_logit(alpha0);
  real<lower=0,upper=1> pi1_cond = inv_logit(alpha1);
  real<lower=0,upper=1> pi1 = (1 - pi0) * pi1_cond;
  real<lower=0,upper=1> pi_binom = 1 - pi0 - pi1;
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

  // ZOIB priors (logit scale)
  alpha0 ~ normal(0, 1.5);
  alpha1 ~ normal(0, 1.5);

  // Probability on natural scale
  prob = inv_logit(eta);

  // Zero-One Inflated Binomial likelihood
  for (n in 1:N) {
    real binom_lp = binomial_lpmf(y[n] | trials[n], prob[n]);

    if (y[n] == 0) {
      // y = 0: mixture of zero-inflation and binomial
      target += log_sum_exp(
        log(pi0),
        log(pi_binom) + binom_lp
      );
    } else if (y[n] == trials[n]) {
      // y = n: mixture of one-inflation and binomial
      target += log_sum_exp(
        log(pi1),
        log(pi_binom) + binom_lp
      );
    } else {
      // 0 < y < n: just binomial with pi_binom weight
      target += log(pi_binom) + binom_lp;
    }
  }
}

generated quantities {
  vector[N] log_lik;

  for (n in 1:N) {
    real eta_n = X[n] * beta + re[group_idx[n]];
    real prob_n = inv_logit(eta_n);
    real binom_lp = binomial_lpmf(y[n] | trials[n], prob_n);

    if (y[n] == 0) {
      log_lik[n] = log_sum_exp(log(pi0), log(pi_binom) + binom_lp);
    } else if (y[n] == trials[n]) {
      log_lik[n] = log_sum_exp(log(pi1), log(pi_binom) + binom_lp);
    } else {
      log_lik[n] = log(pi_binom) + binom_lp;
    }
  }
}
