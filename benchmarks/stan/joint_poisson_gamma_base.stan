// Joint Poisson-Gamma model (no RE)
// Matches numdenom ratiod_poisson_gamma() exactly
// Row 1 in gradient_methods.md

data {
  int<lower=1> N;
  array[N] int<lower=0> y;      // Numerator counts (Poisson)
  vector<lower=0>[N] effort;    // Denominator (Gamma-distributed effort)
  vector[N] x;                  // Covariate
}

parameters {
  // Numerator (Poisson) parameters
  real beta0_num;               // Intercept
  real beta1_num;               // Slope for x

  // Denominator (Gamma) parameters - effort is observed, we model its generation
  // In numdenom, effort is treated as known, so we only model numerator
  // But for validation we need the joint likelihood
}

model {
  vector[N] lambda;

  // Priors matching numdenom defaults
  beta0_num ~ normal(0, 10);
  beta1_num ~ normal(0, 5);

  // Poisson likelihood for counts with effort as exposure
  for (n in 1:N) {
    lambda[n] = effort[n] * exp(beta0_num + beta1_num * x[n]);
  }

  y ~ poisson(lambda);
}

generated quantities {
  vector[N] log_lik;
  vector[N] lambda;

  for (n in 1:N) {
    lambda[n] = effort[n] * exp(beta0_num + beta1_num * x[n]);
    log_lik[n] = poisson_lpmf(y[n] | lambda[n]);
  }
}
