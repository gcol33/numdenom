# Diagnostic script to identify the root cause of gamma_gamma validation failure

library(numdenom)

set.seed(42)

# Simple test case - no RE
N <- 100
x <- rnorm(N)
true_beta_num <- c(2.0, 0.3)  # intercept, slope
true_beta_denom <- c(3.0, 0.2)
shape_num <- 5
shape_denom <- 8

# Generate mu for each observation
eta_num <- cbind(1, x) %*% true_beta_num
eta_denom <- cbind(1, x) %*% true_beta_denom
mu_num <- exp(eta_num)
mu_denom <- exp(eta_denom)

# Generate gamma data
y_num <- rgamma(N, shape = shape_num, rate = shape_num / mu_num)
y_denom <- rgamma(N, shape = shape_denom, rate = shape_denom / mu_denom)

df <- data.frame(y = y_num, denom = y_denom, x = x)

cat("True parameters:\n")
cat("  beta_num = (", true_beta_num[1], ",", true_beta_num[2], ")\n")
cat("  beta_denom = (", true_beta_denom[1], ",", true_beta_denom[2], ")\n")
cat("  shape_num =", shape_num, "\n")
cat("  shape_denom =", shape_denom, "\n")

# Fit with numdenom
cat("\nFitting numdenom...\n")
fit <- tratio(
  y | denom ~ x,
  data = df,
  family = ratiod_gamma_gamma(),
  control = list(iter = 1000, warmup = 500, chains = 1, verbose = TRUE)
)

cat("\n\nNumdenom estimates:\n")
print(summary(fit))

# Check draws
draws <- as.matrix(fit$draws)
cat("\n\nPosterior means from numdenom:\n")
cat("  beta_num[1] =", mean(draws[,"beta_num[1]"]), "\n")
cat("  beta_num[2] =", mean(draws[,"beta_num[2]"]), "\n")
cat("  beta_denom[1] =", mean(draws[,"beta_denom[1]"]), "\n")
cat("  beta_denom[2] =", mean(draws[,"beta_denom[2]"]), "\n")
if ("phi_num" %in% colnames(draws)) {
  cat("  phi_num =", mean(draws[,"phi_num"]), "(true:", shape_num, ")\n")
}
if ("phi_denom" %in% colnames(draws)) {
  cat("  phi_denom =", mean(draws[,"phi_denom"]), "(true:", shape_denom, ")\n")
}

# Check if betas are recovering the truth
cat("\n\nComparison to truth:\n")
cat("  beta_num[2] error:", mean(draws[,"beta_num[2]"]) - true_beta_num[2], "\n")
cat("  beta_denom[2] error:", mean(draws[,"beta_denom[2]"]) - true_beta_denom[2], "\n")

# Now let's check what the Stan model recovers with SAME priors
cat("\n\n=== Now comparing with Stan using same priors ===\n")

library(cmdstanr)

# Write Stan model with numdenom priors
stan_code <- "
data {
  int<lower=1> N;
  vector<lower=0>[N] y_num;
  vector<lower=0>[N] y_denom;
  int<lower=1> p;
  matrix[N, p] X;
}

parameters {
  vector[p] beta_num;
  vector[p] beta_denom;
  real<lower=0> shape_num;
  real<lower=0> shape_denom;
}

model {
  vector[N] mu_num = exp(X * beta_num);
  vector[N] mu_denom = exp(X * beta_denom);

  // Use numdenom priors:
  // - beta ~ normal(0, 10)
  // - shape ~ gamma(2, 0.5)  <- numdenom default for gamma_gamma
  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  shape_num ~ gamma(2, 0.5);
  shape_denom ~ gamma(2, 0.5);

  // Likelihoods (same as numdenom)
  for (n in 1:N) {
    if (y_num[n] > 0) {
      target += gamma_lpdf(y_num[n] | shape_num, shape_num / mu_num[n]);
    }
    if (y_denom[n] > 0) {
      target += gamma_lpdf(y_denom[n] | shape_denom, shape_denom / mu_denom[n]);
    }
  }
}
"

writeLines(stan_code, "benchmarks/debug_gg_same_priors.stan")
stan_model <- cmdstan_model("benchmarks/debug_gg_same_priors.stan")

stan_data <- list(
  N = N,
  y_num = y_num,
  y_denom = y_denom,
  p = 2,
  X = cbind(1, x)
)

cat("Fitting Stan model with same priors...\n")
fit_stan <- stan_model$sample(
  data = stan_data,
  iter_sampling = 500,
  iter_warmup = 500,
  chains = 1,
  refresh = 100
)

draws_stan <- fit_stan$draws(format = "df")

cat("\n\nStan posterior means (same priors):\n")
cat("  beta_num[1] =", mean(draws_stan$`beta_num[1]`), "\n")
cat("  beta_num[2] =", mean(draws_stan$`beta_num[2]`), "\n")
cat("  beta_denom[1] =", mean(draws_stan$`beta_denom[1]`), "\n")
cat("  beta_denom[2] =", mean(draws_stan$`beta_denom[2]`), "\n")
cat("  shape_num =", mean(draws_stan$shape_num), "(true:", shape_num, ")\n")
cat("  shape_denom =", mean(draws_stan$shape_denom), "(true:", shape_denom, ")\n")

cat("\n\nStan comparison to truth:\n")
cat("  beta_num[2] error:", mean(draws_stan$`beta_num[2]`) - true_beta_num[2], "\n")
cat("  beta_denom[2] error:", mean(draws_stan$`beta_denom[2]`) - true_beta_denom[2], "\n")

cat("\n\n=== Summary ===\n")
cat("numdenom beta_num[2]:", mean(draws[,"beta_num[2]"]), "\n")
cat("Stan beta_num[2]:    ", mean(draws_stan$`beta_num[2]`), "\n")
cat("Difference:          ", mean(draws[,"beta_num[2]"]) - mean(draws_stan$`beta_num[2]`), "\n")
