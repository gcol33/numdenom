# Detailed diagnostic script for gamma_gamma

library(numdenom)

set.seed(42)

# Simple test case
N <- 50  # Small for faster debugging
x <- rnorm(N)
true_beta_num <- c(2.0, 0.3)
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

cat("=== Data Summary ===\n")
cat("y_num range:", range(df$y), "\n")
cat("y_denom range:", range(df$denom), "\n")
cat("Any zeros in y_num?", any(df$y <= 0), "\n")
cat("Any zeros in y_denom?", any(df$denom <= 0), "\n")

# Parse the formula
parsed <- numdenom:::ratiod_formula(y | denom ~ x, data = df)
family <- ratiod_gamma_gamma()

cat("\n=== Parsed Formula ===\n")
cat("Model type:", family$name, "\n")
cat("Numerator response:", head(parsed$numerator$response), "...\n")
cat("Denominator response:", head(parsed$denominator$response), "...\n")
cat("X_num dims:", dim(parsed$numerator$X), "\n")
cat("X_denom dims:", dim(parsed$denominator$X), "\n")

# Check what prepare_hmc_data returns
cat("\n=== HMC Data Preparation ===\n")

# This calls prepare_hmc_data internally
# Let's trace what happens

# Manually call prepare_hmc_data with model_type
hmc_data <- numdenom:::prepare_hmc_data(parsed, family, df, model_type = "gamma_gamma")

cat("y_num (first 5):", hmc_data$y_num[1:5], "\n")
cat("y_denom (first 5):", hmc_data$y_denom[1:5], "\n")
cat("y_num_cont (first 5):", hmc_data$y_num_cont[1:5], "\n")
cat("y_denom_cont (first 5):", hmc_data$y_denom_cont[1:5], "\n")

cat("\n=== CRITICAL CHECK ===\n")
cat("For gamma_gamma, y_num should be 0 (dummy int)\n")
cat("y_num[1:5]:", hmc_data$y_num[1:5], "\n")
cat("y_num_cont should have the actual data\n")
cat("y_num_cont[1:5]:", hmc_data$y_num_cont[1:5], "\n")
cat("Original df$y[1:5]:", df$y[1:5], "\n")
cat("\nAre they equal?", all.equal(hmc_data$y_num_cont, as.numeric(df$y)), "\n")

# Now test with cpp_test_log_post if available
cat("\n=== Testing log_post_gamma_gamma ===\n")

# Let me compute what log_lik_gamma should be for observation 1
y1 <- df$y[1]
mu1 <- exp(true_beta_num[1] + true_beta_num[2] * df$x[1])
shape <- 5  # true shape

log_lik_gamma <- function(y, shape, mu) {
  rate <- shape / mu
  shape * log(rate) + (shape - 1) * log(y) - rate * y - lgamma(shape)
}

cat("For obs 1:\n")
cat("  y =", y1, "\n")
cat("  eta =", log(mu1), "\n")
cat("  mu =", mu1, "\n")
cat("  shape =", shape, "\n")
cat("  Expected log_lik =", log_lik_gamma(y1, shape, mu1), "\n")

# Fit the model and print detailed diagnostics
cat("\n=== Fitting Model ===\n")
fit <- tratio(
  y | denom ~ x,
  data = df,
  family = ratiod_gamma_gamma(),
  control = list(iter = 500, warmup = 250, chains = 1, verbose = TRUE)
)

# Check what parameters were stored
draws <- as.matrix(fit$draws)
cat("\n=== Parameter Names ===\n")
cat(colnames(draws), "\n")

cat("\n=== Posterior Stats ===\n")
for (param in colnames(draws)) {
  cat(sprintf("%s: mean=%.4f, sd=%.4f, ESS=%.1f\n",
              param, mean(draws[,param]), sd(draws[,param]),
              posterior::ess_basic(draws[,param])))
}

# Compare log_lik from numdenom vs manual calculation
cat("\n=== Log-likelihood check ===\n")
# At posterior mean
beta_num_hat <- c(mean(draws[,"beta_num[1]"]), mean(draws[,"beta_num[2]"]))
beta_denom_hat <- c(mean(draws[,"beta_denom[1]"]), mean(draws[,"beta_denom[2]"]))
shape_num_hat <- if ("phi_num" %in% colnames(draws)) mean(draws[,"phi_num"]) else NA
shape_denom_hat <- if ("phi_denom" %in% colnames(draws)) mean(draws[,"phi_denom"]) else NA

cat("Posterior mean estimates:\n")
cat("  beta_num:", beta_num_hat, "(true:", true_beta_num, ")\n")
cat("  beta_denom:", beta_denom_hat, "(true:", true_beta_denom, ")\n")
cat("  shape_num:", shape_num_hat, "(true:", shape_num, ")\n")
cat("  shape_denom:", shape_denom_hat, "(true:", shape_denom, ")\n")
