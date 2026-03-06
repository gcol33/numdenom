# Simple gamma_gamma test with known parameters

library(numdenom)

set.seed(123)

# Simpler test - intercept only
N <- 200

# True parameters
beta_num <- c(2.0)  # just intercept
beta_denom <- c(3.0)  # just intercept
shape_num <- 4
shape_denom <- 6

# Generate data
mu_num <- exp(beta_num[1])
mu_denom <- exp(beta_denom[1])

y_num <- rgamma(N, shape = shape_num, rate = shape_num / mu_num)
y_denom <- rgamma(N, shape = shape_denom, rate = shape_denom / mu_denom)

df <- data.frame(y = y_num, denom = y_denom)

cat("=== True parameters ===\n")
cat("beta_num[1] =", beta_num[1], "=> mu_num =", mu_num, "\n")
cat("beta_denom[1] =", beta_denom[1], "=> mu_denom =", mu_denom, "\n")
cat("shape_num =", shape_num, "\n")
cat("shape_denom =", shape_denom, "\n")

cat("\n=== Data summary ===\n")
cat("mean(y_num) =", mean(y_num), "(expected:", mu_num, ")\n")
cat("mean(y_denom) =", mean(y_denom), "(expected:", mu_denom, ")\n")
cat("log(mean(y_num)) =", log(mean(y_num)), "(expected:", beta_num[1], ")\n")
cat("log(mean(y_denom)) =", log(mean(y_denom)), "(expected:", beta_denom[1], ")\n")

# Fit with longer chains
cat("\n=== Fitting with H gradient ===\n")
fit <- ratiod(
  y | denom ~ 1,
  data = df,
  family = ratiod_gamma_gamma(),
  iter = 2000, warmup = 1000, chains = 2,
  gradient_mode = "H",
  seed = 42
)

# Extract draws
draws <- as.matrix(fit$draws)

cat("\n=== Posterior summary ===\n")
cat("beta_num[1]:   mean =", mean(draws[,"beta_num[1]"]),
    ", sd =", sd(draws[,"beta_num[1]"]),
    ", true =", beta_num[1], "\n")
cat("beta_denom[1]: mean =", mean(draws[,"beta_denom[1]"]),
    ", sd =", sd(draws[,"beta_denom[1]"]),
    ", true =", beta_denom[1], "\n")
cat("shape_num:     mean =", mean(draws[,"shape_num"]),
    ", sd =", sd(draws[,"shape_num"]),
    ", true =", shape_num, "\n")
cat("shape_denom:   mean =", mean(draws[,"shape_denom"]),
    ", sd =", sd(draws[,"shape_denom"]),
    ", true =", shape_denom, "\n")

# Check convergence
cat("\n=== Diagnostics ===\n")
n_div <- sum(fit$divergences)
cat("Divergences:", n_div, "\n")

# Compute Rhat and ESS
library(posterior)
summ <- summarise_draws(as_draws(fit$draws))
print(summ)
