# Debug lognormal - check if there's a num/denom swap issue
library(numdenom)

set.seed(42)

# Generate data where sigma_num and sigma_denom are very different
N <- 100
mu_num <- 2.0
mu_denom <- 4.0
sigma_num <- 0.2  # SMALL
sigma_denom <- 1.0  # LARGE (very different from sigma_num)

y_num <- exp(rnorm(N, mu_num, sigma_num))
y_denom <- exp(rnorm(N, mu_denom, sigma_denom))

dat <- data.frame(y_num = y_num, y_denom = y_denom)

cat("True values:\n")
cat("  beta_num:", mu_num, "\n")
cat("  beta_denom:", mu_denom, "\n")
cat("  sigma_num:", sigma_num, "(SMALL)\n")
cat("  sigma_denom:", sigma_denom, "(LARGE)\n\n")

cat("Empirical (from data):\n")
cat("  mean(log(y_num)):", mean(log(y_num)), "\n")
cat("  mean(log(y_denom)):", mean(log(y_denom)), "\n")
cat("  sd(log(y_num)):", sd(log(y_num)), "\n")
cat("  sd(log(y_denom)):", sd(log(y_denom)), "\n\n")

# Fit model
devtools::load_all()
fit <- tratio(
  y_num | y_denom ~ 1,
  data = dat,
  family = ratiod_lognormal(),
  control = list(iter = 2000, warmup = 1000, chains = 1, seed = 123)
)

print(summary(fit))

# Check if sigma_num and sigma_denom got swapped
draws <- as.matrix(fit$draws)
cat("\nEstimates:\n")
cat("  sigma_num:", mean(draws[,"sigma_num"]), "(should be ~0.2)\n")
cat("  sigma_denom:", mean(draws[,"sigma_denom"]), "(should be ~1.0)\n")
cat("  beta_num:", mean(draws[,"beta_num[1]"]), "(should be ~2.0)\n")
cat("  beta_denom:", mean(draws[,"beta_denom[1]"]), "(should be ~4.0)\n")
