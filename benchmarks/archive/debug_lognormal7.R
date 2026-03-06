# Debug lognormal - check step size adaptation
library(numdenom)

set.seed(42)
devtools::load_all()

N <- 50
mu_num <- 2.0
mu_denom <- 4.0
sigma_num <- 0.5
sigma_denom <- 0.3

y_num <- exp(rnorm(N, mu_num, sigma_num))
y_denom <- exp(rnorm(N, mu_denom, sigma_denom))

dat <- data.frame(y_num = y_num, y_denom = y_denom)

# Fit with very short chain and check trace
cat("=== Fitting with verbose sampling ===\n")
fit <- ratiod(
  y_num | y_denom ~ 1,
  data = dat,
  family = ratiod_lognormal(),
  iter = 50,
  warmup = 25,
  chains = 1,
  seed = 123
)

# Look at actual draws
draws <- as.matrix(fit$draws)
cat("\nFirst 10 draws:\n")
print(draws[1:10,])

cat("\nLast 10 draws:\n")
print(draws[(nrow(draws)-9):nrow(draws),])

cat("\nParameter ranges:\n")
for (col in colnames(draws)) {
  cat("  ", col, ": [", min(draws[,col]), ",", max(draws[,col]), "]\n")
}

# Check acceptance rate more carefully
cat("\nDiagnostics from fit object:\n")
cat("  Acceptance rate:", fit$diagnostics$accept_rate, "\n")

# Try with very different step size
cat("\n=== Testing with custom step size ===\n")
fit2 <- ratiod(
  y_num | y_denom ~ 1,
  data = dat,
  family = ratiod_lognormal(),
  iter = 50,
  warmup = 25,
  chains = 1,
  seed = 123,
  step_size = 0.01  # Much smaller step size
)

draws2 <- as.matrix(fit2$draws)
cat("\nWith step_size=0.01, parameter ranges:\n")
for (col in colnames(draws2)) {
  cat("  ", col, ": [", min(draws2[,col]), ",", max(draws2[,col]), "]\n")
}
