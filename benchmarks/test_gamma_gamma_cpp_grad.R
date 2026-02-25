# Direct test of C++ gradient computation for gamma_gamma

library(numdenom)

set.seed(42)

# Small test
N <- 50
y_num <- rgamma(N, shape = 4, rate = 4 / 7)  # mean = 7
y_denom <- rgamma(N, shape = 6, rate = 6 / 20)  # mean = 20

df <- data.frame(y = y_num, denom = y_denom)

# Fit a very short chain to see initial gradients
cat("=== Testing C++ gradient computation ===\n")

# Use verbose mode and check if there are issues
fit <- ratiod(
  y | denom ~ 1,
  data = df,
  family = ratiod_gamma_gamma(),
  iter = 100, warmup = 50, chains = 1,
  gradient_mode = "H",
  seed = 42,
  verbose = TRUE
)

cat("\n=== Checking draws ===\n")
draws <- as.matrix(fit$draws)
cat("shape_num range:", range(draws[,"shape_num"]), "\n")
cat("shape_denom range:", range(draws[,"shape_denom"]), "\n")

# Let's also try with N mode (numerical gradients)
cat("\n=== Fitting with N (numerical) gradients ===\n")
fit_N <- ratiod(
  y | denom ~ 1,
  data = df,
  family = ratiod_gamma_gamma(),
  iter = 100, warmup = 50, chains = 1,
  gradient_mode = "N",
  seed = 42,
  verbose = TRUE
)

draws_N <- as.matrix(fit_N$draws)
cat("shape_num range (N):", range(draws_N[,"shape_num"]), "\n")
cat("shape_denom range (N):", range(draws_N[,"shape_denom"]), "\n")

# Compare
cat("\n=== Comparison ===\n")
cat("H mode - shape_num mean:", mean(draws[,"shape_num"]), "\n")
cat("N mode - shape_num mean:", mean(draws_N[,"shape_num"]), "\n")
cat("H mode - shape_denom mean:", mean(draws[,"shape_denom"]), "\n")
cat("N mode - shape_denom mean:", mean(draws_N[,"shape_denom"]), "\n")
