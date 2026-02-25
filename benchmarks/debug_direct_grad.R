# Directly compare gradients at the same point

library(numdenom)

set.seed(42)

# Minimal test case
N <- 10
x <- rnorm(N)
y_num <- rgamma(N, shape = 5, rate = 5 / exp(2 + 0.3 * x))
y_denom <- rgamma(N, shape = 8, rate = 8 / exp(3 + 0.2 * x))

df <- data.frame(y = y_num, denom = y_denom, x = x)

# Export a test function from C++ if available
if (exists("cpp_test_gradient_comparison", where = "package:numdenom")) {
  cat("cpp_test_gradient_comparison found!\n")
} else {
  cat("cpp_test_gradient_comparison not found - using workaround\n")
  
  # Parse formula and prepare data
  parsed <- numdenom:::ratiod_formula(y | denom ~ x, data = df)
  family <- ratiod_gamma_gamma()
  
  # Run 1 iteration with each mode and print verbose output
  cat("\n=== Single iteration with N mode ===\n")
  fit_N <- ratiod(
    y | denom ~ x,
    data = df,
    family = ratiod_gamma_gamma(),
    iter = 2, warmup = 1, chains = 1,
    gradient_mode = "N",
    verbose = TRUE
  )
  
  cat("\n=== Single iteration with A mode ===\n")
  fit_A <- ratiod(
    y | denom ~ x,
    data = df,
    family = ratiod_gamma_gamma(),
    iter = 2, warmup = 1, chains = 1,
    gradient_mode = "A",
    verbose = TRUE
  )
}
