# Debug time scaling between numdenom and Stan

library(numdenom)

set.seed(42)

N_OBS <- 200
N_TIMES <- 10

# Setup data
x <- rnorm(N_OBS)
time <- rep(1:N_TIMES, length.out = N_OBS)

df <- data.frame(
  y = rpois(N_OBS, 10),
  denom = rpois(N_OBS, 20),
  x = x, time = time
)

# Create temporal GP spec
temp <- temporal_gp(time_var = "time", cov = "exponential")

# Validate it to see what time_values are used
temp_validated <- numdenom:::validate_temporal(temp, df)

cat("numdenom time_values (unique times):\n")
print(temp_validated$time_values)

cat("\nStan scaled time values:\n")
time_unique_scaled <- (1:N_TIMES - mean(1:N_TIMES)) / sd(1:N_TIMES)
print(time_unique_scaled)

cat("\nTime differences:\n")
cat("numdenom dt (consecutive):", diff(temp_validated$time_values), "\n")
cat("Stan dt (consecutive):     ", diff(time_unique_scaled), "\n")
