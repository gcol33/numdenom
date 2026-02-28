# Verify multiscale temporal gradients match across modes
# Check if H matches A_t (which had 0 divergent transitions)

library(numdenom)

N_OBS <- 100
N_TIMES <- 10

set.seed(42)

# Generate temporal data
time_idx <- rep(1:N_TIMES, length.out = N_OBS)
x <- rnorm(N_OBS)
eta <- 2 + 0.3 * x + cumsum(rnorm(N_TIMES, 0, 0.1))[time_idx]
y <- rpois(N_OBS, exp(eta))
effort <- rgamma(N_OBS, 10, 1)

df <- data.frame(
  y = y,
  effort = effort,
  x = x,
  time = factor(time_idx)
)

cat("\n", strrep("=", 60), "\n")
cat("Multiscale Temporal Gradient Verification\n")
cat(strrep("=", 60), "\n\n")

# Test with A_t mode (which had 0 divergent)
cat("Testing gradient_mode = 'A_t' (0 divergent in benchmark)...\n")
tryCatch({
  fit_at <- ratiod(
    y | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    iter = 100,
    chains = 1,
    gradient_mode = "A_t",
    temporal = temporal_multiscale(time_var = "time"),
    refresh = 0
  )
  cat(sprintf("  Divergent: %d\n", fit_at$divergent_count))
}, error = function(e) {
  cat(sprintf("  ERROR: %s\n", conditionMessage(e)))
})

cat("\n")

# Test with H mode
cat("Testing gradient_mode = 'H'...\n")
tryCatch({
  fit_h <- ratiod(
    y | effort ~ x,
    data = df,
    family = ratiod_poisson_gamma(),
    iter = 100,
    chains = 1,
    gradient_mode = "H",
    temporal = temporal_multiscale(time_var = "time"),
    refresh = 0
  )
  cat(sprintf("  Divergent: %d\n", fit_h$divergent_count))
}, error = function(e) {
  cat(sprintf("  ERROR: %s\n", conditionMessage(e)))
})

cat("\n")
cat("If A_t has 0 divergent but H has many, the H gradient is incorrect.\n")
cat("Need to verify: max(|grad_H - grad_A_t|) < 1e-5\n")
