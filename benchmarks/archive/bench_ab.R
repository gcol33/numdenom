# A/B benchmark script for momentum pool optimization
# Run this script twice: once before and once after the change
library(numdenom)

N <- 500; S <- 50
bench <- function(label, family_fn, data, formula_str, reps = 3) {
  times <- numeric(reps)
  for (i in 1:reps) {
    set.seed(100 + i)  # Different seeds to average NUTS randomness
    times[i] <- system.time({
      ratiod(as.formula(formula_str), data = data,
             family = family_fn(),
             iter = 500, warmup = 250, chains = 1, verbose = FALSE)
    })["elapsed"]
  }
  cat(sprintf("%-12s  median=%.2fs  [%s]\n", label,
      median(times), paste(sprintf("%.2f", times), collapse=", ")))
}

set.seed(42)
df <- data.frame(
  y_num = rnbinom(N, mu = 5, size = 2),
  y_denom = rnbinom(N, mu = 10, size = 3),
  x = rnorm(N),
  site = rep(1:S, each = N / S)
)

cat("=== A/B Benchmark (N=500, 500 iter, 3 reps) ===\n")
bench("PG base",  ratiod_poisson_gamma, df, "y_num | y_denom ~ x")
bench("NB base",  ratiod_negbin_negbin, df, "y_num | y_denom ~ x")
bench("PG+RE",    ratiod_poisson_gamma, df, "y_num | y_denom ~ x + (1|site)")
bench("NB+RE",    ratiod_negbin_negbin, df, "y_num | y_denom ~ x + (1|site)")
