# bench_3fixes.R
# Benchmark the three performance fixes:
# 1. Fused digamma+lgamma
# 2. Function pointer gradient dispatch
# 3. Vectorized exp via Eigen
#
# Compare NB base, NB+RE, PG base, PG+RE timings against known Stan baselines

library(numdenom)

set.seed(42)
N <- 500
S <- 50

# Simulate data
df <- data.frame(
  x = rnorm(N),
  site = factor(sample(1:S, N, replace=TRUE))
)

# NB data
beta_num <- c(1.5, 0.3)
beta_denom <- c(2.0, -0.2)
eta_num <- beta_num[1] + beta_num[2] * df$x
eta_denom <- beta_denom[1] + beta_denom[2] * df$x
df$y_num <- rnbinom(N, mu = exp(eta_num), size = 3)
df$y_denom <- rnbinom(N, mu = exp(eta_denom), size = 5)

# PG data
df$count <- rpois(N, lambda = exp(eta_num))
df$effort <- rgamma(N, shape = 5, rate = 5 / exp(eta_denom))

cat(paste(rep("=", 60), collapse=""), "\n")
cat("BENCHMARK: 3 Fixes (N=", N, ", iter=500, chains=1)\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

bench_model <- function(name, formula, data, family, gradient_mode = "H") {
  t <- system.time({
    fit <- ratiod(formula, data = data, family = family,
                  mode = "hmc", gradient_mode = gradient_mode,
                  iter = 500, warmup = 250, chains = 1,
                  refresh = 0)
  })["elapsed"]
  div <- sum(fit$diagnostics$divergent)
  td <- mean(fit$diagnostics$treedepth)
  cat(sprintf("  %-25s %5.1fs  div=%d  td=%.1f\n", name, t, div, td))
  invisible(t)
}

cat("--- NegBin models ---\n")
t_nb_base <- bench_model("NB base (H)", y_num | y_denom ~ x, df,
                          ratiod_negbin_negbin())
t_nb_re <- bench_model("NB + RE (H)", y_num | y_denom ~ x + (1|site), df,
                        ratiod_negbin_negbin())

cat("\n--- Poisson-Gamma models ---\n")
t_pg_base <- bench_model("PG base (H)", count | effort ~ x, df,
                          ratiod_poisson_gamma())
t_pg_re <- bench_model("PG + RE (H)", count | effort ~ x + (1|site), df,
                        ratiod_poisson_gamma())

cat("\n--- Stan baselines (from previous benchmarks) ---\n")
cat(sprintf("  %-25s %5.1fs\n", "NB base (Stan)", 1.5))
cat(sprintf("  %-25s %5.1fs\n", "NB + RE (Stan)", 2.9))
cat(sprintf("  %-25s %5.1fs\n", "PG base (Stan)", 1.2))
cat(sprintf("  %-25s %5.1fs\n", "PG + RE (Stan)", 2.5))

cat("\n--- Speedup ratios (numdenom / Stan) ---\n")
cat(sprintf("  NB base:  %.2fx  (was 1.80x)\n", t_nb_base / 1.5))
cat(sprintf("  NB + RE:  %.2fx  (was 1.79x)\n", t_nb_re / 2.9))
cat(sprintf("  PG base:  %.2fx  (was 0.92x)\n", t_pg_base / 1.2))
cat(sprintf("  PG + RE:  %.2fx  (was 1.92x)\n", t_pg_re / 2.5))
