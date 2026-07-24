# bench_opt_speedup.R
# Benchmark the digamma table + reciprocal optimizations vs Stan gap.
# Measures: wall time, average tree depth, total leapfrog steps.
#
# Run after: devtools::clean_dll(); devtools::load_all()

library(numdenom)
set.seed(42)

N_OBS   <- 500
N_ITER  <- 500
N_WARMUP <- 250
N_CHAINS <- 1
N_SITES <- 50

cat(strrep("=", 60), "\n")
cat("OPTIMIZATION BENCHMARK: NB base, NB+RE, PG+RE\n")
cat("N =", N_OBS, " iter =", N_ITER, " chains =", N_CHAINS, "\n")
cat(strrep("=", 60), "\n\n")

# --- Data generation ---
df <- data.frame(
  site = rep(1:N_SITES, each = N_OBS / N_SITES),
  x = rnorm(N_OBS)
)
eta <- 1.5 + 0.3 * df$x + rnorm(N_SITES, 0, 0.3)[df$site]
df$y_num   <- rnbinom(N_OBS, size = 5, mu = exp(eta))
df$y_denom <- rnbinom(N_OBS, size = 5, mu = exp(eta + 0.5))
df$y_denom <- pmax(df$y_denom, 1)
# PG data: continuous denominator
df$effort <- rgamma(N_OBS, shape = 5, rate = 5 / exp(eta + 0.5))
df$effort <- pmax(df$effort, 0.01)

run_bench <- function(label, ...) {
  cat(sprintf("\n>>> %s <<<\n", label))
  gc()
  time <- system.time({
    fit <- tratio(...,
                  control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H", verbose = FALSE))
  })["elapsed"]

  # Extract tree depth diagnostics
  td <- fit$diagnostics$treedepth
  if (is.null(td)) td <- fit$treedepth
  if (is.list(td)) td <- td[[1]]
  # Only sampling iterations (after warmup)
  if (length(td) >= N_ITER) {
    td_sample <- td[(N_WARMUP + 1):N_ITER]
  } else {
    td_sample <- td
  }
  avg_depth <- mean(td_sample, na.rm = TRUE)
  total_lf  <- sum(2^td_sample, na.rm = TRUE)

  cat(sprintf("  Time: %.2f s\n", time))
  cat(sprintf("  Avg tree depth (sampling): %.1f\n", avg_depth))
  cat(sprintf("  Total leapfrog steps (sampling): %d\n", total_lf))
  cat(sprintf("  Time per 1000 leapfrog: %.2f ms\n", 1000 * time / total_lf))

  invisible(list(time = time, avg_depth = avg_depth, total_lf = total_lf))
}

# --- Model 1: NB base (no RE) ---
r1 <- tryCatch(
  run_bench("NB base (row 1)",
    y_num | y_denom ~ x, data = df,
    family = ratiod_negbin_negbin()
  ),
  error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
)

# --- Model 2: NB + RE ---
r2 <- tryCatch(
  run_bench("NB + RE (row 2)",
    y_num | y_denom ~ x + (1 | site), data = df,
    family = ratiod_negbin_negbin()
  ),
  error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
)

# --- Model 3: PG + RE ---
r3 <- tryCatch(
  run_bench("PG + RE (row 42)",
    y_num | effort ~ x + (1 | site), data = df,
    family = ratiod_poisson_gamma()
  ),
  error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
)

# --- Summary ---
cat("\n", strrep("=", 60), "\n")
cat("SUMMARY (compare with gradient_methods.md Stan times)\n")
cat(strrep("-", 60), "\n")
cat(sprintf("%-20s  %8s  %8s  %8s\n", "Model", "Time(s)", "AvgDepth", "ms/1kLF"))
cat(strrep("-", 60), "\n")
if (!is.null(r1)) cat(sprintf("%-20s  %8.2f  %8.1f  %8.2f\n",
  "NB base", r1$time, r1$avg_depth, 1000 * r1$time / r1$total_lf))
if (!is.null(r2)) cat(sprintf("%-20s  %8.2f  %8.1f  %8.2f\n",
  "NB + RE", r2$time, r2$avg_depth, 1000 * r2$time / r2$total_lf))
if (!is.null(r3)) cat(sprintf("%-20s  %8.2f  %8.1f  %8.2f\n",
  "PG + RE", r3$time, r3$avg_depth, 1000 * r3$time / r3$total_lf))
cat(strrep("-", 60), "\n")
cat("Previous: NB base=2.7s, NB+RE=5.2s, PG+RE=4.8s\n")
cat("Stan:     NB base=1.5s, NB+RE=2.9s, PG+RE=2.5s\n")
cat(strrep("=", 60), "\n")
