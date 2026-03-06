#!/usr/bin/env Rscript
# Benchmark the 5 rows where Stan was faster, now with -O3
# Compare to previous -O2 results
library(numdenom)

set.seed(42)
N <- 500
S <- 50

# --- Simulate data ---
site <- rep(1:S, length.out = N)
region <- rep(1:10, each = N/10)
x1 <- rnorm(N)

# NB data
lambda_num <- exp(1 + 0.3 * x1)
lambda_denom <- exp(2 + 0.1 * x1)
y_num <- rnbinom(N, mu = lambda_num, size = 5)
y_denom <- rnbinom(N, mu = lambda_denom, size = 5)
y_denom[y_denom == 0] <- 1

df <- data.frame(
  y_num = y_num,
  y_denom = y_denom,
  x1 = x1,
  site = factor(site),
  region = factor(region)
)

cat("=== numdenom -O3 Benchmark (N=500, 500 iter, 1 chain) ===\n\n")

# Previous -O2 results and Stan times for comparison
prev <- list(
  pg_base = list(o2 = 1.1, stan = 1.2),
  pg_re = list(o2 = 4.8, stan = 2.5),
  pg_crossed = list(o2 = 11.3, stan = 2.9),
  nb_base = list(o2 = 2.7, stan = 1.5),
  nb_re = list(o2 = 5.2, stan = 2.9),
  nb_crossed = list(o2 = 20.3, stan = 9.9)
)

results <- list()

# Row 1: PG base
cat("--- Row 1: PG base ---\n")
t1 <- system.time({
  fit <- tryCatch(
    ratiod(y_num | y_denom ~ x1, data = df,
           family = ratiod_poisson_gamma(),
           mode = "hmc", iter = 500, warmup = 250, chains = 1,
           gradient_mode = "H", verbose = FALSE),
    error = function(e) { cat("ERROR:", e$message, "\n"); NULL }
  )
})["elapsed"]
results$pg_base <- t1
cat(sprintf("  O3: %.1fs  (O2: %.1fs, Stan: %.1fs)\n\n", t1, prev$pg_base$o2, prev$pg_base$stan))

# Row 2: PG + RE
cat("--- Row 2: PG + RE ---\n")
t2 <- system.time({
  fit <- tryCatch(
    ratiod(y_num | y_denom ~ x1 + (1 | site), data = df,
           family = ratiod_poisson_gamma(),
           mode = "hmc", iter = 500, warmup = 250, chains = 1,
           gradient_mode = "H", verbose = FALSE),
    error = function(e) { cat("ERROR:", e$message, "\n"); NULL }
  )
})["elapsed"]
results$pg_re <- t2
cat(sprintf("  O3: %.1fs  (O2: %.1fs, Stan: %.1fs)\n\n", t2, prev$pg_re$o2, prev$pg_re$stan))

# Row 4: PG + crossed
cat("--- Row 4: PG + crossed ---\n")
t3 <- system.time({
  fit <- tryCatch(
    ratiod(y_num | y_denom ~ x1 + (1 | site) + (1 | region), data = df,
           family = ratiod_poisson_gamma(),
           mode = "hmc", iter = 500, warmup = 250, chains = 1,
           gradient_mode = "H", verbose = FALSE),
    error = function(e) { cat("ERROR:", e$message, "\n"); NULL }
  )
})["elapsed"]
results$pg_crossed <- t3
cat(sprintf("  O3: %.1fs  (O2: %.1fs, Stan: %.1fs)\n\n", t3, prev$pg_crossed$o2, prev$pg_crossed$stan))

# Row 31: NB base
cat("--- Row 31: NB base ---\n")
t4 <- system.time({
  fit <- tryCatch(
    ratiod(y_num | y_denom ~ x1, data = df,
           family = ratiod_negbin_negbin(),
           mode = "hmc", iter = 500, warmup = 250, chains = 1,
           gradient_mode = "H", verbose = FALSE),
    error = function(e) { cat("ERROR:", e$message, "\n"); NULL }
  )
})["elapsed"]
results$nb_base <- t4
cat(sprintf("  O3: %.1fs  (O2: %.1fs, Stan: %.1fs)\n\n", t4, prev$nb_base$o2, prev$nb_base$stan))

# Row 32: NB + RE
cat("--- Row 32: NB + RE ---\n")
t5 <- system.time({
  fit <- tryCatch(
    ratiod(y_num | y_denom ~ x1 + (1 | site), data = df,
           family = ratiod_negbin_negbin(),
           mode = "hmc", iter = 500, warmup = 250, chains = 1,
           gradient_mode = "H", verbose = FALSE),
    error = function(e) { cat("ERROR:", e$message, "\n"); NULL }
  )
})["elapsed"]
results$nb_re <- t5
cat(sprintf("  O3: %.1fs  (O2: %.1fs, Stan: %.1fs)\n\n", t5, prev$nb_re$o2, prev$nb_re$stan))

# Row 34: NB + crossed
cat("--- Row 34: NB + crossed ---\n")
t6 <- system.time({
  fit <- tryCatch(
    ratiod(y_num | y_denom ~ x1 + (1 | site) + (1 | region), data = df,
           family = ratiod_negbin_negbin(),
           mode = "hmc", iter = 500, warmup = 250, chains = 1,
           gradient_mode = "H", verbose = FALSE),
    error = function(e) { cat("ERROR:", e$message, "\n"); NULL }
  )
})["elapsed"]
results$nb_crossed <- t6
cat(sprintf("  O3: %.1fs  (O2: %.1fs, Stan: %.1fs)\n\n", t6, prev$nb_crossed$o2, prev$nb_crossed$stan))

# Summary table
cat("\n=== SUMMARY ===\n")
cat(sprintf("%-15s  %6s  %6s  %6s  %8s  %8s\n",
            "Model", "O3(s)", "O2(s)", "Stan(s)", "O3/Stan", "O2->O3"))
cat(paste(rep("-", 65), collapse = ""), "\n")
for (nm in names(results)) {
  o3 <- results[[nm]]
  o2 <- prev[[nm]]$o2
  stan <- prev[[nm]]$stan
  cat(sprintf("%-15s  %6.1f  %6.1f  %6.1f  %8.2fx  %7.0f%%\n",
              nm, o3, o2, stan, o3/stan, (1 - o3/o2) * 100))
}
