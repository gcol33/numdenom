# Benchmark: TreeStats momentum pool optimization
# Compare NB base and PG base timing (N=500, 500 iter, 1 chain)
# Run AFTER devtools::load_all() with the momentum pool changes

library(numdenom)

set.seed(42)
N <- 500
S <- 50
df <- data.frame(
  y_num = rnbinom(N, mu = 5, size = 2),
  y_denom = rnbinom(N, mu = 10, size = 3),
  x = rnorm(N),
  site = rep(1:S, each = N / S)
)

cat("=== TreeStats Momentum Pool Benchmark ===\n")
cat(sprintf("N=%d, iter=500, warmup=250, chains=1\n\n", N))

# PG base
cat("PG base (poisson_gamma)...\n")
t_pg <- system.time({
  fit_pg <- tratio(y_num | y_denom ~ x, data = df,
                   family = ratiod_poisson_gamma(),
                   control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))
})["elapsed"]
cat(sprintf("  Time: %.2fs\n", t_pg))

# NB base
cat("NB base (negbin_negbin)...\n")
t_nb <- system.time({
  fit_nb <- tratio(y_num | y_denom ~ x, data = df,
                   family = ratiod_negbin_negbin(),
                   control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))
})["elapsed"]
cat(sprintf("  Time: %.2fs\n", t_nb))

# PG + RE
cat("PG + RE...\n")
t_pg_re <- system.time({
  fit_pg_re <- tratio(y_num | y_denom ~ x + (1 | site), data = df,
                      family = ratiod_poisson_gamma(),
                      control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))
})["elapsed"]
cat(sprintf("  Time: %.2fs\n", t_pg_re))

# NB + RE
cat("NB + RE...\n")
t_nb_re <- system.time({
  fit_nb_re <- tratio(y_num | y_denom ~ x + (1 | site), data = df,
                      family = ratiod_negbin_negbin(),
                      control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))
})["elapsed"]
cat(sprintf("  Time: %.2fs\n", t_nb_re))

# Binomial base
df_bin <- data.frame(
  y_num = rbinom(N, size = 20, prob = 0.3),
  y_denom = rep(20L, N),
  x = rnorm(N)
)
cat("Binomial base...\n")
t_bin <- system.time({
  fit_bin <- tratio(y_num | y_denom ~ x, data = df_bin,
                    family = ratiod_binomial(),
                    control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))
})["elapsed"]
cat(sprintf("  Time: %.2fs\n", t_bin))

cat("\n=== Summary ===\n")
cat(sprintf("PG base:     %.2fs\n", t_pg))
cat(sprintf("NB base:     %.2fs\n", t_nb))
cat(sprintf("PG + RE:     %.2fs\n", t_pg_re))
cat(sprintf("NB + RE:     %.2fs\n", t_nb_re))
cat(sprintf("Binomial:    %.2fs\n", t_bin))
cat("\nReference (before momentum pool):\n")
cat("PG base:  1.1s  |  NB base: 2.7s  |  PG+RE: 4.8s  |  NB+RE: 5.2s  |  Bin: ~0.5s\n")
