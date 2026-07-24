# bench_vectorized.R
# Benchmark vectorized gradient dispatch + momentum pool + function pointer
# Compare against previous results from bench_3fixes.R
#
# Targets: base models (where fused path fires) and +RE models
# N=500, 500 iter, 1 chain, H mode

library(numdenom)

cat("=== Vectorized Gradient + Momentum Pool Benchmark ===\n")
cat("Date:", format(Sys.time()), "\n\n")

N <- 500
S <- 50
n_iter <- 500
n_warmup <- 250
n_chains <- 1

set.seed(42)

# --- Simulate data ---
x <- rnorm(N)
site <- sample(1:S, N, replace = TRUE)
group2 <- sample(1:10, N, replace = TRUE)

# PG data
lambda_pg <- exp(0.5 + 0.3 * x)
y_num_pg <- rpois(N, lambda_pg)
y_denom_pg <- rgamma(N, shape = 5, rate = 5 / exp(0.2 + 0.1 * x))

# NB data
mu_nb <- exp(1.0 + 0.5 * x)
y_num_nb <- rnbinom(N, mu = mu_nb, size = 3)
y_denom_nb <- rnbinom(N, mu = exp(0.8 + 0.2 * x), size = 5)

# Binomial data
n_trials <- sample(10:50, N, replace = TRUE)
p_binom <- plogis(0.3 + 0.5 * x)
y_binom <- rbinom(N, n_trials, p_binom)

df <- data.frame(
  y_num_pg = y_num_pg, y_denom_pg = y_denom_pg,
  y_num_nb = y_num_nb, y_denom_nb = y_denom_nb,
  y_binom = y_binom, n_trials = n_trials,
  x = x, site = factor(site), group2 = factor(group2)
)

# --- Helper: timed fit ---
timed_fit <- function(desc, expr) {
  cat(sprintf("  %-30s", desc))
  gc(FALSE)
  t <- system.time(tryCatch(
    fit <- eval(expr),
    error = function(e) { cat(" ERROR:", e$message, "\n"); return(NULL) }
  ))["elapsed"]
  cat(sprintf(" %6.1fs\n", t))
  invisible(t)
}

results <- list()

# =============================================
# 1. BASE MODELS (fused single-pass path, p<=4)
# =============================================
cat("\n--- Base models (fused single-pass, p<=4) ---\n")

results$pg_base <- timed_fit("PG base", quote(
  tratio(y_num_pg | y_denom_pg ~ x, data = df,
         family = ratiod_poisson_gamma(),
         control = list(iter = n_iter, warmup = n_warmup, chains = n_chains, gradient_mode = "H", verbose = FALSE))
))

results$nb_base <- timed_fit("NB base", quote(
  tratio(y_num_nb | y_denom_nb ~ x, data = df,
         family = ratiod_negbin_negbin(),
         control = list(iter = n_iter, warmup = n_warmup, chains = n_chains, gradient_mode = "H", verbose = FALSE))
))

results$bin_base <- timed_fit("Binomial base", quote(
  tratio(y_binom | n_trials ~ x, data = df,
         family = ratiod_binomial(),
         control = list(iter = n_iter, warmup = n_warmup, chains = n_chains, gradient_mode = "H", verbose = FALSE))
))

# =============================================
# 2. +RE MODELS (fused single-pass with RE scatter)
# =============================================
cat("\n--- +RE models (fused with RE scatter) ---\n")

results$pg_re <- timed_fit("PG + RE", quote(
  tratio(y_num_pg | y_denom_pg ~ x + (1 | site), data = df,
         family = ratiod_poisson_gamma(),
         control = list(iter = n_iter, warmup = n_warmup, chains = n_chains, gradient_mode = "H", verbose = FALSE))
))

results$nb_re <- timed_fit("NB + RE", quote(
  tratio(y_num_nb | y_denom_nb ~ x + (1 | site), data = df,
         family = ratiod_negbin_negbin(),
         control = list(iter = n_iter, warmup = n_warmup, chains = n_chains, gradient_mode = "H", verbose = FALSE))
))

results$bin_re <- timed_fit("Binomial + RE", quote(
  tratio(y_binom | n_trials ~ x + (1 | site), data = df,
         family = ratiod_binomial(),
         control = list(iter = n_iter, warmup = n_warmup, chains = n_chains, gradient_mode = "H", verbose = FALSE))
))

# =============================================
# 3. CROSSED RE (fused with crossed RE scatter)
# =============================================
cat("\n--- Crossed RE models ---\n")

results$pg_crossed <- timed_fit("PG + crossed RE", quote(
  tratio(y_num_pg | y_denom_pg ~ x + (1 | site) + (1 | group2), data = df,
         family = ratiod_poisson_gamma(),
         control = list(iter = n_iter, warmup = n_warmup, chains = n_chains, gradient_mode = "H", verbose = FALSE))
))

results$nb_crossed <- timed_fit("NB + crossed RE", quote(
  tratio(y_num_nb | y_denom_nb ~ x + (1 | site) + (1 | group2), data = df,
         family = ratiod_negbin_negbin(),
         control = list(iter = n_iter, warmup = n_warmup, chains = n_chains, gradient_mode = "H", verbose = FALSE))
))

# =============================================
# Summary
# =============================================
cat("\n=== Summary ===\n")
cat("Previous results (bench_3fixes.R, 5-rep median):\n")
cat("  NB base:    1.36s (Stan 1.5s)\n")
cat("  PG base:    1.64s (Stan 1.2s)\n")
cat("  NB+RE:      5.18s (Stan 2.9s)\n")
cat("  PG+RE:      3.56s (Stan 2.5s)\n\n")

cat("Current results:\n")
for (nm in names(results)) {
  cat(sprintf("  %-20s %6.1fs\n", nm, results[[nm]]))
}
cat("\nDone.\n")
