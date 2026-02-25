# bench_nuts_diagnostics.R
# Compare NUTS diagnostics between numdenom and Stan on models where Stan is faster
# Records per-iteration: treedepth, n_leapfrog, step size (epsilon)
# Run before/after warmup adaptation fixes to quantify improvement

library(numdenom)
library(cmdstanr)

set.seed(42)

# =====================================================================
# Helper: run numdenom and extract NUTS diagnostics
# =====================================================================

run_numdenom <- function(formula, data, family, label, ...) {
  cat("\n=== numdenom:", label, "===\n")
  t0 <- proc.time()["elapsed"]
  fit <- ratiod(formula, data = data, family = family,
                mode = "hmc", iter = 500, warmup = 250, chains = 1,
                verbose = FALSE, ...)
  elapsed <- proc.time()["elapsed"] - t0

  # Extract diagnostics from the fit object
  diag <- list(
    label = label,
    time = elapsed,
    epsilon = fit$epsilon,
    treedepth = fit$diagnostics$treedepth,
    n_leapfrog = fit$diagnostics$n_leapfrog,
    divergent = fit$diagnostics$divergent
  )

  if (!is.null(diag$treedepth)) {
    cat(sprintf("  Time: %.1fs\n", elapsed))
    cat(sprintf("  Epsilon: %.5f\n", diag$epsilon))
    cat(sprintf("  Mean tree depth: %.2f\n", mean(diag$treedepth)))
    cat(sprintf("  Mean leapfrog: %.1f\n", mean(diag$n_leapfrog)))
    cat(sprintf("  Max tree depth: %d\n", max(diag$treedepth)))
    cat(sprintf("  Divergent: %d\n", sum(diag$divergent)))
  } else {
    cat(sprintf("  Time: %.1fs (no per-iteration diagnostics)\n", elapsed))
  }

  diag
}


# =====================================================================
# Helper: run Stan and extract NUTS diagnostics
# =====================================================================

run_stan <- function(stan_file, stan_data, label) {
  cat("\n=== Stan:", label, "===\n")

  mod <- cmdstan_model(stan_file)
  fit <- mod$sample(
    data = stan_data,
    iter_warmup = 250,
    iter_sampling = 250,
    chains = 1,
    seed = 42,
    refresh = 0,
    show_messages = FALSE
  )

  # Extract sampler diagnostics
  np <- fit$sampler_diagnostics(inc_warmup = FALSE)
  elapsed <- fit$time()$total

  diag <- list(
    label = label,
    time = elapsed,
    epsilon = mean(np[, , "stepsize__"]),
    treedepth = as.integer(np[, , "treedepth__"]),
    n_leapfrog = as.integer(np[, , "n_leapfrog__"]),
    divergent = as.integer(np[, , "divergent__"])
  )

  cat(sprintf("  Time: %.1fs\n", elapsed))
  cat(sprintf("  Epsilon: %.5f\n", diag$epsilon))
  cat(sprintf("  Mean tree depth: %.2f\n", mean(diag$treedepth)))
  cat(sprintf("  Mean leapfrog: %.1f\n", mean(diag$n_leapfrog)))
  cat(sprintf("  Max tree depth: %d\n", max(diag$treedepth)))
  cat(sprintf("  Divergent: %d\n", sum(diag$divergent)))

  diag
}


# =====================================================================
# Helper: compare diagnostics
# =====================================================================

compare_diagnostics <- function(nd_diag, stan_diag) {
  cat(sprintf("\n--- %s: numdenom vs Stan ---\n", nd_diag$label))
  cat(sprintf("  Time:         %.1fs vs %.1fs (ratio: %.2fx)\n",
              nd_diag$time, stan_diag$time,
              nd_diag$time / stan_diag$time))
  cat(sprintf("  Epsilon:      %.5f vs %.5f (ratio: %.2fx)\n",
              nd_diag$epsilon, stan_diag$epsilon,
              nd_diag$epsilon / stan_diag$epsilon))

  if (!is.null(nd_diag$treedepth) && !is.null(stan_diag$treedepth)) {
    cat(sprintf("  Tree depth:   %.2f vs %.2f\n",
                mean(nd_diag$treedepth), mean(stan_diag$treedepth)))
    cat(sprintf("  Leapfrog:     %.1f vs %.1f (ratio: %.2fx)\n",
                mean(nd_diag$n_leapfrog), mean(stan_diag$n_leapfrog),
                mean(nd_diag$n_leapfrog) / mean(stan_diag$n_leapfrog)))
    cat(sprintf("  Divergent:    %d vs %d\n",
                sum(nd_diag$divergent), sum(stan_diag$divergent)))
  }
}


# =====================================================================
# Model definitions: rows where Stan was faster
# =====================================================================

N <- 500

# --- Row 1: pg_base (Poisson-Gamma, intercept only) ---
cat("\n\n========== ROW 1: pg_base ==========\n")
x1 <- rnorm(N)
data_pg <- data.frame(
  y_num = rpois(N, exp(1.0 + 0.3 * x1)),
  y_denom = rgamma(N, shape = 5, rate = 5 / exp(0.5)),
  x1 = x1
)
nd1 <- tryCatch(
  run_numdenom(y_num | y_denom ~ x1, data_pg, ratiod_poisson_gamma(), "pg_base"),
  error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
)

# --- Row 2: pg_re (Poisson-Gamma with RE) ---
cat("\n\n========== ROW 2: pg_re ==========\n")
n_groups <- 20
data_pg_re <- data.frame(
  y_num = rpois(N, exp(1.0 + 0.3 * x1)),
  y_denom = rgamma(N, shape = 5, rate = 5 / exp(0.5)),
  x1 = x1,
  group = factor(sample(1:n_groups, N, replace = TRUE))
)
nd2 <- tryCatch(
  run_numdenom(y_num | y_denom ~ x1 + (1 | group), data_pg_re,
               ratiod_poisson_gamma(), "pg_re"),
  error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
)

# --- Row 4: pg_icar (Poisson-Gamma with ICAR spatial) ---
cat("\n\n========== ROW 4: pg_icar ==========\n")
n_sites <- 50
# Create simple adjacency (chain graph)
adj_row <- c()
adj_col <- c()
for (i in 1:(n_sites - 1)) {
  adj_row <- c(adj_row, i, i + 1)
  adj_col <- c(adj_col, i + 1, i)
}
W <- Matrix::sparseMatrix(i = adj_row, j = adj_col, x = 1,
                           dims = c(n_sites, n_sites))

data_pg_icar <- data.frame(
  y_num = rpois(N, exp(1.0 + 0.3 * x1)),
  y_denom = rgamma(N, shape = 5, rate = 5 / exp(0.5)),
  x1 = x1,
  site = factor(sample(1:n_sites, N, replace = TRUE))
)
nd4 <- tryCatch(
  run_numdenom(y_num | y_denom ~ x1, data_pg_icar,
               ratiod_poisson_gamma(), "pg_icar",
               spatial = spatial_car(W, level = "group", group_var = "site")),
  error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
)

# --- Row 31: nb_base (NegBin, intercept only) ---
cat("\n\n========== ROW 31: nb_base ==========\n")
data_nb <- data.frame(
  y_num = rnbinom(N, mu = exp(1.0 + 0.3 * x1), size = 5),
  y_denom = rnbinom(N, mu = exp(0.5), size = 5),
  x1 = x1
)
nd31 <- tryCatch(
  run_numdenom(y_num | y_denom ~ x1, data_nb, ratiod_negbin_negbin(), "nb_base"),
  error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
)

# --- Row 32: nb_re (NegBin with RE) ---
cat("\n\n========== ROW 32: nb_re ==========\n")
data_nb_re <- data.frame(
  y_num = rnbinom(N, mu = exp(1.0 + 0.3 * x1), size = 5),
  y_denom = rnbinom(N, mu = exp(0.5), size = 5),
  x1 = x1,
  group = factor(sample(1:n_groups, N, replace = TRUE))
)
nd32 <- tryCatch(
  run_numdenom(y_num | y_denom ~ x1 + (1 | group), data_nb_re,
               ratiod_negbin_negbin(), "nb_re"),
  error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
)

# --- Row 34: nb_icar (NegBin with ICAR spatial) ---
cat("\n\n========== ROW 34: nb_icar ==========\n")
data_nb_icar <- data.frame(
  y_num = rnbinom(N, mu = exp(1.0 + 0.3 * x1), size = 5),
  y_denom = rnbinom(N, mu = exp(0.5), size = 5),
  x1 = x1,
  site = factor(sample(1:n_sites, N, replace = TRUE))
)
nd34 <- tryCatch(
  run_numdenom(y_num | y_denom ~ x1, data_nb_icar,
               ratiod_negbin_negbin(), "nb_icar",
               spatial = spatial_car(W, level = "group", group_var = "site")),
  error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
)


# =====================================================================
# Summary table
# =====================================================================

cat("\n\n========================================\n")
cat("NUTS Diagnostics Summary (numdenom)\n")
cat("========================================\n")

results <- list(nd1, nd2, nd4, nd31, nd32, nd34)
results <- results[!sapply(results, is.null)]

for (r in results) {
  if (!is.null(r$treedepth)) {
    cat(sprintf("%-12s  eps=%.5f  depth=%.2f  leapfrog=%.1f  time=%.1fs  div=%d\n",
                r$label, r$epsilon,
                mean(r$treedepth), mean(r$n_leapfrog),
                r$time, sum(r$divergent)))
  } else {
    cat(sprintf("%-12s  eps=%.5f  time=%.1fs\n",
                r$label, r$epsilon, r$time))
  }
}

cat("\n\nDone. Run with custom Stan models for full comparison.\n")
cat("Stan models: benchmarks/stan/joint_pg_base.stan, joint_nb_base.stan, etc.\n")
