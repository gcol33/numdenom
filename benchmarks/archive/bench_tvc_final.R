#!/usr/bin/env Rscript
# Final TVC benchmark: all 3 families, N=500, 500 iter, 1 chain
# After: (1) tau PC prior fix, (2) rho Uniform fix, (3) AR1 stationary fix,
#         (4) OAS shrinkage floor, (5) TVC removed from needs_dense
setwd("C:/Users/Gilles Colling/Documents/dev/numdenom")
devtools::load_all(quiet = TRUE)

N_OBS <- 500
N_ITER <- 500
N_WARMUP <- 250

set.seed(42)
S <- 50; T_val <- 20
site <- rep(1:S, each = N_OBS / S)
time <- rep(1:T_val, times = N_OBS / T_val)
x <- rnorm(N_OBS)
re_site <- rnorm(S, sd = 0.3)
tvc_effect <- sin(2 * pi * (1:T_val) / T_val) * 0.5

y_num <- rpois(N_OBS, exp(0.5 + 0.3 * x + re_site[site] + tvc_effect[time] * x))
y_denom <- rpois(N_OBS, exp(1 + 0.2 * x + re_site[site] * 0.5)) + 1
y_trials <- rpois(N_OBS, 50) + 10
y_success <- rbinom(N_OBS, y_trials, plogis(-0.5 + 0.3 * x + re_site[site] + tvc_effect[time] * x))

df <- data.frame(y_num = y_num, y_denom = y_denom,
                 y_success = y_success, y_trials = y_trials,
                 x = x, site = factor(site), time = factor(time))

run_tvc <- function(label, formula, family, pre_fix_time) {
  cat(sprintf("\n=== %s (pre-fix: %.1fs) ===\n", label, pre_fix_time))
  t0 <- proc.time()
  fit <- tryCatch(
    tratio(formula, data = df, family = family,
           temporal = temporal_tvc("time", terms = 2, structure = "rw1"),
           control = list(iter = N_ITER, warmup = N_WARMUP, chains = 1, gradient_mode = "H", verbose = TRUE)),
    error = function(e) { cat("ERROR:", e$message, "\n"); NULL }
  )
  elapsed <- (proc.time() - t0)[3]
  if (!is.null(fit)) {
    div <- fit$diagnostics$n_divergent
    td <- mean(fit$diagnostics$treedepth)
    cat(sprintf("  RESULT: %.1fs (%.1fx speedup), div=%d, td=%.1f\n",
                elapsed, pre_fix_time / elapsed, div, td))
  } else {
    cat(sprintf("  FAILED after %.1fs\n", elapsed))
  }
  invisible(fit)
}

fit_pg <- run_tvc("PG + RE + TVC (row 27)",
                  y_num | y_denom ~ x + (1 | site),
                  ratiod_poisson_gamma(), 68.0)

fit_nb <- run_tvc("NB + RE + TVC (row 57)",
                  y_num | y_denom ~ x + (1 | site),
                  ratiod_negbin_negbin(), 183.2)

fit_bin <- run_tvc("Bin + RE + TVC (row 89)",
                   y_success | y_trials ~ x + (1 | site),
                   ratiod_binomial(), 24.4)
