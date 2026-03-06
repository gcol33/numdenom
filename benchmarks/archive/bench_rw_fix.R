.libPaths(c("C:/Users/Gilles Colling/AppData/Local/R/win-library/4.5", .libPaths()))
devtools::load_all(quiet = TRUE)

set.seed(123)
N <- 500; T_times <- 20
time_id <- rep(1:T_times, length.out = N)
x1 <- rnorm(N)
true_temporal <- sin(2 * pi * (1:T_times) / T_times)
eta_num <- 1.5 + 0.3 * x1 + true_temporal[time_id]
eta_denom <- 1.0 + 0.1 * x1 + 0.5 * true_temporal[time_id]
# NB data for NB family
y_num_nb <- rnbinom(N, mu = exp(eta_num), size = 5)
y_denom_nb <- rnbinom(N, mu = exp(eta_denom), size = 5)
# PG data for PG family
y_num_pg <- rpois(N, lambda = exp(eta_num))
y_denom_pg <- rgamma(N, shape = 5, rate = 5 / exp(eta_denom))
df_nb <- data.frame(y_num = y_num_nb, y_denom = y_denom_nb, x1, time_id)
df_pg <- data.frame(y_num = y_num_pg, y_denom = y_denom_pg, x1, time_id)

run_bench <- function(label, ...) {
  cat(sprintf("=== %s ===\n", label))
  t <- system.time({ fit <- ratiod(...) })[3]
  d <- fit$diagnostics
  n_div <- if (!is.null(d$n_divergent)) sum(d$n_divergent) else 0
  avg_acc <- if (!is.null(d$avg_accept_prob)) mean(d$avg_accept_prob) else NA
  n_lf <- if (!is.null(d$n_leapfrog)) d$n_leapfrog else NA
  avg_td <- if (!is.null(n_lf)) mean(log2(pmax(n_lf, 1))) else NA
  cat(sprintf("  Time: %.1fs, div=%d, avg_td=%.1f, acc=%.3f\n",
      t, n_div, avg_td, avg_acc))
}

run_bench("NB+RW1",
  y_num | y_denom ~ x1,
  data = df_nb, family = ratiod_negbin_negbin(),
  temporal = temporal_rw1("time_id"),
  iter = 500, warmup = 250, chains = 1, cores = 1, mode = "hmc", metric = "auto")

run_bench("PG+RW1",
  y_num | y_denom ~ x1,
  data = df_pg, family = ratiod_poisson_gamma(),
  temporal = temporal_rw1("time_id"),
  iter = 500, warmup = 250, chains = 1, cores = 1, mode = "hmc", metric = "auto")

run_bench("NB+RW2",
  y_num | y_denom ~ x1,
  data = df_nb, family = ratiod_negbin_negbin(),
  temporal = temporal_rw2("time_id"),
  iter = 500, warmup = 250, chains = 1, cores = 1, mode = "hmc", metric = "auto")

run_bench("NB+AR1",
  y_num | y_denom ~ x1,
  data = df_nb, family = ratiod_negbin_negbin(),
  temporal = temporal_ar1("time_id"),
  iter = 500, warmup = 250, chains = 1, cores = 1, mode = "hmc", metric = "auto")

run_bench("PG+AR1",
  y_num | y_denom ~ x1,
  data = df_pg, family = ratiod_poisson_gamma(),
  temporal = temporal_ar1("time_id"),
  iter = 500, warmup = 250, chains = 1, cores = 1, mode = "hmc", metric = "auto")

run_bench("PG+RW2",
  y_num | y_denom ~ x1,
  data = df_pg, family = ratiod_poisson_gamma(),
  temporal = temporal_rw2("time_id"),
  iter = 500, warmup = 250, chains = 1, cores = 1, mode = "hmc", metric = "auto")
