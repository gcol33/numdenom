# bench_pg_profile.R — profile PG vs NB per-leapfrog cost
# Uses short warmup+sampling to isolate gradient cost from adaptation
library(numdenom)
set.seed(42)

N <- 500; S <- 50
df <- data.frame(site = rep(1:S, each = N/S), x = rnorm(N))
eta <- 1.5 + 0.3 * df$x + rnorm(S, 0, 0.3)[df$site]
df$y_num <- rnbinom(N, size = 5, mu = exp(eta))
df$y_denom <- pmax(rnbinom(N, size = 5, mu = exp(eta + 0.5)), 1)
df$effort <- pmax(rgamma(N, shape = 5, rate = 5 / exp(eta + 0.5)), 0.01)

cat("\n=== NB+RE: 500 iter (standard) ===\n")
t1 <- system.time({
  fit_nb <- ratiod(y_num | y_denom ~ x + (1 | site), data = df,
    family = ratiod_negbin_negbin(), iter = 500, warmup = 250,
    chains = 1, gradient_mode = "H", verbose = FALSE)
})[["elapsed"]]
n_lf1 <- fit_nb$diagnostics$n_leapfrog
if (is.list(n_lf1)) n_lf1 <- n_lf1[[1]]
cat(sprintf("  Time: %.2f s  Total sampling LF: %d  ms/kLF: %.2f\n",
  t1, sum(n_lf1), 1000 * t1 / sum(n_lf1)))

cat("\n=== PG+RE: 500 iter (standard) ===\n")
t2 <- system.time({
  fit_pg <- ratiod(y_num | effort ~ x + (1 | site), data = df,
    family = ratiod_poisson_gamma(), iter = 500, warmup = 250,
    chains = 1, gradient_mode = "H", verbose = FALSE)
})[["elapsed"]]
n_lf2 <- fit_pg$diagnostics$n_leapfrog
if (is.list(n_lf2)) n_lf2 <- n_lf2[[1]]
cat(sprintf("  Time: %.2f s  Total sampling LF: %d  ms/kLF: %.2f\n",
  t2, sum(n_lf2), 1000 * t2 / sum(n_lf2)))

# Now test with many iterations and very short warmup to isolate gradient cost
cat("\n=== NB+RE: 1000 iter, 50 warmup (gradient-dominated) ===\n")
t3 <- system.time({
  fit_nb2 <- ratiod(y_num | y_denom ~ x + (1 | site), data = df,
    family = ratiod_negbin_negbin(), iter = 1000, warmup = 50,
    chains = 1, gradient_mode = "H", verbose = FALSE)
})[["elapsed"]]
n_lf3 <- fit_nb2$diagnostics$n_leapfrog
if (is.list(n_lf3)) n_lf3 <- n_lf3[[1]]
cat(sprintf("  Time: %.2f s  Total sampling LF: %d  ms/kLF: %.2f\n",
  t3, sum(n_lf3), 1000 * t3 / sum(n_lf3)))

cat("\n=== PG+RE: 1000 iter, 50 warmup (gradient-dominated) ===\n")
t4 <- system.time({
  fit_pg2 <- ratiod(y_num | effort ~ x + (1 | site), data = df,
    family = ratiod_poisson_gamma(), iter = 1000, warmup = 50,
    chains = 1, gradient_mode = "H", verbose = FALSE)
})[["elapsed"]]
n_lf4 <- fit_pg2$diagnostics$n_leapfrog
if (is.list(n_lf4)) n_lf4 <- n_lf4[[1]]
cat(sprintf("  Time: %.2f s  Total sampling LF: %d  ms/kLF: %.2f\n",
  t4, sum(n_lf4), 1000 * t4 / sum(n_lf4)))

# Compare tree depths
cat("\n=== Tree depth comparison ===\n")
td_nb <- fit_nb$diagnostics$treedepth
if (is.list(td_nb)) td_nb <- td_nb[[1]]
td_pg <- fit_pg$diagnostics$treedepth
if (is.list(td_pg)) td_pg <- td_pg[[1]]
cat(sprintf("  NB+RE: mean depth=%.2f, median=%d, max=%d\n",
  mean(td_nb), median(td_nb), max(td_nb)))
cat(sprintf("  PG+RE: mean depth=%.2f, median=%d, max=%d\n",
  mean(td_pg), median(td_pg), max(td_pg)))

# Parameter counts
cat("\n=== Parameter counts ===\n")
cat(sprintf("  NB+RE params: %d\n", length(fit_nb$draws[[1]][1,])))
cat(sprintf("  PG+RE params: %d\n", length(fit_pg$draws[[1]][1,])))
