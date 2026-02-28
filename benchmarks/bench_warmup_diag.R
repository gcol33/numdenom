# bench_warmup_diag.R
# Diagnose warmup vs sampling leapfrog for NB+RE and PG+RE
library(numdenom)
set.seed(42)

N <- 500; S <- 50
df <- data.frame(site = rep(1:S, each = N/S), x = rnorm(N))
eta <- 1.5 + 0.3 * df$x + rnorm(S, 0, 0.3)[df$site]
df$y_num <- rnbinom(N, size = 5, mu = exp(eta))
df$y_denom <- pmax(rnbinom(N, size = 5, mu = exp(eta + 0.5)), 1)
df$effort <- pmax(rgamma(N, shape = 5, rate = 5 / exp(eta + 0.5)), 0.01)

cat("\n=== NB+RE ===\n")
t1 <- system.time({
  fit_nb <- ratiod(y_num | y_denom ~ x + (1 | site), data = df,
    family = ratiod_negbin_negbin(), iter = 500, warmup = 250,
    chains = 1, gradient_mode = "H", verbose = FALSE)
})["elapsed"]
td <- fit_nb$diagnostics$treedepth
if (is.list(td)) td <- td[[1]]
warmup_lf <- sum(2^td[1:250])
sample_lf <- sum(2^td[251:500])
cat(sprintf("  Time: %.2f s\n", t1))
cat(sprintf("  Warmup: depth=%.1f, LF=%d\n", mean(td[1:250]), warmup_lf))
cat(sprintf("  Sampling: depth=%.1f, LF=%d\n", mean(td[251:500]), sample_lf))
cat(sprintf("  Total LF: %d\n", warmup_lf + sample_lf))
cat(sprintf("  True ms/kLF: %.2f\n", 1000 * t1 / (warmup_lf + sample_lf)))

cat("\n=== PG+RE ===\n")
t2 <- system.time({
  fit_pg <- ratiod(y_num | effort ~ x + (1 | site), data = df,
    family = ratiod_poisson_gamma(), iter = 500, warmup = 250,
    chains = 1, gradient_mode = "H", verbose = FALSE)
})["elapsed"]
td2 <- fit_pg$diagnostics$treedepth
if (is.list(td2)) td2 <- td2[[1]]
warmup_lf2 <- sum(2^td2[1:250])
sample_lf2 <- sum(2^td2[251:500])
cat(sprintf("  Time: %.2f s\n", t2))
cat(sprintf("  Warmup: depth=%.1f, LF=%d\n", mean(td2[1:250]), warmup_lf2))
cat(sprintf("  Sampling: depth=%.1f, LF=%d\n", mean(td2[251:500]), sample_lf2))
cat(sprintf("  Total LF: %d\n", warmup_lf2 + sample_lf2))
cat(sprintf("  True ms/kLF: %.2f\n", 1000 * t2 / (warmup_lf2 + sample_lf2)))
