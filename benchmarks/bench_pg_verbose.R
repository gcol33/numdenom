# bench_pg_verbose.R — diagnose PG+RE warmup overhead
library(numdenom)
set.seed(42)

N <- 500; S <- 50
df <- data.frame(site = rep(1:S, each = N/S), x = rnorm(N))
eta <- 1.5 + 0.3 * df$x + rnorm(S, 0, 0.3)[df$site]
df$y_num <- rnbinom(N, size = 5, mu = exp(eta))
df$y_denom <- pmax(rnbinom(N, size = 5, mu = exp(eta + 0.5)), 1)
df$effort <- pmax(rgamma(N, shape = 5, rate = 5 / exp(eta + 0.5)), 0.01)

cat("\n=== PG+RE verbose ===\n")
t1 <- system.time({
  fit <- ratiod(y_num | effort ~ x + (1 | site), data = df,
    family = ratiod_poisson_gamma(), iter = 500, warmup = 250,
    chains = 1, gradient_mode = "H", verbose = TRUE)
})["elapsed"]
cat(sprintf("\nTotal time: %.2f s\n", t1))
