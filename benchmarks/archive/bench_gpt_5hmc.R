library(numdenom)
set.seed(123)
N <- 500L; N_s <- 50L; N_t <- 20L
site <- factor(rep(1:N_s, length.out = N))
x <- rnorm(N)
time_num <- rep(1:N_t, length.out = N)
y <- rpois(N, exp(2 + 0.5 * x))
d <- rgamma(N, 10, 1)
df <- data.frame(y = y, denom = d, x = x, site = site, time_num = time_num)

cat("PG+GP_t (same data, 5 HMC seeds):\n")
times <- numeric(5)
for (s in 1:5) {
  t <- system.time(fit <- tratio(y | denom ~ x + (1 | site), data = df,
    family = ratiod_poisson_gamma(), temporal = temporal_gp("time_num"),
    control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE, seed = s)))[["elapsed"]]
  times[s] <- t
  cat(sprintf("  seed=%d: %.1fs\n", s, t))
}
cat(sprintf("  MEDIAN: %.1fs\n", median(times)))
