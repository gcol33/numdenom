#!/usr/bin/env Rscript
# 3-rep benchmark for stable timings
library(numdenom)

set.seed(42)
N <- 500; S <- 50
site <- rep(1:S, length.out = N)
region <- rep(1:10, each = N/10)
x1 <- rnorm(N)
lambda_num <- exp(1 + 0.3 * x1)
lambda_denom <- exp(2 + 0.1 * x1)
y_num <- rnbinom(N, mu = lambda_num, size = 5)
y_denom <- rnbinom(N, mu = lambda_denom, size = 5)
y_denom[y_denom == 0] <- 1
df <- data.frame(y_num=y_num, y_denom=y_denom, x1=x1,
                 site=factor(site), region=factor(region))

models <- list(
  pg_base = list(f = y_num | y_denom ~ x1, fam = ratiod_poisson_gamma(), stan = 1.2),
  pg_re = list(f = y_num | y_denom ~ x1 + (1 | site), fam = ratiod_poisson_gamma(), stan = 2.5),
  pg_crossed = list(f = y_num | y_denom ~ x1 + (1 | site) + (1 | region), fam = ratiod_poisson_gamma(), stan = 2.9),
  nb_base = list(f = y_num | y_denom ~ x1, fam = ratiod_negbin_negbin(), stan = 1.5),
  nb_re = list(f = y_num | y_denom ~ x1 + (1 | site), fam = ratiod_negbin_negbin(), stan = 2.9),
  nb_crossed = list(f = y_num | y_denom ~ x1 + (1 | site) + (1 | region), fam = ratiod_negbin_negbin(), stan = 9.9)
)

cat("=== 3-rep Benchmark (N=500, 500 iter, 1 chain, -O3 + code fixes) ===\n\n")
cat(sprintf("%-15s  %6s  %6s  %6s  %6s  %8s\n", "Model", "r1", "r2", "r3", "median", "Stan"))
cat(paste(rep("-", 60), collapse = ""), "\n")

for (nm in names(models)) {
  m <- models[[nm]]
  times <- numeric(3)
  for (r in 1:3) {
    set.seed(100 + r)
    times[r] <- system.time({
      tryCatch(
        tratio(m$f, data = df, family = m$fam, mode = "hmc",
               control = list(iter = 500, warmup = 250, chains = 1, gradient_mode = "H", verbose = FALSE)),
        error = function(e) { cat("ERROR:", nm, e$message, "\n"); NULL }
      )
    })["elapsed"]
  }
  med <- median(times)
  ratio <- med / m$stan
  marker <- if (ratio <= 1.0) " <-- BEATS STAN" else ""
  cat(sprintf("%-15s  %6.1f  %6.1f  %6.1f  %6.1f  %6.1f  (%.2fx)%s\n",
              nm, times[1], times[2], times[3], med, m$stan, ratio, marker))
}
