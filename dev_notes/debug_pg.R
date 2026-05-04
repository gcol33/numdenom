suppressPackageStartupMessages({
  devtools::load_all(".", quiet = TRUE)
})

set.seed(42)
N <- 100
n_groups <- 6
group <- factor(rep(seq_len(n_groups), length.out = N))
x_fix <- rnorm(N)
x_re  <- rnorm(N)
eta_n <- 0.3 + 0.4 * x_fix +
         rnorm(n_groups, sd = 0.5)[as.integer(group)]
eta_d <- 0.5 + 0.2 * x_fix
mu_n <- exp(eta_n); mu_d <- exp(eta_d)
y_n  <- rpois(N, mu_n)
y_d  <- rgamma(N, shape = 2, rate = 2 / mu_d)

df <- data.frame(y_num = y_n, y_denom = y_d,
                 x_fix = x_fix, x_re = x_re, group = group)

cat("LEGACY poisson_gamma intercept-only RE:\n")
options(tulpaRatio.use_specs = FALSE)
fit <- ratiod(y_num | y_denom ~ x_fix + (1 | group),
              data = df, family = ratiod_poisson_gamma(),
              mode = "hmc", iter = 200L, warmup = 100L,
              chains = 1L, seed = 42L, verbose = FALSE)
draws <- fit$draws; if (is.list(draws) && !is.matrix(draws)) draws <- draws[[1]]
cat("  n_params=", ncol(draws), "\n")
print(head(round(colMeans(draws), 3), 6))
