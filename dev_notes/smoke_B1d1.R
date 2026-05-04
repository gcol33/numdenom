suppressPackageStartupMessages({
  devtools::load_all(".", quiet = TRUE)
})

set.seed(42)
N <- 150L
n_groups <- 6L
group <- factor(rep(seq_len(n_groups), length.out = N))
x_fix <- rnorm(N)
x_re  <- rnorm(N)
trials <- rpois(N, 8) + 5L

# Intercept-only RE
eta1 <- 0.3 + 0.4 * x_fix +
        rnorm(n_groups, sd = 0.5)[as.integer(group)]
y1 <- rbinom(N, trials, plogis(eta1))
df1 <- data.frame(successes = y1, trials = trials,
                  x_fix = x_fix, x_re = x_re, group = group)

cat("Smoke 1: binomial intercept-only RE, spec path\n")
options(tulpaRatio.use_specs = TRUE)
fit1 <- ratiod(successes | trials ~ x_fix + (1 | group),
               data = df1, family = ratiod_binomial(),
               mode = "hmc", iter = 200L, warmup = 100L,
               chains = 1L, seed = 42L, verbose = FALSE)
draws1 <- fit1$draws
if (is.list(draws1) && !is.matrix(draws1)) draws1 <- draws1[[1]]
cat("  n_params=", ncol(draws1), "\n")
print(head(round(colMeans(draws1), 3), 6))

# Intercept + slope RE
eta2 <- 0.3 + 0.4 * x_fix +
        rnorm(n_groups, sd = 0.5)[as.integer(group)] +
        rnorm(n_groups, sd = 0.3)[as.integer(group)] * x_re
y2 <- rbinom(N, trials, plogis(eta2))
df2 <- data.frame(successes = y2, trials = trials,
                  x_fix = x_fix, x_re = x_re, group = group)

cat("\nSmoke 2: binomial intercept+slope RE, spec path\n")
fit2 <- ratiod(successes | trials ~ x_fix + (1 + x_re || group),
               data = df2, family = ratiod_binomial(),
               mode = "hmc", iter = 200L, warmup = 100L,
               chains = 1L, seed = 42L, verbose = FALSE)
draws2 <- fit2$draws
if (is.list(draws2) && !is.matrix(draws2)) draws2 <- draws2[[1]]
cat("  n_params=", ncol(draws2), "\n")
print(head(round(colMeans(draws2), 3), 8))

cat("\nSmoke OK\n")
