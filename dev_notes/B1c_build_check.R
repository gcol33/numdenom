#!/usr/bin/env Rscript
# B1c build verification: load the package, sanity-check that the spec path
# accepts ZI/HURDLE/OI configurations end-to-end with finite output.

setwd("C:/Users/Gilles Colling/Documents/dev/tulpaRatio")

suppressPackageStartupMessages({
  library(devtools)
})

cat("--- compileAttributes ---\n")
Rcpp::compileAttributes()

cat("--- load_all ---\n")
devtools::load_all(quiet = TRUE)

# Quick smoke: simulate ZI binomial and run BOTH legacy and spec paths
set.seed(20260504)
n <- 100
x1 <- rnorm(n)
nt <- sample(5:20, n, replace = TRUE)
eta <- 0.5 + 0.6 * x1
p_succ <- plogis(eta)
is_zero <- runif(n) < plogis(-1.0)
y <- ifelse(is_zero, 0L, rbinom(n, nt, p_succ))
df <- data.frame(y = y, n_trials = nt, x1 = x1)

cat("--- legacy ZI binomial fit ---\n")
options(tulpaRatio.use_specs = FALSE)
fit_leg <- ratiod(y | n_trials ~ x1, data = df,
                  family = ratiod_zibinomial(),
                  mode = "hmc", iter = 400L, warmup = 200L,
                  chains = 1L, seed = 42L, verbose = FALSE,
                  gradient_mode = "A_r")
cat("  legacy posterior means:\n")
print(round(colMeans(fit_leg$draws), 4))

cat("--- spec ZI binomial fit ---\n")
options(tulpaRatio.use_specs = TRUE)
fit_spec <- ratiod(y | n_trials ~ x1, data = df,
                   family = ratiod_zibinomial(),
                   mode = "hmc", iter = 400L, warmup = 200L,
                   chains = 1L, seed = 42L, verbose = FALSE,
                   gradient_mode = "A_r")
cat("  spec posterior means:\n")
print(round(colMeans(fit_spec$draws), 4))

cat("\n--- max abs diff ---\n")
print(round(max(abs(colMeans(fit_leg$draws) - colMeans(fit_spec$draws))), 5))

options(tulpaRatio.use_specs = FALSE)
cat("OK\n")
