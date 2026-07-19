# Minimal reproduction: ratiod summary()/mcmc_diagnostics() report rhat ~2.12
# and ess_bulk ~1 for a well-mixed fit whose draws give rhat ~1.008 via posterior.
#
# Run: Rscript dev_notes/repro_rhat_ess_bug.R

suppressMessages({ library(tulpaRatio); stopifnot(requireNamespace("posterior", quietly = TRUE)) })

set.seed(123)
N <- 500
x <- rnorm(N); trials <- sample(10:50, N, replace = TRUE)
y <- rbinom(N, trials, plogis(0.5 + 0.3 * x))
df <- data.frame(y = y, trials = trials, x = x)

fit <- ratiod(y | trials ~ x, data = df, family = ratiod_binomial(),
              iter = 1000, warmup = 500, chains = 4, verbose = FALSE)

cat("backend =", fit$backend, " chains =", fit$chains,
    " nrow(draws) =", nrow(fit$draws), "\n\n")

cat("=== 1. what tulpaRatio reports ===\n")
print(mcmc_diagnostics(fit))

cat("\n=== 2. posterior:: applied to get_draws_array()'s own array ===\n")
da <- tulpaRatio:::get_draws_array(fit)
cat("dim(draws_array) =", paste(dim(da$draws), collapse = " x "),
    " (expect iter x chain x param =", nrow(fit$draws) / fit$chains, "x",
    fit$chains, "x", ncol(fit$draws), ")\n")
arr <- da$draws
for (p in seq_len(dim(arr)[3])) {
  m <- arr[, , p]
  cat(sprintf("  %-12s rhat=%.4f ess_bulk=%.1f\n", dimnames(arr)[[3]][p],
              posterior::rhat(m), posterior::ess_bulk(m)))
}

cat("\n=== 3. independent manual reshape (contiguous chain blocks) ===\n")
dr <- as.matrix(fit$draws); per <- nrow(dr) / fit$chains
for (p in seq_len(ncol(dr))) {
  m <- matrix(dr[, p], nrow = per, ncol = fit$chains)
  cat(sprintf("  %-12s rhat=%.4f ess_bulk=%.1f\n", colnames(dr)[p],
              posterior::rhat(m), posterior::ess_bulk(m)))
}

cat("\n=== 4. per-chain means (chains agree => well mixed) ===\n")
for (c in seq_len(fit$chains)) {
  blk <- dr[((c - 1) * per + 1):(c * per), , drop = FALSE]
  cat(sprintf("  chain %d: beta_num[2] mean=%.4f sd=%.4f\n",
              c, mean(blk[, "beta_num[2]"]), sd(blk[, "beta_num[2]"])))
}
