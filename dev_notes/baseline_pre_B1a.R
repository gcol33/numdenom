# dev_notes/baseline_pre_B1a.R
# Baseline: legacy tulpaRatio binomial fit before any LikelihoodSpec PoC work.
# Captures posterior means + runtime for parity comparison after B1a wires the
# LikelihoodSpec path. Tiny model (~200 obs, 2 covariates, 200 iter) so the run
# is fast enough to repeat.

set.seed(20260503)

n <- 200
x1 <- rnorm(n)
x2 <- rnorm(n)
eta <- 0.4 + 0.8 * x1 - 0.5 * x2
p <- plogis(eta)
n_trials <- sample(5:20, n, replace = TRUE)
y <- rbinom(n, size = n_trials, prob = p)

dat <- data.frame(y = y, n_trials = n_trials, x1 = x1, x2 = x2)

formula_str <- "y | n_trials ~ x1 + x2"
seed <- 42L
n_iter <- 200L
n_warmup <- 100L

suppressPackageStartupMessages({
  library(tulpaRatio)
})

cat("=== tulpaRatio binomial baseline ===\n")
cat("formula:", formula_str, "\n")
cat("n_obs:", n, " seed:", seed, " iter:", n_iter, " warmup:", n_warmup, "\n\n")

t0 <- Sys.time()
fit <- ratiod(
  formula = y | n_trials ~ x1 + x2,
  data    = dat,
  family  = ratiod_binomial(),
  mode    = "hmc",
  iter    = n_iter,
  warmup  = n_warmup,
  chains  = 1L,
  seed    = seed,
  verbose = FALSE,
  gradient_mode = "A_r"
)
t1 <- Sys.time()
runtime <- as.numeric(difftime(t1, t0, units = "secs"))

cat("=== runtime ===\n")
cat(sprintf("legacy_seconds = %.3f\n\n", runtime))

post <- summary(fit)
cat("=== summary ===\n")
print(post)

# Extract posterior means by walking the draws structure.
draws <- fit$draws
if (is.list(draws)) {
  flat <- unlist(lapply(names(draws), function(nm) {
    x <- draws[[nm]]
    if (is.matrix(x)) {
      m <- colMeans(x)
      if (is.null(names(m))) names(m) <- paste0(nm, "[", seq_along(m), "]")
      m
    } else if (is.numeric(x)) {
      setNames(mean(x), nm)
    } else {
      NULL
    }
  }))
} else if (is.matrix(draws)) {
  flat <- colMeans(draws)
} else {
  flat <- numeric(0)
}

cat("\n=== posterior means (flat) ===\n")
for (nm in names(flat)) {
  cat(sprintf("%-40s %12.6f\n", nm, flat[[nm]]))
}

baseline <- list(
  formula = formula_str,
  seed    = seed,
  n_obs   = n,
  n_iter  = n_iter,
  n_warmup = n_warmup,
  runtime_seconds = runtime,
  posterior_means = flat
)

saveRDS(baseline, file.path("dev_notes", "baseline_pre_B1a.rds"))
cat("\nSaved baseline to dev_notes/baseline_pre_B1a.rds\n")
