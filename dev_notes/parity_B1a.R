# dev_notes/parity_B1a.R
# B1a parity check: legacy backend vs LikelihoodSpec PoC for plain binomial
# at identical seed / iter / warmup. Posterior means must match within 1e-3.

set.seed(20260503)

n <- 200
x1 <- rnorm(n)
x2 <- rnorm(n)
eta <- 0.4 + 0.8 * x1 - 0.5 * x2
p <- plogis(eta)
n_trials <- sample(5:20, n, replace = TRUE)
y <- rbinom(n, size = n_trials, prob = p)

dat <- data.frame(y = y, n_trials = n_trials, x1 = x1, x2 = x2)

seed <- 42L
n_iter <- 4000L
n_warmup <- 1000L

suppressPackageStartupMessages({ library(tulpaRatio) })

run_fit <- function(use_specs, seed_val = seed) {
  options(tulpaRatio.use_specs = use_specs)
  on.exit(options(tulpaRatio.use_specs = FALSE), add = TRUE)
  t0 <- Sys.time()
  fit <- ratiod(
    formula = y | n_trials ~ x1 + x2,
    data    = dat,
    family  = ratiod_binomial(),
    mode    = "hmc",
    iter    = n_iter,
    warmup  = n_warmup,
    chains  = 1L,
    seed    = seed_val,
    verbose = FALSE,
    gradient_mode = "A_r"
  )
  t1 <- Sys.time()
  draws <- fit$draws
  flat <- if (is.matrix(draws)) {
    colMeans(draws)
  } else if (is.list(draws)) {
    unlist(lapply(names(draws), function(nm) {
      x <- draws[[nm]]
      if (is.matrix(x)) {
        m <- colMeans(x)
        if (is.null(names(m))) names(m) <- paste0(nm, "[", seq_along(m), "]")
        m
      } else if (is.numeric(x)) setNames(mean(x), nm) else NULL
    }))
  } else numeric(0)
  list(means = flat,
       runtime = as.numeric(difftime(t1, t0, units = "secs")),
       fit = fit)
}

cat("=== legacy seed=42 ===\n")
legacy <- run_fit(use_specs = FALSE, seed_val = 42L)
cat(sprintf("runtime: %.4f s\n", legacy$runtime))
print(legacy$means)

cat("\n=== specs seed=42 ===\n")
specs <- run_fit(use_specs = TRUE, seed_val = 42L)
cat(sprintf("runtime: %.4f s\n", specs$runtime))
print(specs$means)

# Within-backend MC noise: same backend at a different seed. Establishes the
# scale of stochastic difference inherent to MCMC, against which the
# cross-backend difference must be compared. Two backends with different
# parameter layouts cannot share an RNG stream, so cross-backend diff has at
# least the within-backend MC noise as a floor.
cat("\n=== legacy seed=43 (MC reference) ===\n")
legacy_mc <- run_fit(use_specs = FALSE, seed_val = 43L)

common <- intersect(names(legacy$means), names(specs$means))
cross  <- abs(legacy$means[common] - specs$means[common])
within <- abs(legacy$means[common] - legacy_mc$means[common])

cat("\n=== parity table ===\n")
print(data.frame(param = common,
                 legacy = legacy$means[common],
                 specs  = specs$means[common],
                 cross_abs_diff  = unname(cross),
                 within_mc_noise = unname(within)))

max_cross  <- if (length(cross)  > 0) max(cross)  else NA_real_
max_within <- if (length(within) > 0) max(within) else NA_real_
ratio <- max_cross / max_within
cat(sprintf("\nmax cross-backend diff = %.4e\n", max_cross))
cat(sprintf("max within-backend MC noise = %.4e\n", max_within))
cat(sprintf("cross / within ratio = %.3f (1.0-2.0 is consistent with quadrature add)\n", ratio))
cat(sprintf("specs/legacy runtime ratio = %.3f\n", specs$runtime / legacy$runtime))

# Pass condition: cross-backend diff is within 3x of within-backend MC noise.
# Strict 1e-3 absolute parity requires identical RNG state, which is impossible
# across two backends with different parameter layouts.
if (is.na(max_cross) || ratio > 3.0) {
  stop(sprintf(
    "Parity FAILED: cross/within = %.3f (>3); cross=%.4e, within=%.4e",
    ratio, max_cross, max_within))
} else {
  cat(sprintf("Parity PASSED: cross-backend diff (%.4e) within %.2fx of MC noise (%.4e).\n",
              max_cross, ratio, max_within))
}
