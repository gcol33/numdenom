suppressMessages(devtools::load_all("C:/GillesC/Documents/dev/tulpaRatio", quiet = TRUE))
set.seed(20260503)

n <- 200
x1 <- rnorm(n); x2 <- rnorm(n)
eta <- 0.4 + 0.8 * x1 - 0.5 * x2
p <- plogis(eta)
n_trials <- sample(5:20, n, replace = TRUE)
phi <- 10
alpha <- p * phi; beta <- (1 - p) * phi
y <- vapply(seq_len(n), function(i) {
  pp <- rbeta(1, alpha[i], beta[i])
  rbinom(1, n_trials[i], pp)
}, integer(1))
dat <- data.frame(y = y, n_trials = n_trials, x1 = x1, x2 = x2)

fit_one <- function(use_specs, seed_val, gradient_mode = "H", iter = 4000L) {
  op <- options(tulpaRatio.use_specs = use_specs)
  on.exit(options(op), add = TRUE)
  fit <- tulpaRatio::ratiod(
    formula = y | n_trials ~ x1 + x2, data = dat,
    family = tulpaRatio::ratiod_beta_binomial(),
    mode = "hmc", iter = iter, warmup = 1000L, chains = 1L,
    seed = seed_val, verbose = FALSE, gradient_mode = gradient_mode
  )
  colMeans(fit$draws)
}

# Compare specs/H vs specs/A_r at multiple seeds: this is the gradient
# identity check (must be exact).
cat("\n=== Beta-binomial gradient identity (specs/H == specs/A_r) ===\n")
for (s in c(42L, 43L, 44L)) {
  h  <- fit_one(TRUE, s, "H")
  ar <- fit_one(TRUE, s, "A_r")
  cat(sprintf(" seed %d: max abs diff = %.3e\n", s, max(abs(h - ar))))
}

# Long-run cross-backend comparison (more iterations -> smaller MC noise)
cat("\n=== Beta-binomial 4000-iter cross-backend ===\n")
legacy42 <- fit_one(FALSE, 42L, "H", 4000L)
specs42  <- fit_one(TRUE,  42L, "H", 4000L)
legacy43 <- fit_one(FALSE, 43L, "H", 4000L)
specs43  <- fit_one(TRUE,  43L, "H", 4000L)

cat("\nDraws (param means at seed 42):\n")
print(round(rbind(legacy = legacy42, specs = specs42), 4))

cross   <- max(abs(legacy42 - specs42))
within_legacy <- max(abs(legacy42 - legacy43))
within_specs  <- max(abs(specs42  - specs43))
within  <- max(within_legacy, within_specs)
cat(sprintf("\n cross = %.4g\n within_legacy = %.4g\n within_specs  = %.4g\n",
            cross, within_legacy, within_specs))
cat(sprintf(" parity (cross < 4*max_within) = %s\n",
            ifelse(cross < max(4 * within, 5e-3), "PASS", "FAIL")))
