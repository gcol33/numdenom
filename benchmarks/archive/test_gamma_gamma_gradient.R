# Test gamma_gamma gradient correctness: H vs A mode comparison

library(numdenom)

set.seed(123)

# Generate simple test data
N <- 100
x <- rnorm(N)

# True parameters
beta_num <- c(2.0, 0.3)
beta_denom <- c(3.0, 0.2)
shape_num <- 5
shape_denom <- 8

# Generate data
eta_num <- cbind(1, x) %*% beta_num
eta_denom <- cbind(1, x) %*% beta_denom
mu_num <- exp(eta_num)
mu_denom <- exp(eta_denom)

y_num <- rgamma(N, shape = shape_num, rate = shape_num / mu_num)
y_denom <- rgamma(N, shape = shape_denom, rate = shape_denom / mu_denom)

df <- data.frame(y = y_num, denom = y_denom, x = x)

cat("=== Testing gamma_gamma gradient modes ===\n\n")

# Test with H gradient (hand-coded)
cat("Fitting with H (hand-coded) gradient...\n")
time_H <- system.time({
  fit_H <- ratiod(
    y | denom ~ x,
    data = df,
    family = ratiod_gamma_gamma(),
    iter = 500, warmup = 250, chains = 1,
    gradient_mode = "H",
    verbose = FALSE
  )
})
cat("H mode time:", time_H["elapsed"], "s\n")

# Test with A (forward autodiff) gradient
cat("\nFitting with A (forward autodiff) gradient...\n")
time_A <- system.time({
  fit_A <- ratiod(
    y | denom ~ x,
    data = df,
    family = ratiod_gamma_gamma(),
    iter = 500, warmup = 250, chains = 1,
    gradient_mode = "A",
    verbose = FALSE
  )
})
cat("A mode time:", time_A["elapsed"], "s\n")

# Compare results
draws_H <- as.matrix(fit_H$draws)
draws_A <- as.matrix(fit_A$draws)

cat("\n=== Posterior Comparison (H vs A) ===\n")
params <- c("beta_num[1]", "beta_num[2]", "beta_denom[1]", "beta_denom[2]",
            "shape_num", "shape_denom")

for (p in params) {
  if (p %in% colnames(draws_H) && p %in% colnames(draws_A)) {
    mean_H <- mean(draws_H[, p])
    mean_A <- mean(draws_A[, p])
    se_H <- sd(draws_H[, p]) / sqrt(nrow(draws_H))
    se_A <- sd(draws_A[, p]) / sqrt(nrow(draws_A))
    diff <- abs(mean_H - mean_A)
    diff_se <- diff / ((se_H + se_A) / 2)
    cat(sprintf("  %s: H=%.4f, A=%.4f, diff=%.4f (%.1f SE)\n",
                p, mean_H, mean_A, diff, diff_se))
  }
}

cat("\n=== Speed Comparison ===\n")
cat(sprintf("H: %.1fs, A: %.1fs, Speedup: %.1fx\n",
            time_H["elapsed"], time_A["elapsed"],
            time_A["elapsed"] / time_H["elapsed"]))

cat("\n=== True vs Estimated ===\n")
cat(sprintf("  beta_num[1]: true=%.2f, H=%.4f, A=%.4f\n",
            beta_num[1], mean(draws_H[,"beta_num[1]"]), mean(draws_A[,"beta_num[1]"])))
cat(sprintf("  beta_num[2]: true=%.2f, H=%.4f, A=%.4f\n",
            beta_num[2], mean(draws_H[,"beta_num[2]"]), mean(draws_A[,"beta_num[2]"])))
cat(sprintf("  beta_denom[1]: true=%.2f, H=%.4f, A=%.4f\n",
            beta_denom[1], mean(draws_H[,"beta_denom[1]"]), mean(draws_A[,"beta_denom[1]"])))
cat(sprintf("  beta_denom[2]: true=%.2f, H=%.4f, A=%.4f\n",
            beta_denom[2], mean(draws_H[,"beta_denom[2]"]), mean(draws_A[,"beta_denom[2]"])))
cat(sprintf("  shape_num: true=%.2f, H=%.4f, A=%.4f\n",
            shape_num, mean(draws_H[,"shape_num"]), mean(draws_A[,"shape_num"])))
cat(sprintf("  shape_denom: true=%.2f, H=%.4f, A=%.4f\n",
            shape_denom, mean(draws_H[,"shape_denom"]), mean(draws_A[,"shape_denom"])))

# Check for divergences
n_div_H <- sum(fit_H$divergences)
n_div_A <- sum(fit_A$divergences)
cat(sprintf("\nDivergences: H=%d, A=%d\n", n_div_H, n_div_A))
