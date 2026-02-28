# Test gamma_gamma gradient correctness: ALL modes comparison (H, A, A_t, N)

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

cat("=== Testing gamma_gamma: ALL gradient modes ===\n")
cat("True params: beta_num = (", paste(beta_num, collapse=", "), ")\n")
cat("             beta_denom = (", paste(beta_denom, collapse=", "), ")\n")
cat("             shape_num =", shape_num, ", shape_denom =", shape_denom, "\n\n")

# Common settings
common_args <- list(
  formula = y | denom ~ x,
  data = df,
  family = ratiod_gamma_gamma(),
  iter = 1000, warmup = 500, chains = 1,
  verbose = FALSE,
  seed = 42
)

modes <- c("H", "A", "A_t", "N")
results <- list()

for (mode in modes) {
  cat("Fitting with", mode, "mode...\n")
  args <- c(common_args, list(gradient_mode = mode))
  time <- system.time({
    fit <- do.call(ratiod, args)
  })
  results[[mode]] <- list(
    fit = fit,
    time = time["elapsed"],
    draws = as.matrix(fit$draws)
  )
  cat("  Time:", round(time["elapsed"], 1), "s\n")
}

cat("\n=== Posterior Means Comparison ===\n")
params <- c("beta_num[1]", "beta_num[2]", "beta_denom[1]", "beta_denom[2]",
            "shape_num", "shape_denom")
true_vals <- c(beta_num, beta_denom, shape_num, shape_denom)

# Print header
cat(sprintf("%15s | %8s | %8s | %8s | %8s | %8s\n",
            "Parameter", "True", "H", "A", "A_t", "N"))
cat(paste(rep("-", 70), collapse=""), "\n")

for (i in seq_along(params)) {
  p <- params[i]
  vals <- sapply(modes, function(m) {
    if (p %in% colnames(results[[m]]$draws)) {
      mean(results[[m]]$draws[, p])
    } else {
      NA
    }
  })
  cat(sprintf("%15s | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f\n",
              p, true_vals[i], vals["H"], vals["A"], vals["A_t"], vals["N"]))
}

cat("\n=== Timing Summary ===\n")
for (mode in modes) {
  cat(sprintf("  %s: %.1fs\n", mode, results[[mode]]$time))
}

cat("\n=== Divergences ===\n")
for (mode in modes) {
  n_div <- sum(results[[mode]]$fit$divergences)
  cat(sprintf("  %s: %d divergences\n", mode, n_div))
}

# Check H-N gradient agreement (gold standard check)
cat("\n=== H vs N difference (should be near 0 if H is correct) ===\n")
for (i in seq_along(params)) {
  p <- params[i]
  if (p %in% colnames(results[["H"]]$draws) && p %in% colnames(results[["N"]]$draws)) {
    mean_H <- mean(results[["H"]]$draws[, p])
    mean_N <- mean(results[["N"]]$draws[, p])
    diff <- mean_H - mean_N
    cat(sprintf("  %s: H-N = %.4f\n", p, diff))
  }
}
