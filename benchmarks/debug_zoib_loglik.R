# Debug ZOIB: Check if log-posterior changes with ZI/OI parameters
library(numdenom)

set.seed(42)

N_OBS <- 100
N_SITES <- 10

# Setup
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)
trials <- sample(30:60, N_OBS, replace = TRUE)

# True parameters
true_intercept <- 0.5
true_slope <- 0.3
true_sigma_re <- 0.3
site_effects <- rnorm(N_SITES, 0, true_sigma_re)

# ZOIB parameters
true_pi0 <- 0.15
true_pi1 <- 0.12

# Generate data
eta <- true_intercept + true_slope * x + site_effects[as.integer(site)]
prob <- plogis(eta)
is_struct_zero <- rbinom(N_OBS, 1, true_pi0)
is_struct_one <- ifelse(is_struct_zero == 1, 0, rbinom(N_OBS, 1, true_pi1))
successes <- ifelse(is_struct_zero == 1, 0,
              ifelse(is_struct_one == 1, trials,
                     rbinom(N_OBS, trials, prob)))

df <- data.frame(successes = successes, trials = trials, x = x, site = site)

cat("Data summary:\n")
cat(sprintf("  Zeros: %d (%.1f%%)\n", sum(successes == 0), 100 * mean(successes == 0)))
cat(sprintf("  Ones:  %d (%.1f%%)\n", sum(successes == trials), 100 * mean(successes == trials)))

# Fit a short run and extract the log-posterior values
cat("\n\nFitting ZOIB model (short run) with A_t mode...\n")
fit <- ratiod(successes | trials ~ x + (1 | site), data = df,
         family = ratiod_zoibinomial(),
         iter = 100, warmup = 50, chains = 1, verbose = FALSE,
         gradient_mode = "A_t")

draws <- as.matrix(fit$draws)
cat("\nParameter names:\n")
print(colnames(draws))

cat("\n\nFinal parameter values:\n")
cat(sprintf("beta_num[1] (intercept): %.4f\n", mean(draws[,"beta_num[1]"])))
cat(sprintf("beta_num[2] (slope): %.4f\n", mean(draws[,"beta_num[2]"])))
cat(sprintf("sigma_re: %.4f\n", mean(draws[,"sigma_re"])))
if ("beta_zi[1]" %in% colnames(draws)) {
  cat(sprintf("beta_zi[1]: %.4f (true: %.4f)\n", mean(draws[,"beta_zi[1]"]), qlogis(true_pi0)))
}
if ("beta_oi[1]" %in% colnames(draws)) {
  cat(sprintf("beta_oi[1]: %.4f (true: %.4f)\n", mean(draws[,"beta_oi[1]"]), qlogis(true_pi1)))
}

# Now manually compute the ZOIB log-likelihood for comparison
cat("\n\n=== Manual log-likelihood calculation ===\n")

# Use the posterior means from the fit
intercept <- mean(draws[,"beta_num[1]"])
slope <- mean(draws[,"beta_num[2]"])

# Compute eta and p for each observation
eta_manual <- intercept + slope * x  # Ignoring RE for simplicity
p_manual <- plogis(eta_manual)

# Compute log-likelihood with different ZI/OI values
compute_zoib_ll <- function(data, p, logit_zi, logit_oi) {
  y <- data$successes
  n <- data$trials

  zi <- plogis(logit_zi)
  oi <- plogis(logit_oi)

  ll <- 0
  for (i in 1:nrow(data)) {
    if (y[i] == 0) {
      # P(Y=0) = zi + (1-zi)*(1-oi)*Binom(0|n,p)
      p_binom_zero <- (1 - p[i])^n[i]
      prob <- zi + (1 - zi) * (1 - oi) * p_binom_zero
    } else if (y[i] == n[i]) {
      # P(Y=n) = (1-zi)*(oi + (1-oi)*Binom(n|n,p))
      p_binom_n <- p[i]^n[i]
      prob <- (1 - zi) * (oi + (1 - oi) * p_binom_n)
    } else {
      # P(Y=y) = (1-zi)*(1-oi)*Binom(y|n,p)
      prob <- (1 - zi) * (1 - oi) * dbinom(y[i], n[i], p[i])
    }
    ll <- ll + log(max(prob, 1e-300))
  }
  ll
}

# Test with different ZI/OI values
cat("\nLog-likelihood with different ZI/OI values:\n")
for (logit_zi in c(-3, -1.735, 0, 1)) {
  for (logit_oi in c(-3, -1.992, 0, 1)) {
    ll <- compute_zoib_ll(df, p_manual, logit_zi, logit_oi)
    cat(sprintf("  logit_zi=%.2f, logit_oi=%.2f -> ll=%.2f\n", logit_zi, logit_oi, ll))
  }
}

cat("\nNote: True logit_zi = %.3f, true logit_oi = %.3f\n", qlogis(true_pi0), qlogis(true_pi1))
