# Debug lognormal - test log-posterior directly with cpp_test_log_post if available

library(numdenom)

set.seed(42)

# Generate minimal intercept-only data
N <- 100
mu_num <- 2.0  # mean on log scale (intercept)
mu_denom <- 4.0
sigma_num <- 0.5
sigma_denom <- 0.3

y_num <- exp(rnorm(N, mu_num, sigma_num))
y_denom <- exp(rnorm(N, mu_denom, sigma_denom))

dat <- data.frame(y_num = y_num, y_denom = y_denom)

# Compute log-likelihood manually for lognormal
lognormal_ll <- function(y, mu, sigma) {
  # log(Y) ~ N(mu, sigma^2)
  # log_pdf = -log(y) - log(sigma) - 0.5*((log(y) - mu)/sigma)^2
  log_y <- log(y)
  z <- (log_y - mu) / sigma
  sum(-log(y) - log(sigma) - 0.5 * z^2)
}

# Compute at true values
ll_true <- lognormal_ll(y_num, mu_num, sigma_num) + lognormal_ll(y_denom, mu_denom, sigma_denom)
cat("LL at TRUE values:\n")
cat("  LL =", ll_true, "\n\n")

# Compute at values the sampler is stuck at
ll_stuck <- lognormal_ll(y_num, 1.75, 0.60) + lognormal_ll(y_denom, 4.01, 0.19)
cat("LL at STUCK values (beta_num=1.75, sigma_num=0.60, beta_denom=4.01, sigma_denom=0.19):\n")
cat("  LL =", ll_stuck, "\n\n")

# Compare
cat("Difference (stuck - true):", ll_stuck - ll_true, "\n")

# Wait, is the issue that the NUMERATOR and DENOMINATOR sigmas are swapped?
# Let me check if sigma_num=0.3 and sigma_denom=0.5 would give better LL
ll_swapped_sigma <- lognormal_ll(y_num, 1.75, 0.30) + lognormal_ll(y_denom, 4.01, 0.50)
cat("\nLL with SWAPPED sigmas (sigma_num=0.30, sigma_denom=0.50):\n")
cat("  LL =", ll_swapped_sigma, "\n")

# What if the issue is that the data y_num and y_denom are swapped?
ll_swapped_data <- lognormal_ll(y_denom, mu_num, sigma_num) + lognormal_ll(y_num, mu_denom, sigma_denom)
cat("\nLL with SWAPPED DATA (fitting y_denom as num, y_num as denom):\n")
cat("  LL =", ll_swapped_data, "\n")

# What about MLE for each?
cat("\n=== MLE CHECK ===\n")
cat("For y_num:\n")
cat("  mean(log(y_num)) =", mean(log(y_num)), " (true mu_num =", mu_num, ")\n")
cat("  sd(log(y_num)) =", sd(log(y_num)), " (true sigma_num =", sigma_num, ")\n")

cat("\nFor y_denom:\n")
cat("  mean(log(y_denom)) =", mean(log(y_denom)), " (true mu_denom =", mu_denom, ")\n")
cat("  sd(log(y_denom)) =", sd(log(y_denom)), " (true sigma_denom =", sigma_denom, ")\n")

# What does R think the LL is at stuck values (with empirical sigma)?
emp_sigma_num <- sd(log(y_num))
emp_sigma_denom <- sd(log(y_denom))
ll_emp_stuck <- lognormal_ll(y_num, 1.75, emp_sigma_num) + lognormal_ll(y_denom, 4.01, emp_sigma_denom)
cat("\nLL at stuck mu values but with empirical sigmas:\n")
cat("  LL =", ll_emp_stuck, "\n")

# Is there maybe some parameter ordering issue?
# Let's check what parameters the sampler sees
cat("\n=== Parameter order check ===\n")
form <- numdenom:::ratiod_formula(y_num | y_denom ~ 1, data = dat)
model_type <- numdenom:::get_hmc_model_type(ratiod_lognormal())
hmc_data <- numdenom:::prepare_hmc_data(form, dat, ratiod_lognormal(), model_type)

cat("p_num =", hmc_data$p_num, "\n")
cat("p_denom =", hmc_data$p_denom, "\n")
cat("y_num_cont (first 5):", head(hmc_data$y_num_cont, 5), "\n")
cat("y_denom_cont (first 5):", head(hmc_data$y_denom_cont, 5), "\n")

cat("\nX_num:\n")
print(head(hmc_data$X_num))
cat("\nX_denom:\n")
print(head(hmc_data$X_denom))
