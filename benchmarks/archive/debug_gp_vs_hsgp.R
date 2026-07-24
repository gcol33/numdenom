# Debug GP vs HSGP parameter recovery
# GP shows slopes stuck near zero while HSGP works
library(numdenom)
library(posterior)

set.seed(42)

# Generate simple test data
N <- 100
coords <- data.frame(
  x = runif(N, 0, 1),
  y = runif(N, 0, 1)
)

# True parameters
true_intercept <- 1.0
true_slope <- 0.3

# Generate spatial effect (smooth field)
dist_mat <- as.matrix(dist(coords))
Sigma <- exp(-dist_mat / 0.2)  # Exponential covariance with range 0.2
L <- chol(Sigma + diag(1e-6, N))
gp_effect <- as.vector(t(L) %*% rnorm(N)) * 0.5  # spatial SD = 0.5

# Generate data
x <- rnorm(N)
eta <- true_intercept + true_slope * x + gp_effect
trials <- sample(20:50, N, replace = TRUE)
successes <- rbinom(N, trials, plogis(eta))

df <- data.frame(
  successes = successes,
  trials = trials,
  x = x,
  coord_x = coords$x,
  coord_y = coords$y
)

cat("True parameters: intercept =", true_intercept, ", slope =", true_slope, "\n\n")

# Test HSGP
cat("===== HSGP =====\n")
fit_hsgp <- tratio(successes | trials ~ x, data = df, family = ratiod_binomial(),
                    spatial = spatial_hsgp(coords = c("coord_x", "coord_y"), m = 5),
                    control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))

draws_hsgp <- as.matrix(fit_hsgp$draws)
cat("Column names:", head(colnames(draws_hsgp), 10), "\n")
cat("beta_num[1] (intercept):", sprintf("%.3f (SD %.3f)", mean(draws_hsgp[,1]), sd(draws_hsgp[,1])), "\n")
cat("beta_num[2] (slope):", sprintf("%.3f (SD %.3f)", mean(draws_hsgp[,2]), sd(draws_hsgp[,2])), "\n")

# Test GP (NNGP)
cat("\n===== GP (NNGP) =====\n")
fit_gp <- tratio(successes | trials ~ x, data = df, family = ratiod_binomial(),
                  spatial = spatial_gp(coords = c("coord_x", "coord_y"), nn = 10),
                  control = list(iter = 500, warmup = 250, chains = 1, verbose = FALSE))

draws_gp <- as.matrix(fit_gp$draws)
cat("Column names:", head(colnames(draws_gp), 10), "\n")
cat("beta_num[1] (intercept):", sprintf("%.3f (SD %.3f)", mean(draws_gp[,1]), sd(draws_gp[,1])), "\n")
cat("beta_num[2] (slope):", sprintf("%.3f (SD %.3f)", mean(draws_gp[,2]), sd(draws_gp[,2])), "\n")

# Check raw column values
cat("\n===== Raw sample inspection =====\n")
cat("HSGP - first 5 rows of columns 1-5:\n")
print(head(draws_hsgp[, 1:min(5, ncol(draws_hsgp))], 5))

cat("\nGP - first 5 rows of columns 1-5:\n")
print(head(draws_gp[, 1:min(5, ncol(draws_gp))], 5))

# Check if GP slopes are truly stuck
cat("\n===== Slope movement analysis =====\n")
cat("HSGP slope range:", range(draws_hsgp[,2]), "\n")
cat("GP slope range:", range(draws_gp[,2]), "\n")

# Check GP spatial effects
if ("gp_w[1]" %in% colnames(draws_gp)) {
  gp_w_cols <- grep("^gp_w\\[", colnames(draws_gp))
  cat("\nGP spatial effects (gp_w) first 3:", head(colnames(draws_gp)[gp_w_cols], 3), "\n")
  cat("gp_w[1] range:", range(draws_gp[, gp_w_cols[1]]), "\n")
}
