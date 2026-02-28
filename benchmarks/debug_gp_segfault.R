# Debug spatial GP segfault - check nn_neighbor_dist array dimensions
devtools::load_all()

set.seed(42)

N <- 20  # Small for debugging
coords <- cbind(runif(N, 0, 10), runif(N, 0, 10))

beta_num <- c(1.0, 0.3)
beta_denom <- c(2.0, -0.2)
phi_num <- 5
phi_denom <- 5

# Generate spatial effect using GP
dist_mat <- as.matrix(dist(coords))
gp_sigma <- 0.4
gp_length <- 3.0
Sigma_gp <- gp_sigma^2 * exp(-dist_mat / gp_length)
diag(Sigma_gp) <- diag(Sigma_gp) + 1e-6
L_gp <- chol(Sigma_gp)
gp_effect <- as.vector(t(L_gp) %*% rnorm(N))

x <- rnorm(N)
X <- cbind(1, x)

eta_num <- X %*% beta_num + gp_effect
eta_denom <- X %*% beta_denom + gp_effect

mu_num <- exp(eta_num)
mu_denom <- exp(eta_denom)

y_num <- rnbinom(N, size = phi_num, mu = mu_num)
y_denom <- rnbinom(N, size = phi_denom, mu = mu_denom)

dat <- data.frame(y_num = y_num, y_denom = y_denom, x = x,
                  lon = coords[,1], lat = coords[,2])

# Create GP object and validate
gp <- spatial_gp(~ lon + lat, cov = "exponential")
gp$nn <- 10L  # Default neighbor count

# Validate GP to get neighbor info
validated <- numdenom:::validate_gp(gp, dat)

cat("=== GP Neighbor Info ===\n")
cat("N =", N, "\n")
cat("nn =", gp$nn, "\n")
cat("nn_idx dimensions:", dim(validated$neighbor_info$nn_idx), "\n")
cat("nn_dist dimensions:", dim(validated$neighbor_info$nn_dist), "\n")
cat("nn_neighbor_dist dimensions:", dim(validated$neighbor_info$nn_neighbor_dist), "\n")
cat("nn_order length:", length(validated$neighbor_info$nn_order), "\n")
cat("nn_order_inv length:", length(validated$neighbor_info$nn_order_inv), "\n")

# Check the flattening
nn_info <- validated$neighbor_info
nn_neighbor_dist_flat <- as.vector(aperm(nn_info$nn_neighbor_dist, c(3, 2, 1)))
cat("\nnn_neighbor_dist_flat length:", length(nn_neighbor_dist_flat), "\n")
cat("Expected length (N * nn * nn):", N * gp$nn * gp$nn, "\n")

# Check C++ access pattern
# C++ does: gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2]
# With i in [0, N-1], j1 in [0, nn-1], j2 in [0, nn-1]
# Max index would be (N-1) * nn * nn + (nn-1) * nn + (nn-1) = N * nn * nn - 1
cat("\nMax C++ index:", (N-1) * gp$nn * gp$nn + (gp$nn-1) * gp$nn + (gp$nn-1), "\n")
cat("Array length - 1:", length(nn_neighbor_dist_flat) - 1, "\n")

if ((N-1) * gp$nn * gp$nn + (gp$nn-1) * gp$nn + (gp$nn-1) >= length(nn_neighbor_dist_flat)) {
  cat("\n*** BUG: C++ max index exceeds array bounds! ***\n")
}

# Test access pattern manually
cat("\n=== Testing access pattern ===\n")
# aperm(x, c(3, 2, 1)) means: new[k, j, i] = old[i, j, k]
# When flattened row-major: index = k * (J * I) + j * I + i
# But C++ expects: index = i * (nn * nn) + j1 * nn + j2

# The aperm reorders dimensions: (N, k, k) -> (k, k, N)
# So flattened: element [i, j1, j2] of original ends up at position:
#   j2 * (k * N) + j1 * N + i
# But C++ accesses: i * k * k + j1 * k + j2

cat("R array dim before aperm: (", N, ",", gp$nn, ",", gp$nn, ")\n")
cat("R array dim after aperm: (", gp$nn, ",", gp$nn, ",", N, ")\n")
cat("\nFor element [i=2, j1=1, j2=3] (0-indexed):\n")
cat("  Original position in R (1-indexed): [3, 2, 4]\n")
cat("  After aperm, row-major flat index: 3 * (", gp$nn, "*", N, ") + 1 *", N, "+ 2 =",
    3 * (gp$nn * N) + 1 * N + 2, "\n")
cat("  C++ expected index: 2 *", gp$nn * gp$nn, "+ 1 *", gp$nn, "+ 3 =",
    2 * gp$nn * gp$nn + 1 * gp$nn + 3, "\n")
cat("\n*** These don't match! The aperm permutation is wrong for C++ row-major access ***\n")
