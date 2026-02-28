# Debug GP crash - check data structures before calling C++
devtools::load_all()

set.seed(42)

N <- 20  # Very small for debugging
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

# Create the spatial_gp object
gp <- spatial_gp(~ lon + lat, cov = 'exponential')
gp$nn <- 10L

# Manually run validation
validated <- numdenom:::validate_gp(gp, dat)
nn_info <- validated$neighbor_info

cat("=== GP Data Structures Debug ===\n")
cat("N =", N, "\n")
cat("nn =", gp$nn, "\n")
cat("\n")

cat("nn_idx (R matrix): dim =", dim(nn_info$nn_idx), "\n")
cat("  Expected: (N, nn) = (", N, ",", gp$nn, ")\n")
cat("  Values range: [", min(nn_info$nn_idx), ",", max(nn_info$nn_idx), "]\n")
cat("  First row:", nn_info$nn_idx[1,], "\n")
cat("\n")

cat("nn_dist (R matrix): dim =", dim(nn_info$nn_dist), "\n")
cat("  Expected: (N, nn) = (", N, ",", gp$nn, ")\n")
cat("\n")

cat("nn_order: length =", length(nn_info$nn_order), "\n")
cat("  Expected: N =", N, "\n")
cat("  Values range: [", min(nn_info$nn_order), ",", max(nn_info$nn_order), "]\n")
cat("\n")

cat("nn_order_inv: length =", length(nn_info$nn_order_inv), "\n")
cat("  Expected: N =", N, "\n")
cat("\n")

cat("nn_neighbor_dist: dim =", dim(nn_info$nn_neighbor_dist), "\n")
cat("  Expected: (N, nn, nn) = (", N, ",", gp$nn, ",", gp$nn, ")\n")
cat("\n")

# Check flattening
nn_idx_flat <- as.integer(as.vector(t(nn_info$nn_idx)))
nn_dist_flat <- as.vector(t(nn_info$nn_dist))
nn_neighbor_dist_flat <- as.vector(aperm(nn_info$nn_neighbor_dist, c(3, 2, 1)))

cat("=== Flattened vectors for C++ ===\n")
cat("nn_idx_flat: length =", length(nn_idx_flat), " (expected:", N * gp$nn, ")\n")
cat("nn_dist_flat: length =", length(nn_dist_flat), " (expected:", N * gp$nn, ")\n")
cat("nn_neighbor_dist_flat: length =", length(nn_neighbor_dist_flat), " (expected:", N * gp$nn * gp$nn, ")\n")
cat("\n")

# C++ access pattern check
cat("=== C++ access pattern check ===\n")
# For observation i=2 (0-indexed), neighbor j=3:
# C++ accesses nn_idx with: i * nn + j = 2 * 10 + 3 = 23
# R transposed matrix: t(nn_info$nn_idx)[j+1, i+1] = column (j+1), row (i+1)
# After as.vector(t()): index = (i) * nn + j = same

i_test <- 2  # 0-indexed
j_test <- 3
cpp_idx <- i_test * gp$nn + j_test
cat("Test access i=", i_test, ", j=", j_test, ":\n")
cat("  C++ index:", cpp_idx, "\n")
cat("  Value in R (transposed):", t(nn_info$nn_idx)[j_test + 1, i_test + 1], "\n")
cat("  Value in flattened vector:", nn_idx_flat[cpp_idx + 1], "\n")  # +1 for R indexing
cat("\n")

# Check coords flattening
coords_flat <- as.vector(t(coords))
cat("coords_flat: length =", length(coords_flat), " (expected:", 2 * N, ")\n")
cat("  First point: (", coords_flat[1], ",", coords_flat[2], ")\n")
cat("  From R matrix: (", coords[1, 1], ",", coords[1, 2], ")\n")
cat("\n")

# Now try to fit the model
cat("=== Attempting to fit model... ===\n")
tryCatch({
  fit_nd <- ratiod(
    y_num | y_denom ~ x,
    data = dat,
    family = ratiod_negbin_negbin(),
    spatial = spatial_gp(~ lon + lat, cov = 'exponential'),
    iter = 10,  # Just a few iterations
    warmup = 5,
    chains = 1,
    seed = 123
  )
  cat("SUCCESS!\n")
}, error = function(e) {
  cat("ERROR:", conditionMessage(e), "\n")
})
