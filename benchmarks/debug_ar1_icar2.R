# Debug script to understand AR1+ICAR failure - Part 2
# Check if spatial effects are being applied

library(numdenom)

set.seed(42)

# Small test
N_OBS <- 100
N_SITES <- 10
N_TIMES <- 5

# Setup
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)
time_factor <- factor(rep(1:N_TIMES, length.out = N_OBS))

# Spatial grid and adjacency
n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_SITES, ]
adj_mat <- matrix(0, N_SITES, N_SITES)
for (i in 1:N_SITES) {
  for (j in 1:N_SITES) {
    if (i != j) {
      dist <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
      if (dist <= 1.5) adj_mat[i, j] <- 1
    }
  }
}
spatial_site <- factor(rep(1:N_SITES, length.out = N_OBS))

# Generate data
eta_num <- 2 + 0.3 * x
eta_denom <- 4 + 0.2 * x
df <- data.frame(
  y = rnbinom(N_OBS, mu = exp(eta_num), size = 5),
  denom = rnbinom(N_OBS, mu = exp(eta_denom), size = 10),
  x = x,
  site = site,
  spatial_site = spatial_site,
  time_factor = time_factor
)
df$denom[df$denom == 0] <- 1

cat("=== Test 1: ICAR only (no temporal) ===\n")
fit1 <- ratiod(
  y | denom ~ x + (1|site),
  data = df,
  family = ratiod_negbin_negbin(),
  spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
  iter = 500, warmup = 250, chains = 1,
  verbose = FALSE
)
draws1 <- as.matrix(fit1$draws)
cat("Params:", paste(head(colnames(draws1), 20), collapse = ", "), "\n")
spatial_params1 <- colnames(draws1)[grep("^spatial\\[", colnames(draws1))]
cat("Spatial params:", length(spatial_params1), "\n\n")

cat("=== Test 2: RW1 only (no spatial) ===\n")
fit2 <- ratiod(
  y | denom ~ x + (1|site),
  data = df,
  family = ratiod_negbin_negbin(),
  temporal = temporal_rw1("time_factor"),
  iter = 500, warmup = 250, chains = 1,
  verbose = FALSE
)
draws2 <- as.matrix(fit2$draws)
cat("Params:", paste(head(colnames(draws2), 20), collapse = ", "), "\n")
temporal_params2 <- colnames(draws2)[grep("^temporal\\[", colnames(draws2))]
cat("Temporal params:", length(temporal_params2), "\n\n")

cat("=== Test 3: AR1 only (no spatial) ===\n")
fit3 <- ratiod(
  y | denom ~ x + (1|site),
  data = df,
  family = ratiod_negbin_negbin(),
  temporal = temporal_ar1("time_factor"),
  iter = 500, warmup = 250, chains = 1,
  verbose = FALSE
)
draws3 <- as.matrix(fit3$draws)
cat("Params:", paste(head(colnames(draws3), 20), collapse = ", "), "\n")
temporal_params3 <- colnames(draws3)[grep("^temporal\\[", colnames(draws3))]
cat("Temporal params:", length(temporal_params3), "\n")
ar1_params <- colnames(draws3)[grep("rho", colnames(draws3))]
cat("AR1 rho params:", paste(ar1_params, collapse = ", "), "\n\n")

cat("=== Test 4: ICAR + RW1 ===\n")
fit4 <- ratiod(
  y | denom ~ x + (1|site),
  data = df,
  family = ratiod_negbin_negbin(),
  spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
  temporal = temporal_rw1("time_factor"),
  iter = 500, warmup = 250, chains = 1,
  verbose = FALSE
)
draws4 <- as.matrix(fit4$draws)
cat("Params:", paste(head(colnames(draws4), 30), collapse = ", "), "\n")
spatial_params4 <- colnames(draws4)[grep("^spatial\\[", colnames(draws4))]
temporal_params4 <- colnames(draws4)[grep("^temporal\\[", colnames(draws4))]
cat("Spatial params:", length(spatial_params4), "\n")
cat("Temporal params:", length(temporal_params4), "\n\n")

cat("=== Test 5: ICAR + AR1 ===\n")
fit5 <- ratiod(
  y | denom ~ x + (1|site),
  data = df,
  family = ratiod_negbin_negbin(),
  spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
  temporal = temporal_ar1("time_factor"),
  iter = 500, warmup = 250, chains = 1,
  verbose = FALSE
)
draws5 <- as.matrix(fit5$draws)
cat("Params:", paste(head(colnames(draws5), 30), collapse = ", "), "\n")
spatial_params5 <- colnames(draws5)[grep("^spatial\\[", colnames(draws5))]
temporal_params5 <- colnames(draws5)[grep("^temporal\\[", colnames(draws5))]
cat("Spatial params:", length(spatial_params5), "\n")
cat("Temporal params:", length(temporal_params5), "\n")

cat("\n=== Summary ===\n")
cat("Test 1 (ICAR only):     spatial =", length(spatial_params1), "\n")
cat("Test 2 (RW1 only):      temporal =", length(temporal_params2), "\n")
cat("Test 3 (AR1 only):      temporal =", length(temporal_params3), "\n")
cat("Test 4 (ICAR + RW1):    spatial =", length(spatial_params4), ", temporal =", length(temporal_params4), "\n")
cat("Test 5 (ICAR + AR1):    spatial =", length(spatial_params5), ", temporal =", length(temporal_params5), "\n")
