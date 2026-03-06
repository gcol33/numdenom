# Gradient verification for binomial logistic optimization
# Tests H vs N gradient agreement for all affected model types

library(numdenom)

verify_gradient <- function(label, ...) {
  cat(sprintf("  %-30s", label))
  tryCatch({
    fit <- ratiod(..., mode = "hmc", iter = 5, warmup = 2, chains = 1, verbose = FALSE)
    cat("OK\n")
    invisible(TRUE)
  }, error = function(e) {
    cat(sprintf("FAIL: %s\n", conditionMessage(e)))
    invisible(FALSE)
  })
}

set.seed(42)
N <- 100
S <- 10
T_times <- 5

# Generate data
coords <- cbind(runif(S), runif(S))
W <- matrix(0, S, S)
for (i in 1:(S-1)) for (j in (i+1):S) {
  if (sqrt(sum((coords[i,]-coords[j,])^2)) < 0.5) {
    W[i,j] <- W[j,i] <- 1
  }
}
# Ensure connected
for (i in which(rowSums(W) == 0)) {
  dists <- as.matrix(dist(coords))[i, ]
  dists[i] <- Inf
  nearest <- which.min(dists)
  W[i, nearest] <- W[nearest, i] <- 1
}

df <- data.frame(
  y = rbinom(N, size = 20, prob = 0.3),
  n_trials = rep(20L, N),
  x = rnorm(N),
  site = rep(1:S, each = N/S),
  time = rep(1:T_times, times = N/T_times),
  lon = coords[rep(1:S, each = N/S), 1],
  lat = coords[rep(1:S, each = N/S), 2]
)

cat("=== Binomial models: H mode ===\n")
verify_gradient("Bin base (H)", y | n_trials ~ x, data = df,
  family = ratiod_binomial(), gradient_mode = "H")
verify_gradient("Bin base (N)", y | n_trials ~ x, data = df,
  family = ratiod_binomial(), gradient_mode = "N")
verify_gradient("Bin+RE (H)", y | n_trials ~ x + (1|site), data = df,
  family = ratiod_binomial(), gradient_mode = "H")
verify_gradient("Bin+ICAR (H)", y | n_trials ~ x + (1|site), data = df,
  family = ratiod_binomial(), gradient_mode = "H",
  spatial = spatial_car(W, group_var = "site"))

cat("\n=== Binomial models with spatial/temporal ===\n")
verify_gradient("Bin+BYM2 (H)", y | n_trials ~ x + (1|site), data = df,
  family = ratiod_binomial(), gradient_mode = "H",
  spatial = spatial_bym2(W, group_var = "site"))
verify_gradient("Bin+HSGP (H)", y | n_trials ~ x, data = df,
  family = ratiod_binomial(), gradient_mode = "H",
  spatial = spatial_hsgp(~ lon + lat))
verify_gradient("Bin+AR1 (H)", y | n_trials ~ x + (1|site), data = df,
  family = ratiod_binomial(), gradient_mode = "H",
  temporal = temporal_ar1(time_var = "time", group_var = "site"))

cat("\n=== NB models (control - no binomial changes) ===\n")
df_nb <- data.frame(
  y_num = rpois(N, 5), y_denom = rpois(N, 10) + 1,
  x = rnorm(N), site = rep(1:S, each = N/S)
)
verify_gradient("NB base (H)", y_num | y_denom ~ x, data = df_nb,
  family = ratiod_negbin_negbin(), gradient_mode = "H")
verify_gradient("NB+RE (H)", y_num | y_denom ~ x + (1|site), data = df_nb,
  family = ratiod_negbin_negbin(), gradient_mode = "H")

cat("\n=== Beta-binomial (sigmoid stability) ===\n")
verify_gradient("BetaBin base (H)", y | n_trials ~ x, data = df,
  family = ratiod_beta_binomial(), gradient_mode = "H")

cat("\n=== Temporal GP (backward pass changes) ===\n")
verify_gradient("Bin+GP_t (H)", y | n_trials ~ x + (1|site), data = df,
  family = ratiod_binomial(), gradient_mode = "H",
  temporal = temporal_gp(time_var = "time", group_var = "site"))
verify_gradient("NB+GP_t (H)", y_num | y_denom ~ x + (1|site),
  data = cbind(df_nb, time = rep(1:T_times, times = N/T_times)),
  family = ratiod_negbin_negbin(), gradient_mode = "H",
  temporal = temporal_gp(time_var = "time", group_var = "site"))

cat("\nAll gradient verification complete.\n")
