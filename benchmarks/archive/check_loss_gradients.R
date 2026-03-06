# Check gradient correctness for 3 remaining LOSS models
# If H-mode gradient is wrong, verify_gradient_runtime() will print
# "Falling back to numerical gradients" warning to stderr

library(numdenom)

N_OBS <- 500L
N_SITES <- 50L
N_TIMES <- 20L
SEED <- 123L
set.seed(SEED)

# Generate data (same as bench_single_row.R)
site <- factor(rep(1:N_SITES, length.out = N_OBS))
time <- rep(1:N_TIMES, length.out = N_OBS)
x <- rnorm(N_OBS)

n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_SITES, ]
site_int <- as.integer(site)
lon_site <- grid$lon[site_int]
lat_site <- grid$lat[site_int]

adj_mat <- matrix(0L, N_SITES, N_SITES)
for (i in 1:N_SITES) {
  for (j in 1:N_SITES) {
    if (i != j) {
      d <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
      if (d <= 1.5) adj_mat[i, j] <- 1L
    }
  }
}

y_pg_num   <- rpois(N_OBS, exp(2 + 0.5 * x))
y_pg_denom <- rgamma(N_OBS, 10, 1)
y_bin_num  <- rbinom(N_OBS, 20, 0.3)
y_bin_denom <- rep(20L, N_OBS)

df_pg <- data.frame(y_num = y_pg_num, y_denom = y_pg_denom, x = x,
                    site = site, time = time, lon = lon_site, lat = lat_site)
df_bin <- data.frame(y = y_bin_num, n_trials = y_bin_denom, x = x,
                     site = site, time = time, lon = lon_site, lat = lat_site)

check_model <- function(label, ...) {
  cat(sprintf("\n=== %s ===\n", label))
  for (mode in c("H", "A")) {
    cat(sprintf("  %s mode: ", mode))
    w <- tryCatch({
      withCallingHandlers(
        {
          fit <- ratiod(..., mode = "hmc", iter = 20, warmup = 10,
                        chains = 1, gradient_mode = mode, verbose = FALSE)
          "OK"
        },
        warning = function(w) {
          if (grepl("gradient mismatch|Falling back", conditionMessage(w))) {
            invokeRestart("muffleWarning")
            return("FALLBACK (gradient mismatch!)")
          }
          invokeRestart("muffleWarning")
        }
      )
    }, error = function(e) {
      sprintf("ERROR: %s", conditionMessage(e))
    })
    cat(w, "\n")
  }
}

cat("Checking gradient correctness for 3 LOSS models\n")
cat("================================================\n")
cat("If 'FALLBACK' appears, the gradient has a BUG.\n")
cat("verify_gradient_runtime() runs at iteration 1.\n\n")

# Row 14: PG+GP_t
check_model("Row 14: PG+GP_t",
  y_num | y_denom ~ x + (1|site), data = df_pg,
  family = ratiod_poisson_gamma(),
  temporal = temporal_gp(time_var = "time", group_var = "site"))

# Row 74: Bin+GP_t
check_model("Row 74: Bin+GP_t",
  y | n_trials ~ x + (1|site), data = df_bin,
  family = ratiod_binomial(),
  temporal = temporal_gp(time_var = "time", group_var = "site"))

# Row 91: Bin+ST_IV
check_model("Row 91: Bin+ST_IV",
  y | n_trials ~ x + (1|site), data = df_bin,
  family = ratiod_binomial(),
  spatial = spatial_car(adj_mat, group_var = "site"),
  temporal = temporal_rw1(time_var = "time", group_var = "site"),
  spatiotemporal = spatiotemporal(type = "IV"))

cat("\n================================================\n")
cat("Done. Any 'FALLBACK' above indicates a gradient bug.\n")
