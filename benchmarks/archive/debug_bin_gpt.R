# Debug: Bin+GP_t gradient mismatch at param 8
# Reproduce exact bench_single_row.R row 74 config
library(numdenom)
set.seed(123)

N_OBS_GP <- 80L
N_SITES_GP <- 20L
N_TIMES_GP <- 10L

# Data generation (matching bench_single_row.R)
sites_gp <- rep(1:N_SITES_GP, each = N_OBS_GP / N_SITES_GP)
times_gp <- rep(1:N_TIMES_GP, length.out = N_OBS_GP)
lon_gp <- rep(runif(N_SITES_GP, 0, 10), each = N_OBS_GP / N_SITES_GP)
lat_gp <- rep(runif(N_SITES_GP, 0, 10), each = N_OBS_GP / N_SITES_GP)
x_gp <- rnorm(N_OBS_GP)

df <- data.frame(
  y = rbinom(N_OBS_GP, size = 20, prob = 0.3),
  trials = rep(20L, N_OBS_GP),
  x = x_gp,
  site = factor(sites_gp),
  time = times_gp,
  lon = lon_gp,
  lat = lat_gp
)

# Try H mode with verbose to see gradient check
cat("=== Bin+GP_t: gradient_mode=H with verbose ===\n")
fit <- tryCatch({
  ratiod(y | trials ~ x + (1 | site), data = df,
         family = ratiod_binomial(),
         temporal = temporal_gp(time_var = "time"),
         iter = 10, warmup = 5, chains = 1,
         gradient_mode = "H", verbose = TRUE)
}, error = function(e) paste0("ERROR: ", e$message))

if (is.character(fit)) {
  cat(fit, "\n")
} else {
  cat("H mode: SUCCESS\n")
}

cat("\nDone.\n")
