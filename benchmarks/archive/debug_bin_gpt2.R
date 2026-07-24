# Debug: Reproduce exact Bin+GP_t (row 74) from bench_single_row.R
library(numdenom)
set.seed(123)

N <- 500L; S <- 50L; T_len <- 20L
site <- factor(rep(1:S, length.out = N))
time <- rep(1:T_len, length.out = N)
x <- rnorm(N)

n_side <- ceiling(sqrt(S))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:S, ]
site_int <- as.integer(site)
lon <- grid$lon[site_int]
lat <- grid$lat[site_int]

y <- rbinom(N, size = 20, prob = plogis(0.5 + 0.3 * x))
trials <- rep(20L, N)

df <- data.frame(y = y, trials = trials, x = x, site = site,
                 time_num = time, lon = lon, lat = lat)

cat("=== Bin+GP_t: N=500, S=50, T=20 ===\n")
cat("Testing H mode...\n")
fit <- tryCatch({
  tratio(y | trials ~ x + (1 | site), data = df,
         family = ratiod_binomial(),
         temporal = temporal_gp(time_var = "time_num"),
         control = list(iter = 10, warmup = 5, chains = 1, gradient_mode = "H", verbose = TRUE))
}, error = function(e) paste0("ERROR: ", e$message))

if (is.character(fit)) cat(fit, "\n") else cat("H mode: SUCCESS\n")
