# Benchmark ST Type IV models only
library(numdenom)
set.seed(42)

N <- 500L
n_sites <- 50L
n_times <- 20L
x <- rnorm(N)
site <- rep(1:n_sites, length.out = N)
time <- rep(1:n_times, length.out = N)

# Adjacency matrix (ring lattice)
W <- matrix(0, n_sites, n_sites)
for (i in 1:(n_sites - 1)) { W[i, i+1] <- 1; W[i+1, i] <- 1 }
W[1, n_sites] <- 1; W[n_sites, 1] <- 1

# PG data
y_pg <- rpois(N, exp(0.5 + 0.3 * x))
eff_pg <- rpois(N, exp(1.0)) + 1
df_pg <- data.frame(y = y_pg, effort = eff_pg, x = x, site = factor(site), time = time)

# NB data
y_nb_num <- rpois(N, exp(0.5 + 0.3 * x))
y_nb_den <- rpois(N, exp(0.5)) + 1
df_nb <- data.frame(y = y_nb_num, denom = y_nb_den, x = x, site = factor(site), time = time)

# Bin data
y_bin <- rbinom(N, size = 10, prob = 0.3)
n_bin <- rep(10L, N)
df_bin <- data.frame(y = y_bin, n = n_bin, x = x, site = factor(site), time = time)

sp <- spatial_car(W, group_var = "site")
tp <- temporal_rw1("time")

cat("=== PG+ST_IV ===\n")
t1 <- system.time({
  fit1 <- tryCatch({
    ratiod(y | effort ~ x, data = df_pg,
           spatial = sp,
           spatiotemporal = spatiotemporal(spatial = sp, temporal = tp, type = "IV"),
           family = ratiod_poisson_gamma(),
           iter = 500, warmup = 250, chains = 1, verbose = TRUE)
    "OK"
  }, error = function(e) paste0("FAIL: ", e$message))
})["elapsed"]
cat(fit1, sprintf("(%.1fs)\n", t1))

cat("\n=== NB+ST_IV ===\n")
t2 <- system.time({
  fit2 <- tryCatch({
    ratiod(y | denom ~ x, data = df_nb,
           spatial = sp,
           spatiotemporal = spatiotemporal(spatial = sp, temporal = tp, type = "IV"),
           family = ratiod_negbin_negbin(),
           iter = 500, warmup = 250, chains = 1, verbose = TRUE)
    "OK"
  }, error = function(e) paste0("FAIL: ", e$message))
})["elapsed"]
cat(fit2, sprintf("(%.1fs)\n", t2))

cat("\n=== Bin+ST_IV ===\n")
t3 <- system.time({
  fit3 <- tryCatch({
    ratiod(y | n ~ x, data = df_bin,
           spatial = sp,
           spatiotemporal = spatiotemporal(spatial = sp, temporal = tp, type = "IV"),
           family = ratiod_binomial(),
           iter = 500, warmup = 250, chains = 1, verbose = TRUE)
    "OK"
  }, error = function(e) paste0("FAIL: ", e$message))
})["elapsed"]
cat(fit3, sprintf("(%.1fs)\n", t3))

cat("\n\n=== SUMMARY ===\n")
cat(sprintf("%-16s %8s %8s %8s %8s\n", "Model", "Before", "After", "Stan", "vs Stan"))
cat(paste(rep("-", 56), collapse=""), "\n")
cat(sprintf("%-16s %7.1fs %7.1fs %7.1fs %7.1fx\n", "PG+ST_IV", 190.8, t1, 82.5, t1/82.5))
cat(sprintf("%-16s %7.1fs %7.1fs %7.1fs %7.1fx\n", "NB+ST_IV", 287.9, t2, 133.5, t2/133.5))
cat(sprintf("%-16s %7.1fs %7.1fs %7.1fs %7.1fx\n", "Bin+ST_IV", 80.7, t3, 56.0, t3/56.0))
