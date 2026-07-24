# Debug: PG+slopes+ICAR segfault
library(numdenom)
set.seed(42)

N <- 500L; S <- 50L
site <- factor(rep(1:S, length.out = N))
x <- rnorm(N)

# Build adjacency matrix (chain structure)
W <- matrix(0, S, S)
for (i in 1:(S-1)) W[i,i+1] <- W[i+1,i] <- 1

y <- rpois(N, exp(0.5 + 0.3*x))
eff <- rpois(N, exp(1.0))
df <- data.frame(y = y, effort = eff, x = x, spatial_site = site)

cat("=== PG+slopes+ICAR: N=500, S=50 ===\n")
cat("Testing with N mode first...\n")
fit_n <- tryCatch({
  tratio(y | effort ~ x + (x | spatial_site), data = df,
         family = ratiod_poisson_gamma(),
         spatial = spatial_car(W, level = "group", group_var = "spatial_site"),
         control = list(iter = 10, warmup = 5, chains = 1, gradient_mode = "N", verbose = TRUE))
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit_n)) cat(fit_n, "\n") else cat("N mode: SUCCESS\n")

cat("\nTesting with H mode...\n")
fit_h <- tryCatch({
  tratio(y | effort ~ x + (x | spatial_site), data = df,
         family = ratiod_poisson_gamma(),
         spatial = spatial_car(W, level = "group", group_var = "spatial_site"),
         control = list(iter = 10, warmup = 5, chains = 1, gradient_mode = "H", verbose = TRUE))
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit_h)) cat(fit_h, "\n") else cat("H mode: SUCCESS\n")
