# Test PG and NB slopes+ICAR with installed package
library(numdenom)
set.seed(42)
N <- 100L
x <- rnorm(N)

# PG data
y <- rpois(N, exp(0.5 + 0.3*x))
eff <- rpois(N, exp(1.0))
cat("Zero effort count:", sum(eff == 0), "\n")

# Spatial structure
n_sites <- 10L
site <- rep(1:n_sites, length.out = N)
W <- matrix(0, n_sites, n_sites)
for (i in 1:(n_sites-1)) { W[i, i+1] <- 1; W[i+1, i] <- 1 }

df <- data.frame(y = y, effort = eff, x = x, site = factor(site))

cat("\n=== PG+ICAR ===\n")
fit1 <- tryCatch({
  tratio(y | effort ~ x, data = df, spatial = spatial_car(W, group_var = "site"),
         family = ratiod_poisson_gamma(),
         control = list(iter = 5, warmup = 2, chains = 1, gradient_mode = "H", verbose = TRUE))
  "SUCCESS"
}, error = function(e) paste0("ERROR: ", e$message))
cat(fit1, "\n")

cat("\n=== PG+slopes ===\n")
fit2 <- tryCatch({
  tratio(y | effort ~ (x | site), data = df,
         family = ratiod_poisson_gamma(),
         control = list(iter = 5, warmup = 2, chains = 1, gradient_mode = "H", verbose = TRUE))
  "SUCCESS"
}, error = function(e) paste0("ERROR: ", e$message))
cat(fit2, "\n")

cat("\n=== PG+slopes+ICAR ===\n")
fit3 <- tryCatch({
  tratio(y | effort ~ (x | site), data = df, spatial = spatial_car(W, group_var = "site"),
         family = ratiod_poisson_gamma(),
         control = list(iter = 5, warmup = 2, chains = 1, gradient_mode = "H", verbose = TRUE))
  "SUCCESS"
}, error = function(e) paste0("ERROR: ", e$message))
cat(fit3, "\n")

# NB data
y_num <- rpois(N, exp(0.5 + 0.3*x))
y_den <- rpois(N, exp(0.5))
y_den[y_den == 0] <- 1
df2 <- data.frame(y = y_num, denom = y_den, x = x, site = factor(site))

cat("\n=== NB+slopes+ICAR ===\n")
fit4 <- tryCatch({
  tratio(y | denom ~ (x | site), data = df2, spatial = spatial_car(W, group_var = "site"),
         family = ratiod_negbin_negbin(),
         control = list(iter = 5, warmup = 2, chains = 1, gradient_mode = "H", verbose = TRUE))
  "SUCCESS"
}, error = function(e) paste0("ERROR: ", e$message))
cat(fit4, "\n")
