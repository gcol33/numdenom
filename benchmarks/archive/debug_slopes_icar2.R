# Debug: isolate whether the bug is in slopes, ICAR, or their combination
library(numdenom)
set.seed(42)

N <- 100L; S <- 10L
site <- factor(rep(1:S, each = N/S))
x <- rnorm(N)

W <- matrix(0, S, S)
for (i in 1:(S-1)) W[i,i+1] <- W[i+1,i] <- 1

y <- rpois(N, exp(0.5 + 0.3*x))
eff <- rpois(N, exp(1.0))
df <- data.frame(y = y, effort = eff, x = x, spatial_site = site)

# Test 1: PG + ICAR only (no slopes) — should work
cat("=== Test 1: PG+ICAR (no slopes) ===\n")
fit1 <- tryCatch({
  tratio(y | effort ~ x + (1 | spatial_site), data = df,
         family = ratiod_poisson_gamma(),
         spatial = spatial_car(W, level = "group", group_var = "spatial_site"),
         control = list(iter = 10, warmup = 5, chains = 1, gradient_mode = "H", verbose = TRUE))
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit1)) cat(fit1, "\n") else cat("ICAR-only: SUCCESS\n")

# Test 2: PG + slopes only (no ICAR)
cat("\n=== Test 2: PG+slopes (no spatial) ===\n")
fit2 <- tryCatch({
  tratio(y | effort ~ x + (x | spatial_site), data = df,
         family = ratiod_poisson_gamma(),
         control = list(iter = 10, warmup = 5, chains = 1, gradient_mode = "H", verbose = TRUE))
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit2)) cat(fit2, "\n") else cat("Slopes-only: SUCCESS\n")

# Test 3: PG + slopes + ICAR (combination)
cat("\n=== Test 3: PG+slopes+ICAR ===\n")
fit3 <- tryCatch({
  tratio(y | effort ~ x + (x | spatial_site), data = df,
         family = ratiod_poisson_gamma(),
         spatial = spatial_car(W, level = "group", group_var = "spatial_site"),
         control = list(iter = 10, warmup = 5, chains = 1, gradient_mode = "H", verbose = TRUE))
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit3)) cat(fit3, "\n") else cat("Slopes+ICAR: SUCCESS\n")

# Test 4: NB + slopes + ICAR (different family)
cat("\n=== Test 4: NB+slopes+ICAR ===\n")
y_nb <- rnbinom(N, size = 5, mu = exp(0.5 + 0.3*x))
y_d <- rnbinom(N, size = 5, mu = exp(1.0))
df_nb <- data.frame(y_num = y_nb, y_denom = y_d, x = x, spatial_site = site)
fit4 <- tryCatch({
  tratio(y_num | y_denom ~ x + (x | spatial_site), data = df_nb,
         family = ratiod_negbin_negbin(),
         spatial = spatial_car(W, level = "group", group_var = "spatial_site"),
         control = list(iter = 10, warmup = 5, chains = 1, gradient_mode = "H", verbose = TRUE))
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit4)) cat(fit4, "\n") else cat("NB+Slopes+ICAR: SUCCESS\n")
