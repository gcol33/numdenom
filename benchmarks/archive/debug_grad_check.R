# Debug gradient check for PG+ICAR using devtools::load_all()
# This ensures we test the CURRENT source, not installed package
devtools::load_all("C:/Users/Gilles Colling/Documents/dev/numdenom", quiet = TRUE)
set.seed(42)

N <- 100L; S <- 10L
site <- factor(rep(1:S, each = N/S))
x <- rnorm(N)

W <- matrix(0, S, S)
for (i in 1:(S-1)) W[i,i+1] <- W[i+1,i] <- 1

y <- rpois(N, exp(0.5 + 0.3*x))
eff <- rpois(N, exp(1.0))
df <- data.frame(y = y, effort = eff, x = x, spatial_site = site)

cat("=== PG+ICAR with devtools::load_all() ===\n")
fit <- tryCatch({
  tratio(y | effort ~ x + (1 | spatial_site), data = df,
         family = ratiod_poisson_gamma(),
         spatial = spatial_car(W, level = "group", group_var = "spatial_site"),
         control = list(iter = 10, warmup = 5, chains = 1, gradient_mode = "H", verbose = TRUE))
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit)) cat(fit, "\n") else cat("H mode: SUCCESS\n")

cat("\n=== PG+slopes with devtools::load_all() ===\n")
fit2 <- tryCatch({
  tratio(y | effort ~ x + (x | spatial_site), data = df,
         family = ratiod_poisson_gamma(),
         control = list(iter = 10, warmup = 5, chains = 1, gradient_mode = "H", verbose = TRUE))
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit2)) cat(fit2, "\n") else cat("Slopes H mode: SUCCESS\n")
