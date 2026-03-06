# Minimal segfault reproduction
library(numdenom)
cat("numdenom loaded\n")
cat("threads:", numdenom:::cpp_get_max_threads(), "\n")

set.seed(42)
N <- 20L
x <- rnorm(N)
y <- rpois(N, exp(0.5 + 0.3*x))
eff <- rpois(N, exp(1.0)) + 1  # ensure no zeros
df <- data.frame(y = y, effort = eff, x = x)

cat("Data created, N =", nrow(df), "\n")
cat("effort range:", range(eff), "\n")

# Try with N mode first (simplest gradient path)
cat("\n=== Testing N mode ===\n")
fit_n <- tryCatch({
  ratiod(y | effort ~ x, data = df,
         family = ratiod_poisson_gamma(),
         iter = 3, warmup = 1, chains = 1,
         gradient_mode = "N", verbose = TRUE)
  "SUCCESS"
}, error = function(e) paste0("ERROR: ", e$message))
cat("N mode:", fit_n, "\n")

# Now try H mode
cat("\n=== Testing H mode ===\n")
fit_h <- tryCatch({
  ratiod(y | effort ~ x, data = df,
         family = ratiod_poisson_gamma(),
         iter = 3, warmup = 1, chains = 1,
         gradient_mode = "H", verbose = TRUE)
  "SUCCESS"
}, error = function(e) paste0("ERROR: ", e$message))
cat("H mode:", fit_h, "\n")
