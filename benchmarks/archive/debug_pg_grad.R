# Debug: isolate PG gradient bug
# Compare H vs N gradient at a specific parameter point
devtools::load_all("C:/Users/Gilles Colling/Documents/dev/numdenom", quiet = TRUE)
set.seed(42)

N <- 10L
x <- rnorm(N)
y <- rpois(N, exp(0.5 + 0.3*x))
eff <- rpois(N, exp(1.0))
# Ensure no zeros in effort (Gamma needs positive values)
eff[eff == 0] <- 1
df <- data.frame(y = y, effort = eff, x = x)

cat("Data:\n")
cat("y:", y, "\n")
cat("effort:", eff, "\n")
cat("x:", round(x, 3), "\n")

# Fit with N mode (numerical) to see if it works
cat("\n=== PG base, N mode ===\n")
fit_n <- tryCatch({
  ratiod(y | effort ~ x, data = df,
         family = ratiod_poisson_gamma(),
         iter = 5, warmup = 2, chains = 1,
         gradient_mode = "N", verbose = TRUE)
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit_n)) cat(fit_n, "\n") else cat("N mode: SUCCESS\n")

# Fit with H mode to see the mismatch
cat("\n=== PG base, H mode ===\n")
fit_h <- tryCatch({
  ratiod(y | effort ~ x, data = df,
         family = ratiod_poisson_gamma(),
         iter = 5, warmup = 2, chains = 1,
         gradient_mode = "H", verbose = TRUE)
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit_h)) cat(fit_h, "\n") else cat("H mode: SUCCESS\n")

# Also test NB base and Bin base for comparison
cat("\n=== NB base, H mode ===\n")
y_nb <- rnbinom(N, size = 5, mu = exp(0.5 + 0.3*x))
y_d <- rnbinom(N, size = 5, mu = exp(1.0))
y_d[y_d == 0] <- 1
df_nb <- data.frame(y_num = y_nb, y_denom = y_d, x = x)
fit_nb <- tryCatch({
  ratiod(y_num | y_denom ~ x, data = df_nb,
         family = ratiod_negbin_negbin(),
         iter = 5, warmup = 2, chains = 1,
         gradient_mode = "H", verbose = TRUE)
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit_nb)) cat(fit_nb, "\n") else cat("NB H mode: SUCCESS\n")
