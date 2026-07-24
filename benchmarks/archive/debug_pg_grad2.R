# Debug: PG gradient bug is data-dependent (works N=10, fails N=100)
devtools::load_all("C:/Users/Gilles Colling/Documents/dev/numdenom", quiet = TRUE)
set.seed(42)

N <- 100L
x <- rnorm(N)
y <- rpois(N, exp(0.5 + 0.3*x))
eff <- rpois(N, exp(1.0))

cat("Number of zero effort values:", sum(eff == 0), "\n")
cat("effort range:", range(eff), "\n")
cat("Zero effort indices:", which(eff == 0), "\n")

df <- data.frame(y = y, effort = eff, x = x)

# Test 1: PG base with zeros in effort
cat("\n=== PG base (with zeros) ===\n")
fit1 <- tryCatch({
  tratio(y | effort ~ x, data = df,
         family = ratiod_poisson_gamma(),
         control = list(iter = 5, warmup = 2, chains = 1, gradient_mode = "H", verbose = FALSE))
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit1)) cat(fit1, "\n") else cat("SUCCESS\n")

# Test 2: PG base with zeros replaced by 1
eff2 <- eff
eff2[eff2 == 0] <- 1
df2 <- data.frame(y = y, effort = eff2, x = x)

cat("\n=== PG base (zeros -> 1) ===\n")
fit2 <- tryCatch({
  tratio(y | effort ~ x, data = df2,
         family = ratiod_poisson_gamma(),
         control = list(iter = 5, warmup = 2, chains = 1, gradient_mode = "H", verbose = FALSE))
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit2)) cat(fit2, "\n") else cat("SUCCESS\n")

# Test 3: PG base with only positive effort (filter out zeros)
keep <- eff > 0
df3 <- data.frame(y = y[keep], effort = eff[keep], x = x[keep])

cat("\n=== PG base (filtered, N=", sum(keep), ") ===\n")
fit3 <- tryCatch({
  tratio(y | effort ~ x, data = df3,
         family = ratiod_poisson_gamma(),
         control = list(iter = 5, warmup = 2, chains = 1, gradient_mode = "H", verbose = FALSE))
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit3)) cat(fit3, "\n") else cat("SUCCESS\n")

# Test 4: N=100 but guarantee no zeros
set.seed(42)
eff4 <- rpois(N, exp(1.5))  # Higher rate = fewer zeros
eff4[eff4 == 0] <- 1
df4 <- data.frame(y = y, effort = eff4, x = x)

cat("\n=== PG base (N=100, guaranteed no zeros) ===\n")
fit4 <- tryCatch({
  tratio(y | effort ~ x, data = df4,
         family = ratiod_poisson_gamma(),
         control = list(iter = 5, warmup = 2, chains = 1, gradient_mode = "H", verbose = FALSE))
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit4)) cat(fit4, "\n") else cat("SUCCESS\n")
