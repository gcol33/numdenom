# Debug: compare H and N gradient values directly
devtools::load_all("C:/Users/Gilles Colling/Documents/dev/numdenom", quiet = TRUE)
set.seed(42)

N <- 100L
x <- rnorm(N)
y <- rpois(N, exp(0.5 + 0.3*x))
eff <- rpois(N, exp(1.0))
# Replace zeros so we isolate the remaining bug from y<=0 issue
eff[eff == 0] <- 1

df <- data.frame(y = y, effort = eff, x = x)

cat("effort range:", range(eff), "\n")
cat("Number of zero effort:", sum(eff == 0), "\n")

# This should work now (no zeros)
cat("\n=== PG base (no zeros), H mode ===\n")
fit <- tryCatch({
  tratio(y | effort ~ x, data = df,
         family = ratiod_poisson_gamma(),
         control = list(iter = 5, warmup = 2, chains = 1, gradient_mode = "H", verbose = TRUE))
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit)) cat(fit, "\n") else cat("SUCCESS\n")
