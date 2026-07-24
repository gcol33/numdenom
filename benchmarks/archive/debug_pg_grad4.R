# Debug: what does N-mode think about zero effort?
# Compare H-mode vs N-mode when data has zeros
devtools::load_all("C:/Users/Gilles Colling/Documents/dev/numdenom", quiet = TRUE)
set.seed(42)

N <- 100L
x <- rnorm(N)
y <- rpois(N, exp(0.5 + 0.3*x))
eff <- rpois(N, exp(1.0))

cat("Zero effort indices:", which(eff == 0), "\n")
cat("Number of zeros:", sum(eff == 0), "\n\n")

# The question: why does N-mode give gradient 3.43 at beta_denom[0]
# when H mode gives -0.01?
#
# N mode: finite diff of compute_log_post
# H mode: analytical gradient from fused path
#
# For y_d = 0: log_lik_gamma(0, ...) = -1e10 (constant)
# So N-mode should see 0 gradient contribution from zero-effort obs
#
# For y_d > 0: d/d(eta_d) = phi*(y/mu - 1)
# At init, beta_denom[0] = log(mean(eff[eff>0])), beta_denom[1] = 0
# So mu_d = mean(eff[eff>0]) for all obs
# sum(phi*(y/mu - 1)) for y > 0 should be near 0

# Let's verify what the initial values are
df <- data.frame(y = y, effort = eff, x = x)
cat("mean(eff[eff>0]):", mean(eff[eff > 0]), "\n")
cat("log(mean(eff[eff>0])):", log(mean(eff[eff > 0])), "\n")

# Now let's test with A_r mode (arena autodiff) to get a third opinion
cat("\n=== PG base with zeros, A_r mode ===\n")
fit_ar <- tryCatch({
  tratio(y | effort ~ x, data = df,
         family = ratiod_poisson_gamma(),
         control = list(iter = 5, warmup = 2, chains = 1, gradient_mode = "A", verbose = TRUE))
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit_ar)) cat(fit_ar, "\n") else cat("A mode: SUCCESS\n")

cat("\n=== PG base with zeros, N mode ===\n")
fit_n <- tryCatch({
  tratio(y | effort ~ x, data = df,
         family = ratiod_poisson_gamma(),
         control = list(iter = 5, warmup = 2, chains = 1, gradient_mode = "N", verbose = TRUE))
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit_n)) cat(fit_n, "\n") else cat("N mode: SUCCESS\n")
