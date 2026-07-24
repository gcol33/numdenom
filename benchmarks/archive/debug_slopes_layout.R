# Debug: check parameter layout for slopes model
library(numdenom)
set.seed(42)
N <- 100L
x <- rnorm(N)
y <- rpois(N, exp(0.5 + 0.3*x))
eff <- rpois(N, exp(1.0)) + 1  # no zeros
n_sites <- 10L
site <- rep(1:n_sites, length.out = N)
df <- data.frame(y = y, effort = eff, x = x, site = factor(site))

# Fit with N mode (works) to get parameter names
cat("=== PG+slopes (N mode) ===\n")
fit <- tratio(y | effort ~ (x | site), data = df,
       family = ratiod_poisson_gamma(),
       control = list(iter = 5, warmup = 2, chains = 1, gradient_mode = "N", verbose = TRUE))

# Extract parameter names
cat("\nParameter names:\n")
pnames <- colnames(fit$draws[,,drop=FALSE])
if (is.null(pnames)) pnames <- colnames(fit$draws)
cat(paste(seq_along(pnames) - 1, pnames, sep = ": "), sep = "\n")
cat("\nTotal params:", length(pnames), "\n")
