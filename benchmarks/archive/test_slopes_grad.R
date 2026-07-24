# Debug: slopes+ICAR gradient mismatch at param 9 for binomial
devtools::load_all(quiet = TRUE)
set.seed(123)

# Reproduce the slopes+ICAR benchmark config (row 79 = Bin+slopes+ICAR)
N <- 500
S <- 50
T <- 20

# Generate spatial adjacency
adj <- matrix(0, S, S)
for (i in 1:(S-1)) {
  adj[i, i+1] <- 1; adj[i+1, i] <- 1
}

# Generate data
site <- rep(1:S, each = N/S)
time <- rep(rep(1:T, each = N/(S*T)), S)
x <- rnorm(N)
lon <- rep(runif(S, 0, 10), each = N/S)
lat <- rep(runif(S, 0, 10), each = N/S)

df <- data.frame(
  y = rbinom(N, size = 20, prob = 0.3),
  trials = rep(20L, N),
  x = x,
  site = factor(site),
  time = time
)

cat("=== Bin+slopes+ICAR: Gradient check ===\n")
cat("Testing with verbose to see parameter layout...\n")

# Run with H mode
fit <- tryCatch({
  tratio(y | trials ~ x + (x | site), data = df,
         family = ratiod_binomial(),
         spatial = spatial_icar(adj),
         control = list(iter = 10, warmup = 5, chains = 1, gradient_mode = "H", verbose = TRUE))
}, error = function(e) paste0("ERROR: ", e$message))

if (is.character(fit)) {
  cat(fit, "\n")
} else {
  cat("H mode: SUCCESS\n")
}

# Try N mode for comparison
cat("\n=== N mode ===\n")
fit_n <- tryCatch({
  tratio(y | trials ~ x + (x | site), data = df,
         family = ratiod_binomial(),
         spatial = spatial_icar(adj),
         control = list(iter = 10, warmup = 5, chains = 1, gradient_mode = "N", verbose = FALSE))
}, error = function(e) paste0("ERROR: ", e$message))
if (is.character(fit_n)) cat(fit_n, "\n") else cat("N mode: SUCCESS\n")
