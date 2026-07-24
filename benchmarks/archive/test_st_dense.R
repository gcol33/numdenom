library(numdenom)
set.seed(42)

N_SITES <- 15; N_TIMES <- 10; N_OBS <- N_SITES * N_TIMES

# Build grid adjacency
adj <- matrix(0, N_SITES, N_SITES)
n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(x = 1:n_side, y = 1:n_side)[1:N_SITES, ]
for (i in 1:(N_SITES - 1)) {
  for (j in (i + 1):N_SITES) {
    if (sqrt((grid$x[i] - grid$x[j])^2 + (grid$y[i] - grid$y[j])^2) <= 1.01) {
      adj[i, j] <- 1; adj[j, i] <- 1
    }
  }
}

site <- factor(rep(1:N_SITES, each = N_TIMES))
time_idx <- rep(1:N_TIMES, N_SITES)
x <- rnorm(N_OBS)

re <- rnorm(N_SITES, 0, 0.3)
phi_s <- cumsum(rnorm(N_SITES, 0, 0.3)); phi_s <- phi_s - mean(phi_s)
phi_t <- cumsum(rnorm(N_TIMES, 0, 0.2)); phi_t <- phi_t - mean(phi_t)
delta <- matrix(rnorm(N_SITES * N_TIMES, 0, 0.15), N_SITES, N_TIMES)
delta <- delta - mean(delta)

shared <- re[as.integer(site)] + phi_s[as.integer(site)] +
  phi_t[time_idx] + delta[cbind(as.integer(site), time_idx)]

eta <- 1.0 + 0.3 * x + shared
trials <- sample(20:50, N_OBS, replace = TRUE)
y <- rbinom(N_OBS, trials, plogis(eta))
df <- data.frame(y = y, trials = trials, x = x,
                 site = site, time = factor(time_idx))

# Test 1: single chain first (avoids parallel issues)
cat("Test 1: Bin+ST-I, 1 chain, 200 iter (quick test)...\n")
t1 <- system.time({
  fit1 <- tryCatch(
    tratio(y | trials ~ x + (1|site), data = df,
           family = ratiod_binomial(),
           spatiotemporal = spatiotemporal(
             spatial = spatial_car(adj, group_var = "site"),
             temporal = temporal_rw1(time_var = "time"),
             type = "I"
           ),
           control = list(iter = 200, warmup = 100, chains = 1, verbose = TRUE)),
    error = function(e) { cat("ERROR:", e$message, "\n"); NULL }
  )
})["elapsed"]
cat(sprintf("Time: %.1fs\n\n", t1))

if (!is.null(fit1)) {
  # Test 2: 2 chains
  cat("Test 2: Bin+ST-I, 2 chains, 1000 iter...\n")
  t2 <- system.time({
    fit2 <- tryCatch(
      tratio(y | trials ~ x + (1|site), data = df,
             family = ratiod_binomial(),
             spatiotemporal = spatiotemporal(
               spatial = spatial_car(adj, group_var = "site"),
               temporal = temporal_rw1(time_var = "time"),
               type = "I"
             ),
             control = list(iter = 1000, warmup = 500, chains = 2, verbose = TRUE)),
      error = function(e) { cat("ERROR:", e$message, "\n"); NULL }
    )
  })["elapsed"]
  cat(sprintf("Time: %.1fs\n", t2))
  if (!is.null(fit2)) {
    draws <- as.matrix(fit2$draws)
    cat(sprintf("Slope: %.4f (SD=%.4f)\n",
                mean(draws[, "beta_num[2]"]), sd(draws[, "beta_num[2]"])))
  }
}
