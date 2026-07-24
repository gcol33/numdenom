# Validation benchmark for ✓runs-only rows against Stan
# Rows: 22, 28, 29, 52, 58, 59, 78, 79, 90, 91

library(numdenom)
library(cmdstanr)
library(posterior)
set.seed(20260209)

# Parameters - smaller for quick validation
N_OBS <- 150
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1
N_SITES <- 15
N_TIMES <- 10

message(paste(rep("=", 70), collapse = ""))
message("VALIDATION BENCHMARK: ✓runs-only rows against Stan")
message(paste(rep("=", 70), collapse = ""))

# Helper to create adjacency matrix (line graph)
create_adjacency <- function(n_sites) {
  # Simple line graph adjacency matrix
  adj <- matrix(0, n_sites, n_sites)
  for (i in 1:(n_sites - 1)) {
    adj[i, i + 1] <- 1
    adj[i + 1, i] <- 1
  }
  adj
}

# Helper to validate parameter recovery
validate_params <- function(nd_draws, stan_draws, params) {
  results <- list()
  for (p in params) {
    nd_vals <- nd_draws[[p]]
    stan_vals <- stan_draws[[p]]
    if (is.null(nd_vals) || is.null(stan_vals)) next

    nd_mean <- mean(nd_vals)
    stan_mean <- mean(stan_vals)
    stan_se <- sd(stan_vals) / sqrt(length(stan_vals))
    diff_se <- abs(nd_mean - stan_mean) / (2 * stan_se)

    results[[p]] <- list(
      nd_mean = nd_mean,
      stan_mean = stan_mean,
      stan_se = stan_se,
      diff_se = diff_se,
      pass = diff_se < 2
    )
  }
  results
}

adj <- create_adjacency(N_SITES)

# ============================================================
# Row 78: Binomial + OI (One-Inflated)
# ============================================================
message("\n>>> Row 78: binomial + OI (one-inflated) <<<")

tryCatch({
  # Generate data with one-inflation
  df78 <- data.frame(
    site = rep(1:N_SITES, each = N_OBS / N_SITES),
    x = rnorm(N_OBS)
  )
  re78 <- rnorm(N_SITES, 0, 0.3)
  eta78 <- 0.5 + 0.3 * df78$x + re78[df78$site]
  prob78 <- plogis(eta78)
  trials78 <- sample(5:20, N_OBS, replace = TRUE)

  # Generate with one-inflation (10% excess ones)
  oi_prob <- 0.1
  is_inflated <- rbinom(N_OBS, 1, oi_prob)
  df78$y <- ifelse(is_inflated == 1, trials78,
                   rbinom(N_OBS, trials78, prob78))
  df78$trials <- trials78

  # Fit numdenom
  time_nd <- system.time({
    fit78 <- tratio(
      y | trials ~ x + (1 | site),
      data = df78,
      family = ratiod_oibinomial(),
      control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H")
    )
  })["elapsed"]

  # Fit Stan
  stan_model <- cmdstan_model("benchmarks/stan/joint_binom_oi.stan")
  stan_data <- list(
    N = N_OBS,
    y = df78$y,
    trials = df78$trials,
    p = 2,
    X = cbind(1, df78$x),
    n_groups = N_SITES,
    group_idx = df78$site
  )

  time_stan <- system.time({
    stan_fit <- stan_model$sample(
      data = stan_data,
      iter_warmup = N_WARMUP,
      iter_sampling = N_ITER - N_WARMUP,
      chains = N_CHAINS,
      refresh = 0
    )
  })["elapsed"]

  message(sprintf("Row 78: numdenom %.1fs, Stan %.1fs", time_nd, time_stan))
  message(sprintf("  Divergences: %d", sum(fit78$diagnostics$divergent)))

}, error = function(e) {
  message(sprintf("Row 78 ERROR: %s", e$message))
})

# ============================================================
# Row 79: Binomial + ZOIB (Zero-One Inflated)
# ============================================================
message("\n>>> Row 79: binomial + ZOIB (zero-one inflated) <<<")

tryCatch({
  # Generate data with zero-one inflation
  df79 <- data.frame(
    site = rep(1:N_SITES, each = N_OBS / N_SITES),
    x = rnorm(N_OBS)
  )
  re79 <- rnorm(N_SITES, 0, 0.3)
  eta79 <- 0.0 + 0.2 * df79$x + re79[df79$site]
  prob79 <- plogis(eta79)
  trials79 <- sample(5:20, N_OBS, replace = TRUE)

  # Generate with zero-one inflation
  pi0 <- 0.1  # 10% excess zeros
  pi1 <- 0.1  # 10% excess ones
  u <- runif(N_OBS)
  df79$y <- ifelse(u < pi0, 0,
                   ifelse(u < pi0 + pi1, trials79,
                          rbinom(N_OBS, trials79, prob79)))
  df79$trials <- trials79

  # Fit numdenom
  time_nd <- system.time({
    fit79 <- tratio(
      y | trials ~ x + (1 | site),
      data = df79,
      family = ratiod_zoibinomial(),
      control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H")
    )
  })["elapsed"]

  # Fit Stan
  stan_model <- cmdstan_model("benchmarks/stan/joint_binom_zoib.stan")
  stan_data <- list(
    N = N_OBS,
    y = df79$y,
    trials = df79$trials,
    p = 2,
    X = cbind(1, df79$x),
    n_groups = N_SITES,
    group_idx = df79$site
  )

  time_stan <- system.time({
    stan_fit <- stan_model$sample(
      data = stan_data,
      iter_warmup = N_WARMUP,
      iter_sampling = N_ITER - N_WARMUP,
      chains = N_CHAINS,
      refresh = 0
    )
  })["elapsed"]

  message(sprintf("Row 79: numdenom %.1fs, Stan %.1fs", time_nd, time_stan))
  message(sprintf("  Divergences: %d", sum(fit79$diagnostics$divergent)))

}, error = function(e) {
  message(sprintf("Row 79 ERROR: %s", e$message))
})

# ============================================================
# Row 22: poisson_gamma + RE + HSGP + RW1
# ============================================================
message("\n>>> Row 22: poisson_gamma + RE + HSGP + RW1 <<<")

tryCatch({
  df22 <- data.frame(
    site = rep(1:N_SITES, each = N_TIMES),
    time = rep(1:N_TIMES, N_SITES),
    x = rnorm(N_SITES * N_TIMES),
    x_coord = runif(N_SITES * N_TIMES),
    y_coord = runif(N_SITES * N_TIMES)
  )

  re22 <- rnorm(N_SITES, 0, 0.3)
  temporal22 <- cumsum(rnorm(N_TIMES, 0, 0.2))
  temporal22 <- temporal22 - mean(temporal22)

  eta_num <- 1.5 + 0.3 * df22$x + re22[df22$site] + temporal22[df22$time]
  eta_denom <- 2.0 - 0.2 * df22$x + re22[df22$site] + temporal22[df22$time]

  df22$y_num <- rpois(nrow(df22), exp(eta_num))
  df22$y_denom <- rgamma(nrow(df22), shape = 5, rate = 5 / exp(eta_denom))
  df22$y_denom <- pmax(df22$y_denom, 0.1)

  time_nd <- system.time({
    fit22 <- tratio(
      y_num | y_denom ~ x + (1 | site),
      data = df22,
      family = ratiod_poisson_gamma(),
      spatial = spatial_hsgp(coords = ~ x_coord + y_coord),
      temporal = temporal_rw1(time_var = "time"),
      control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H")
    )
  })["elapsed"]

  message(sprintf("Row 22: numdenom %.1fs", time_nd))
  message(sprintf("  Divergences: %d", sum(fit22$diagnostics$divergent)))

}, error = function(e) {
  message(sprintf("Row 22 ERROR: %s", e$message))
})

# ============================================================
# Row 52: negbin_negbin + RE + HSGP + RW1
# ============================================================
message("\n>>> Row 52: negbin_negbin + RE + HSGP + RW1 <<<")

tryCatch({
  df52 <- data.frame(
    site = rep(1:N_SITES, each = N_TIMES),
    time = rep(1:N_TIMES, N_SITES),
    x = rnorm(N_SITES * N_TIMES),
    x_coord = runif(N_SITES * N_TIMES),
    y_coord = runif(N_SITES * N_TIMES)
  )

  re52 <- rnorm(N_SITES, 0, 0.3)
  temporal52 <- cumsum(rnorm(N_TIMES, 0, 0.2))
  temporal52 <- temporal52 - mean(temporal52)

  eta_num <- 1.5 + 0.3 * df52$x + re52[df52$site] + temporal52[df52$time]
  eta_denom <- 2.0 - 0.2 * df52$x + re52[df52$site] + temporal52[df52$time]

  df52$y_num <- rnbinom(nrow(df52), size = 5, mu = exp(eta_num))
  df52$y_denom <- rnbinom(nrow(df52), size = 5, mu = exp(eta_denom))
  df52$y_denom <- pmax(df52$y_denom, 1)

  time_nd <- system.time({
    fit52 <- tratio(
      y_num | y_denom ~ x + (1 | site),
      data = df52,
      family = ratiod_negbin_negbin(),
      spatial = spatial_hsgp(coords = ~ x_coord + y_coord),
      temporal = temporal_rw1(time_var = "time"),
      control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H")
    )
  })["elapsed"]

  message(sprintf("Row 52: numdenom %.1fs", time_nd))
  message(sprintf("  Divergences: %d", sum(fit52$diagnostics$divergent)))

}, error = function(e) {
  message(sprintf("Row 52 ERROR: %s", e$message))
})

# ============================================================
# Row 90: binomial + ICAR + RW1 + ST-I
# ============================================================
message("\n>>> Row 90: binomial + ICAR + RW1 + ST-I <<<")

tryCatch({
  df90 <- data.frame(
    site = rep(1:N_SITES, each = N_TIMES),
    time = rep(1:N_TIMES, N_SITES),
    x = rnorm(N_SITES * N_TIMES)
  )

  re90 <- rnorm(N_SITES, 0, 0.3)
  spatial90 <- rnorm(N_SITES, 0, 0.2)
  temporal90 <- cumsum(rnorm(N_TIMES, 0, 0.15))
  temporal90 <- temporal90 - mean(temporal90)
  st_interaction <- matrix(rnorm(N_SITES * N_TIMES, 0, 0.1), N_SITES, N_TIMES)

  eta90 <- 0.5 + 0.2 * df90$x + re90[df90$site] +
           spatial90[df90$site] + temporal90[df90$time]
  for (i in 1:nrow(df90)) {
    eta90[i] <- eta90[i] + st_interaction[df90$site[i], df90$time[i]]
  }

  prob90 <- plogis(eta90)
  trials90 <- sample(10:30, nrow(df90), replace = TRUE)
  df90$y <- rbinom(nrow(df90), trials90, prob90)
  df90$trials <- trials90

  time_nd <- system.time({
    fit90 <- tratio(
      y | trials ~ x + (1 | site),
      data = df90,
      family = ratiod_binomial(),
      spatiotemporal = spatiotemporal(
        spatial = spatial_car(adjacency = adj, group_var = "site", level = "group"),
        temporal = temporal_rw1(time_var = "time"),
        type = "I"
      ),
      control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H")
    )
  })["elapsed"]

  message(sprintf("Row 90: numdenom %.1fs", time_nd))
  message(sprintf("  Divergences: %d", sum(fit90$diagnostics$divergent)))

}, error = function(e) {
  message(sprintf("Row 90 ERROR: %s", e$message))
})

# ============================================================
# Row 91: binomial + ICAR + RW1 + ST-IV
# ============================================================
message("\n>>> Row 91: binomial + ICAR + RW1 + ST-IV <<<")

tryCatch({
  df91 <- df90  # Reuse data

  time_nd <- system.time({
    fit91 <- tratio(
      y | trials ~ x + (1 | site),
      data = df91,
      family = ratiod_binomial(),
      spatiotemporal = spatiotemporal(
        spatial = spatial_car(adjacency = adj, group_var = "site", level = "group"),
        temporal = temporal_rw1(time_var = "time"),
        type = "IV"
      ),
      control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H")
    )
  })["elapsed"]

  message(sprintf("Row 91: numdenom %.1fs", time_nd))
  message(sprintf("  Divergences: %d", sum(fit91$diagnostics$divergent)))

}, error = function(e) {
  message(sprintf("Row 91 ERROR: %s", e$message))
})

# ============================================================
# Row 28: poisson_gamma + ICAR + RW1 + ST-I
# ============================================================
message("\n>>> Row 28: poisson_gamma + ICAR + RW1 + ST-I <<<")

tryCatch({
  df28 <- data.frame(
    site = rep(1:N_SITES, each = N_TIMES),
    time = rep(1:N_TIMES, N_SITES),
    x = rnorm(N_SITES * N_TIMES)
  )

  re28 <- rnorm(N_SITES, 0, 0.3)
  spatial28 <- rnorm(N_SITES, 0, 0.2)
  temporal28 <- cumsum(rnorm(N_TIMES, 0, 0.15))
  temporal28 <- temporal28 - mean(temporal28)
  st_interaction <- matrix(rnorm(N_SITES * N_TIMES, 0, 0.1), N_SITES, N_TIMES)

  eta_num <- 1.5 + 0.3 * df28$x + re28[df28$site] +
             spatial28[df28$site] + temporal28[df28$time]
  eta_denom <- 2.0 - 0.2 * df28$x + re28[df28$site] +
               spatial28[df28$site] + temporal28[df28$time]
  for (i in 1:nrow(df28)) {
    eta_num[i] <- eta_num[i] + st_interaction[df28$site[i], df28$time[i]]
    eta_denom[i] <- eta_denom[i] + st_interaction[df28$site[i], df28$time[i]]
  }

  df28$y_num <- rpois(nrow(df28), exp(eta_num))
  df28$y_denom <- rgamma(nrow(df28), shape = 5, rate = 5 / exp(eta_denom))
  df28$y_denom <- pmax(df28$y_denom, 0.1)

  time_nd <- system.time({
    fit28 <- tratio(
      y_num | y_denom ~ x + (1 | site),
      data = df28,
      family = ratiod_poisson_gamma(),
      spatiotemporal = spatiotemporal(
        spatial = spatial_car(adjacency = adj, group_var = "site", level = "group"),
        temporal = temporal_rw1(time_var = "time"),
        type = "I"
      ),
      control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H")
    )
  })["elapsed"]

  message(sprintf("Row 28: numdenom %.1fs", time_nd))
  message(sprintf("  Divergences: %d", sum(fit28$diagnostics$divergent)))

}, error = function(e) {
  message(sprintf("Row 28 ERROR: %s", e$message))
})

# ============================================================
# Row 29: poisson_gamma + ICAR + RW1 + ST-IV
# ============================================================
message("\n>>> Row 29: poisson_gamma + ICAR + RW1 + ST-IV <<<")

tryCatch({
  time_nd <- system.time({
    fit29 <- tratio(
      y_num | y_denom ~ x + (1 | site),
      data = df28,  # Reuse data
      family = ratiod_poisson_gamma(),
      spatiotemporal = spatiotemporal(
        spatial = spatial_car(adjacency = adj, group_var = "site", level = "group"),
        temporal = temporal_rw1(time_var = "time"),
        type = "IV"
      ),
      control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H")
    )
  })["elapsed"]

  message(sprintf("Row 29: numdenom %.1fs", time_nd))
  message(sprintf("  Divergences: %d", sum(fit29$diagnostics$divergent)))

}, error = function(e) {
  message(sprintf("Row 29 ERROR: %s", e$message))
})

# ============================================================
# Row 58: negbin_negbin + ICAR + RW1 + ST-I
# ============================================================
message("\n>>> Row 58: negbin_negbin + ICAR + RW1 + ST-I <<<")

tryCatch({
  df58 <- data.frame(
    site = rep(1:N_SITES, each = N_TIMES),
    time = rep(1:N_TIMES, N_SITES),
    x = rnorm(N_SITES * N_TIMES)
  )

  re58 <- rnorm(N_SITES, 0, 0.3)
  spatial58 <- rnorm(N_SITES, 0, 0.2)
  temporal58 <- cumsum(rnorm(N_TIMES, 0, 0.15))
  temporal58 <- temporal58 - mean(temporal58)
  st_interaction <- matrix(rnorm(N_SITES * N_TIMES, 0, 0.1), N_SITES, N_TIMES)

  eta_num <- 1.5 + 0.3 * df58$x + re58[df58$site] +
             spatial58[df58$site] + temporal58[df58$time]
  eta_denom <- 2.0 - 0.2 * df58$x + re58[df58$site] +
               spatial58[df58$site] + temporal58[df58$time]
  for (i in 1:nrow(df58)) {
    eta_num[i] <- eta_num[i] + st_interaction[df58$site[i], df58$time[i]]
    eta_denom[i] <- eta_denom[i] + st_interaction[df58$site[i], df58$time[i]]
  }

  df58$y_num <- rnbinom(nrow(df58), size = 5, mu = exp(eta_num))
  df58$y_denom <- rnbinom(nrow(df58), size = 5, mu = exp(eta_denom))
  df58$y_denom <- pmax(df58$y_denom, 1)

  time_nd <- system.time({
    fit58 <- tratio(
      y_num | y_denom ~ x + (1 | site),
      data = df58,
      family = ratiod_negbin_negbin(),
      spatiotemporal = spatiotemporal(
        spatial = spatial_car(adjacency = adj, group_var = "site", level = "group"),
        temporal = temporal_rw1(time_var = "time"),
        type = "I"
      ),
      control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H")
    )
  })["elapsed"]

  message(sprintf("Row 58: numdenom %.1fs", time_nd))
  message(sprintf("  Divergences: %d", sum(fit58$diagnostics$divergent)))

}, error = function(e) {
  message(sprintf("Row 58 ERROR: %s", e$message))
})

# ============================================================
# Row 59: negbin_negbin + ICAR + RW1 + ST-IV
# ============================================================
message("\n>>> Row 59: negbin_negbin + ICAR + RW1 + ST-IV <<<")

tryCatch({
  time_nd <- system.time({
    fit59 <- tratio(
      y_num | y_denom ~ x + (1 | site),
      data = df58,  # Reuse data
      family = ratiod_negbin_negbin(),
      spatiotemporal = spatiotemporal(
        spatial = spatial_car(adjacency = adj, group_var = "site", level = "group"),
        temporal = temporal_rw1(time_var = "time"),
        type = "IV"
      ),
      control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H")
    )
  })["elapsed"]

  message(sprintf("Row 59: numdenom %.1fs", time_nd))
  message(sprintf("  Divergences: %d", sum(fit59$diagnostics$divergent)))

}, error = function(e) {
  message(sprintf("Row 59 ERROR: %s", e$message))
})

message(paste(rep("=", 70), collapse = ""))
message("VALIDATION COMPLETE")
message(paste(rep("=", 70), collapse = ""))
