# Validate remaining 8 rows that have Stan models available
# Rows: 22, 28, 29, 30 (poisson_gamma) + 52, 58, 59, 60 (negbin_negbin)

library(numdenom)
library(cmdstanr)

cat("======================================================================\n")
cat("VALIDATION: Remaining rows with Stan models\n")
cat("======================================================================\n\n")

# Standard parameters (smaller for Stan speed)
N_OBS <- 200
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1
N_SITES <- 30
N_TIMES <- 10
K_LATENT <- 2

set.seed(42)

# Helper to create adjacency matrix
create_adjacency <- function(n_sites) {
  adj <- matrix(0, n_sites, n_sites)
  for (i in 1:(n_sites - 1)) {
    adj[i, i + 1] <- 1
    adj[i + 1, i] <- 1
  }
  adj
}

# Helper to create ICAR components
create_icar_components <- function(adj) {
  n <- nrow(adj)
  node1 <- c()
  node2 <- c()
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      if (adj[i,j] == 1) {
        node1 <- c(node1, i)
        node2 <- c(node2, j)
      }
    }
  }
  list(n_edges = length(node1), node1 = node1, node2 = node2)
}

# Helper for comparison
compare_posteriors <- function(numdenom_fit, stan_fit, param_name, threshold = 2.0) {
  nd_draws <- as.matrix(numdenom_fit$draws[[1]])
  nd_mean <- mean(nd_draws[, param_name])
  nd_se <- sd(nd_draws[, param_name]) / sqrt(nrow(nd_draws))

  stan_draws <- stan_fit$draws(param_name, format = "matrix")
  stan_mean <- mean(stan_draws)
  stan_se <- sd(stan_draws) / sqrt(length(stan_draws))

  combined_se <- sqrt(nd_se^2 + stan_se^2)
  diff_se <- abs(nd_mean - stan_mean) / combined_se

  list(
    nd_mean = nd_mean,
    stan_mean = stan_mean,
    diff_se = diff_se,
    pass = diff_se < threshold
  )
}

results <- list()

# ============================================================================
# Row 22: poisson_gamma + HSGP + RW1
# ============================================================================
cat("\n>>> Row 22: poisson_gamma + HSGP + RW1 <<<\n")
tryCatch({
  coords22 <- data.frame(x = runif(N_OBS, 0, 10), y = runif(N_OBS, 0, 10))
  df22 <- data.frame(
    y_num = rpois(N_OBS, 10),
    y_denom = rgamma(N_OBS, 5, 1),
    x = rnorm(N_OBS),
    site = rep(1:N_SITES, length.out = N_OBS),
    time = rep(1:N_TIMES, each = N_OBS / N_TIMES),
    coord_x = coords22$x,
    coord_y = coords22$y
  )

  # numdenom fit
  t22_nd <- system.time({
    fit22_nd <- ratiod(
      y_num | y_denom ~ x + (1 | site),
      data = df22,
      family = ratiod_poisson_gamma(),
      spatial = spatial_hsgp(coords = c("coord_x", "coord_y")),
      temporal = temporal_rw1(time_var = "time"),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      gradient_mode = "H"
    )
  })["elapsed"]

  # Stan fit
  stan_model22 <- cmdstan_model("benchmarks/stan/hsgp_rw1_pg_joint.stan")
  stan_data22 <- list(
    N = N_OBS,
    M = 5,
    y_num = df22$y_num,
    y_denom = df22$y_denom,
    p = 2,
    X = cbind(1, df22$x),
    coords = as.matrix(coords22),
    n_groups = N_SITES,
    group_idx = df22$site,
    T = N_TIMES,
    time_idx = df22$time,
    c = 1.5
  )

  t22_stan <- system.time({
    fit22_stan <- stan_model22$sample(
      data = stan_data22,
      iter_warmup = N_WARMUP, iter_sampling = N_ITER - N_WARMUP,
      chains = N_CHAINS, refresh = 0
    )
  })["elapsed"]

  comp22 <- compare_posteriors(fit22_nd, fit22_stan, "beta_num[1]")
  div22 <- sum(fit22_nd$diagnostics$divergent)

  cat(sprintf("Row 22: numdenom %.1fs, Stan %.1fs\n", t22_nd, t22_stan))
  cat(sprintf("  Divergences: %d, SE diff: %.2f, PASS: %s\n",
              div22, comp22$diff_se, comp22$pass))

  results$row22 <- list(time_nd = t22_nd, time_stan = t22_stan,
                        div = div22, diff_se = comp22$diff_se,
                        status = ifelse(comp22$pass, "PASS", "FAIL"))
}, error = function(e) {
  cat(sprintf("Row 22: FAILED - %s\n", e$message))
  results$row22 <<- list(status = "ERROR", error = e$message)
})

# ============================================================================
# Row 28: poisson_gamma + ST-I (spatiotemporal Type I)
# ============================================================================
cat("\n>>> Row 28: poisson_gamma + ST-I <<<\n")
tryCatch({
  adj28 <- create_adjacency(N_SITES)
  icar28 <- create_icar_components(adj28)

  df28 <- data.frame(
    y_num = rpois(N_OBS, 10),
    y_denom = rgamma(N_OBS, 5, 1),
    x = rnorm(N_OBS),
    site = rep(1:N_SITES, length.out = N_OBS),
    time = rep(1:N_TIMES, each = N_OBS / N_TIMES)
  )

  # numdenom fit
  t28_nd <- system.time({
    fit28_nd <- ratiod(
      y_num | y_denom ~ x + (1 | site),
      data = df28,
      family = ratiod_poisson_gamma(),
      spatiotemporal = spatiotemporal(
        spatial = spatial_icar(adj = adj28, group_var = "site"),
        temporal = temporal_rw1(time_var = "time"),
        type = "I"
      ),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      gradient_mode = "H"
    )
  })["elapsed"]

  # Stan fit
  stan_model28 <- cmdstan_model("benchmarks/stan/joint_pg_st_i.stan")
  stan_data28 <- list(
    N = N_OBS,
    y_num = df28$y_num,
    y_denom = df28$y_denom,
    p = 2,
    X = cbind(1, df28$x),
    n_groups = N_SITES,
    group_idx = df28$site,
    n_edges = icar28$n_edges,
    node1 = icar28$node1,
    node2 = icar28$node2,
    T = N_TIMES,
    time_idx = df28$time,
    S = N_SITES
  )

  t28_stan <- system.time({
    fit28_stan <- stan_model28$sample(
      data = stan_data28,
      iter_warmup = N_WARMUP, iter_sampling = N_ITER - N_WARMUP,
      chains = N_CHAINS, refresh = 0
    )
  })["elapsed"]

  comp28 <- compare_posteriors(fit28_nd, fit28_stan, "beta_num[1]")
  div28 <- sum(fit28_nd$diagnostics$divergent)

  cat(sprintf("Row 28: numdenom %.1fs, Stan %.1fs\n", t28_nd, t28_stan))
  cat(sprintf("  Divergences: %d, SE diff: %.2f, PASS: %s\n",
              div28, comp28$diff_se, comp28$pass))

  results$row28 <- list(time_nd = t28_nd, time_stan = t28_stan,
                        div = div28, diff_se = comp28$diff_se,
                        status = ifelse(comp28$pass, "PASS", "FAIL"))
}, error = function(e) {
  cat(sprintf("Row 28: FAILED - %s\n", e$message))
  results$row28 <<- list(status = "ERROR", error = e$message)
})

# ============================================================================
# Row 29: poisson_gamma + ST-IV (spatiotemporal Type IV)
# ============================================================================
cat("\n>>> Row 29: poisson_gamma + ST-IV <<<\n")
tryCatch({
  adj29 <- create_adjacency(N_SITES)
  icar29 <- create_icar_components(adj29)

  df29 <- data.frame(
    y_num = rpois(N_OBS, 10),
    y_denom = rgamma(N_OBS, 5, 1),
    x = rnorm(N_OBS),
    site = rep(1:N_SITES, length.out = N_OBS),
    time = rep(1:N_TIMES, each = N_OBS / N_TIMES)
  )

  # numdenom fit
  t29_nd <- system.time({
    fit29_nd <- ratiod(
      y_num | y_denom ~ x + (1 | site),
      data = df29,
      family = ratiod_poisson_gamma(),
      spatiotemporal = spatiotemporal(
        spatial = spatial_icar(adj = adj29, group_var = "site"),
        temporal = temporal_rw1(time_var = "time"),
        type = "IV"
      ),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      gradient_mode = "H"
    )
  })["elapsed"]

  # Stan fit
  stan_model29 <- cmdstan_model("benchmarks/stan/joint_pg_st_iv.stan")
  stan_data29 <- list(
    N = N_OBS,
    y_num = df29$y_num,
    y_denom = df29$y_denom,
    p = 2,
    X = cbind(1, df29$x),
    n_groups = N_SITES,
    group_idx = df29$site,
    n_edges = icar29$n_edges,
    node1 = icar29$node1,
    node2 = icar29$node2,
    T = N_TIMES,
    time_idx = df29$time,
    S = N_SITES
  )

  t29_stan <- system.time({
    fit29_stan <- stan_model29$sample(
      data = stan_data29,
      iter_warmup = N_WARMUP, iter_sampling = N_ITER - N_WARMUP,
      chains = N_CHAINS, refresh = 0
    )
  })["elapsed"]

  comp29 <- compare_posteriors(fit29_nd, fit29_stan, "beta_num[1]")
  div29 <- sum(fit29_nd$diagnostics$divergent)

  cat(sprintf("Row 29: numdenom %.1fs, Stan %.1fs\n", t29_nd, t29_stan))
  cat(sprintf("  Divergences: %d, SE diff: %.2f, PASS: %s\n",
              div29, comp29$diff_se, comp29$pass))

  results$row29 <- list(time_nd = t29_nd, time_stan = t29_stan,
                        div = div29, diff_se = comp29$diff_se,
                        status = ifelse(comp29$pass, "PASS", "FAIL"))
}, error = function(e) {
  cat(sprintf("Row 29: FAILED - %s\n", e$message))
  results$row29 <<- list(status = "ERROR", error = e$message)
})

# ============================================================================
# Row 30: poisson_gamma + latent factor
# ============================================================================
cat("\n>>> Row 30: poisson_gamma + latent factor <<<\n")
tryCatch({
  N_SMALL <- 50  # Latent factor models are slow with large N

  df30 <- data.frame(
    y_num = rpois(N_SMALL, 10),
    y_denom = rgamma(N_SMALL, 5, 1),
    x = rnorm(N_SMALL),
    site = rep(1:10, length.out = N_SMALL)
  )

  # numdenom fit
  t30_nd <- system.time({
    fit30_nd <- ratiod(
      y_num | y_denom ~ x + (1 | site),
      data = df30,
      family = ratiod_poisson_gamma(),
      latent = latent_factor(n_factors = K_LATENT),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      gradient_mode = "H"
    )
  })["elapsed"]

  # Stan fit
  stan_model30 <- cmdstan_model("benchmarks/stan/joint_pg_latent.stan")
  stan_data30 <- list(
    N = N_SMALL,
    y_num = df30$y_num,
    y_denom = df30$y_denom,
    p = 2,
    X = cbind(1, df30$x),
    K = K_LATENT,
    n_groups = 10,
    group_idx = df30$site
  )

  t30_stan <- system.time({
    fit30_stan <- stan_model30$sample(
      data = stan_data30,
      iter_warmup = N_WARMUP, iter_sampling = N_ITER - N_WARMUP,
      chains = N_CHAINS, refresh = 0
    )
  })["elapsed"]

  comp30 <- compare_posteriors(fit30_nd, fit30_stan, "beta_num[1]")
  div30 <- sum(fit30_nd$diagnostics$divergent)

  cat(sprintf("Row 30: numdenom %.1fs, Stan %.1fs\n", t30_nd, t30_stan))
  cat(sprintf("  Divergences: %d, SE diff: %.2f, PASS: %s\n",
              div30, comp30$diff_se, comp30$pass))

  results$row30 <- list(time_nd = t30_nd, time_stan = t30_stan,
                        div = div30, diff_se = comp30$diff_se,
                        status = ifelse(comp30$pass, "PASS", "FAIL"))
}, error = function(e) {
  cat(sprintf("Row 30: FAILED - %s\n", e$message))
  results$row30 <<- list(status = "ERROR", error = e$message)
})

# ============================================================================
# Row 52: negbin_negbin + HSGP + RW1
# ============================================================================
cat("\n>>> Row 52: negbin_negbin + HSGP + RW1 <<<\n")
tryCatch({
  coords52 <- data.frame(x = runif(N_OBS, 0, 10), y = runif(N_OBS, 0, 10))
  df52 <- data.frame(
    y_num = rnbinom(N_OBS, size = 5, mu = 10),
    y_denom = rnbinom(N_OBS, size = 5, mu = 20),
    x = rnorm(N_OBS),
    site = rep(1:N_SITES, length.out = N_OBS),
    time = rep(1:N_TIMES, each = N_OBS / N_TIMES),
    coord_x = coords52$x,
    coord_y = coords52$y
  )

  # numdenom fit
  t52_nd <- system.time({
    fit52_nd <- ratiod(
      y_num | y_denom ~ x + (1 | site),
      data = df52,
      family = ratiod_negbin_negbin(),
      spatial = spatial_hsgp(coords = c("coord_x", "coord_y")),
      temporal = temporal_rw1(time_var = "time"),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      gradient_mode = "H"
    )
  })["elapsed"]

  # Stan fit
  stan_model52 <- cmdstan_model("benchmarks/stan/hsgp_rw1_nb_joint.stan")
  stan_data52 <- list(
    N = N_OBS,
    M = 5,
    y_num = df52$y_num,
    y_denom = df52$y_denom,
    p = 2,
    X = cbind(1, df52$x),
    coords = as.matrix(coords52),
    n_groups = N_SITES,
    group_idx = df52$site,
    T = N_TIMES,
    time_idx = df52$time,
    c = 1.5
  )

  t52_stan <- system.time({
    fit52_stan <- stan_model52$sample(
      data = stan_data52,
      iter_warmup = N_WARMUP, iter_sampling = N_ITER - N_WARMUP,
      chains = N_CHAINS, refresh = 0
    )
  })["elapsed"]

  comp52 <- compare_posteriors(fit52_nd, fit52_stan, "beta_num[1]")
  div52 <- sum(fit52_nd$diagnostics$divergent)

  cat(sprintf("Row 52: numdenom %.1fs, Stan %.1fs\n", t52_nd, t52_stan))
  cat(sprintf("  Divergences: %d, SE diff: %.2f, PASS: %s\n",
              div52, comp52$diff_se, comp52$pass))

  results$row52 <- list(time_nd = t52_nd, time_stan = t52_stan,
                        div = div52, diff_se = comp52$diff_se,
                        status = ifelse(comp52$pass, "PASS", "FAIL"))
}, error = function(e) {
  cat(sprintf("Row 52: FAILED - %s\n", e$message))
  results$row52 <<- list(status = "ERROR", error = e$message)
})

# ============================================================================
# Row 58: negbin_negbin + ST-I
# ============================================================================
cat("\n>>> Row 58: negbin_negbin + ST-I <<<\n")
tryCatch({
  adj58 <- create_adjacency(N_SITES)
  icar58 <- create_icar_components(adj58)

  df58 <- data.frame(
    y_num = rnbinom(N_OBS, size = 5, mu = 10),
    y_denom = rnbinom(N_OBS, size = 5, mu = 20),
    x = rnorm(N_OBS),
    site = rep(1:N_SITES, length.out = N_OBS),
    time = rep(1:N_TIMES, each = N_OBS / N_TIMES)
  )

  # numdenom fit
  t58_nd <- system.time({
    fit58_nd <- ratiod(
      y_num | y_denom ~ x + (1 | site),
      data = df58,
      family = ratiod_negbin_negbin(),
      spatiotemporal = spatiotemporal(
        spatial = spatial_icar(adj = adj58, group_var = "site"),
        temporal = temporal_rw1(time_var = "time"),
        type = "I"
      ),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      gradient_mode = "H"
    )
  })["elapsed"]

  # Stan fit
  stan_model58 <- cmdstan_model("benchmarks/stan/joint_nb_st_i.stan")
  stan_data58 <- list(
    N = N_OBS,
    y_num = df58$y_num,
    y_denom = df58$y_denom,
    p = 2,
    X = cbind(1, df58$x),
    n_groups = N_SITES,
    group_idx = df58$site,
    n_edges = icar58$n_edges,
    node1 = icar58$node1,
    node2 = icar58$node2,
    T = N_TIMES,
    time_idx = df58$time,
    S = N_SITES
  )

  t58_stan <- system.time({
    fit58_stan <- stan_model58$sample(
      data = stan_data58,
      iter_warmup = N_WARMUP, iter_sampling = N_ITER - N_WARMUP,
      chains = N_CHAINS, refresh = 0
    )
  })["elapsed"]

  comp58 <- compare_posteriors(fit58_nd, fit58_stan, "beta_num[1]")
  div58 <- sum(fit58_nd$diagnostics$divergent)

  cat(sprintf("Row 58: numdenom %.1fs, Stan %.1fs\n", t58_nd, t58_stan))
  cat(sprintf("  Divergences: %d, SE diff: %.2f, PASS: %s\n",
              div58, comp58$diff_se, comp58$pass))

  results$row58 <- list(time_nd = t58_nd, time_stan = t58_stan,
                        div = div58, diff_se = comp58$diff_se,
                        status = ifelse(comp58$pass, "PASS", "FAIL"))
}, error = function(e) {
  cat(sprintf("Row 58: FAILED - %s\n", e$message))
  results$row58 <<- list(status = "ERROR", error = e$message)
})

# ============================================================================
# Row 59: negbin_negbin + ST-IV
# ============================================================================
cat("\n>>> Row 59: negbin_negbin + ST-IV <<<\n")
tryCatch({
  adj59 <- create_adjacency(N_SITES)
  icar59 <- create_icar_components(adj59)

  df59 <- data.frame(
    y_num = rnbinom(N_OBS, size = 5, mu = 10),
    y_denom = rnbinom(N_OBS, size = 5, mu = 20),
    x = rnorm(N_OBS),
    site = rep(1:N_SITES, length.out = N_OBS),
    time = rep(1:N_TIMES, each = N_OBS / N_TIMES)
  )

  # numdenom fit
  t59_nd <- system.time({
    fit59_nd <- ratiod(
      y_num | y_denom ~ x + (1 | site),
      data = df59,
      family = ratiod_negbin_negbin(),
      spatiotemporal = spatiotemporal(
        spatial = spatial_icar(adj = adj59, group_var = "site"),
        temporal = temporal_rw1(time_var = "time"),
        type = "IV"
      ),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      gradient_mode = "H"
    )
  })["elapsed"]

  # Stan fit
  stan_model59 <- cmdstan_model("benchmarks/stan/joint_nb_st_iv.stan")
  stan_data59 <- list(
    N = N_OBS,
    y_num = df59$y_num,
    y_denom = df59$y_denom,
    p = 2,
    X = cbind(1, df59$x),
    n_groups = N_SITES,
    group_idx = df59$site,
    n_edges = icar59$n_edges,
    node1 = icar59$node1,
    node2 = icar59$node2,
    T = N_TIMES,
    time_idx = df59$time,
    S = N_SITES
  )

  t59_stan <- system.time({
    fit59_stan <- stan_model59$sample(
      data = stan_data59,
      iter_warmup = N_WARMUP, iter_sampling = N_ITER - N_WARMUP,
      chains = N_CHAINS, refresh = 0
    )
  })["elapsed"]

  comp59 <- compare_posteriors(fit59_nd, fit59_stan, "beta_num[1]")
  div59 <- sum(fit59_nd$diagnostics$divergent)

  cat(sprintf("Row 59: numdenom %.1fs, Stan %.1fs\n", t59_nd, t59_stan))
  cat(sprintf("  Divergences: %d, SE diff: %.2f, PASS: %s\n",
              div59, comp59$diff_se, comp59$pass))

  results$row59 <- list(time_nd = t59_nd, time_stan = t59_stan,
                        div = div59, diff_se = comp59$diff_se,
                        status = ifelse(comp59$pass, "PASS", "FAIL"))
}, error = function(e) {
  cat(sprintf("Row 59: FAILED - %s\n", e$message))
  results$row59 <<- list(status = "ERROR", error = e$message)
})

# ============================================================================
# Row 60: negbin_negbin + latent factor
# ============================================================================
cat("\n>>> Row 60: negbin_negbin + latent factor <<<\n")
tryCatch({
  N_SMALL <- 50  # Latent factor models are slow with large N

  df60 <- data.frame(
    y_num = rnbinom(N_SMALL, size = 5, mu = 10),
    y_denom = rnbinom(N_SMALL, size = 5, mu = 20),
    x = rnorm(N_SMALL),
    site = rep(1:10, length.out = N_SMALL)
  )

  # numdenom fit
  t60_nd <- system.time({
    fit60_nd <- ratiod(
      y_num | y_denom ~ x + (1 | site),
      data = df60,
      family = ratiod_negbin_negbin(),
      latent = latent_factor(n_factors = K_LATENT),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      gradient_mode = "H"
    )
  })["elapsed"]

  # Stan fit
  stan_model60 <- cmdstan_model("benchmarks/stan/joint_nb_latent.stan")
  stan_data60 <- list(
    N = N_SMALL,
    y_num = df60$y_num,
    y_denom = df60$y_denom,
    p = 2,
    X = cbind(1, df60$x),
    K = K_LATENT,
    n_groups = 10,
    group_idx = df60$site
  )

  t60_stan <- system.time({
    fit60_stan <- stan_model60$sample(
      data = stan_data60,
      iter_warmup = N_WARMUP, iter_sampling = N_ITER - N_WARMUP,
      chains = N_CHAINS, refresh = 0
    )
  })["elapsed"]

  comp60 <- compare_posteriors(fit60_nd, fit60_stan, "beta_num[1]")
  div60 <- sum(fit60_nd$diagnostics$divergent)

  cat(sprintf("Row 60: numdenom %.1fs, Stan %.1fs\n", t60_nd, t60_stan))
  cat(sprintf("  Divergences: %d, SE diff: %.2f, PASS: %s\n",
              div60, comp60$diff_se, comp60$pass))

  results$row60 <- list(time_nd = t60_nd, time_stan = t60_stan,
                        div = div60, diff_se = comp60$diff_se,
                        status = ifelse(comp60$pass, "PASS", "FAIL"))
}, error = function(e) {
  cat(sprintf("Row 60: FAILED - %s\n", e$message))
  results$row60 <<- list(status = "ERROR", error = e$message)
})

# ============================================================================
# Summary
# ============================================================================
cat("\n======================================================================\n")
cat("VALIDATION SUMMARY\n")
cat("======================================================================\n\n")

for (name in names(results)) {
  r <- results[[name]]
  if (r$status %in% c("PASS", "FAIL")) {
    cat(sprintf("  %s: %s (%.2f SE diff, %d div, nd=%.1fs, stan=%.1fs)\n",
                name, r$status, r$diff_se, r$div, r$time_nd, r$time_stan))
  } else {
    cat(sprintf("  %s: ERROR - %s\n", name, r$error))
  }
}

# Count results
pass_count <- sum(sapply(results, function(x) x$status == "PASS"))
fail_count <- sum(sapply(results, function(x) x$status == "FAIL"))
error_count <- sum(sapply(results, function(x) x$status == "ERROR"))

cat(sprintf("\nTotal: %d PASS, %d FAIL, %d ERROR out of %d\n",
            pass_count, fail_count, error_count, length(results)))
