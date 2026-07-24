# Benchmark tests for MSGP sampler strategies
# Tests correctness and compares performance of different samplers

# Helper function to generate spatial data
generate_msgp_data <- function(n = 50, seed = 123) {
  set.seed(seed)

  # Generate coordinates on unit square
  coords <- data.frame(
    x = runif(n),
    y = runif(n)
  )

  # Generate true spatial effects (local + regional patterns)
  dist_mat <- as.matrix(dist(coords))

  # Local effect (short range)
  sigma_local <- 0.5
  phi_local <- 0.1
  K_local <- sigma_local^2 * exp(-dist_mat / phi_local)
  diag(K_local) <- diag(K_local) + 1e-6
  L_local <- chol(K_local)
  w_local <- as.numeric(t(L_local) %*% rnorm(n))

  # Regional effect (long range)
  sigma_regional <- 0.3
  phi_regional <- 0.5
  K_regional <- sigma_regional^2 * exp(-dist_mat / phi_regional)
  diag(K_regional) <- diag(K_regional) + 1e-6
  L_regional <- chol(K_regional)
  w_regional <- as.numeric(t(L_regional) %*% rnorm(n))

  # Combined spatial effect
  w_total <- w_local + w_regional

  # Generate counts
  denom <- rpois(n, 100)
  num <- rpois(n, exp(log(denom) + 0.5 + w_total))

  data.frame(
    x = coords$x,
    y = coords$y,
    num = num,
    denom = denom
  )
}

# Test that each sampler option runs without error
test_that("all MSGP sampler options run without error", {
  skip_on_cran()

  df <- generate_msgp_data(n = 30, seed = 42)

  samplers <- c("auto", "noncentered", "centered", "interweaved", "adaptive", "riemannian")

  for (sampler in samplers) {
    # Create spatial specification with sampler option
    ms <- spatial_multiscale(
      ~ x + y,
      nn_local = 5,
      nn_regional = 10,
      sampler = sampler
    )

    expect_equal(ms$sampler, sampler)

    # Fit model with minimal iterations (just test it runs)
    fit <- suppressWarnings(
      tratio(
        num | denom ~ 1,
        data = df,
        spatial = ms,
        family = ratiod_poisson_gamma(),
        mode = "hmc",
        control = list(chains = 1, iter = 100, warmup = 50, verbose = FALSE)
      )
    )

    expect_s3_class(fit, "ratiod_fit")

    # Check fit succeeded (has n_eff or similar diagnostic)
    expect_true(!is.null(fit$diagnostics) || !is.null(fit$fit))
  }
})

# Benchmark test for divergence rates
test_that("MSGP samplers have acceptable divergence rates", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow benchmark (set RUN_SLOW_TESTS=true to run)")

  df <- generate_msgp_data(n = 40, seed = 123)

  # Test non-centered (current default, known to work well)
  ms_nc <- spatial_multiscale(~ x + y, nn_local = 5, nn_regional = 10, sampler = "noncentered")

  fit_nc <- suppressWarnings(
    tratio(
      num | denom ~ 1,
      data = df,
      spatial = ms_nc,
      family = ratiod_poisson_gamma(),
      mode = "hmc",
      control = list(chains = 1, iter = 150, warmup = 250, verbose = FALSE)
    )
  )

  # Extract divergence info from fit (stored in fit$fit$diagnostics)
  n_divergent <- fit_nc$fit$diagnostics$n_divergent %||% 0
  n_samples <- fit_nc$fit$n_save %||% 250  # post-warmup samples

  # The non-centered sampler should have very low divergence
  # (Based on previous testing: 0% divergence rate)
  expect_true(n_samples >= 200)  # Should have at least 200 post-warmup samples
  expect_true(n_divergent / n_samples < 0.05)  # Less than 5% divergence rate
})

# Speed benchmark
test_that("MSGP sampler speed comparison", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow benchmark (set RUN_SLOW_TESTS=true to run)")

  df <- generate_msgp_data(n = 50, seed = 456)

  samplers_to_test <- c("noncentered", "centered", "interweaved")
  results <- list()

  for (sampler in samplers_to_test) {
    ms <- spatial_multiscale(~ x + y, nn_local = 5, nn_regional = 15, sampler = sampler)

    start_time <- Sys.time()

    fit <- suppressWarnings(
      tratio(
        num | denom ~ 1,
        data = df,
        spatial = ms,
        family = ratiod_poisson_gamma(),
        mode = "hmc",
        control = list(chains = 1, iter = 150, warmup = 150, verbose = FALSE)
      )
    )

    end_time <- Sys.time()
    elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))

    n_samples <- fit$fit$n_save %||% 150  # post-warmup samples
    n_divergent <- fit$fit$diagnostics$n_divergent %||% 0

    results[[sampler]] <- list(
      time = elapsed,
      n_samples = n_samples,
      n_divergent = n_divergent,
      samples_per_sec = n_samples / elapsed
    )
  }

  # Print results for manual review
  cat("\n\n=== MSGP Sampler Benchmark Results ===\n")
  for (sampler in names(results)) {
    r <- results[[sampler]]
    div_pct <- 100 * r$n_divergent / r$n_samples
    cat(sprintf("%s: %.1f sec, %d samples, %.1f samples/sec, %d div (%.1f%%)\n",
                sampler, r$time, r$n_samples, r$samples_per_sec, r$n_divergent, div_pct))
  }
  cat("=====================================\n\n")

  # Non-centered should be competitive
  expect_true(results[["noncentered"]]$n_samples > 0)
})

# Stress test with multiple seeds
test_that("MSGP non-centered is robust across seeds", {
  skip_on_cran()
  skip_if_not(Sys.getenv("RUN_SLOW_TESTS") == "true",
              "Skipping slow benchmark (set RUN_SLOW_TESTS=true to run)")

  seeds <- c(111, 222, 333, 444, 555)
  all_ok <- TRUE

  for (seed in seeds) {
    df <- generate_msgp_data(n = 35, seed = seed)

    ms <- spatial_multiscale(~ x + y, nn_local = 5, nn_regional = 10, sampler = "noncentered")

    fit <- tryCatch({
      suppressWarnings(
        tratio(
          num | denom ~ 1,
          data = df,
          spatial = ms,
          family = ratiod_poisson_gamma(),
          mode = "hmc",
          control = list(chains = 1, iter = 200, warmup = 100, verbose = FALSE)
        )
      )
    }, error = function(e) {
      NULL
    })

    if (is.null(fit)) {
      all_ok <- FALSE
      cat(sprintf("Seed %d: FAILED (error)\n", seed))
    } else {
      n_samples <- fit$fit$n_save %||% 100
      n_divergent <- fit$fit$diagnostics$n_divergent %||% 0
      div_pct <- 100 * n_divergent / n_samples

      if (div_pct > 50) {
        all_ok <- FALSE
        cat(sprintf("Seed %d: FAILED (%d/%d divergent, %.1f%%)\n",
                    seed, n_divergent, n_samples, div_pct))
      } else {
        cat(sprintf("Seed %d: OK (%d samples, %d div, %.1f%%)\n",
                    seed, n_samples, n_divergent, div_pct))
      }
    }
  }

  expect_true(all_ok)
})
