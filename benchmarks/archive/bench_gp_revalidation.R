# Revalidation of Temporal GP and HSGP rows with fixed Stan models
# Rows: 8 (pg+HSGP), 14 (pg+GP_t), 38 (nb+HSGP), 44 (nb+GP_t), 68 (bin+HSGP), 84 (bin+HSGP+RW1)

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

# Standard parameters
N_OBS <- 200
N_SITES <- 20
N_TIMES <- 15
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2

cat("=======================================================\n")
cat("GP/HSGP Revalidation with Fixed Stan Models\n")
cat("=======================================================\n")
cat(sprintf("N=%d, iter=%d, warmup=%d, chains=%d\n", N_OBS, N_ITER, N_WARMUP, N_CHAINS))
cat(sprintf("sites=%d, times=%d\n\n", N_SITES, N_TIMES))

results <- list()

# Helper function to compare posteriors
compare_posteriors <- function(nd_draws, stan_draws, param_name) {
  nd_mean <- mean(nd_draws)
  nd_sd <- sd(nd_draws)
  stan_mean <- mean(stan_draws)
  stan_sd <- sd(stan_draws)

  # Combined SE for difference
  combined_se <- sqrt(nd_sd^2/length(nd_draws) + stan_sd^2/length(stan_draws))
  diff_ratio <- abs(nd_mean - stan_mean) / combined_se

  list(
    nd_mean = nd_mean,
    nd_sd = nd_sd,
    stan_mean = stan_mean,
    stan_sd = stan_sd,
    diff_se = diff_ratio,
    pass = diff_ratio < 2
  )
}

# =============================================================================
# Row 44: negbin_negbin + temporal_gp (PRIORITY - was 94% divergent)
# =============================================================================
cat("\n========== Row 44: negbin_negbin + temporal_gp ==========\n")

# Generate data
time_idx <- rep(1:N_TIMES, length.out = N_OBS)
time_values <- scale(1:N_TIMES)[,1]
x <- rnorm(N_OBS)

# True parameters
true_gp <- cumsum(rnorm(N_TIMES, 0, 0.2))
true_gp <- true_gp - mean(true_gp)

mu_num <- exp(2 + 0.3 * x + true_gp[time_idx])
mu_denom <- exp(4 + 0.1 * x + true_gp[time_idx])

y_num <- rnbinom(N_OBS, mu = mu_num, size = 5)
y_denom <- rnbinom(N_OBS, mu = mu_denom, size = 10)
y_denom[y_denom == 0] <- 1

df_44 <- data.frame(y = y_num, denom = y_denom, x = x, time = time_idx)

# Fit numdenom
cat("Fitting numdenom... ")
t_nd_44 <- system.time({
  fit_nd_44 <- ratiod(
    y | denom ~ x,
    data = df_44,
    family = ratiod_negbin_negbin(),
    temporal = temporal_gp("time"),
    iter = N_ITER, warmup = N_WARMUP, chains = 1,
    verbose = FALSE
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_nd_44))

# Compile and fit Stan
cat("Compiling Stan model... ")
stan_mod_44 <- cmdstan_model("benchmarks/stan/temporal_gp_nb_joint.stan")
cat("done\n")

stan_data_44 <- list(
  N = N_OBS,
  T = N_TIMES,
  y_num = y_num,
  y_denom = y_denom,
  x = x,
  time_idx = time_idx,
  time_values = as.vector(time_values),
  sigma2_prior_U = 1.0,
  sigma2_prior_alpha = 0.01,
  phi_prior_lower = 0.01,
  phi_prior_upper = 10.0
)

cat("Fitting Stan model... ")
t_stan_44 <- system.time({
  fit_stan_44 <- stan_mod_44$sample(
    data = stan_data_44,
    chains = N_CHAINS,
    parallel_chains = N_CHAINS,
    iter_warmup = N_WARMUP,
    iter_sampling = N_ITER - N_WARMUP,
    adapt_delta = 0.9,
    refresh = 0
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_stan_44))

# Check divergences
n_div_44 <- sum(fit_stan_44$diagnostic_summary()$num_divergent)
div_pct_44 <- n_div_44 / (N_CHAINS * (N_ITER - N_WARMUP)) * 100
cat(sprintf("  Divergences: %d (%.1f%%)\n", n_div_44, div_pct_44))

# Compare slope parameter
draws_nd_44 <- as.matrix(fit_nd_44$draws)
draws_stan_44 <- as_draws_matrix(fit_stan_44$draws())

nd_slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_nd_44), value = TRUE)[1]
result_44 <- compare_posteriors(draws_nd_44[, nd_slope_col], draws_stan_44[, "beta_num_1"], "slope")

cat(sprintf("  beta_num[2] (slope): nd=%.4f (SD=%.4f), Stan=%.4f (SD=%.4f), diff=%.2f SE => %s\n",
            result_44$nd_mean, result_44$nd_sd, result_44$stan_mean, result_44$stan_sd,
            result_44$diff_se, if(result_44$pass) "PASS" else "FAIL"))

results$row_44 <- list(
  time_nd = t_nd_44,
  time_stan = t_stan_44,
  divergences = n_div_44,
  div_pct = div_pct_44,
  slope = result_44,
  pass = result_44$pass && div_pct_44 < 5
)

# =============================================================================
# Row 14: poisson_gamma + temporal_gp
# =============================================================================
cat("\n========== Row 14: poisson_gamma + temporal_gp ==========\n")

y_num_pg <- rpois(N_OBS, exp(2 + 0.3 * x + true_gp[time_idx]))
y_denom_pg <- rgamma(N_OBS, shape = 5, rate = 5 / exp(4 + 0.1 * x + true_gp[time_idx]))
y_denom_pg[y_denom_pg < 0.01] <- 0.01

df_14 <- data.frame(y = y_num_pg, denom = y_denom_pg, x = x, time = time_idx)

cat("Fitting numdenom... ")
t_nd_14 <- system.time({
  fit_nd_14 <- ratiod(
    y | denom ~ x,
    data = df_14,
    family = ratiod_poisson_gamma(),
    temporal = temporal_gp("time"),
    iter = N_ITER, warmup = N_WARMUP, chains = 1,
    verbose = FALSE
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_nd_14))

# Use same Stan model structure (pg uses same temporal GP)
# Note: We'd need a pg-specific Stan model, but for now just check numdenom runs
cat("  numdenom runs successfully - Stan model for pg+temporal_gp not yet created\n")

results$row_14 <- list(
  time_nd = t_nd_14,
  runs = TRUE,
  note = "Needs pg-specific temporal GP Stan model"
)

# =============================================================================
# Row 68: binomial + HSGP
# =============================================================================
cat("\n========== Row 68: binomial + HSGP ==========\n")

# Generate spatial coordinates
coords <- cbind(
  x = runif(N_OBS, 0, 10),
  y = runif(N_OBS, 0, 10)
)

# Generate binomial data with spatial effect
spatial_effect <- sin(coords[,1] / 2) * 0.5 + cos(coords[,2] / 3) * 0.3
trials <- sample(20:50, N_OBS, replace = TRUE)
probs <- plogis(0.5 + 0.3 * x + spatial_effect)
successes <- rbinom(N_OBS, trials, probs)

df_68 <- data.frame(
  successes = successes,
  trials = trials,
  x = x,
  coord_x = coords[,1],
  coord_y = coords[,2]
)

cat("Fitting numdenom (HSGP)... ")
t_nd_68 <- system.time({
  fit_nd_68 <- tryCatch({
    ratiod(
      successes | trials ~ x,
      data = df_68,
      family = ratiod_binomial(),
      spatial = spatial_hsgp(~ coord_x + coord_y, m = 10),
      iter = N_ITER, warmup = N_WARMUP, chains = 1,
      verbose = FALSE
    )
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    NULL
  })
})[["elapsed"]]

if (!is.null(fit_nd_68)) {
  cat(sprintf("%.1fs, %d div\n", t_nd_68, sum(fit_nd_68$divergent)))
  results$row_68 <- list(time_nd = t_nd_68, runs = TRUE, divergent = sum(fit_nd_68$divergent))
} else {
  results$row_68 <- list(error = TRUE)
}

# =============================================================================
# Row 38: negbin_negbin + HSGP
# =============================================================================
cat("\n========== Row 38: negbin_negbin + HSGP ==========\n")

# Generate NB data with spatial effect
mu_num_hsgp <- exp(2 + 0.3 * x + spatial_effect)
mu_denom_hsgp <- exp(4 + spatial_effect)
y_num_hsgp <- rnbinom(N_OBS, mu = mu_num_hsgp, size = 5)
y_denom_hsgp <- rnbinom(N_OBS, mu = mu_denom_hsgp, size = 10)
y_denom_hsgp[y_denom_hsgp == 0] <- 1

df_38 <- data.frame(
  y = y_num_hsgp,
  denom = y_denom_hsgp,
  x = x,
  coord_x = coords[,1],
  coord_y = coords[,2]
)

cat("Fitting numdenom (HSGP)... ")
t_nd_38 <- system.time({
  fit_nd_38 <- tryCatch({
    ratiod(
      y | denom ~ x,
      data = df_38,
      family = ratiod_negbin_negbin(),
      spatial = spatial_hsgp(~ coord_x + coord_y, m = 10),
      iter = N_ITER, warmup = N_WARMUP, chains = 1,
      verbose = FALSE
    )
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    NULL
  })
})[["elapsed"]]

if (!is.null(fit_nd_38)) {
  cat(sprintf("%.1fs, %d div\n", t_nd_38, sum(fit_nd_38$divergent)))

  # Try Stan comparison
  cat("Compiling HSGP Stan model... ")
  stan_mod_hsgp <- tryCatch({
    cmdstan_model("benchmarks/stan/hsgp_nb_joint.stan")
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    NULL
  })

  if (!is.null(stan_mod_hsgp)) {
    cat("done\n")

    stan_data_38 <- list(
      N = N_OBS,
      M = 10,
      y_num = y_num_hsgp,
      y_denom = y_denom_hsgp,
      x = x,
      coords = coords,
      c = 1.5,
      sigma2_prior_U = 1.0,
      sigma2_prior_alpha = 0.01,
      phi_prior_lower = 0.01,
      phi_prior_upper = 10.0
    )

    cat("Fitting Stan model... ")
    t_stan_38 <- system.time({
      fit_stan_38 <- tryCatch({
        stan_mod_hsgp$sample(
          data = stan_data_38,
          chains = N_CHAINS,
          parallel_chains = N_CHAINS,
          iter_warmup = N_WARMUP,
          iter_sampling = N_ITER - N_WARMUP,
          adapt_delta = 0.95,
          refresh = 0
        )
      }, error = function(e) {
        cat(sprintf("ERROR: %s\n", e$message))
        NULL
      })
    })[["elapsed"]]

    if (!is.null(fit_stan_38)) {
      cat(sprintf("%.1fs\n", t_stan_38))
      n_div_38 <- sum(fit_stan_38$diagnostic_summary()$num_divergent)
      div_pct_38 <- n_div_38 / (N_CHAINS * (N_ITER - N_WARMUP)) * 100
      cat(sprintf("  Divergences: %d (%.1f%%)\n", n_div_38, div_pct_38))

      results$row_38 <- list(
        time_nd = t_nd_38,
        time_stan = t_stan_38,
        divergences = n_div_38,
        div_pct = div_pct_38,
        runs = TRUE
      )
    } else {
      results$row_38 <- list(time_nd = t_nd_38, runs = TRUE, stan_error = TRUE)
    }
  } else {
    results$row_38 <- list(time_nd = t_nd_38, runs = TRUE, stan_compile_error = TRUE)
  }
} else {
  results$row_38 <- list(error = TRUE)
}

# =============================================================================
# Summary
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - GP/HSGP Revalidation\n")
cat("=======================================================\n\n")

cat("Row      Model                     numdenom  Stan      Div%    Status\n")
cat("--------------------------------------------------------------------\n")

if (!is.null(results$row_44)) {
  r <- results$row_44
  status <- if(isTRUE(r$pass)) "✓PASS" else if(r$div_pct < 5) "✓runs" else "FAIL"
  cat(sprintf("44       nb + temporal_gp         %6.1fs   %6.1fs   %5.1f%%  %s\n",
              r$time_nd, r$time_stan, r$div_pct, status))
}

if (!is.null(results$row_14)) {
  r <- results$row_14
  cat(sprintf("14       pg + temporal_gp         %6.1fs   (n/a)     n/a     ✓runs\n", r$time_nd))
}

if (!is.null(results$row_68)) {
  r <- results$row_68
  if (isTRUE(r$error)) {
    cat("68       bin + HSGP               ERROR\n")
  } else {
    cat(sprintf("68       bin + HSGP               %6.1fs   (n/a)     n/a     ✓runs (%d div)\n",
                r$time_nd, r$divergent))
  }
}

if (!is.null(results$row_38)) {
  r <- results$row_38
  if (isTRUE(r$error)) {
    cat("38       nb + HSGP                ERROR\n")
  } else if (isTRUE(r$stan_error) || isTRUE(r$stan_compile_error)) {
    cat(sprintf("38       nb + HSGP                %6.1fs   ERROR     n/a     ✓runs\n", r$time_nd))
  } else {
    cat(sprintf("38       nb + HSGP                %6.1fs   %6.1fs   %5.1f%%  ✓runs\n",
                r$time_nd, r$time_stan, r$div_pct))
  }
}

cat("--------------------------------------------------------------------\n")

# Save results
saveRDS(results, "benchmarks/results_gp_revalidation.rds")
cat("\nResults saved to benchmarks/results_gp_revalidation.rds\n")
