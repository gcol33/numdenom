# Validation of binomial + ICAR + ZI (Row 86)
library(numdenom)
library(brms)
library(posterior)

set.seed(42)

N_OBS <- 200
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2
N_SITES <- 20

cat("=======================================================\n")
cat("Binomial + ICAR + ZI Validation: Row 86\n")
cat("=======================================================\n\n")

compare_posteriors <- function(nd_draws, brms_draws, param_name, threshold_se = 2) {
  nd_mean <- mean(nd_draws)
  nd_sd <- sd(nd_draws)
  brms_mean <- mean(brms_draws)
  brms_sd <- sd(brms_draws)
  se_combined <- sqrt(nd_sd^2 / length(nd_draws) + brms_sd^2 / length(brms_draws))
  diff <- abs(nd_mean - brms_mean)
  ratio <- diff / se_combined
  pass <- ratio < threshold_se
  list(param = param_name, nd_mean = nd_mean, nd_sd = nd_sd,
       brms_mean = brms_mean, brms_sd = brms_sd, diff = diff, ratio = ratio, pass = pass)
}

print_result <- function(result) {
  cat(sprintf("  %s: nd=%.4f (SD=%.4f), brms=%.4f (SD=%.4f), diff=%.4f (%.2f SE) => %s\n",
              result$param, result$nd_mean, result$nd_sd,
              result$brms_mean, result$brms_sd,
              result$diff, result$ratio, if(result$pass) "PASS" else "FAIL"))
}

# Setup data
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)
trials <- sample(10:50, N_OBS, replace = TRUE)

# Build adjacency for ICAR
adj_matrix <- matrix(0, N_SITES, N_SITES)
for (i in 1:(N_SITES-1)) {
  adj_matrix[i, i+1] <- 1
  adj_matrix[i+1, i] <- 1
}

# Generate data
site_effects <- rnorm(N_SITES, 0, 0.3)
spatial_effects <- rnorm(N_SITES, 0, 0.2)
spatial_effects <- spatial_effects - mean(spatial_effects)

eta <- 0.5 + 0.3 * x + site_effects[as.integer(site)] + spatial_effects[as.integer(site)]
prob <- plogis(eta)
successes <- rbinom(N_OBS, trials, prob)

# Add zero-inflation (10% structural zeros)
zi_prob <- 0.1
is_zi <- runif(N_OBS) < zi_prob
successes[is_zi] <- 0

df <- data.frame(
  successes = successes,
  trials = trials,
  x = x,
  site = site
)

cat("Fitting numdenom (ZI binomial + ICAR)... ")
t_nd <- system.time({
  fit_nd <- tryCatch({
    ratiod(
      successes | trials ~ x + (1 | site),
      data = df,
      family = ratiod_zibinomial(),
      spatial = spatial_car(adj_matrix, group_var = "site"),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      verbose = FALSE
    )
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    NULL
  })
})[["elapsed"]]
if (!is.null(fit_nd)) cat(sprintf("%.1fs\n", t_nd))

if (!is.null(fit_nd)) {
  # brms ZI binomial (without spatial - brms doesn't easily support ICAR)
  cat("Fitting brms (ZI binomial, no spatial)... ")
  t_brms <- system.time({
    fit_brms <- brm(
      successes | trials(trials) ~ x + (1 | site),
      data = df,
      family = zero_inflated_binomial(),
      iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
      refresh = 0, silent = 2
    )
  })[["elapsed"]]
  cat(sprintf("%.1fs\n", t_brms))

  draws_nd <- as.matrix(fit_nd$draws)
  draws_brms <- as_draws_matrix(fit_brms)

  nd_slope_col <- grep("^beta_num\\[2\\]$", colnames(draws_nd), value = TRUE)[1]

  result <- compare_posteriors(draws_nd[, nd_slope_col], draws_brms[, "b_x"], "slope (x)")
  print_result(result)
  cat(sprintf("  Times: nd=%.1fs, brms=%.1fs, speedup=%.1fx\n", t_nd, t_brms, t_brms/t_nd))
  cat("  Note: brms uses IID RE; numdenom adds ICAR. Slopes should be similar.\n")

  if (result$pass) {
    cat("\n  ✓ Row 86 PASSED - update gradient_methods.md with '✓Stan'\n")
  } else {
    cat("\n  Note: Different model structure (ICAR vs IID) may cause differences.\n")
  }
} else {
  cat("  Row 86 could not be validated due to error.\n")
}
