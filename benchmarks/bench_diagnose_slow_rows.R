# Diagnostic benchmark: WHY are rows 14, 27, 38 slow?
# Measures tree depth, leapfrog counts, and per-phase timing

library(numdenom)

set.seed(42)

cat("\n", strrep("=", 70), "\n")
cat("DIAGNOSTIC: Tree depth & leapfrog profiling\n")
cat(strrep("=", 70), "\n\n")

N_OBS <- 500
N_ITER <- 500
N_WARMUP <- 250
N_CHAINS <- 1
N_SITES <- 50
N_TIMES <- 20

site <- factor(rep(1:N_SITES, length.out = N_OBS))
time_idx <- rep(1:N_TIMES, length.out = N_OBS)
x <- rnorm(N_OBS)

eta_num <- 2.0 + 0.3 * x
eta_denom <- 1.5 + 0.2 * x

diagnose <- function(label, fit, elapsed) {
  cat(sprintf("\n--- %s (%.1fs) ---\n", label, elapsed))

  # Tree depth
  td <- fit$diagnostics$treedepth
  if (!is.null(td)) {
    cat(sprintf("  Tree depth:  mean=%.1f, median=%d, max=%d\n",
                mean(td), median(td), max(td)))
    cat(sprintf("  Tree depth distribution: "))
    tt <- table(td)
    for (i in seq_along(tt)) {
      cat(sprintf("%s:%d ", names(tt)[i], tt[i]))
    }
    cat("\n")
    # % at max tree depth
    max_td <- max(td)
    pct_max <- 100 * sum(td == max_td) / length(td)
    cat(sprintf("  At max depth (%d): %.1f%%\n", max_td, pct_max))
  }

  # Leapfrog steps
  nlf <- fit$diagnostics$n_leapfrog
  if (!is.null(nlf)) {
    cat(sprintf("  Leapfrog/iter: mean=%.0f, median=%.0f, max=%d, total=%d\n",
                mean(nlf), median(nlf), max(nlf), sum(nlf)))
    cat(sprintf("  Time per 1000 LF: %.2fs\n", 1000 * elapsed / sum(nlf)))
  }

  # Divergences
  div <- fit$diagnostics$divergent
  if (is.numeric(div) && length(div) == 1) {
    cat(sprintf("  Divergences: %d\n", div))
  } else if (!is.null(div)) {
    cat(sprintf("  Divergences: %d\n", sum(div)))
  }

  # Step size
  eps <- fit$diagnostics$epsilon
  if (!is.null(eps)) {
    cat(sprintf("  Step size (epsilon): %.4f\n", eps))
  }

  # Effective samples
  summ <- summary(fit)
  cat(sprintf("  beta_num intercept: %.3f (true: 2.0)\n", summ$fixed[1, "mean"]))
}

# =============================================================================
# Row 2: pg + RE (FAST baseline for comparison)
# =============================================================================
cat("\n========== Row 2: pg + RE (fast baseline) ==========\n")
df_2 <- data.frame(
  y = rpois(N_OBS, exp(eta_num)),
  effort = rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_denom)),
  x = x, site = site
)

t_2 <- system.time({
  fit_2 <- ratiod(
    y | effort ~ x + (1 | site), data = df_2,
    family = ratiod_poisson_gamma(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    gradient_mode = "H", verbose = TRUE, refresh = 0
  )
})["elapsed"]
diagnose("Row 2: pg+RE", fit_2, t_2)

# =============================================================================
# Row 27: pg + RE + TVC (SLOW)
# =============================================================================
cat("\n========== Row 27: pg + RE + TVC ==========\n")
df_27 <- data.frame(
  y = rpois(N_OBS, exp(eta_num)),
  effort = rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_denom)),
  x = x, site = site, time = factor(time_idx)
)

t_27 <- system.time({
  fit_27 <- ratiod(
    y | effort ~ x + (1 | site), data = df_27,
    family = ratiod_poisson_gamma(),
    temporal = temporal_tvc(time_var = "time", terms = "x"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    gradient_mode = "H", verbose = TRUE, refresh = 0
  )
})["elapsed"]
diagnose("Row 27: pg+RE+TVC", fit_27, t_27)

# =============================================================================
# Row 38: nb + RE + HSGP (SLOW)
# =============================================================================
cat("\n========== Row 38: nb + RE + HSGP ==========\n")
n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = seq(0, 1, length.out = n_side),
                    lat = seq(0, 1, length.out = n_side))[1:N_SITES, ]
site_int <- as.integer(site)
lon <- grid$lon[site_int]
lat <- grid$lat[site_int]

eta_38n <- 2.0 + 0.3 * x
eta_38d <- 4.0 + 0.2 * x

y_38n <- rnbinom(N_OBS, mu = exp(eta_38n), size = 5)
y_38d <- rnbinom(N_OBS, mu = exp(eta_38d), size = 10)
y_38d[y_38d == 0] <- 1L

df_38 <- data.frame(
  y = y_38n, denom = y_38d,
  x = x, site = site, lon = lon, lat = lat
)

t_38 <- system.time({
  fit_38 <- ratiod(
    y | denom ~ x + (1 | site), data = df_38,
    family = ratiod_negbin_negbin(),
    spatial = spatial_hsgp(coords = c("lon", "lat"), m = 6),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    gradient_mode = "H", verbose = TRUE, refresh = 0
  )
})["elapsed"]
diagnose("Row 38: nb+RE+HSGP", fit_38, t_38)

# =============================================================================
# Row 14: pg + RE + temporal GP (SLOW)
# =============================================================================
cat("\n========== Row 14: pg + RE + temporal GP ==========\n")
true_sigma_gp <- 0.5
true_phi_gp <- 2.0
dist_mat <- as.matrix(dist(1:N_TIMES))
K <- true_sigma_gp^2 * exp(-dist_mat / true_phi_gp)
gp_effects <- MASS::mvrnorm(1, rep(0, N_TIMES), K + diag(1e-6, N_TIMES))
gp_by_obs <- gp_effects[time_idx]

eta_14n <- 2.0 + 0.3 * x + gp_by_obs
eta_14d <- 1.5 + 0.2 * x + gp_by_obs

df_14 <- data.frame(
  y = rpois(N_OBS, exp(eta_14n)),
  effort = rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_14d)),
  x = x, site = site, time = time_idx
)
df_14$effort[df_14$effort < 0.01] <- 0.01

t_14 <- system.time({
  fit_14 <- ratiod(
    y | effort ~ x + (1 | site), data = df_14,
    family = ratiod_poisson_gamma(),
    temporal = temporal_gp(time_var = "time", cov = "exponential"),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    gradient_mode = "H", verbose = TRUE, refresh = 0
  )
})["elapsed"]
diagnose("Row 14: pg+RE+temporal GP", fit_14, t_14)

# =============================================================================
# COMPARISON
# =============================================================================
cat("\n", strrep("=", 70), "\n")
cat("COMPARISON\n")
cat(strrep("=", 70), "\n\n")

cat(sprintf("%-30s %8s %10s %12s %12s\n",
            "Model", "Time", "Mean TD", "Mean LF", "ms/LF"))
cat(strrep("-", 75), "\n")

for (info in list(
  list("Row 2: pg+RE", fit_2, t_2),
  list("Row 27: pg+RE+TVC", fit_27, t_27),
  list("Row 38: nb+RE+HSGP", fit_38, t_38),
  list("Row 14: pg+RE+temporal GP", fit_14, t_14)
)) {
  td <- info[[2]]$diagnostics$treedepth
  nlf <- info[[2]]$diagnostics$n_leapfrog
  mean_td <- if (!is.null(td)) sprintf("%.1f", mean(td)) else "?"
  mean_lf <- if (!is.null(nlf)) sprintf("%.0f", mean(nlf)) else "?"
  ms_lf <- if (!is.null(nlf) && sum(nlf) > 0) {
    sprintf("%.2f", 1000 * info[[3]] / sum(nlf))
  } else "?"
  cat(sprintf("%-30s %7.1fs %10s %12s %12s\n",
              info[[1]], info[[3]], mean_td, mean_lf, ms_lf))
}
