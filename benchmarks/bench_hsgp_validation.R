# HSGP Validation - Rows 8, 38 (poisson_gamma and negbin_negbin with HSGP spatial)
# Quick validation script for HSGP models only

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

N_OBS <- 200
N_ITER <- 2000
N_WARMUP <- 1000
N_CHAINS <- 2
N_SITES <- 20

cat("=======================================================\n")
cat("HSGP Validation: Rows 8, 38\n")
cat("=======================================================\n\n")

compare_posteriors <- function(nd_draws, stan_draws, param_name, threshold_se = 2) {
  nd_mean <- mean(nd_draws)
  nd_sd <- sd(nd_draws)
  stan_mean <- mean(stan_draws)
  stan_sd <- sd(stan_draws)

  se_combined <- sqrt(nd_sd^2 / length(nd_draws) + stan_sd^2 / length(stan_draws))
  diff <- abs(nd_mean - stan_mean)
  ratio <- diff / se_combined
  pass <- ratio < threshold_se

  list(
    param = param_name,
    nd_mean = nd_mean, nd_sd = nd_sd,
    stan_mean = stan_mean, stan_sd = stan_sd,
    diff = diff, ratio = ratio, pass = pass
  )
}

print_result <- function(result) {
  cat(sprintf("  %s: nd=%.4f (SD=%.4f), Stan=%.4f (SD=%.4f), diff=%.4f (%.2f SE) => %s\n",
              result$param, result$nd_mean, result$nd_sd,
              result$stan_mean, result$stan_sd,
              result$diff, result$ratio, if(result$pass) "PASS" else "FAIL"))
}

# Setup data
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)

# Spatial coordinates
n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = seq(0, 1, length.out = n_side),
                    lat = seq(0, 1, length.out = n_side))[1:N_SITES, ]
site_int <- as.integer(site)
lon <- grid$lon[site_int]
lat <- grid$lat[site_int]

eta_num <- 2 + 0.3 * x
eta_denom <- 4 + 0.2 * x

results <- list()

# =============================================================================
# Row 38: negbin_negbin + HSGP
# =============================================================================
cat("\n========== Row 38: negbin_negbin + HSGP ==========\n")

df_nb_hsgp <- data.frame(
  y = rnbinom(N_OBS, mu = exp(eta_num), size = 5),
  denom = rnbinom(N_OBS, mu = exp(eta_denom), size = 10),
  x = x, lon = lon, lat = lat
)
df_nb_hsgp$denom[df_nb_hsgp$denom == 0] <- 1

cat("Fitting numdenom (HSGP)... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x, data = df_nb_hsgp,
    family = ratiod_negbin_negbin(),
    spatial = spatial_hsgp(coords = c("lon", "lat"), m = 6),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_nd))

cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/hsgp_nb_joint.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS, M = 6L,
  y_num = df_nb_hsgp$y, y_denom = df_nb_hsgp$denom,
  x = df_nb_hsgp$x,
  coords = cbind(df_nb_hsgp$lon, df_nb_hsgp$lat),
  c = 1.5,
  sigma2_prior_U = 1.0, sigma2_prior_alpha = 0.01,
  phi_prior_lower = 0.01, phi_prior_upper = 100.0
)

cat("Fitting Stan model... ")
t_stan <- system.time({
  fit_stan <- stan_model$sample(
    data = stan_data,
    iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
    chains = N_CHAINS, parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE, adapt_delta = 0.9
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_stan))

draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

nd_beta_col <- grep("^beta_num\\[2\\]$", colnames(draws_nd), value = TRUE)[1]
results$row_38 <- list(
  beta_num_1 = compare_posteriors(
    draws_nd[, nd_beta_col], draws_stan$beta_num_1, "beta_num[2] (slope)"
  ),
  time_nd = t_nd, time_stan = t_stan
)
print_result(results$row_38$beta_num_1)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd, t_stan, t_stan/t_nd))

# =============================================================================
# Row 8: poisson_gamma + HSGP
# =============================================================================
cat("\n========== Row 8: poisson_gamma + HSGP ==========\n")

df_pg_hsgp <- data.frame(
  y = rpois(N_OBS, exp(eta_num)),
  denom = rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_denom)),
  x = x, lon = lon, lat = lat
)
df_pg_hsgp$denom[df_pg_hsgp$denom < 0.01] <- 0.01

cat("Fitting numdenom (HSGP)... ")
t_nd_pg <- system.time({
  fit_nd_pg <- ratiod(
    y | denom ~ x, data = df_pg_hsgp,
    family = ratiod_poisson_gamma(),
    spatial = spatial_hsgp(coords = c("lon", "lat"), m = 6),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_nd_pg))

cat("Compiling Stan model... ")
stan_model_pg <- cmdstan_model("benchmarks/stan/hsgp_pg_joint.stan")
cat("done\n")

stan_data_pg <- list(
  N = N_OBS, M = 6L,
  y_num = df_pg_hsgp$y, y_denom = df_pg_hsgp$denom,
  x = df_pg_hsgp$x,
  coords = cbind(df_pg_hsgp$lon, df_pg_hsgp$lat),
  c = 1.5,
  sigma2_prior_U = 1.0, sigma2_prior_alpha = 0.01,
  phi_prior_lower = 0.01, phi_prior_upper = 100.0
)

cat("Fitting Stan model... ")
t_stan_pg <- system.time({
  fit_stan_pg <- stan_model_pg$sample(
    data = stan_data_pg,
    iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
    chains = N_CHAINS, parallel_chains = N_CHAINS,
    refresh = 0, show_messages = FALSE, adapt_delta = 0.9
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_stan_pg))

draws_nd_pg <- as.matrix(fit_nd_pg$draws)
draws_stan_pg <- fit_stan_pg$draws(format = "df")

nd_beta_col_pg <- grep("^beta_num\\[2\\]$", colnames(draws_nd_pg), value = TRUE)[1]
results$row_8 <- list(
  beta_num_1 = compare_posteriors(
    draws_nd_pg[, nd_beta_col_pg], draws_stan_pg$beta_num_1, "beta_num[2] (slope)"
  ),
  time_nd = t_nd_pg, time_stan = t_stan_pg
)
print_result(results$row_8$beta_num_1)
cat(sprintf("  Times: nd=%.1fs, Stan=%.1fs, speedup=%.1fx\n", t_nd_pg, t_stan_pg, t_stan_pg/t_nd_pg))

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - HSGP Validation (Rows 8, 38)\n")
cat("=======================================================\n\n")

cat(sprintf("%-8s %-30s %10s %10s %8s %s\n",
            "Row", "Model", "numdenom", "Stan", "Speedup", "Status"))
cat(paste(rep("-", 80), collapse = ""), "\n")

r38 <- results$row_38
cat(sprintf("%-8s %-30s %10.1fs %10.1fs %8.1fx %s\n",
            "38", "nb + HSGP", r38$time_nd, r38$time_stan, r38$time_stan / r38$time_nd,
            if(r38$beta_num_1$pass) "PASS" else "FAIL"))

r8 <- results$row_8
cat(sprintf("%-8s %-30s %10.1fs %10.1fs %8.1fx %s\n",
            "8", "pg + HSGP", r8$time_nd, r8$time_stan, r8$time_stan / r8$time_nd,
            if(r8$beta_num_1$pass) "PASS" else "FAIL"))

cat(paste(rep("-", 80), collapse = ""), "\n")

saveRDS(results, "benchmarks/results_hsgp_validation.rds")
cat("\nResults saved to benchmarks/results_hsgp_validation.rds\n")

cat("\n\nUPDATE gradient_methods.md:\n")
if (r38$beta_num_1$pass) cat("  Row 38: PASS - add '✓Stan (joint)' to Notes\n")
if (r8$beta_num_1$pass) cat("  Row 8: PASS - add '✓Stan (joint)' to Notes\n")
