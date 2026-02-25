# Batch 1: gamma_gamma and lognormal base models (rows 93, 94, 98, 99)
# Goal: 100% validation coverage

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
cat("100% Validation - Batch 1: gamma_gamma & lognormal base\n")
cat("=======================================================\n\n")

compare_params <- function(nd_draws, stan_draws, params, threshold = 2) {
  results <- list()
  for (p in params) {
    nd_name <- p$nd
    stan_name <- p$stan
    label <- p$label

    nd_vals <- nd_draws[, nd_name]
    stan_vals <- stan_draws[[stan_name]]
    nd_mean <- mean(nd_vals); nd_sd <- sd(nd_vals)
    stan_mean <- mean(stan_vals); stan_sd <- sd(stan_vals)
    se <- sqrt(nd_sd^2 / length(nd_vals) + stan_sd^2 / length(stan_vals))
    diff <- abs(nd_mean - stan_mean)
    ratio <- diff / se
    pass <- ratio < threshold
    cat(sprintf("  %s: nd=%.4f (%.4f), stan=%.4f (%.4f), diff=%.2fSE => %s\n",
                label, nd_mean, nd_sd, stan_mean, stan_sd, ratio,
                if(pass) "PASS" else "FAIL"))
    results[[label]] <- list(pass = pass, ratio = ratio)
  }
  results
}

# Setup common data
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)

eta_num <- 2 + 0.3 * x
eta_denom <- 4 + 0.2 * x

results_all <- list()

# =============================================================================
# Row 93: gamma_gamma base (no RE)
# =============================================================================
cat("\n========== Row 93: gamma_gamma (no RE) ==========\n")

df <- data.frame(
  y = rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_num)),
  denom = rgamma(N_OBS, shape = 10, rate = 10 / exp(eta_denom)),
  x = x
)
df$y[df$y < 0.01] <- 0.01
df$denom[df$denom < 0.01] <- 0.01

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x, data = df, family = ratiod_gamma_gamma(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))
draws_nd <- as.matrix(fit_nd$draws)

cat("Fitting Stan... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_gg_base.stan")
stan_data <- list(
  N = N_OBS, y_num = df$y, y_denom = df$denom,
  p = 2, X = cbind(1, df$x)
)
t_stan <- system.time({
  fit_stan <- stan_model$sample(
    data = stan_data, iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
    chains = N_CHAINS, parallel_chains = N_CHAINS, refresh = 0, show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))
draws_stan <- fit_stan$draws(format = "df")

cat("\nComparison:\n")
params <- list(
  list(nd = "beta_num[1]", stan = "beta_num[1]", label = "beta_num[1]"),
  list(nd = "beta_num[2]", stan = "beta_num[2]", label = "beta_num[2]"),
  list(nd = "beta_denom[1]", stan = "beta_denom[1]", label = "beta_denom[1]"),
  list(nd = "beta_denom[2]", stan = "beta_denom[2]", label = "beta_denom[2]")
)
r93 <- compare_params(draws_nd, draws_stan, params)
results_all$row_93 <- list(
  pass = all(sapply(r93, function(x) x$pass)),
  time_nd = t_nd, time_stan = t_stan
)

# =============================================================================
# Row 94: gamma_gamma with RE
# =============================================================================
cat("\n========== Row 94: gamma_gamma + RE ==========\n")

df <- data.frame(
  y = rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_num)),
  denom = rgamma(N_OBS, shape = 10, rate = 10 / exp(eta_denom)),
  x = x, site = site
)
df$y[df$y < 0.01] <- 0.01
df$denom[df$denom < 0.01] <- 0.01

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x + (1|site), data = df, family = ratiod_gamma_gamma(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))
draws_nd <- as.matrix(fit_nd$draws)

cat("Fitting Stan... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_gg_re.stan")
stan_data <- list(
  N = N_OBS, y_num = df$y, y_denom = df$denom,
  p = 2, X = cbind(1, df$x),
  n_groups = N_SITES, group_idx = as.integer(df$site)
)
t_stan <- system.time({
  fit_stan <- stan_model$sample(
    data = stan_data, iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
    chains = N_CHAINS, parallel_chains = N_CHAINS, refresh = 0, show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))
draws_stan <- fit_stan$draws(format = "df")

cat("\nComparison:\n")
r94 <- compare_params(draws_nd, draws_stan, params)
results_all$row_94 <- list(
  pass = all(sapply(r94, function(x) x$pass)),
  time_nd = t_nd, time_stan = t_stan
)

# =============================================================================
# Row 98: lognormal base (no RE)
# =============================================================================
cat("\n========== Row 98: lognormal (no RE) ==========\n")

df <- data.frame(
  y = rlnorm(N_OBS, meanlog = eta_num, sdlog = 0.5),
  denom = rlnorm(N_OBS, meanlog = eta_denom, sdlog = 0.3),
  x = x
)

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x, data = df, family = ratiod_lognormal(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))
draws_nd <- as.matrix(fit_nd$draws)

cat("Fitting Stan... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_ln_base.stan")
stan_data <- list(
  N = N_OBS, y_num = df$y, y_denom = df$denom,
  p = 2, X = cbind(1, df$x)
)
t_stan <- system.time({
  fit_stan <- stan_model$sample(
    data = stan_data, iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
    chains = N_CHAINS, parallel_chains = N_CHAINS, refresh = 0, show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))
draws_stan <- fit_stan$draws(format = "df")

cat("\nComparison:\n")
r98 <- compare_params(draws_nd, draws_stan, params)
results_all$row_98 <- list(
  pass = all(sapply(r98, function(x) x$pass)),
  time_nd = t_nd, time_stan = t_stan
)

# =============================================================================
# Row 99: lognormal with RE
# =============================================================================
cat("\n========== Row 99: lognormal + RE ==========\n")

df <- data.frame(
  y = rlnorm(N_OBS, meanlog = eta_num, sdlog = 0.5),
  denom = rlnorm(N_OBS, meanlog = eta_denom, sdlog = 0.3),
  x = x, site = site
)

cat("Fitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x + (1|site), data = df, family = ratiod_lognormal(),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, verbose = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_nd))
draws_nd <- as.matrix(fit_nd$draws)

cat("Fitting Stan... ")
stan_model <- cmdstan_model("benchmarks/stan/joint_ln_re.stan")
stan_data <- list(
  N = N_OBS, y_num = df$y, y_denom = df$denom,
  p = 2, X = cbind(1, df$x),
  n_groups = N_SITES, group_idx = as.integer(df$site)
)
t_stan <- system.time({
  fit_stan <- stan_model$sample(
    data = stan_data, iter_sampling = N_ITER - N_WARMUP, iter_warmup = N_WARMUP,
    chains = N_CHAINS, parallel_chains = N_CHAINS, refresh = 0, show_messages = FALSE
  )
})["elapsed"]
cat(sprintf("%.1fs\n", t_stan))
draws_stan <- fit_stan$draws(format = "df")

cat("\nComparison:\n")
r99 <- compare_params(draws_nd, draws_stan, params)
results_all$row_99 <- list(
  pass = all(sapply(r99, function(x) x$pass)),
  time_nd = t_nd, time_stan = t_stan
)

# =============================================================================
# Summary
# =============================================================================
cat("\n=======================================================\n")
cat("SUMMARY - Batch 1\n")
cat("=======================================================\n\n")

for (row in names(results_all)) {
  r <- results_all[[row]]
  cat(sprintf("%s: %s (nd=%.1fs, stan=%.1fs)\n",
              row, if(r$pass) "PASS" else "FAIL", r$time_nd, r$time_stan))
}

passed <- sum(sapply(results_all, function(x) x$pass))
total <- length(results_all)
cat(sprintf("\nTotal: %d/%d passed\n", passed, total))
