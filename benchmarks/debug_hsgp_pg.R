# Debug HSGP poisson_gamma validation failure
# Compares all parameters to identify source of discrepancy

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
cat("HSGP Debug: Row 8 (poisson_gamma)\n")
cat("=======================================================\n\n")

# Setup data
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)

n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = seq(0, 1, length.out = n_side),
                    lat = seq(0, 1, length.out = n_side))[1:N_SITES, ]
site_int <- as.integer(site)
lon <- grid$lon[site_int]
lat <- grid$lat[site_int]

eta_num <- 2 + 0.3 * x
eta_denom <- 4 + 0.2 * x

df <- data.frame(
  y = rpois(N_OBS, exp(eta_num)),
  denom = rgamma(N_OBS, shape = 5, rate = 5 / exp(eta_denom)),
  x = x, lon = lon, lat = lat
)
df$denom[df$denom < 0.01] <- 0.01

cat("Data summary:\n")
cat(sprintf("  y: mean=%.2f, sd=%.2f, range=[%d, %d]\n",
            mean(df$y), sd(df$y), min(df$y), max(df$y)))
cat(sprintf("  denom: mean=%.2f, sd=%.2f, range=[%.2f, %.2f]\n",
            mean(df$denom), sd(df$denom), min(df$denom), max(df$denom)))

# Fit numdenom
cat("\nFitting numdenom... ")
t_nd <- system.time({
  fit_nd <- ratiod(
    y | denom ~ x, data = df,
    family = ratiod_poisson_gamma(),
    spatial = spatial_hsgp(coords = c("lon", "lat"), m = 6),
    iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
    verbose = FALSE
  )
})[["elapsed"]]
cat(sprintf("%.1fs\n", t_nd))

# Fit Stan
cat("Compiling Stan model... ")
stan_model <- cmdstan_model("benchmarks/stan/hsgp_pg_joint.stan")
cat("done\n")

stan_data <- list(
  N = N_OBS, M = 6L,
  y_num = df$y, y_denom = df$denom,
  x = df$x,
  coords = cbind(df$lon, df$lat),
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

# Compare posteriors
draws_nd <- as.matrix(fit_nd$draws)
draws_stan <- fit_stan$draws(format = "df")

compare <- function(nd_draws, stan_draws, name) {
  nd_mean <- mean(nd_draws)
  nd_sd <- sd(nd_draws)
  stan_mean <- mean(stan_draws)
  stan_sd <- sd(stan_draws)
  se <- sqrt(nd_sd^2/length(nd_draws) + stan_sd^2/length(stan_draws))
  diff_se <- abs(nd_mean - stan_mean) / se
  cat(sprintf("  %-20s: nd=%.4f (%.4f), Stan=%.4f (%.4f), diff=%.2f SE\n",
              name, nd_mean, nd_sd, stan_mean, stan_sd, diff_se))
}

cat("\n=== Parameter Comparison ===\n")

# Fixed effects
cat("\nFixed effects:\n")
compare(draws_nd[, "beta_num[1]"], draws_stan$beta_num_0, "beta_num[0] (int)")
compare(draws_nd[, "beta_num[2]"], draws_stan$beta_num_1, "beta_num[1] (slope)")
compare(draws_nd[, "beta_denom[1]"], draws_stan$beta_denom_0, "beta_denom[0] (int)")
compare(draws_nd[, "beta_denom[2]"], draws_stan$beta_denom_1, "beta_denom[1] (slope)")

# Dispersion
cat("\nDispersion (phi_denom = Gamma shape):\n")
nd_phi_col <- grep("^phi_denom$|^shape_denom$", colnames(draws_nd), value = TRUE)
if (length(nd_phi_col) > 0) {
  compare(draws_nd[, nd_phi_col[1]], draws_stan$phi_denom, "phi_denom")
} else {
  cat("  phi_denom: NOT FOUND in numdenom draws\n")
  cat("  Available columns: ", paste(head(colnames(draws_nd), 20), collapse = ", "), "...\n")
}

# GP parameters
cat("\nGP parameters:\n")
nd_sigma_col <- grep("sigma.*gp|gp.*sigma|sigma2", colnames(draws_nd), value = TRUE, ignore.case = TRUE)
nd_len_col <- grep("lengthscale|phi.*gp|rho", colnames(draws_nd), value = TRUE, ignore.case = TRUE)

cat("  numdenom GP cols: ", paste(nd_sigma_col, collapse = ", "), "\n")
cat("  numdenom len cols: ", paste(nd_len_col, collapse = ", "), "\n")

if (length(nd_sigma_col) > 0) {
  # Stan has sigma_gp (SD), need to check if numdenom uses sigma or sigma2
  nd_sigma <- draws_nd[, nd_sigma_col[1]]
  stan_sigma2 <- draws_stan$sigma2_gp
  stan_sigma <- draws_stan$sigma_gp

  cat(sprintf("  numdenom %s: mean=%.4f, sd=%.4f\n", nd_sigma_col[1], mean(nd_sigma), sd(nd_sigma)))
  cat(sprintf("  Stan sigma_gp: mean=%.4f, sd=%.4f\n", mean(stan_sigma), sd(stan_sigma)))
  cat(sprintf("  Stan sigma2_gp: mean=%.4f, sd=%.4f\n", mean(stan_sigma2), sd(stan_sigma2)))
}

if (length(nd_len_col) > 0) {
  nd_len <- draws_nd[, nd_len_col[1]]
  stan_len <- draws_stan$lengthscale
  cat(sprintf("  numdenom %s: mean=%.4f, sd=%.4f\n", nd_len_col[1], mean(nd_len), sd(nd_len)))
  cat(sprintf("  Stan lengthscale: mean=%.4f, sd=%.4f\n", mean(stan_len), sd(stan_len)))
}

# Check Stan diagnostics
cat("\n=== Diagnostics ===\n")
diag <- fit_stan$diagnostic_summary()
cat(sprintf("Stan divergences: %d / %d (%.1f%%)\n",
            sum(diag$num_divergent), N_CHAINS * (N_ITER - N_WARMUP),
            100 * sum(diag$num_divergent) / (N_CHAINS * (N_ITER - N_WARMUP))))

cat("\nnumdenom column names (first 30):\n")
cat(paste(head(colnames(draws_nd), 30), collapse = "\n"))
cat("\n")
