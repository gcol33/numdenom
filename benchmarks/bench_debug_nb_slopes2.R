# Debug: Compare numdenom vs Stan for negbin_negbin with random SLOPES
# Fix RE column extraction

library(numdenom)
library(cmdstanr)
library(posterior)

set.seed(42)

N_OBS <- 200
N_ITER <- 4000
N_WARMUP <- 1500
N_CHAINS <- 4
N_SITES <- 20

cat("=== DEBUG: NegBin-NegBin WITH RE slopes ===\n\n")

# Generate data
site <- factor(rep(1:N_SITES, length.out = N_OBS))
x <- rnorm(N_OBS)

true_beta_num <- c(2, 0.5)
true_beta_denom <- c(3, 0.2)
true_phi_num <- 5.0
true_phi_denom <- 8.0
true_sigma_int <- 0.4
true_sigma_slope <- 0.2
true_rho <- 0.3

# Generate correlated RE
L <- matrix(c(1, 0, true_rho, sqrt(1-true_rho^2)), 2, 2)
true_re <- matrix(0, N_SITES, 2)
for (g in 1:N_SITES) {
  z <- rnorm(2)
  true_re[g, ] <- c(true_sigma_int, true_sigma_slope) * (L %*% z)
}

# Compute linear predictors with SHARED RE
eta_num <- true_beta_num[1] + true_beta_num[2] * x
eta_denom <- true_beta_denom[1] + true_beta_denom[2] * x
for (i in 1:N_OBS) {
  g <- as.integer(site[i])
  re_effect <- true_re[g, 1] + true_re[g, 2] * x[i]
  eta_num[i] <- eta_num[i] + re_effect
  eta_denom[i] <- eta_denom[i] + re_effect
}

y_num <- rnbinom(N_OBS, size = true_phi_num, mu = exp(eta_num))
y_denom <- rnbinom(N_OBS, size = true_phi_denom, mu = exp(eta_denom))
y_denom[y_denom == 0] <- 1

df <- data.frame(y_num = y_num, y_denom = y_denom, x = x, site = site)

cat(sprintf("True: beta_num=[%.2f,%.2f], beta_denom=[%.2f,%.2f]\n",
            true_beta_num[1], true_beta_num[2], true_beta_denom[1], true_beta_denom[2]))
cat(sprintf("True: sigma_int=%.2f, sigma_slope=%.2f, rho=%.2f\n",
            true_sigma_int, true_sigma_slope, true_rho))

# Fit numdenom
cat("Fitting numdenom... ")
fit_nd <- ratiod(
  y_num | y_denom ~ x + (1+x|site),
  data = df,
  family = ratiod_negbin_negbin(),
  iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS,
  verbose = FALSE
)
cat("done\n")

# Get numdenom draws
draws_nd <- as.matrix(fit_nd$draws)
nd_cols <- colnames(draws_nd)

# RE naming is re[1,g,intercept] and re[1,g,x]
# Exclude sigma_re columns!
re_int_cols <- nd_cols[grep("^re\\[.*,intercept\\]$", nd_cols)]
re_slope_cols <- nd_cols[grep("^re\\[.*,x\\]$", nd_cols)]

cat("numdenom RE intercept columns:", paste(re_int_cols[1:3], collapse=", "), "...\n")
cat("numdenom RE slope columns:", paste(re_slope_cols[1:3], collapse=", "), "...\n")
cat("Number of intercept cols:", length(re_int_cols), "\n")
cat("Number of slope cols:", length(re_slope_cols), "\n")

nd_mean_re_int <- rowMeans(draws_nd[, re_int_cols, drop=FALSE])
nd_mean_re_slope <- rowMeans(draws_nd[, re_slope_cols, drop=FALSE])

# Fit Stan model
stan_code <- "
data {
  int<lower=1> N;
  array[N] int<lower=0> y_num;
  array[N] int<lower=0> y_denom;
  matrix[N, 2] X;
  int<lower=1> n_groups;
  array[N] int<lower=1,upper=n_groups> group_idx;
  vector[N] x_slope;
}
transformed data {
  real sigma_re_scale = 2.5;
}
parameters {
  vector[2] beta_num;
  vector[2] beta_denom;
  real<lower=0> phi_num;
  real<lower=0> phi_denom;
  real<lower=0> sigma_intercept;
  real<lower=0> sigma_slope;
  real<lower=-1, upper=1> L21;
  matrix[n_groups, 2] re;
}
transformed parameters {
  matrix[2, 2] L_Omega;
  L_Omega[1, 1] = 1.0;
  L_Omega[1, 2] = 0.0;
  L_Omega[2, 1] = L21;
  L_Omega[2, 2] = sqrt(1.0 - L21 * L21);
}
model {
  vector[N] eta_num;
  vector[N] eta_denom;

  eta_num = X * beta_num;
  eta_denom = X * beta_denom;
  for (n in 1:N) {
    int g = group_idx[n];
    real re_effect = re[g, 1] + re[g, 2] * x_slope[n];
    eta_num[n] += re_effect;
    eta_denom[n] += re_effect;
  }

  beta_num ~ normal(0, 10);
  beta_denom ~ normal(0, 10);
  phi_num ~ gamma(2, 0.1);
  phi_denom ~ gamma(2, 0.1);
  sigma_intercept ~ cauchy(0, sigma_re_scale);
  sigma_slope ~ cauchy(0, sigma_re_scale);
  target += 1.5 * log(1 - L21 * L21);

  // MVN prior on RE (centered)
  for (g in 1:n_groups) {
    real y1 = re[g, 1] / sigma_intercept;
    real y2 = re[g, 2] / sigma_slope;
    real L22 = L_Omega[2, 2];
    real z1 = y1;
    real z2 = (y2 - L21 * z1) / L22;
    target += -0.5 * (z1 * z1 + z2 * z2);
  }
  target += -n_groups * (log(sigma_intercept) + log(sigma_slope) + log(L_Omega[2, 2]));

  y_num ~ neg_binomial_2_log(eta_num, phi_num);
  y_denom ~ neg_binomial_2_log(eta_denom, phi_denom);
}
generated quantities {
  real mean_re_int = mean(re[,1]);
  real mean_re_slope = mean(re[,2]);
  real eff_int_num = beta_num[1] + mean_re_int;
  real eff_int_denom = beta_denom[1] + mean_re_int;
  real eff_slope_num = beta_num[2] + mean_re_slope;
  real eff_slope_denom = beta_denom[2] + mean_re_slope;
}
"

writeLines(stan_code, "stan/joint_nb_slopes_gq.stan")
stan_mod <- cmdstan_model("stan/joint_nb_slopes_gq.stan")

stan_data <- list(
  N = N_OBS,
  y_num = df$y_num,
  y_denom = df$y_denom,
  X = cbind(1, df$x),
  n_groups = N_SITES,
  group_idx = as.numeric(df$site),
  x_slope = df$x
)

cat("Fitting Stan... ")
fit_stan <- stan_mod$sample(
  data = stan_data,
  iter_sampling = N_ITER - N_WARMUP,
  iter_warmup = N_WARMUP,
  chains = N_CHAINS,
  parallel_chains = N_CHAINS,
  refresh = 0,
  show_messages = FALSE,
  adapt_delta = 0.95
)
cat("done\n\n")

draws_stan <- fit_stan$draws(format = "df")

cat("=== Mean RE comparison ===\n")
cat(sprintf("mean(RE intercept): nd=%.4f, stan=%.4f\n",
            mean(nd_mean_re_int), mean(draws_stan$mean_re_int)))
cat(sprintf("mean(RE slope): nd=%.4f, stan=%.4f\n",
            mean(nd_mean_re_slope), mean(draws_stan$mean_re_slope)))

cat("\n=== Effective intercept comparison ===\n")
nd_eff_int_num <- draws_nd[, "beta_num[1]"] + nd_mean_re_int
nd_eff_int_denom <- draws_nd[, "beta_denom[1]"] + nd_mean_re_int

cat(sprintf("effective_intercept_num:\n"))
cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", mean(nd_eff_int_num), sd(nd_eff_int_num)))
cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", mean(draws_stan$eff_int_num), sd(draws_stan$eff_int_num)))

se_eff <- sqrt(sd(nd_eff_int_num)^2/length(nd_eff_int_num) + sd(draws_stan$eff_int_num)^2/length(draws_stan$eff_int_num))
diff_se <- abs(mean(nd_eff_int_num) - mean(draws_stan$eff_int_num)) / se_eff
cat(sprintf("  Diff: %.2f SE => %s\n", diff_se, if(diff_se < 2) "PASS" else "FAIL"))

cat(sprintf("\neffective_intercept_denom:\n"))
cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", mean(nd_eff_int_denom), sd(nd_eff_int_denom)))
cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", mean(draws_stan$eff_int_denom), sd(draws_stan$eff_int_denom)))

se_eff_d <- sqrt(sd(nd_eff_int_denom)^2/length(nd_eff_int_denom) + sd(draws_stan$eff_int_denom)^2/length(draws_stan$eff_int_denom))
diff_se_d <- abs(mean(nd_eff_int_denom) - mean(draws_stan$eff_int_denom)) / se_eff_d
cat(sprintf("  Diff: %.2f SE => %s\n", diff_se_d, if(diff_se_d < 2) "PASS" else "FAIL"))

cat("\n=== Effective slope comparison ===\n")
nd_eff_slope_num <- draws_nd[, "beta_num[2]"] + nd_mean_re_slope
nd_eff_slope_denom <- draws_nd[, "beta_denom[2]"] + nd_mean_re_slope

cat(sprintf("effective_slope_num:\n"))
cat(sprintf("  numdenom: %.4f (SD=%.4f)\n", mean(nd_eff_slope_num), sd(nd_eff_slope_num)))
cat(sprintf("  Stan:     %.4f (SD=%.4f)\n", mean(draws_stan$eff_slope_num), sd(draws_stan$eff_slope_num)))

se_slope <- sqrt(sd(nd_eff_slope_num)^2/length(nd_eff_slope_num) + sd(draws_stan$eff_slope_num)^2/length(draws_stan$eff_slope_num))
diff_slope <- abs(mean(nd_eff_slope_num) - mean(draws_stan$eff_slope_num)) / se_slope
cat(sprintf("  Diff: %.2f SE => %s\n", diff_slope, if(diff_slope < 2) "PASS" else "FAIL"))

cat("\n=== Raw parameter comparison (for diagnostics) ===\n")
cat(sprintf("beta_num[1]: nd=%.4f, stan=%.4f\n", mean(draws_nd[, "beta_num[1]"]), mean(draws_stan$`beta_num[1]`)))
cat(sprintf("beta_num[2]: nd=%.4f, stan=%.4f\n", mean(draws_nd[, "beta_num[2]"]), mean(draws_stan$`beta_num[2]`)))
cat(sprintf("beta_denom[1]: nd=%.4f, stan=%.4f\n", mean(draws_nd[, "beta_denom[1]"]), mean(draws_stan$`beta_denom[1]`)))
cat(sprintf("beta_denom[2]: nd=%.4f, stan=%.4f\n", mean(draws_nd[, "beta_denom[2]"]), mean(draws_stan$`beta_denom[2]`)))
cat(sprintf("phi_num: nd=%.4f, stan=%.4f\n", mean(draws_nd[, "phi_num"]), mean(draws_stan$phi_num)))
cat(sprintf("phi_denom: nd=%.4f, stan=%.4f\n", mean(draws_nd[, "phi_denom"]), mean(draws_stan$phi_denom)))

# Also compare sigma and rho
sigma_int_col <- nd_cols[grep("sigma_re\\[1,intercept\\]", nd_cols)]
sigma_slope_col <- nd_cols[grep("sigma_re\\[1,x\\]", nd_cols)]
chol_col <- nd_cols[grep("L_chol\\[1,1\\]", nd_cols)]

if (length(sigma_int_col) > 0) {
  cat(sprintf("\nsigma_intercept: nd=%.4f, stan=%.4f\n",
              mean(draws_nd[, sigma_int_col]), mean(draws_stan$sigma_intercept)))
}
if (length(sigma_slope_col) > 0) {
  cat(sprintf("sigma_slope: nd=%.4f, stan=%.4f\n",
              mean(draws_nd[, sigma_slope_col]), mean(draws_stan$sigma_slope)))
}
if (length(chol_col) > 0) {
  cat(sprintf("L21 (correlation): nd=%.4f, stan=%.4f\n",
              mean(draws_nd[, chol_col]), mean(draws_stan$L21)))
}
