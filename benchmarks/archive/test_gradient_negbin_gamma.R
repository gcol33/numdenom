# Quick check: is the A_r gradient mismatch specific to NEGBIN_GAMMA or general?
library(numdenom)
set.seed(42)

cat("=== A_r gradient check across families with +RE ===\n\n")

families <- list(
  list(name="poisson_gamma", fam=ratiod_poisson_gamma(), phi_num=NULL, phi_denom=3),
  list(name="negbin_negbin", fam=ratiod_negbin_negbin(), phi_num=5, phi_denom=4),
  list(name="negbin_gamma", fam=ratiod_negbin_gamma(), phi_num=5, phi_denom=3)
)

for (f in families) {
  cat(sprintf("--- %s + RE ---\n", f$name))
  sim <- sim_ratiod(n=200, family=f$fam,
                    beta_num=c(2, 0.3), beta_denom=c(1, 0.2),
                    sigma_re=0.5, phi_num=f$phi_num, phi_denom=f$phi_denom,
                    n_groups=10, seed=42)
  cat("  A_r mode (watch for gradient mismatch warning):\n")
  fit <- tratio(y_num | y_denom ~ x1 + (1|group), data=sim$data,
                family=f$fam,
                control = list(iter=10, warmup=5, chains=1, gradient_mode="A_r", verbose=FALSE))
  cat("\n")
}
