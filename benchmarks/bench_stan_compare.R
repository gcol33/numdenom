# Stan comparison benchmark
# Compares numdenom with brms (Stan) for validation
# Validation: posterior means within 2 SE

library(numdenom)
library(brms)

set.seed(123)

# Standard parameters
N <- 500
N_SITES <- 50
x <- rnorm(N)
site <- factor(rep(1:N_SITES, length.out = N))

# Generate data - poisson-gamma style
y_count <- rpois(N, exp(2 + 0.3*x))
denom <- rgamma(N, shape=10, rate=0.1)  # mean ~100
denom[denom < 1] <- 1
df <- data.frame(y=y_count, denom=denom, x=x, site=site)

# Also generate negbin data
y_nb <- rnbinom(N, mu=exp(2 + 0.3*x), size=5)
denom_nb <- rnbinom(N, mu=100, size=10)
denom_nb[denom_nb == 0] <- 1
df_nb <- data.frame(y=y_nb, denom=denom_nb, x=x, site=site)

# Binomial data
trials <- rpois(N, 50) + 10
prob <- plogis(-0.5 + 0.3*x)
y_bin <- rbinom(N, trials, prob)
df_bin <- data.frame(y=y_bin, trials=trials, x=x, site=site)

results <- list()

# ============ Test 1: Simple poisson-gamma (no RE) ============
cat("\n=== Test 1: Poisson-gamma (no RE) ===\n")

# numdenom
time_nd <- system.time({
  fit_nd <- ratiod(y | denom ~ x, data=df, family=ratiod_poisson_gamma(),
                   iter=1000, warmup=500, chains=2, gradient_mode="H")
})["elapsed"]

# brms: model y ~ Poisson with log(denom) offset
time_brms <- system.time({
  fit_brms <- brm(y ~ x + offset(log(denom)),
                  data=df, family=poisson(),
                  iter=1000, warmup=500, chains=2,
                  backend="cmdstanr", silent=2, refresh=0)
})["elapsed"]

# Compare posteriors
nd_summary <- summary(fit_nd)
brms_summary <- summary(fit_brms)$fixed

cat(sprintf("  numdenom: x effect = %.3f (SE=%.3f), time=%.1fs\n",
            nd_summary$fixed["x", "mean"],
            nd_summary$fixed["x", "se_mean"],
            time_nd))
cat(sprintf("  brms:     x effect = %.3f (SE=%.3f), time=%.1fs\n",
            brms_summary["x", "Estimate"],
            brms_summary["x", "Est.Error"]/sqrt(brms_summary["x", "Bulk_ESS"]),
            time_brms))

diff <- abs(nd_summary$fixed["x", "mean"] - brms_summary["x", "Estimate"])
se_combined <- sqrt(nd_summary$fixed["x", "se_mean"]^2 +
                    (brms_summary["x", "Est.Error"]/sqrt(brms_summary["x", "Bulk_ESS"]))^2)
within_2se <- diff < 2 * se_combined

results$test1 <- list(
  numdenom_time = time_nd,
  brms_time = time_brms,
  numdenom_x = nd_summary$fixed["x", "mean"],
  brms_x = brms_summary["x", "Estimate"],
  diff = diff,
  within_2se = within_2se,
  speedup = time_brms / time_nd
)

cat(sprintf("  Difference: %.4f (%.1f combined SE) - %s\n",
            diff, diff/se_combined, if(within_2se) "PASS" else "FAIL"))
cat(sprintf("  Speedup: %.1fx\n", time_brms/time_nd))


# ============ Test 2: Negbin with RE ============
cat("\n=== Test 2: Negbin with RE ===\n")

# numdenom
time_nd <- system.time({
  fit_nd <- ratiod(y | denom ~ x + (1|site), data=df_nb, family=ratiod_negbin_negbin(),
                   iter=1000, warmup=500, chains=2, gradient_mode="H")
})["elapsed"]

# brms: negbinomial with offset
time_brms <- system.time({
  fit_brms <- brm(y ~ x + (1|site) + offset(log(denom)),
                  data=df_nb, family=negbinomial(),
                  iter=1000, warmup=500, chains=2,
                  backend="cmdstanr", silent=2, refresh=0)
})["elapsed"]

nd_summary <- summary(fit_nd)
brms_summary <- summary(fit_brms)$fixed

cat(sprintf("  numdenom: x effect = %.3f (SE=%.3f), time=%.1fs\n",
            nd_summary$fixed["x", "mean"],
            nd_summary$fixed["x", "se_mean"],
            time_nd))
cat(sprintf("  brms:     x effect = %.3f (SE=%.3f), time=%.1fs\n",
            brms_summary["x", "Estimate"],
            brms_summary["x", "Est.Error"]/sqrt(brms_summary["x", "Bulk_ESS"]),
            time_brms))

diff <- abs(nd_summary$fixed["x", "mean"] - brms_summary["x", "Estimate"])
se_combined <- sqrt(nd_summary$fixed["x", "se_mean"]^2 +
                    (brms_summary["x", "Est.Error"]/sqrt(brms_summary["x", "Bulk_ESS"]))^2)
within_2se <- diff < 2 * se_combined

results$test2 <- list(
  numdenom_time = time_nd,
  brms_time = time_brms,
  numdenom_x = nd_summary$fixed["x", "mean"],
  brms_x = brms_summary["x", "Estimate"],
  diff = diff,
  within_2se = within_2se,
  speedup = time_brms / time_nd
)

cat(sprintf("  Difference: %.4f (%.1f combined SE) - %s\n",
            diff, diff/se_combined, if(within_2se) "PASS" else "FAIL"))
cat(sprintf("  Speedup: %.1fx\n", time_brms/time_nd))


# ============ Test 3: Binomial ============
cat("\n=== Test 3: Binomial with RE ===\n")

# numdenom
time_nd <- system.time({
  fit_nd <- ratiod(y | trials ~ x + (1|site), data=df_bin, family=ratiod_binomial(),
                   iter=1000, warmup=500, chains=2, gradient_mode="H")
})["elapsed"]

# brms
time_brms <- system.time({
  fit_brms <- brm(y | trials(trials) ~ x + (1|site),
                  data=df_bin, family=binomial(),
                  iter=1000, warmup=500, chains=2,
                  backend="cmdstanr", silent=2, refresh=0)
})["elapsed"]

nd_summary <- summary(fit_nd)
brms_summary <- summary(fit_brms)$fixed

cat(sprintf("  numdenom: x effect = %.3f (SE=%.3f), time=%.1fs\n",
            nd_summary$fixed["x", "mean"],
            nd_summary$fixed["x", "se_mean"],
            time_nd))
cat(sprintf("  brms:     x effect = %.3f (SE=%.3f), time=%.1fs\n",
            brms_summary["x", "Estimate"],
            brms_summary["x", "Est.Error"]/sqrt(brms_summary["x", "Bulk_ESS"]),
            time_brms))

diff <- abs(nd_summary$fixed["x", "mean"] - brms_summary["x", "Estimate"])
se_combined <- sqrt(nd_summary$fixed["x", "se_mean"]^2 +
                    (brms_summary["x", "Est.Error"]/sqrt(brms_summary["x", "Bulk_ESS"]))^2)
within_2se <- diff < 2 * se_combined

results$test3 <- list(
  numdenom_time = time_nd,
  brms_time = time_brms,
  numdenom_x = nd_summary$fixed["x", "mean"],
  brms_x = brms_summary["x", "Estimate"],
  diff = diff,
  within_2se = within_2se,
  speedup = time_brms / time_nd
)

cat(sprintf("  Difference: %.4f (%.1f combined SE) - %s\n",
            diff, diff/se_combined, if(within_2se) "PASS" else "FAIL"))
cat(sprintf("  Speedup: %.1fx\n", time_brms/time_nd))


# ============ Summary ============
cat("\n=== SUMMARY ===\n")
cat(sprintf("Test 1 (PG no RE):    %s  Speedup: %.1fx\n",
            if(results$test1$within_2se) "PASS" else "FAIL",
            results$test1$speedup))
cat(sprintf("Test 2 (NB with RE):  %s  Speedup: %.1fx\n",
            if(results$test2$within_2se) "PASS" else "FAIL",
            results$test2$speedup))
cat(sprintf("Test 3 (Bin with RE): %s  Speedup: %.1fx\n",
            if(results$test3$within_2se) "PASS" else "FAIL",
            results$test3$speedup))

saveRDS(results, "benchmarks/stan_compare_results.rds")
cat("\nResults saved to benchmarks/stan_compare_results.rds\n")
