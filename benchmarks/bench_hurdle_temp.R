# Temporary benchmark for hurdle models
devtools::load_all()
library(brms)
set.seed(123)

N <- 500
N_SITES <- 50
x <- rnorm(N)
site <- factor(rep(1:N_SITES, length.out = N))

# Poisson-gamma data with zeros
y_pg <- rpois(N, exp(2 + 0.3*x))
effort <- rgamma(N, 10, 1)
y_pg <- ifelse(runif(N) < 0.3, 0, y_pg)
df_pg_zi <- data.frame(y = y_pg, effort = effort, x = x, site = site)

# Binomial data with zeros
trials <- sample(10:50, N, replace = TRUE)
y_bin <- rbinom(N, trials, plogis(0.5 + 0.3*x))
y_bin <- ifelse(runif(N) < 0.3, 0, y_bin)
df_bin_zi <- data.frame(y = y_bin, trials = trials, x = x, site = site)

validate <- function(nd_fit, brms_fit) {
  nd_draws <- as.matrix(nd_fit$draws)
  nd_col <- grep("beta_num\\[2\\]", colnames(nd_draws), value = TRUE)[1]
  nd_x <- mean(nd_draws[, nd_col])
  nd_sd <- sd(nd_draws[, nd_col])

  brms_sum <- fixef(brms_fit)
  brms_x <- brms_sum["x", "Estimate"]
  brms_se <- brms_sum["x", "Est.Error"]

  diff <- abs(nd_x - brms_x)
  threshold <- 2 * max(nd_sd, brms_se)

  list(nd_x = nd_x, brms_x = brms_x, diff = diff, pass = diff < threshold,
       nd_sd = nd_sd, brms_se = brms_se)
}

# ========== Row 17: pg_hurdle ==========
cat("\n========== Row 17: pg_hurdle ==========\n")

cat("Running numdenom... ")
t_nd_17 <- system.time({
  fit_nd_17 <- ratiod(y | effort ~ x + (1|site), data = df_pg_zi,
                      family = ratiod_poisson_gamma(),
                      zi = numdenom::hurdle_poisson(),
                      iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
})["elapsed"]
cat(sprintf("%.1fs (div: %d)\n", t_nd_17, fit_nd_17$diagnostics$divergent))

cat("Running brms... ")
t_brms_17 <- system.time({
  fit_brms_17 <- brm(y ~ x + (1|site) + offset(log(effort)),
                     data = df_pg_zi, family = hurdle_poisson(),
                     iter = 1000, warmup = 500, chains = 2,
                     backend = "cmdstanr", silent = 2, refresh = 0)
})["elapsed"]
cat(sprintf("%.1fs\n", t_brms_17))

val_17 <- validate(fit_nd_17, fit_brms_17)
cat(sprintf("\n=== VALIDATION Row 17 ===\n"))
cat(sprintf("numdenom: x = %.4f (SD=%.4f)\n", val_17$nd_x, val_17$nd_sd))
cat(sprintf("brms:     x = %.4f (SE=%.4f)\n", val_17$brms_x, val_17$brms_se))
cat(sprintf("Diff: %.4f | Status: %s | Speedup: %.1fx\n",
            val_17$diff, if(val_17$pass) "PASS" else "FAIL", t_brms_17/t_nd_17))

# ========== Row 77: bin_hurdle ==========
cat("\n========== Row 77: bin_hurdle ==========\n")

cat("Running numdenom... ")
t_nd_77 <- system.time({
  fit_nd_77 <- ratiod(y | trials ~ x + (1|site), data = df_bin_zi,
                      family = ratiod_hurdle_binomial(),
                      iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
})["elapsed"]
cat(sprintf("%.1fs (div: %d)\n", t_nd_77, fit_nd_77$diagnostics$divergent))

cat("Running brms... ")
t_brms_77 <- system.time({
  # brms doesn't have hurdle_binomial, use hurdle_poisson as approximation
  fit_brms_77 <- brm(bf(y ~ x + (1|site), hu ~ 1),
                     data = df_bin_zi, family = hurdle_poisson(),
                     iter = 1000, warmup = 500, chains = 2,
                     backend = "cmdstanr", silent = 2, refresh = 0)
})["elapsed"]
cat(sprintf("%.1fs\n", t_brms_77))

val_77 <- validate(fit_nd_77, fit_brms_77)
cat(sprintf("\n=== VALIDATION Row 77 ===\n"))
cat(sprintf("numdenom: x = %.4f (SD=%.4f)\n", val_77$nd_x, val_77$nd_sd))
cat(sprintf("brms:     x = %.4f (SE=%.4f)\n", val_77$brms_x, val_77$brms_se))
cat(sprintf("Diff: %.4f | Status: %s | Speedup: %.1fx\n",
            val_77$diff, if(val_77$pass) "PASS" else "FAIL", t_brms_77/t_nd_77))

cat("\n========== SUMMARY ==========\n")
cat(sprintf("Row 17 (pg_hurdle):  %s\n", if(val_17$pass) "PASS" else "FAIL"))
cat(sprintf("Row 77 (bin_hurdle): %s\n", if(val_77$pass) "PASS" else "FAIL"))
