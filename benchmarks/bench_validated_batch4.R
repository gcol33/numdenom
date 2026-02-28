# Batch 4: Validate next 25 model configurations against Stan/brms
# Focus: binomial family (rows 61-85)

library(devtools)
load_all()
library(brms)
set.seed(42)

# Standard parameters
N <- 200
N_SITES <- 30
N_TIMES <- 15

results <- list()

# Helper function for validation
validate_model <- function(name, nd_fit, brms_fit, params_map) {
  cat("\n===", name, "===\n")

  nd_draws <- nd_fit$draws
  brms_draws <- as_draws_df(brms_fit)

  all_pass <- TRUE
  for (nd_param in names(params_map)) {
    brms_param <- params_map[[nd_param]]

    nd_mean <- mean(nd_draws[, nd_param])
    nd_sd <- sd(nd_draws[, nd_param])
    brms_mean <- mean(brms_draws[[brms_param]])
    brms_sd <- sd(brms_draws[[brms_param]])

    diff <- abs(nd_mean - brms_mean)
    pooled_se <- sqrt(nd_sd^2 + brms_sd^2)
    ratio <- diff / pooled_se

    pass <- ratio < 2
    all_pass <- all_pass && pass

    cat(sprintf("  %s: nd=%.3f(%.3f) brms=%.3f(%.3f) diff=%.2fSE %s\n",
                nd_param, nd_mean, nd_sd, brms_mean, brms_sd, ratio,
                if(pass) "PASS" else "FAIL"))
  }

  list(name = name, pass = all_pass)
}

# Generate base data
site <- rep(1:N_SITES, length.out = N)
time <- rep(1:N_TIMES, each = ceiling(N/N_TIMES))[1:N]
x <- rnorm(N)

# Create adjacency for spatial models
adj <- matrix(0, N_SITES, N_SITES)
for (i in 1:(N_SITES-1)) {
  adj[i, i+1] <- adj[i+1, i] <- 1
}

# =============================================================================
# Row 61: binomial, no RE
# =============================================================================
cat("\n\n========== ROW 61: binomial, no RE ==========\n")

trials <- sample(20:50, N, replace = TRUE)
prob <- plogis(0.5 + 0.3 * x)
successes <- rbinom(N, trials, prob)
df61 <- data.frame(successes = successes, trials = trials, x = x)

t_nd <- system.time({
  fit_nd <- ratiod(successes | trials ~ x, data = df61,
                   family = ratiod_binomial(),
                   iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
})

t_brms <- system.time({
  fit_brms <- brm(successes | trials(trials) ~ x, data = df61,
                  family = binomial(), iter = 1000, warmup = 500, chains = 2,
                  silent = 2, refresh = 0)
})

results[[61]] <- validate_model("Row 61: binomial basic",
                                 fit_nd, fit_brms,
                                 list("beta_num[1]" = "b_Intercept",
                                      "beta_num[2]" = "b_x"))
results[[61]]$nd_time <- t_nd["elapsed"]
results[[61]]$brms_time <- t_brms["elapsed"]

# =============================================================================
# Row 62: binomial + RE
# =============================================================================
cat("\n\n========== ROW 62: binomial + RE ==========\n")

re_effect <- rnorm(N_SITES, 0, 0.5)
prob <- plogis(0.5 + 0.3 * x + re_effect[site])
successes <- rbinom(N, trials, prob)
df62 <- data.frame(successes = successes, trials = trials, x = x, site = factor(site))

t_nd <- system.time({
  fit_nd <- ratiod(successes | trials ~ x + (1|site), data = df62,
                   family = ratiod_binomial(),
                   iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
})

t_brms <- system.time({
  fit_brms <- brm(successes | trials(trials) ~ x + (1|site), data = df62,
                  family = binomial(), iter = 1000, warmup = 500, chains = 2,
                  silent = 2, refresh = 0)
})

results[[62]] <- validate_model("Row 62: binomial + RE",
                                 fit_nd, fit_brms,
                                 list("beta_num[1]" = "b_Intercept",
                                      "beta_num[2]" = "b_x"))
results[[62]]$nd_time <- t_nd["elapsed"]
results[[62]]$brms_time <- t_brms["elapsed"]

# =============================================================================
# Row 63: binomial + random slopes
# =============================================================================
cat("\n\n========== ROW 63: binomial + slopes ==========\n")

slope_effect <- rnorm(N_SITES, 0.3, 0.2)
prob <- plogis(0.5 + slope_effect[site] * x + re_effect[site])
successes <- rbinom(N, trials, prob)
df63 <- data.frame(successes = successes, trials = trials, x = x, site = factor(site))

t_nd <- system.time({
  fit_nd <- ratiod(successes | trials ~ x + (1 + x|site), data = df63,
                   family = ratiod_binomial(),
                   iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
})

t_brms <- system.time({
  fit_brms <- brm(successes | trials(trials) ~ x + (1 + x|site), data = df63,
                  family = binomial(), iter = 1000, warmup = 500, chains = 2,
                  silent = 2, refresh = 0)
})

results[[63]] <- validate_model("Row 63: binomial + slopes",
                                 fit_nd, fit_brms,
                                 list("beta_num[1]" = "b_Intercept",
                                      "beta_num[2]" = "b_x"))
results[[63]]$nd_time <- t_nd["elapsed"]
results[[63]]$brms_time <- t_brms["elapsed"]

# =============================================================================
# Row 71: binomial + RW1
# =============================================================================
cat("\n\n========== ROW 71: binomial + RW1 ==========\n")

rw_effect <- cumsum(rnorm(N_TIMES, 0, 0.2))
rw_effect <- rw_effect - mean(rw_effect)
prob <- plogis(0.5 + 0.3 * x + rw_effect[time])
successes <- rbinom(N, trials, prob)
df71 <- data.frame(successes = successes, trials = trials, x = x, time = time)

t_nd <- system.time({
  fit_nd <- ratiod(successes | trials ~ x, data = df71,
                   family = ratiod_binomial(),
                   temporal = temporal_rw1(time_var = "time"),
                   iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
})

t_brms <- system.time({
  fit_brms <- brm(successes | trials(trials) ~ x + s(time, bs = "re"),
                  data = df71, family = binomial(),
                  iter = 1000, warmup = 500, chains = 2,
                  silent = 2, refresh = 0)
})

results[[71]] <- validate_model("Row 71: binomial + RW1",
                                 fit_nd, fit_brms,
                                 list("beta_num[1]" = "b_Intercept",
                                      "beta_num[2]" = "b_x"))
results[[71]]$nd_time <- t_nd["elapsed"]
results[[71]]$brms_time <- t_brms["elapsed"]

# =============================================================================
# Row 72: binomial + RW2
# =============================================================================
cat("\n\n========== ROW 72: binomial + RW2 ==========\n")

t_nd <- system.time({
  fit_nd <- ratiod(successes | trials ~ x, data = df71,
                   family = ratiod_binomial(),
                   temporal = temporal_rw2(time_var = "time"),
                   iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
})

results[[72]] <- list(name = "Row 72: binomial + RW2",
                      nd_time = t_nd["elapsed"],
                      pass = NA,
                      note = "No direct brms equivalent for RW2")

# =============================================================================
# Row 73: binomial + AR1
# =============================================================================
cat("\n\n========== ROW 73: binomial + AR1 ==========\n")

t_nd <- system.time({
  fit_nd <- ratiod(successes | trials ~ x, data = df71,
                   family = ratiod_binomial(),
                   temporal = temporal_ar1(time_var = "time"),
                   iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
})

t_brms <- system.time({
  fit_brms <- brm(successes | trials(trials) ~ x + ar(time = time, gr = 1),
                  data = transform(df71, gr = 1), family = binomial(),
                  iter = 1000, warmup = 500, chains = 2,
                  silent = 2, refresh = 0)
})

results[[73]] <- validate_model("Row 73: binomial + AR1",
                                 fit_nd, fit_brms,
                                 list("beta_num[1]" = "b_Intercept",
                                      "beta_num[2]" = "b_x"))
results[[73]]$nd_time <- t_nd["elapsed"]
results[[73]]$brms_time <- t_brms["elapsed"]

# =============================================================================
# Rows 76-79: ZI variants
# =============================================================================
cat("\n\n========== ROW 76: binomial + ZI ==========\n")

# ZI binomial
zi_prob <- 0.2
prob <- plogis(0.5 + 0.3 * x)
is_zero <- rbinom(N, 1, zi_prob)
successes <- ifelse(is_zero == 1, 0, rbinom(N, trials, prob))
df76 <- data.frame(successes = successes, trials = trials, x = x)

t_nd <- system.time({
  fit_nd <- ratiod(successes | trials ~ x, data = df76,
                   family = ratiod_zibinomial(),
                   iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
})

results[[76]] <- list(name = "Row 76: ZI binomial",
                      nd_time = t_nd["elapsed"],
                      pass = NA,
                      note = "ZI binomial needs custom Stan model")

cat("\n\n========== ROW 77: binomial + Hurdle ==========\n")

t_nd <- system.time({
  fit_nd <- ratiod(successes | trials ~ x, data = df76,
                   family = ratiod_hurdle_binomial(),
                   iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
})

results[[77]] <- list(name = "Row 77: hurdle binomial",
                      nd_time = t_nd["elapsed"],
                      pass = NA,
                      note = "Hurdle binomial needs custom Stan model")

cat("\n\n========== ROW 78: binomial + OI ==========\n")

t_nd <- system.time({
  fit_nd <- ratiod(successes | trials ~ x, data = df76,
                   family = ratiod_oibinomial(),
                   iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
})

results[[78]] <- list(name = "Row 78: OI binomial",
                      nd_time = t_nd["elapsed"],
                      pass = NA,
                      note = "OI binomial needs custom Stan model")

cat("\n\n========== ROW 79: ZOIB ==========\n")

# ZOIB data
y_zoib <- rbeta(N, 2, 5)
y_zoib[sample(N, 20)] <- 0
y_zoib[sample(N, 10)] <- 1
df79 <- data.frame(y = y_zoib, x = x)

t_nd <- system.time({
  fit_nd <- tryCatch({
    ratiod(y ~ x, data = df79,
           family = ratiod_zoib(),
           iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
  }, error = function(e) NULL)
})

results[[79]] <- list(name = "Row 79: ZOIB",
                      nd_time = if(is.null(fit_nd)) NA else t_nd["elapsed"],
                      pass = NA,
                      note = "ZOIB needs custom Stan model")

# =============================================================================
# Rows 65-66: binomial + spatial (ICAR, BYM2)
# =============================================================================
cat("\n\n========== ROW 65: binomial + ICAR ==========\n")

# Generate spatial effect
Q <- diag(rowSums(adj)) - adj
Q_inv <- MASS::ginv(Q + 0.01 * diag(N_SITES))
spatial_effect <- MASS::mvrnorm(1, rep(0, N_SITES), 0.5 * Q_inv)
spatial_effect <- spatial_effect - mean(spatial_effect)

prob <- plogis(0.5 + 0.3 * x + spatial_effect[site])
successes <- rbinom(N, trials, prob)
df65 <- data.frame(successes = successes, trials = trials, x = x, site = factor(site))

t_nd <- system.time({
  fit_nd <- ratiod(successes | trials ~ x, data = df65,
                   family = ratiod_binomial(),
                   spatial = spatial_car(adj = adj, group_var = "site"),
                   iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
})

results[[65]] <- list(name = "Row 65: binomial + ICAR",
                      nd_time = t_nd["elapsed"],
                      pass = NA,
                      note = "ICAR validation needs custom Stan or mgcv comparison")

cat("\n\n========== ROW 66: binomial + BYM2 ==========\n")

t_nd <- system.time({
  fit_nd <- ratiod(successes | trials ~ x, data = df65,
                   family = ratiod_binomial(),
                   spatial = spatial_bym2(adj = adj, group_var = "site"),
                   iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
})

results[[66]] <- list(name = "Row 66: binomial + BYM2",
                      nd_time = t_nd["elapsed"],
                      pass = NA,
                      note = "BYM2 validation needs custom Stan model")

# =============================================================================
# Rows 80-82: binomial + spatial + temporal
# =============================================================================
cat("\n\n========== ROW 80: binomial + ICAR + RW1 ==========\n")

df80 <- data.frame(successes = successes, trials = trials, x = x,
                   site = factor(site), time = time)

t_nd <- system.time({
  fit_nd <- ratiod(successes | trials ~ x, data = df80,
                   family = ratiod_binomial(),
                   spatial = spatial_car(adj = adj, group_var = "site"),
                   temporal = temporal_rw1(time_var = "time"),
                   iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
})

results[[80]] <- list(name = "Row 80: binomial + ICAR + RW1",
                      nd_time = t_nd["elapsed"],
                      pass = NA,
                      note = "Spatiotemporal needs custom Stan")

cat("\n\n========== ROW 81: binomial + BYM2 + RW1 ==========\n")

t_nd <- system.time({
  fit_nd <- ratiod(successes | trials ~ x, data = df80,
                   family = ratiod_binomial(),
                   spatial = spatial_bym2(adj = adj, group_var = "site"),
                   temporal = temporal_rw1(time_var = "time"),
                   iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
})

results[[81]] <- list(name = "Row 81: binomial + BYM2 + RW1",
                      nd_time = t_nd["elapsed"],
                      pass = NA,
                      note = "Spatiotemporal needs custom Stan")

cat("\n\n========== ROW 82: binomial + ICAR + AR1 ==========\n")

t_nd <- system.time({
  fit_nd <- ratiod(successes | trials ~ x, data = df80,
                   family = ratiod_binomial(),
                   spatial = spatial_car(adj = adj, group_var = "site"),
                   temporal = temporal_ar1(time_var = "time"),
                   iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
})

results[[82]] <- list(name = "Row 82: binomial + ICAR + AR1",
                      nd_time = t_nd["elapsed"],
                      pass = NA,
                      note = "Spatiotemporal needs custom Stan")

# =============================================================================
# More negbin rows that need validation (31, 32, 34, 35, 36, 46)
# =============================================================================
cat("\n\n========== ROW 31: negbin_negbin basic ==========\n")

y_num <- rnbinom(N, size = 5, mu = exp(0.5 + 0.3 * x) * 10)
y_denom <- rnbinom(N, size = 5, mu = 20)
df31 <- data.frame(num = y_num, denom = y_denom, x = x)

t_nd <- system.time({
  fit_nd <- ratiod(num | denom ~ x, data = df31,
                   family = ratiod_negbin_negbin(),
                   iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
})

t_brms <- system.time({
  fit_brms <- brm(num ~ x, data = df31,
                  family = negbinomial(), iter = 1000, warmup = 500, chains = 2,
                  silent = 2, refresh = 0)
})

results[[31]] <- validate_model("Row 31: negbin basic",
                                 fit_nd, fit_brms,
                                 list("beta_num[2]" = "b_x"))
results[[31]]$nd_time <- t_nd["elapsed"]
results[[31]]$brms_time <- t_brms["elapsed"]
results[[31]]$note <- "Intercept differs (joint vs marginal model)"

cat("\n\n========== ROW 32: negbin_negbin + RE ==========\n")

re_effect <- rnorm(N_SITES, 0, 0.3)
y_num <- rnbinom(N, size = 5, mu = exp(0.5 + 0.3 * x + re_effect[site]) * 10)
df32 <- data.frame(num = y_num, denom = y_denom, x = x, site = factor(site))

t_nd <- system.time({
  fit_nd <- ratiod(num | denom ~ x + (1|site), data = df32,
                   family = ratiod_negbin_negbin(),
                   iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
})

t_brms <- system.time({
  fit_brms <- brm(num ~ x + (1|site), data = df32,
                  family = negbinomial(), iter = 1000, warmup = 500, chains = 2,
                  silent = 2, refresh = 0)
})

results[[32]] <- validate_model("Row 32: negbin + RE",
                                 fit_nd, fit_brms,
                                 list("beta_num[2]" = "b_x"))
results[[32]]$nd_time <- t_nd["elapsed"]
results[[32]]$brms_time <- t_brms["elapsed"]

cat("\n\n========== ROW 34: negbin_negbin + crossed RE ==========\n")

site2 <- sample(1:10, N, replace = TRUE)
df34 <- data.frame(num = y_num, denom = y_denom, x = x,
                   site = factor(site), site2 = factor(site2))

t_nd <- system.time({
  fit_nd <- ratiod(num | denom ~ x + (1|site) + (1|site2), data = df34,
                   family = ratiod_negbin_negbin(),
                   iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
})

t_brms <- system.time({
  fit_brms <- brm(num ~ x + (1|site) + (1|site2), data = df34,
                  family = negbinomial(), iter = 1000, warmup = 500, chains = 2,
                  silent = 2, refresh = 0)
})

results[[34]] <- validate_model("Row 34: negbin + crossed",
                                 fit_nd, fit_brms,
                                 list("beta_num[2]" = "b_x"))
results[[34]]$nd_time <- t_nd["elapsed"]
results[[34]]$brms_time <- t_brms["elapsed"]

cat("\n\n========== ROW 46: negbin_negbin + ZI ==========\n")

zi_prob <- 0.15
y_num_zi <- ifelse(rbinom(N, 1, zi_prob) == 1, 0, y_num)
df46 <- data.frame(num = y_num_zi, denom = y_denom, x = x)

t_nd <- system.time({
  fit_nd <- ratiod(num | denom ~ x, data = df46,
                   family = ratiod_zinegbin_negbin(),
                   iter = 1000, warmup = 500, chains = 2, verbose = FALSE)
})

results[[46]] <- list(name = "Row 46: ZI negbin",
                      nd_time = t_nd["elapsed"],
                      pass = NA,
                      note = "ZI negbin needs custom Stan model")

# =============================================================================
# Summary
# =============================================================================
cat("\n\n")
cat("="^60, "\n")
cat("BATCH 4 SUMMARY\n")
cat("="^60, "\n\n")

for (row in sort(as.numeric(names(results)))) {
  r <- results[[row]]
  status <- if(is.na(r$pass)) "TIMING" else if(r$pass) "PASS" else "FAIL"
  time_str <- if(!is.null(r$nd_time) && !is.na(r$nd_time))
    sprintf("%.1fs", r$nd_time) else "N/A"
  brms_str <- if(!is.null(r$brms_time) && !is.na(r$brms_time))
    sprintf("%.1fs", r$brms_time) else ""

  cat(sprintf("Row %2d: %-30s [%s] nd=%s %s\n",
              row, r$name, status, time_str, brms_str))
  if (!is.null(r$note)) cat(sprintf("        Note: %s\n", r$note))
}

cat("\n\nDone!\n")
