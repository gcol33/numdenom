# Batch 2: Next 25 Stan Validations
# Models: slopes, crossed RE, GP spatial, combined spatial+temporal, ZI+spatial
# Standard parameters: N=500, iter=1000, warmup=500, chains=2

devtools::load_all()
library(brms)
set.seed(123)

# Setup data (same as batch 1)
N <- 500
N_SITES <- 50
N_TIMES <- 20
x <- rnorm(N)
site <- factor(rep(1:N_SITES, length.out = N))
time <- factor(rep(1:N_TIMES, length.out = N))

# Spatial grid for GP
n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_SITES, ]
site_int <- as.integer(site)
lon <- grid$lon[site_int]
lat <- grid$lat[site_int]

# Adjacency matrix
adj_mat <- matrix(0, N_SITES, N_SITES)
for (i in 1:N_SITES) {
  for (j in 1:N_SITES) {
    if (i != j) {
      dist <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
      if (dist <= 1.5) adj_mat[i, j] <- 1
    }
  }
}
spatial_site <- factor(rep(1:N_SITES, length.out = N))

# Generate data for each family
y_pg <- rpois(N, exp(2 + 0.3*x))
effort <- rgamma(N, 10, 1)
df_pg <- data.frame(y = y_pg, effort = effort, x = x, site = site, time = time,
                    lon = lon, lat = lat, spatial_site = spatial_site)

y_nb <- rnbinom(N, mu = exp(2 + 0.3*x), size = 5)
denom <- rnbinom(N, mu = 100, size = 10)
denom[denom == 0] <- 1
df_nb <- data.frame(y = y_nb, denom = denom, x = x, site = site, time = time,
                    lon = lon, lat = lat, spatial_site = spatial_site)

trials <- sample(10:50, N, replace = TRUE)
y_bin <- rbinom(N, trials, plogis(0.5 + 0.3*x))
df_bin <- data.frame(y = y_bin, trials = trials, x = x, site = site, time = time,
                     lon = lon, lat = lat, spatial_site = spatial_site)

# ZI data (add zeros)
df_pg_zi <- df_pg
df_pg_zi$y <- ifelse(runif(N) < 0.3, 0, df_pg_zi$y)

df_nb_zi <- df_nb
df_nb_zi$y <- ifelse(runif(N) < 0.3, 0, df_nb_zi$y)

df_bin_zi <- df_bin
df_bin_zi$y <- ifelse(runif(N) < 0.3, 0, df_bin_zi$y)

# Validation function
validate_against_stan <- function(nd_fit, brms_fit, param_name = "x") {
  nd_draws <- as.matrix(nd_fit$draws)
  nd_col <- grep(paste0("beta_num\\[2\\]"), colnames(nd_draws), value = TRUE)[1]
  if (is.na(nd_col)) nd_col <- grep(param_name, colnames(nd_draws), value = TRUE)[1]
  nd_x <- mean(nd_draws[, nd_col])
  nd_x_sd <- sd(nd_draws[, nd_col])

  brms_sum <- fixef(brms_fit)
  brms_x <- brms_sum[param_name, "Estimate"]
  brms_x_se <- brms_sum[param_name, "Est.Error"]

  diff <- abs(nd_x - brms_x)
  threshold <- 2 * max(nd_x_sd, brms_x_se)
  pass <- diff < threshold

  list(
    nd_x = nd_x,
    nd_sd = nd_x_sd,
    brms_x = brms_x,
    brms_se = brms_x_se,
    diff = diff,
    pass = pass
  )
}

# Main benchmark function
bench_validated <- function(name, row, nd_call, brms_call, true_x = 0.3, skip_stan = FALSE) {
  cat(sprintf("\n\n========== Row %d: %s ==========\n", row, name))
  cat(sprintf("True x = %.2f\n\n", true_x))

  result <- list(row = row, name = name, true_x = true_x)

  # Run numdenom
  cat("Running numdenom... ")
  flush.console()
  tryCatch({
    time_nd <- system.time({
      fit_nd <- eval(nd_call)
    })["elapsed"]
    result$time_nd <- time_nd
    result$divergent <- fit_nd$diagnostics$divergent
    cat(sprintf("%.1fs (div: %d)\n", time_nd, result$divergent))
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", conditionMessage(e)))
    result$error_nd <<- conditionMessage(e)
  })

  # Run brms/Stan
  if (!skip_stan) {
    cat("Running brms/Stan... ")
    flush.console()
    tryCatch({
      time_brms <- system.time({
        fit_brms <- eval(brms_call)
      })["elapsed"]
      result$time_brms <- time_brms
      cat(sprintf("%.1fs\n", time_brms))
    }, error = function(e) {
      cat(sprintf("ERROR: %s\n", conditionMessage(e)))
      result$error_brms <<- conditionMessage(e)
    })
  }

  # Validate
  if (!is.null(result$time_nd) && !is.null(result$time_brms)) {
    val <- validate_against_stan(fit_nd, fit_brms)
    result$nd_x <- val$nd_x
    result$brms_x <- val$brms_x
    result$diff <- val$diff
    result$pass <- val$pass
    result$speedup <- result$time_brms / result$time_nd

    cat(sprintf("\n=== VALIDATION ===\n"))
    cat(sprintf("numdenom: x = %.4f (SD=%.4f)\n", val$nd_x, val$nd_sd))
    cat(sprintf("brms:     x = %.4f (SE=%.4f)\n", val$brms_x, val$brms_se))
    cat(sprintf("Diff: %.4f | Status: %s | Speedup: %.1fx\n",
                val$diff, if(val$pass) "PASS" else "FAIL", result$speedup))
  }

  result
}

results <- list()

# ============================================================
# SECTION 1: RANDOM SLOPES AND CROSSED RE
# ============================================================

# Row 3: poisson_gamma + random slopes
results[["3"]] <- bench_validated("pg_slopes", 3,
  nd_call = quote(tratio(y | effort ~ x + (x|site), data = df_pg, family = ratiod_poisson_gamma(),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y ~ x + (x|site) + offset(log(effort)), data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 4: poisson_gamma + crossed RE
results[["4"]] <- bench_validated("pg_crossed", 4,
  nd_call = quote(tratio(y | effort ~ x + (1|site) + (1|time), data = df_pg, family = ratiod_poisson_gamma(),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y ~ x + (1|site) + (1|time) + offset(log(effort)), data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 33: negbin_negbin + random slopes
results[["33"]] <- bench_validated("nb_slopes", 33,
  nd_call = quote(tratio(y | denom ~ x + (x|site), data = df_nb, family = ratiod_negbin_negbin(),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y ~ x + (x|site) + offset(log(denom)), data = df_nb, family = negbinomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 34: negbin_negbin + crossed RE
results[["34"]] <- bench_validated("nb_crossed", 34,
  nd_call = quote(tratio(y | denom ~ x + (1|site) + (1|time), data = df_nb, family = ratiod_negbin_negbin(),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y ~ x + (1|site) + (1|time) + offset(log(denom)), data = df_nb, family = negbinomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 63: binomial + random slopes
results[["63"]] <- bench_validated("bin_slopes", 63,
  nd_call = quote(tratio(y | trials ~ x + (x|site), data = df_bin, family = ratiod_binomial(),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y | trials(trials) ~ x + (x|site), data = df_bin, family = binomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 64: binomial + crossed RE
results[["64"]] <- bench_validated("bin_crossed", 64,
  nd_call = quote(tratio(y | trials ~ x + (1|site) + (1|time), data = df_bin, family = ratiod_binomial(),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|time), data = df_bin, family = binomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# SECTION 2: GP SPATIAL (slower - brms gp() term)
# ============================================================

# Row 7: poisson_gamma + GP
results[["7"]] <- bench_validated("pg_gp", 7,
  nd_call = quote(tratio(y | effort ~ x + (1|site), data = df_pg, family = ratiod_poisson_gamma(),
                         spatial = spatial_gp(coords = cbind(lon, lat), level = "group", group_var = "spatial_site"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y ~ x + (1|site) + gp(lon, lat) + offset(log(effort)), data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 37: negbin_negbin + GP
results[["37"]] <- bench_validated("nb_gp", 37,
  nd_call = quote(tratio(y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin(),
                         spatial = spatial_gp(coords = cbind(lon, lat), level = "group", group_var = "spatial_site"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y ~ x + (1|site) + gp(lon, lat) + offset(log(denom)), data = df_nb, family = negbinomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 67: binomial + GP
results[["67"]] <- bench_validated("bin_gp", 67,
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
                         spatial = spatial_gp(coords = cbind(lon, lat), level = "group", group_var = "spatial_site"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + gp(lon, lat), data = df_bin, family = binomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# SECTION 3: ADDITIONAL TEMPORAL (RW2, AR1)
# ============================================================

# Row 42: negbin_negbin + RW2
results[["42"]] <- bench_validated("nb_rw2", 42,
  nd_call = quote(tratio(y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin(),
                         temporal = temporal_rw2("time"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y ~ x + (1|site) + (1|time) + offset(log(denom)), data = df_nb, family = negbinomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 43: negbin_negbin + AR1
results[["43"]] <- bench_validated("nb_ar1", 43,
  nd_call = quote(tratio(y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin(),
                         temporal = temporal_ar1("time"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y ~ x + (1|site) + (1|time) + offset(log(denom)), data = df_nb, family = negbinomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 72: binomial + RW2
results[["72"]] <- bench_validated("bin_rw2", 72,
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
                         temporal = temporal_rw2("time"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|time), data = df_bin, family = binomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 73: binomial + AR1
results[["73"]] <- bench_validated("bin_ar1", 73,
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
                         temporal = temporal_ar1("time"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|time), data = df_bin, family = binomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# SECTION 4: SPATIAL + TEMPORAL COMBINATIONS
# ============================================================

# Row 19: poisson_gamma + BYM2 + RW1
results[["19"]] <- bench_validated("pg_bym2_rw1", 19,
  nd_call = quote(tratio(y | effort ~ x + (1|site), data = df_pg, family = ratiod_poisson_gamma(),
                         spatial = spatial_bym2(adj_mat, level = "group", group_var = "spatial_site"),
                         temporal = temporal_rw1("time"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y ~ x + (1|site) + (1|spatial_site) + (1|time) + offset(log(effort)),
                        data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 20: poisson_gamma + ICAR + AR1
results[["20"]] <- bench_validated("pg_icar_ar1", 20,
  nd_call = quote(tratio(y | effort ~ x + (1|site), data = df_pg, family = ratiod_poisson_gamma(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                         temporal = temporal_ar1("time"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y ~ x + (1|site) + (1|spatial_site) + (1|time) + offset(log(effort)),
                        data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 49: negbin_negbin + BYM2 + RW1
results[["49"]] <- bench_validated("nb_bym2_rw1", 49,
  nd_call = quote(tratio(y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin(),
                         spatial = spatial_bym2(adj_mat, level = "group", group_var = "spatial_site"),
                         temporal = temporal_rw1("time"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y ~ x + (1|site) + (1|spatial_site) + (1|time) + offset(log(denom)),
                        data = df_nb, family = negbinomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 50: negbin_negbin + ICAR + AR1
results[["50"]] <- bench_validated("nb_icar_ar1", 50,
  nd_call = quote(tratio(y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                         temporal = temporal_ar1("time"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y ~ x + (1|site) + (1|spatial_site) + (1|time) + offset(log(denom)),
                        data = df_nb, family = negbinomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 81: binomial + BYM2 + RW1
results[["81"]] <- bench_validated("bin_bym2_rw1", 81,
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
                         spatial = spatial_bym2(adj_mat, level = "group", group_var = "spatial_site"),
                         temporal = temporal_rw1("time"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|spatial_site) + (1|time),
                        data = df_bin, family = binomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 82: binomial + ICAR + AR1
results[["82"]] <- bench_validated("bin_icar_ar1", 82,
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                         temporal = temporal_ar1("time"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|spatial_site) + (1|time),
                        data = df_bin, family = binomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# SECTION 5: ZI + SPATIAL
# ============================================================

# Row 24: poisson_gamma + ICAR + ZI
results[["24"]] <- bench_validated("pg_icar_zi", 24,
  nd_call = quote(tratio(y | effort ~ x + (1|site), data = df_pg_zi, family = ratiod_poisson_gamma(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                         zi = numdenom::zi_poisson(),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y ~ x + (1|site) + (1|spatial_site) + offset(log(effort)),
                        data = df_pg_zi, family = zero_inflated_poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 54: negbin_negbin + ICAR + ZI
results[["54"]] <- bench_validated("nb_icar_zi", 54,
  nd_call = quote(tratio(y | denom ~ x + (1|site), data = df_nb_zi, family = ratiod_negbin_negbin(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                         zi = numdenom::zi_negbin(),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y ~ x + (1|site) + (1|spatial_site) + offset(log(denom)),
                        data = df_nb_zi, family = zero_inflated_negbinomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 86: binomial + ICAR + ZI
results[["86"]] <- bench_validated("bin_icar_zi", 86,
  nd_call = quote(tratio(y | trials ~ x + (1|site), data = df_bin_zi, family = ratiod_zibinomial(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|spatial_site),
                        data = df_bin_zi, family = zero_inflated_binomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# SECTION 6: SLOPES + SPATIAL
# ============================================================

# Row 25: poisson_gamma + slopes + ICAR
results[["25"]] <- bench_validated("pg_slopes_icar", 25,
  nd_call = quote(tratio(y | effort ~ x + (x|site), data = df_pg, family = ratiod_poisson_gamma(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y ~ x + (x|site) + (1|spatial_site) + offset(log(effort)),
                        data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 55: negbin_negbin + slopes + ICAR
results[["55"]] <- bench_validated("nb_slopes_icar", 55,
  nd_call = quote(tratio(y | denom ~ x + (x|site), data = df_nb, family = ratiod_negbin_negbin(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y ~ x + (x|site) + (1|spatial_site) + offset(log(denom)),
                        data = df_nb, family = negbinomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 87: binomial + slopes + ICAR
results[["87"]] <- bench_validated("bin_slopes_icar", 87,
  nd_call = quote(tratio(y | trials ~ x + (x|site), data = df_bin, family = ratiod_binomial(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y | trials(trials) ~ x + (x|site) + (1|spatial_site),
                        data = df_bin, family = binomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# SECTION 7: NEGBIN HURDLE
# ============================================================

# Row 47: negbin_negbin + Hurdle
results[["47"]] <- bench_validated("nb_hurdle", 47,
  nd_call = quote(tratio(y | denom ~ x + (1|site), data = df_nb_zi, family = ratiod_negbin_negbin(),
                         zi = numdenom::hurdle_negbin(),
                         control = list(iter = 1000, warmup = 500, chains = 2, verbose = FALSE))),
  brms_call = quote(brm(y ~ x + (1|site) + offset(log(denom)),
                        data = df_nb_zi, family = brms::hurdle_negbinomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# Summary
# ============================================================
cat("\n\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("BATCH 2 VALIDATED BENCHMARK SUMMARY\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat(sprintf("%-8s %-20s %8s %8s %8s %6s %s\n",
            "Row", "Model", "numdenom", "Stan", "Speedup", "Diff", "Status"))
cat(paste(rep("-", 80), collapse = ""), "\n")

n_pass <- 0
n_fail <- 0
n_error <- 0

for (r in results) {
  if (!is.null(r$pass)) {
    if (r$pass) {
      n_pass <- n_pass + 1
      status <- "PASS"
    } else {
      n_fail <- n_fail + 1
      status <- "FAIL"
    }
    cat(sprintf("%-8s %-20s %8.1fs %8.1fs %8.1fx %6.4f %s\n",
                r$row, r$name, r$time_nd, r$time_brms, r$speedup, r$diff, status))
  } else if (!is.null(r$error_nd) || !is.null(r$error_brms)) {
    n_error <- n_error + 1
    cat(sprintf("%-8s %-20s %s\n", r$row, r$name, "ERROR"))
  }
}

cat(paste(rep("-", 80), collapse = ""), "\n")
cat(sprintf("Passed: %d | Failed: %d | Error: %d | Total: %d\n",
            n_pass, n_fail, n_error, length(results)))

if (n_fail > 0) {
  cat("\n*** WARNING: Some models failed validation! ***\n")
}

# Merge with existing results
if (file.exists("benchmarks/results_validated.rds")) {
  old_results <- readRDS("benchmarks/results_validated.rds")
  all_results <- c(old_results, results)
} else {
  all_results <- results
}

saveRDS(all_results, "benchmarks/results_validated.rds")
cat(sprintf("\nResults saved (total: %d models)\n", length(all_results)))
