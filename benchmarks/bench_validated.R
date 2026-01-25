# Integrated Benchmark + Stan Validation
# Every benchmark includes Stan comparison for correctness validation
# Standard parameters: N=500, iter=1000, warmup=500, chains=2

library(numdenom)
library(brms)
set.seed(123)

# Setup data
N <- 500
N_SITES <- 50
N_TIMES <- 20
x <- rnorm(N)
site <- factor(rep(1:N_SITES, length.out = N))
time <- factor(rep(1:N_TIMES, length.out = N))

# Spatial grid
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

# Validation function
validate_against_stan <- function(nd_fit, brms_fit, param_name = "x") {
  # Extract numdenom draws
  nd_draws <- as.matrix(nd_fit$draws)
  nd_col <- grep(paste0("beta_num\\[2\\]"), colnames(nd_draws), value = TRUE)[1]
  if (is.na(nd_col)) nd_col <- grep(param_name, colnames(nd_draws), value = TRUE)[1]
  nd_x <- mean(nd_draws[, nd_col])
  nd_x_sd <- sd(nd_draws[, nd_col])

  # Extract brms fixed effects
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
bench_validated <- function(name, row, nd_call, brms_call, true_x = 0.3) {
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
    cat(sprintf("%.1fs (divergent: %d)\n", time_nd, result$divergent))
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", conditionMessage(e)))
    result$error_nd <<- conditionMessage(e)
  })

  # Run brms/Stan
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
# Row 1: poisson_gamma base (no RE)
# ============================================================
results[["1"]] <- bench_validated("pg_base", 1,
  nd_call = quote(ratiod(y | effort ~ x, data = df_pg, family = ratiod_poisson_gamma(),
                         iter = 1000, warmup = 500, chains = 2, verbose = FALSE)),
  brms_call = quote(brm(y ~ x + offset(log(effort)), data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# Row 2: poisson_gamma + RE
# ============================================================
results[["2"]] <- bench_validated("pg_re", 2,
  nd_call = quote(ratiod(y | effort ~ x + (1|site), data = df_pg, family = ratiod_poisson_gamma(),
                         iter = 1000, warmup = 500, chains = 2, verbose = FALSE)),
  brms_call = quote(brm(y ~ x + (1|site) + offset(log(effort)), data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# Row 5: poisson_gamma + ICAR
# ============================================================
results[["5"]] <- bench_validated("pg_icar", 5,
  nd_call = quote(ratiod(y | effort ~ x + (1|site), data = df_pg, family = ratiod_poisson_gamma(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                         iter = 1000, warmup = 500, chains = 2, verbose = FALSE)),
  brms_call = quote(brm(y ~ x + (1|site) + (1|spatial_site) + offset(log(effort)),
                        data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# Row 11: poisson_gamma + RW1 temporal
# ============================================================
results[["11"]] <- bench_validated("pg_rw1", 11,
  nd_call = quote(ratiod(y | effort ~ x + (1|site), data = df_pg, family = ratiod_poisson_gamma(),
                         temporal = temporal_rw1("time"),
                         iter = 1000, warmup = 500, chains = 2, verbose = FALSE)),
  brms_call = quote(brm(y ~ x + (1|site) + (1|time) + offset(log(effort)),
                        data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# Row 31: negbin_negbin base
# ============================================================
results[["31"]] <- bench_validated("nb_base", 31,
  nd_call = quote(ratiod(y | denom ~ x, data = df_nb, family = ratiod_negbin_negbin(),
                         iter = 1000, warmup = 500, chains = 2, verbose = FALSE)),
  brms_call = quote(brm(y ~ x + offset(log(denom)), data = df_nb, family = negbinomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# Row 32: negbin_negbin + RE
# ============================================================
results[["32"]] <- bench_validated("nb_re", 32,
  nd_call = quote(ratiod(y | denom ~ x + (1|site), data = df_nb, family = ratiod_negbin_negbin(),
                         iter = 1000, warmup = 500, chains = 2, verbose = FALSE)),
  brms_call = quote(brm(y ~ x + (1|site) + offset(log(denom)), data = df_nb, family = negbinomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# Row 61: binomial base
# ============================================================
results[["61"]] <- bench_validated("bin_base", 61,
  nd_call = quote(ratiod(y | trials ~ x, data = df_bin, family = ratiod_binomial(),
                         iter = 1000, warmup = 500, chains = 2, verbose = FALSE)),
  brms_call = quote(brm(y | trials(trials) ~ x, data = df_bin, family = binomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# Row 62: binomial + RE
# ============================================================
results[["62"]] <- bench_validated("bin_re", 62,
  nd_call = quote(ratiod(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
                         iter = 1000, warmup = 500, chains = 2, verbose = FALSE)),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site), data = df_bin, family = binomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# Row 65: binomial + ICAR
# ============================================================
results[["65"]] <- bench_validated("bin_icar", 65,
  nd_call = quote(ratiod(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                         iter = 1000, warmup = 500, chains = 2, verbose = FALSE)),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|spatial_site),
                        data = df_bin, family = binomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# Row 71: binomial + RW1 temporal
# ============================================================
results[["71"]] <- bench_validated("bin_rw1", 71,
  nd_call = quote(ratiod(y | trials ~ x + (1|site), data = df_bin, family = ratiod_binomial(),
                         temporal = temporal_rw1("time"),
                         iter = 1000, warmup = 500, chains = 2, verbose = FALSE)),
  brms_call = quote(brm(y | trials(trials) ~ x + (1|site) + (1|time),
                        data = df_bin, family = binomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# Summary
# ============================================================
cat("\n\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
cat("VALIDATED BENCHMARK SUMMARY\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
cat(sprintf("%-8s %-15s %8s %8s %8s %6s %s\n",
            "Row", "Model", "numdenom", "Stan", "Speedup", "Diff", "Status"))
cat(paste(rep("-", 70), collapse = ""), "\n")

n_pass <- 0
n_total <- 0

for (r in results) {
  if (!is.null(r$pass)) {
    n_total <- n_total + 1
    if (r$pass) n_pass <- n_pass + 1
    status <- if(r$pass) "PASS" else "FAIL"
    cat(sprintf("%-8s %-15s %8.1fs %8.1fs %8.1fx %6.4f %s\n",
                r$row, r$name, r$time_nd, r$time_brms, r$speedup, r$diff, status))
  } else if (!is.null(r$error_nd) || !is.null(r$error_brms)) {
    cat(sprintf("%-8s %-15s %s\n", r$row, r$name, "ERROR"))
  }
}

cat(paste(rep("-", 70), collapse = ""), "\n")
cat(sprintf("Validated: %d/%d passed\n", n_pass, n_total))

if (n_pass < n_total) {
  cat("\n*** WARNING: Some models failed validation! ***\n")
}

saveRDS(results, "benchmarks/results_validated.rds")
cat("\nResults saved to benchmarks/results_validated.rds\n")
