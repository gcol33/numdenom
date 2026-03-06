# Validated Benchmark Batch 3: Next 25 Models
# 4-mode gradient timing (N, A, A_t, H) + Stan/brms validation
# Standard parameters: N=500, iter=500, warmup=250, chains=1 (for timing)
# Validation uses: iter=1000, warmup=500, chains=2

# Load package from source (avoids installation issues)
devtools::load_all(quiet = TRUE)
library(brms)
set.seed(123)

# ============================================================
# DATA SETUP (same as bench_validated.R)
# ============================================================

N <- 500
N_SITES <- 50
N_TIMES <- 20
x <- rnorm(N)
x2 <- rnorm(N)  # second covariate for slopes
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

# Crossed RE
site2 <- factor(sample(1:20, N, replace = TRUE))

# poisson_gamma data
y_pg <- rpois(N, exp(2 + 0.3*x + 0.2*x2))
effort <- rgamma(N, 10, 1)
df_pg <- data.frame(y = y_pg, effort = effort, x = x, x2 = x2, site = site,
                    site2 = site2, time = time, lon = lon, lat = lat,
                    spatial_site = spatial_site)

# negbin_negbin data
y_nb <- rnbinom(N, mu = exp(2 + 0.3*x + 0.2*x2), size = 5)
denom <- rnbinom(N, mu = 100, size = 10)
denom[denom == 0] <- 1
df_nb <- data.frame(y = y_nb, denom = denom, x = x, x2 = x2, site = site,
                    site2 = site2, time = time, lon = lon, lat = lat,
                    spatial_site = spatial_site)

# binomial data
trials <- sample(10:50, N, replace = TRUE)
y_bin <- rbinom(N, trials, plogis(0.5 + 0.3*x + 0.2*x2))
df_bin <- data.frame(y = y_bin, trials = trials, x = x, x2 = x2, site = site,
                     site2 = site2, time = time, lon = lon, lat = lat,
                     spatial_site = spatial_site)

# ZI data (add structural zeros)
df_pg_zi <- df_pg
df_pg_zi$y <- ifelse(runif(N) < 0.3, 0, df_pg_zi$y)

df_nb_zi <- df_nb
df_nb_zi$y <- ifelse(runif(N) < 0.3, 0, df_nb_zi$y)

df_bin_zi <- df_bin
df_bin_zi$y <- ifelse(runif(N) < 0.3, 0, df_bin_zi$y)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

# 2-mode gradient timing (H and A_t only - skip slow A and N modes)
# A_t provides gradient verification against H; A and N are too slow for batch runs
time_4modes <- function(nd_call, timeout = 300) {
  modes <- c("H", "A_t")  # Skip A (O(p*N), very slow) and N (also slow)
  times <- setNames(rep(NA_real_, 4), c("H", "A", "A_t", "N"))

  for (mode in modes) {
    timing_call <- nd_call
    timing_call$gradient_mode <- mode
    timing_call$iter <- 500
    timing_call$warmup <- 250
    timing_call$chains <- 1
    timing_call$verbose <- FALSE

    tryCatch({
      t <- system.time({ eval(timing_call) })["elapsed"]
      times[mode] <- round(t, 1)
      cat(sprintf("  %s: %.1fs", mode, t))
    }, error = function(e) {
      cat(sprintf("  %s: ERR", mode))
    })
  }
  cat("\n")
  times
}

# Stan validation - robust extraction for all model types
validate_stan <- function(nd_fit, brms_fit, param = "x") {
  # Defensive checks
  if (is.null(nd_fit) || is.null(nd_fit$draws)) {
    return(list(nd = NA, brms = NA, diff = NA, pass = NA,
                error = "numdenom fit or draws is NULL"))
  }

  nd_draws <- tryCatch(as.matrix(nd_fit$draws), error = function(e) NULL)
  if (is.null(nd_draws)) {
    return(list(nd = NA, brms = NA, diff = NA, pass = NA,
                error = "Could not convert draws to matrix"))
  }

  cols <- colnames(nd_draws)
  if (is.null(cols) || length(cols) == 0) {
    return(list(nd = NA, brms = NA, diff = NA, pass = NA,
                error = "Draws matrix has no column names"))
  }

  # Try multiple patterns to find the x coefficient
  nd_col <- NULL
  patterns <- c(
    "^beta_num\\[2\\]$",           # Standard: beta_num[2]
    "^beta_num_x$",                 # Named: beta_num_x
    paste0("^", param, "$"),        # Exact match
    paste0("beta_num.*", param),    # beta_num containing param
    "^beta\\[2\\]$"                 # Simple beta[2]
  )

  for (pat in patterns) {
    match <- grep(pat, cols, value = TRUE)
    if (length(match) > 0) {
      nd_col <- match[1]
      break
    }
  }

  # If still not found, return error
  if (is.null(nd_col) || is.na(nd_col)) {
    return(list(nd = NA, brms = NA, diff = NA, pass = NA,
                error = paste("Could not find param in:", paste(head(cols, 10), collapse=", "))))
  }

  nd_est <- mean(nd_draws[, nd_col])
  nd_sd <- sd(nd_draws[, nd_col])

  brms_sum <- fixef(brms_fit)
  brms_est <- brms_sum[param, "Estimate"]
  brms_se <- brms_sum[param, "Est.Error"]

  diff <- abs(nd_est - brms_est)
  threshold <- 2 * max(nd_sd, brms_se)

  list(nd = nd_est, brms = brms_est, diff = diff, pass = diff < threshold)
}

# Main benchmark function
# val_chains: number of chains for validation (default 2, use 1 for models with parallel bugs)
run_bench <- function(row, name, nd_call, brms_call = NULL, skip_stan = FALSE, val_chains = 2) {
  cat(sprintf("\n========== Row %d: %s ==========\n", row, name))
  result <- list(row = row, name = name)

  # 4-mode timing
  cat("4-mode timing:")
  result$times <- time_4modes(nd_call)

  # Full run for validation
  if (!skip_stan && !is.null(brms_call)) {
    cat(sprintf("Validation run (numdenom, chains=%d)... ", val_chains))
    flush.console()
    tryCatch({
      # Properly modify quoted call by converting to list, modifying, then back
      nd_call_full <- as.call(c(as.list(nd_call),
                                iter = 1000, warmup = 500, chains = val_chains, verbose = FALSE))
      t_nd <- system.time({ fit_nd <- eval(nd_call_full) })["elapsed"]
      result$time_nd <- round(t_nd, 1)
      cat(sprintf("%.1fs\n", t_nd))

      cat("Validation run (brms)... ")
      flush.console()
      t_brms <- system.time({ fit_brms <- eval(brms_call) })["elapsed"]
      result$time_brms <- round(t_brms, 1)
      cat(sprintf("%.1fs\n", t_brms))

      val <- validate_stan(fit_nd, fit_brms)

      # Handle extraction errors
      if (!is.null(val$error)) {
        cat(sprintf("  EXTRACTION ERROR: %s\n", val$error))
        result$error <- val$error
        result$speedup <- round(result$time_brms / result$time_nd, 1)
      } else {
        result$diff <- round(val$diff, 4)
        result$pass <- val$pass
        result$speedup <- round(result$time_brms / result$time_nd, 1)

        cat(sprintf("  numdenom=%.3f, brms=%.3f, diff=%.4f => %s (%.1fx)\n",
                    val$nd, val$brms, val$diff, if(val$pass) "PASS" else "FAIL", result$speedup))
      }
    }, error = function(e) {
      cat(sprintf("ERROR: %s\n", conditionMessage(e)))
      result$error <<- conditionMessage(e)
    })
  }

  result
}

# ============================================================
# BATCH 3: 25 MODELS
# ============================================================

results <- list()

# ----- POISSON_GAMMA FAMILY -----

# Row 3: pg + random slopes
results[["3"]] <- run_bench(3, "pg_slopes",
  nd_call = quote(ratiod(y | effort ~ x + (x | site), data = df_pg,
                         family = ratiod_poisson_gamma())),
  brms_call = quote(brm(y ~ x + (x | site) + offset(log(effort)), data = df_pg,
                        family = poisson(), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 4: pg + crossed RE
results[["4"]] <- run_bench(4, "pg_crossed",
  nd_call = quote(ratiod(y | effort ~ x + (1 | site) + (1 | site2), data = df_pg,
                         family = ratiod_poisson_gamma())),
  brms_call = quote(brm(y ~ x + (1 | site) + (1 | site2) + offset(log(effort)),
                        data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 10: pg + pCAR (proper CAR)
results[["10"]] <- run_bench(10, "pg_pcar",
  nd_call = quote(ratiod(y | effort ~ x + (1 | site), data = df_pg,
                         family = ratiod_poisson_gamma(),
                         spatial = spatial_car(adj_mat, proper = TRUE, level = "group",
                                               group_var = "spatial_site"))),
  brms_call = quote(brm(y ~ x + (1 | site) + (1 | spatial_site) + offset(log(effort)),
                        data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 11: pg + RW1 (needs 4-mode, has Stan)
results[["11"]] <- run_bench(11, "pg_rw1",
  nd_call = quote(ratiod(y | effort ~ x + (1 | site), data = df_pg,
                         family = ratiod_poisson_gamma(),
                         temporal = temporal_rw1("time"))),
  brms_call = quote(brm(y ~ x + (1 | site) + (1 | time) + offset(log(effort)),
                        data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 12: pg + RW2
results[["12"]] <- run_bench(12, "pg_rw2",
  nd_call = quote(ratiod(y | effort ~ x + (1 | site), data = df_pg,
                         family = ratiod_poisson_gamma(),
                         temporal = temporal_rw2("time"))),
  brms_call = quote(brm(y ~ x + (1 | site) + (1 | time) + offset(log(effort)),
                        data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 13: pg + AR1
results[["13"]] <- run_bench(13, "pg_ar1",
  nd_call = quote(ratiod(y | effort ~ x + (1 | site), data = df_pg,
                         family = ratiod_poisson_gamma(),
                         temporal = temporal_ar1("time"))),
  brms_call = quote(brm(y ~ x + (1 | site) + (1 | time) + offset(log(effort)),
                        data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 14: pg + temporal GP (use val_chains=1 due to parallel bug in temporal_gp)
results[["14"]] <- run_bench(14, "pg_gp_t",
  nd_call = quote(ratiod(y | effort ~ x + (1 | site), data = df_pg,
                         family = ratiod_poisson_gamma(),
                         temporal = temporal_gp("time"))),
  brms_call = quote(brm(y ~ x + (1 | site) + (1 | time) + offset(log(effort)),
                        data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 1,
                        backend = "cmdstanr", silent = 2, refresh = 0)),
  val_chains = 1  # temporal_gp has parallel chain bug
)

# Row 15: pg + multi-scale temporal (use val_chains=1 due to parallel bug)
results[["15"]] <- run_bench(15, "pg_ms_t",
  nd_call = quote(ratiod(y | effort ~ x + (1 | site), data = df_pg,
                         family = ratiod_poisson_gamma(),
                         temporal = temporal_multiscale("time"))),
  brms_call = quote(brm(y ~ x + (1 | site) + (1 | time) + offset(log(effort)),
                        data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 1,
                        backend = "cmdstanr", silent = 2, refresh = 0)),
  val_chains = 1  # temporal_multiscale has parallel chain bug
)

# Row 16: pg + ZI (needs 4-mode)
results[["16"]] <- run_bench(16, "pg_zi",
  nd_call = quote(ratiod(y | effort ~ x + (1 | site), data = df_pg_zi,
                         family = ratiod_poisson_gamma(),
                         zi = zi_poisson())),
  brms_call = quote(brm(y ~ x + (1 | site) + offset(log(effort)),
                        data = df_pg_zi, family = zero_inflated_poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 17: pg + Hurdle (needs 4-mode)
results[["17"]] <- run_bench(17, "pg_hurdle",
  nd_call = quote(ratiod(y | effort ~ x + (1 | site), data = df_pg_zi,
                         family = ratiod_poisson_gamma(),
                         zi = numdenom::hurdle_poisson())),
  brms_call = quote(brm(y ~ x + (1 | site) + offset(log(effort)),
                        data = df_pg_zi, family = brms::hurdle_poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 19: pg + BYM2 + RW1
results[["19"]] <- run_bench(19, "pg_bym2_rw1",
  nd_call = quote(ratiod(y | effort ~ x + (1 | site), data = df_pg,
                         family = ratiod_poisson_gamma(),
                         spatial = spatial_bym2(adj_mat, level = "group", group_var = "spatial_site"),
                         temporal = temporal_rw1("time"))),
  brms_call = quote(brm(y ~ x + (1 | site) + (1 | spatial_site) + (1 | time) + offset(log(effort)),
                        data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 20: pg + ICAR + AR1
results[["20"]] <- run_bench(20, "pg_icar_ar1",
  nd_call = quote(ratiod(y | effort ~ x + (1 | site), data = df_pg,
                         family = ratiod_poisson_gamma(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                         temporal = temporal_ar1("time"))),
  brms_call = quote(brm(y ~ x + (1 | site) + (1 | spatial_site) + (1 | time) + offset(log(effort)),
                        data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 24: pg + ICAR + ZI
results[["24"]] <- run_bench(24, "pg_icar_zi",
  nd_call = quote(ratiod(y | effort ~ x + (1 | site), data = df_pg_zi,
                         family = ratiod_poisson_gamma(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                         zi = zi_poisson())),
  brms_call = quote(brm(y ~ x + (1 | site) + (1 | spatial_site) + offset(log(effort)),
                        data = df_pg_zi, family = zero_inflated_poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 25: pg + slopes + ICAR
results[["25"]] <- run_bench(25, "pg_slopes_icar",
  nd_call = quote(ratiod(y | effort ~ x + (x | site), data = df_pg,
                         family = ratiod_poisson_gamma(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"))),
  brms_call = quote(brm(y ~ x + (x | site) + (1 | spatial_site) + offset(log(effort)),
                        data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 27: pg + TVC
results[["27"]] <- run_bench(27, "pg_tvc",
  nd_call = quote(ratiod(y | effort ~ x + (1 | site), data = df_pg,
                         family = ratiod_poisson_gamma(),
                         temporal = temporal_tvc("time", ~ x))),
  brms_call = quote(brm(y ~ x + (1 | site) + (1 | time) + offset(log(effort)),
                        data = df_pg, family = poisson(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ----- NEGBIN_NEGBIN FAMILY -----

# Row 33: nb + random slopes
results[["33"]] <- run_bench(33, "nb_slopes",
  nd_call = quote(ratiod(y | denom ~ x + (x | site), data = df_nb,
                         family = ratiod_negbin_negbin())),
  brms_call = quote(brm(y ~ x + (x | site) + offset(log(denom)), data = df_nb,
                        family = negbinomial(), iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 40: nb + pCAR
results[["40"]] <- run_bench(40, "nb_pcar",
  nd_call = quote(ratiod(y | denom ~ x + (1 | site), data = df_nb,
                         family = ratiod_negbin_negbin(),
                         spatial = spatial_car(adj_mat, proper = TRUE, level = "group",
                                               group_var = "spatial_site"))),
  brms_call = quote(brm(y ~ x + (1 | site) + (1 | spatial_site) + offset(log(denom)),
                        data = df_nb, family = negbinomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 41: nb + RW1
results[["41"]] <- run_bench(41, "nb_rw1",
  nd_call = quote(ratiod(y | denom ~ x + (1 | site), data = df_nb,
                         family = ratiod_negbin_negbin(),
                         temporal = temporal_rw1("time"))),
  brms_call = quote(brm(y ~ x + (1 | site) + (1 | time) + offset(log(denom)),
                        data = df_nb, family = negbinomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 42: nb + RW2
results[["42"]] <- run_bench(42, "nb_rw2",
  nd_call = quote(ratiod(y | denom ~ x + (1 | site), data = df_nb,
                         family = ratiod_negbin_negbin(),
                         temporal = temporal_rw2("time"))),
  brms_call = quote(brm(y ~ x + (1 | site) + (1 | time) + offset(log(denom)),
                        data = df_nb, family = negbinomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 43: nb + AR1
results[["43"]] <- run_bench(43, "nb_ar1",
  nd_call = quote(ratiod(y | denom ~ x + (1 | site), data = df_nb,
                         family = ratiod_negbin_negbin(),
                         temporal = temporal_ar1("time"))),
  brms_call = quote(brm(y ~ x + (1 | site) + (1 | time) + offset(log(denom)),
                        data = df_nb, family = negbinomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 44: nb + temporal GP (use val_chains=1 due to parallel bug in temporal_gp)
results[["44"]] <- run_bench(44, "nb_gp_t",
  nd_call = quote(ratiod(y | denom ~ x + (1 | site), data = df_nb,
                         family = ratiod_negbin_negbin(),
                         temporal = temporal_gp("time"))),
  brms_call = quote(brm(y ~ x + (1 | site) + (1 | time) + offset(log(denom)),
                        data = df_nb, family = negbinomial(),
                        iter = 1000, warmup = 500, chains = 1,
                        backend = "cmdstanr", silent = 2, refresh = 0)),
  val_chains = 1  # temporal_gp has parallel chain bug
)

# Row 45: nb + multi-scale temporal (use val_chains=1 due to parallel bug)
results[["45"]] <- run_bench(45, "nb_ms_t",
  nd_call = quote(ratiod(y | denom ~ x + (1 | site), data = df_nb,
                         family = ratiod_negbin_negbin(),
                         temporal = temporal_multiscale("time"))),
  brms_call = quote(brm(y ~ x + (1 | site) + (1 | time) + offset(log(denom)),
                        data = df_nb, family = negbinomial(),
                        iter = 1000, warmup = 500, chains = 1,
                        backend = "cmdstanr", silent = 2, refresh = 0)),
  val_chains = 1  # temporal_multiscale has parallel chain bug
)

# Row 47: nb + Hurdle
results[["47"]] <- run_bench(47, "nb_hurdle",
  nd_call = quote(ratiod(y | denom ~ x + (1 | site), data = df_nb_zi,
                         family = ratiod_negbin_negbin(),
                         zi = numdenom::hurdle_negbin())),
  brms_call = quote(brm(y ~ x + (1 | site) + offset(log(denom)),
                        data = df_nb_zi, family = brms::hurdle_negbinomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 49: nb + BYM2 + RW1
results[["49"]] <- run_bench(49, "nb_bym2_rw1",
  nd_call = quote(ratiod(y | denom ~ x + (1 | site), data = df_nb,
                         family = ratiod_negbin_negbin(),
                         spatial = spatial_bym2(adj_mat, level = "group", group_var = "spatial_site"),
                         temporal = temporal_rw1("time"))),
  brms_call = quote(brm(y ~ x + (1 | site) + (1 | spatial_site) + (1 | time) + offset(log(denom)),
                        data = df_nb, family = negbinomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# Row 50: nb + ICAR + AR1
results[["50"]] <- run_bench(50, "nb_icar_ar1",
  nd_call = quote(ratiod(y | denom ~ x + (1 | site), data = df_nb,
                         family = ratiod_negbin_negbin(),
                         spatial = spatial_car(adj_mat, level = "group", group_var = "spatial_site"),
                         temporal = temporal_ar1("time"))),
  brms_call = quote(brm(y ~ x + (1 | site) + (1 | spatial_site) + (1 | time) + offset(log(denom)),
                        data = df_nb, family = negbinomial(),
                        iter = 1000, warmup = 500, chains = 2,
                        backend = "cmdstanr", silent = 2, refresh = 0))
)

# ============================================================
# SUMMARY
# ============================================================

cat("\n\n")
cat(paste(rep("=", 90), collapse = ""), "\n")
cat("BATCH 3 RESULTS SUMMARY\n")
cat(paste(rep("=", 90), collapse = ""), "\n")

# 4-mode timing table
cat("\n4-MODE GRADIENT TIMING (seconds):\n")
cat(sprintf("%-6s %-20s %8s %8s %8s %8s\n", "Row", "Model", "H", "A", "A_t", "N"))
cat(paste(rep("-", 70), collapse = ""), "\n")
for (r in results) {
  if (!is.null(r$times)) {
    cat(sprintf("%-6s %-20s %8.1f %8.1f %8.1f %8.1f\n",
                r$row, r$name,
                ifelse(is.na(r$times["H"]), NA, r$times["H"]),
                ifelse(is.na(r$times["A"]), NA, r$times["A"]),
                ifelse(is.na(r$times["A_t"]), NA, r$times["A_t"]),
                ifelse(is.na(r$times["N"]), NA, r$times["N"])))
  }
}

# Validation table
cat("\n\nSTAN VALIDATION:\n")
cat(sprintf("%-6s %-20s %10s %10s %8s %8s %s\n",
            "Row", "Model", "numdenom", "brms", "Speedup", "Diff", "Status"))
cat(paste(rep("-", 80), collapse = ""), "\n")

n_pass <- 0
n_total <- 0

for (r in results) {
  if (!is.null(r$pass)) {
    n_total <- n_total + 1
    if (r$pass) n_pass <- n_pass + 1
    cat(sprintf("%-6s %-20s %10.1fs %10.1fs %8.1fx %8.4f %s\n",
                r$row, r$name, r$time_nd, r$time_brms, r$speedup, r$diff,
                if(r$pass) "PASS" else "FAIL"))
  } else if (!is.null(r$error)) {
    cat(sprintf("%-6s %-20s ERROR: %s\n", r$row, r$name, r$error))
  }
}

cat(paste(rep("-", 80), collapse = ""), "\n")
cat(sprintf("Validated: %d/%d passed\n", n_pass, n_total))

# Save results
saveRDS(results, "benchmarks/results_batch3.rds")
cat("\nResults saved to benchmarks/results_batch3.rds\n")

# Print markdown table for gradient_methods.md update
cat("\n\nMARKDOWN TABLE FOR gradient_methods.md:\n")
cat("| # | Family | RE | Spatial | Temporal | ZI | Grad | H(s) | A(s) | A_t(s) | N(s) | Notes |\n")
for (r in results) {
  if (!is.null(r$times)) {
    cat(sprintf("| %s | ... | ... | ... | ... | ... | H | %.1f | %.1f | %.1f | %.1f | |\n",
                r$row,
                ifelse(is.na(r$times["H"]), NA, r$times["H"]),
                ifelse(is.na(r$times["A"]), NA, r$times["A"]),
                ifelse(is.na(r$times["A_t"]), NA, r$times["A_t"]),
                ifelse(is.na(r$times["N"]), NA, r$times["N"])))
  }
}
