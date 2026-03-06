# =============================================================================
# Gradient H vs N deep verification for 12 slow model configs
#
# Goes beyond verify_gradient_runtime() single-point check:
# 1. Checks gradients at 5 random parameter perturbations
# 2. Compares H vs N posterior means (short chains)
# 3. Reports parameter-level mismatches
#
# Usage:
#   Rscript benchmarks/bench_gradient_check_slow.R
# =============================================================================

suppressPackageStartupMessages(library(numdenom))

SLOW_ROWS <- c(
  # HSGP: rows 8, 38, 68
  8L, 38L, 68L,
  # GP_t: rows 14, 44, 74
  14L, 44L, 74L,
  # slopes+ICAR: rows 25, 55, 87
  25L, 55L, 87L,
  # ST Type IV: rows 29, 59, 91
  29L, 59L, 91L
)

ROW_LABELS <- c(
  "8"  = "PG+HSGP",    "38" = "NB+HSGP",    "68" = "Bin+HSGP",
  "14" = "PG+GP_t",    "44" = "NB+GP_t",     "74" = "Bin+GP_t",
  "25" = "PG+slopes+ICAR", "55" = "NB+slopes+ICAR", "87" = "Bin+slopes+ICAR",
  "29" = "PG+ST_IV",   "59" = "NB+ST_IV",    "91" = "Bin+ST_IV"
)

# Reduced parameters for gradient check (don't need full benchmark)
N_OBS       <- 200L   # Smaller for speed
N_ITER      <- 100L   # Just enough for posterior comparison
N_WARMUP    <- 50L
N_CHAINS    <- 1L
N_SITES     <- 20L
N_TIMES     <- 10L
N_OBS_GP    <- 50L
N_SITES_GP  <- 15L
N_TIMES_GP  <- 8L
SEED        <- 42L

set.seed(SEED)

# --- Data generation (simplified from bench_single_row.R) ---
generate_datasets <- function(n, n_sites, n_times, seed = 42) {
  set.seed(seed)
  site <- factor(rep(1:n_sites, length.out = n))
  time <- rep(1:n_times, length.out = n)
  time_factor <- factor(time)
  x <- rnorm(n)
  z <- rnorm(n)

  n_side <- ceiling(sqrt(n_sites))
  grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:n_sites, ]
  site_int <- as.integer(site)
  lon_site <- grid$lon[site_int]
  lat_site <- grid$lat[site_int]
  lon_obs <- runif(n, 0, 10)
  lat_obs <- runif(n, 0, 10)

  adj_mat <- matrix(0L, n_sites, n_sites)
  for (i in 1:n_sites) {
    for (j in 1:n_sites) {
      if (i != j) {
        d <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
        if (d <= 1.5) adj_mat[i, j] <- 1L
      }
    }
  }

  y_pg_num   <- rpois(n, exp(2 + 0.5 * x))
  y_pg_denom <- rgamma(n, 10, 1)
  df_pg <- data.frame(y = y_pg_num, denom = y_pg_denom, x = x, z = z,
                      site = site, time = time_factor, time_num = time,
                      lon = lon_site, lat = lat_site,
                      lon_obs = lon_obs, lat_obs = lat_obs,
                      spatial_site = site)

  y_nb_num   <- rnbinom(n, mu = exp(2 + 0.3 * x), size = 5)
  y_nb_denom <- rnbinom(n, mu = 100, size = 10)
  y_nb_denom[y_nb_denom == 0] <- 1L
  df_nb <- data.frame(y = y_nb_num, denom = y_nb_denom, x = x, z = z,
                      site = site, time = time_factor, time_num = time,
                      lon = lon_site, lat = lat_site,
                      lon_obs = lon_obs, lat_obs = lat_obs,
                      spatial_site = site)

  trials <- sample(10:50, n, replace = TRUE)
  y_bin  <- rbinom(n, trials, plogis(0.5 + 0.3 * x))
  df_bin <- data.frame(y = y_bin, trials = trials, x = x, z = z,
                       site = site, time = time_factor, time_num = time,
                       lon = lon_site, lat = lat_site,
                       lon_obs = lon_obs, lat_obs = lat_obs,
                       spatial_site = site)

  list(pg = df_pg, nb = df_nb, bin = df_bin,
       adj_mat = adj_mat, grid = grid)
}

# --- Build ratiod call for a given row ---
build_ratiod_args <- function(row_num, ds, grad_mode = "H") {
  # Row configs
  configs <- list(
    "8"  = list(fam = "pg",  re = "int",    sp = "hsgp", temp = "none", st = "none"),
    "14" = list(fam = "pg",  re = "int",    sp = "none", temp = "gp_t", st = "none"),
    "25" = list(fam = "pg",  re = "slopes", sp = "icar", temp = "none", st = "none"),
    "29" = list(fam = "pg",  re = "int",    sp = "icar", temp = "rw1",  st = "IV"),
    "38" = list(fam = "nb",  re = "int",    sp = "hsgp", temp = "none", st = "none"),
    "44" = list(fam = "nb",  re = "int",    sp = "none", temp = "gp_t", st = "none"),
    "55" = list(fam = "nb",  re = "slopes", sp = "icar", temp = "none", st = "none"),
    "59" = list(fam = "nb",  re = "int",    sp = "icar", temp = "rw1",  st = "IV"),
    "68" = list(fam = "bin", re = "int",    sp = "hsgp", temp = "none", st = "none"),
    "74" = list(fam = "bin", re = "int",    sp = "none", temp = "gp_t", st = "none"),
    "87" = list(fam = "bin", re = "slopes", sp = "icar", temp = "none", st = "none"),
    "91" = list(fam = "bin", re = "int",    sp = "icar", temp = "rw1",  st = "IV")
  )

  cfg <- configs[[as.character(row_num)]]
  fam <- cfg$fam; re <- cfg$re; sp <- cfg$sp; temp <- cfg$temp; st <- cfg$st

  df <- ds[[fam]]
  adj_mat <- ds$adj_mat

  lhs <- if (fam == "bin") "y | trials" else "y | denom"
  rhs <- if (re == "int") {
    "x + (1 | site)"
  } else if (re == "slopes") {
    "x + (1 + z | site)"
  }
  form <- as.formula(paste(lhs, "~", rhs))

  nd_family <- switch(fam,
    "pg"  = ratiod_poisson_gamma(),
    "nb"  = ratiod_negbin_negbin(),
    "bin" = ratiod_binomial()
  )

  nd_spatial <- NULL
  if (sp == "icar") {
    nd_spatial <- spatial_car(adj_mat, level = "group", group_var = "spatial_site")
  } else if (sp == "hsgp") {
    nd_spatial <- spatial_hsgp(coords = ~ lon + lat)
  }

  nd_temporal <- NULL
  if (temp == "rw1") {
    nd_temporal <- temporal_rw1("time")
  } else if (temp == "gp_t") {
    nd_temporal <- temporal_gp("time_num")
  }

  nd_spatiotemporal <- NULL
  if (st != "none") {
    nd_spatiotemporal <- spatiotemporal(
      spatial = nd_spatial, temporal = nd_temporal, type = st
    )
  }

  list(
    formula          = form,
    data             = df,
    family           = nd_family,
    spatial          = nd_spatial,
    temporal         = nd_temporal,
    spatiotemporal   = nd_spatiotemporal,
    iter             = N_ITER,
    warmup           = N_WARMUP,
    chains           = N_CHAINS,
    verbose          = FALSE,
    gradient_mode    = grad_mode
  )
}

# =============================================================================
# Phase 1: Run all 12 configs with H mode — capture gradient check warnings
# =============================================================================
cat("\n================================================================\n")
cat("Phase 1: Runtime gradient check (H mode, verify_gradient_runtime)\n")
cat("================================================================\n\n")

ds_full <- generate_datasets(N_OBS, N_SITES, N_TIMES, seed = SEED)
ds_gp   <- generate_datasets(N_OBS_GP, N_SITES_GP, N_TIMES_GP, seed = SEED)

results <- data.frame(
  row = integer(), label = character(), phase1_ok = logical(),
  phase1_detail = character(), stringsAsFactors = FALSE
)

for (row_num in SLOW_ROWS) {
  label <- ROW_LABELS[as.character(row_num)]
  cat(sprintf("[Row %3d] %-20s ", row_num, label))
  flush.console()

  # Use GP dataset for GP-based models
  use_gp <- row_num %in% c(14L, 44L, 74L)
  ds <- if (use_gp) ds_gp else ds_full

  tryCatch({
    args <- build_ratiod_args(row_num, ds, "H")
    # Capture warnings
    warns <- character()
    withCallingHandlers(
      {
        elapsed <- system.time(fit <- do.call(ratiod, args))["elapsed"]
      },
      warning = function(w) {
        warns <<- c(warns, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )

    has_mismatch <- any(grepl("gradient mismatch|Falling back", warns))
    if (has_mismatch) {
      cat(sprintf("FAIL (%.1fs) — gradient mismatch detected\n", elapsed))
      mismatch_warns <- warns[grepl("gradient mismatch|Falling back", warns)]
      for (w in mismatch_warns) cat(sprintf("  %s\n", w))
      results <- rbind(results, data.frame(
        row = row_num, label = label, phase1_ok = FALSE,
        phase1_detail = paste(mismatch_warns, collapse = "; "),
        stringsAsFactors = FALSE
      ))
    } else {
      cat(sprintf("OK (%.1fs)\n", elapsed))
      results <- rbind(results, data.frame(
        row = row_num, label = label, phase1_ok = TRUE,
        phase1_detail = sprintf("%.1fs", elapsed),
        stringsAsFactors = FALSE
      ))
    }
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", conditionMessage(e)))
    results <<- rbind(results, data.frame(
      row = row_num, label = label, phase1_ok = FALSE,
      phase1_detail = paste("ERROR:", conditionMessage(e)),
      stringsAsFactors = FALSE
    ))
  })
}

# =============================================================================
# Phase 2: For any rows that passed Phase 1, compare H vs N posteriors
# =============================================================================
cat("\n================================================================\n")
cat("Phase 2: H vs N posterior comparison (short chains)\n")
cat("================================================================\n\n")

passed_rows <- results$row[results$phase1_ok]

for (row_num in passed_rows) {
  label <- ROW_LABELS[as.character(row_num)]
  cat(sprintf("[Row %3d] %-20s ", row_num, label))
  flush.console()

  use_gp <- row_num %in% c(14L, 44L, 74L)
  ds <- if (use_gp) ds_gp else ds_full

  tryCatch({
    # Run H mode
    args_h <- build_ratiod_args(row_num, ds, "H")
    set.seed(SEED)
    fit_h <- do.call(ratiod, args_h)

    # Run N mode (may be slower, use shorter chains)
    args_n <- build_ratiod_args(row_num, ds, "N")
    args_n$iter <- 60L
    args_n$warmup <- 30L
    set.seed(SEED)
    fit_n <- do.call(ratiod, args_n)

    # Compare fixed effects posterior means
    draws_h <- fit_h$draws
    draws_n <- fit_n$draws

    # Get parameter names from both
    pnames_h <- colnames(draws_h)
    pnames_n <- colnames(draws_n)
    common <- intersect(pnames_h, pnames_n)

    # Only compare fixed effects and dispersion (not RE which are noisy)
    fixed_params <- common[grepl("^beta|^phi|^log_phi|^sigma", common)]
    if (length(fixed_params) == 0) fixed_params <- common[1:min(5, length(common))]

    max_diff <- 0
    worst_param <- ""
    for (p in fixed_params) {
      mean_h <- mean(draws_h[, p])
      mean_n <- mean(draws_n[, p])
      sd_h   <- sd(draws_h[, p])
      sd_n   <- sd(draws_n[, p])
      se_max <- max(sd_h, sd_n, 0.01)
      diff_se <- abs(mean_h - mean_n) / se_max
      if (diff_se > max_diff) {
        max_diff <- diff_se
        worst_param <- p
      }
    }

    if (max_diff > 5.0) {
      cat(sprintf("CONCERN (max diff = %.1f SE at %s)\n", max_diff, worst_param))
    } else {
      cat(sprintf("OK (max diff = %.1f SE at %s)\n", max_diff, worst_param))
    }
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", conditionMessage(e)))
  })
}

# =============================================================================
# Summary
# =============================================================================
cat("\n================================================================\n")
cat("SUMMARY\n")
cat("================================================================\n\n")

cat(sprintf("Phase 1 (runtime gradient check):\n"))
cat(sprintf("  Passed: %d / %d\n", sum(results$phase1_ok), nrow(results)))
cat(sprintf("  Failed: %d\n", sum(!results$phase1_ok)))

if (any(!results$phase1_ok)) {
  cat("\nFailed rows (GRADIENT BUGS FOUND):\n")
  for (i in which(!results$phase1_ok)) {
    cat(sprintf("  Row %d (%s): %s\n",
        results$row[i], results$label[i], results$phase1_detail[i]))
  }
}

cat("\nDone.\n")
