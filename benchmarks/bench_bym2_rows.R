# =============================================================================
# BYM2 Rows Benchmark — OAS shrinkage + Riebler reparameterization
# Rows: 6, 19, 36, 49, 66, 81
# Captures: timing, divergences, treedepth, mass matrix type, shrinkage
# =============================================================================

devtools::load_all()

N_OBS    <- 500L
N_ITER   <- 500L
N_WARMUP <- 250L
N_CHAINS <- 1L
N_SITES  <- 50L
N_TIMES  <- 20L
SEEDS    <- c(42L, 99L, 123L, 314L)

set.seed(42)

# --- Data generation ---
generate_data <- function(n, n_sites, n_times, seed = 42) {
  set.seed(seed)
  site <- factor(rep(1:n_sites, length.out = n))
  time <- rep(1:n_times, length.out = n)
  time_factor <- factor(time)
  x <- rnorm(n)

  n_side <- ceiling(sqrt(n_sites))
  grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:n_sites, ]
  site_int <- as.integer(site)
  lon_site <- grid$lon[site_int]
  lat_site <- grid$lat[site_int]

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
  df_pg <- data.frame(y = y_pg_num, denom = y_pg_denom, x = x,
                      site = site, time = time_factor,
                      lon = lon_site, lat = lat_site,
                      spatial_site = site)

  y_nb_num   <- rnbinom(n, mu = exp(2 + 0.3 * x), size = 5)
  y_nb_denom <- rnbinom(n, mu = 100, size = 10)
  y_nb_denom[y_nb_denom == 0] <- 1L
  df_nb <- data.frame(y = y_nb_num, denom = y_nb_denom, x = x,
                      site = site, time = time_factor,
                      lon = lon_site, lat = lat_site,
                      spatial_site = site)

  trials <- sample(10:50, n, replace = TRUE)
  y_bin  <- rbinom(n, trials, plogis(0.5 + 0.3 * x))
  df_bin <- data.frame(y = y_bin, trials = trials, x = x,
                       site = site, time = time_factor,
                       lon = lon_site, lat = lat_site,
                       spatial_site = site)

  list(pg = df_pg, nb = df_nb, bin = df_bin, adj_mat = adj_mat)
}

# --- Row definitions ---
bym2_rows <- list(
  list(row = 6,  name = "PG+BYM2",      fam = "pg",  temp = FALSE),
  list(row = 19, name = "PG+BYM2+RW1",  fam = "pg",  temp = TRUE),
  list(row = 36, name = "NB+BYM2",      fam = "nb",  temp = FALSE),
  list(row = 49, name = "NB+BYM2+RW1",  fam = "nb",  temp = TRUE),
  list(row = 66, name = "Bin+BYM2",     fam = "bin", temp = FALSE),
  list(row = 81, name = "Bin+BYM2+RW1", fam = "bin", temp = TRUE)
)

# --- Run benchmarks ---
results <- list()

for (row_cfg in bym2_rows) {
  for (seed in SEEDS) {
    ds <- generate_data(N_OBS, N_SITES, N_TIMES, seed = seed)
    df <- ds[[row_cfg$fam]]
    adj_mat <- ds$adj_mat

    nd_family <- switch(row_cfg$fam,
      "pg"  = ratiod_poisson_gamma(),
      "nb"  = ratiod_negbin_negbin(),
      "bin" = ratiod_binomial()
    )

    lhs <- if (row_cfg$fam == "bin") "y | trials" else "y | denom"
    form <- as.formula(paste(lhs, "~ x + (1 | site)"))

    nd_spatial <- spatial_bym2(adj_mat, level = "group", group_var = "spatial_site")
    nd_temporal <- if (row_cfg$temp) temporal_rw1("time") else NULL

    label <- sprintf("Row %d (%s) seed=%d", row_cfg$row, row_cfg$name, seed)
    cat(sprintf("\n--- %s ---\n", label))

    tryCatch({
      elapsed <- system.time({
        fit <- ratiod(
          formula   = form,
          data      = df,
          family    = nd_family,
          spatial   = nd_spatial,
          temporal  = nd_temporal,
          iter      = N_ITER,
          warmup    = N_WARMUP,
          chains    = N_CHAINS,
          seed      = seed,
          verbose   = TRUE,
          gradient_mode = "H"
        )
      })["elapsed"]

      # Extract diagnostics
      diag <- fit$diagnostics
      n_div <- if (!is.null(diag$n_divergent)) diag$n_divergent else NA
      avg_td <- if (!is.null(diag$avg_treedepth)) diag$avg_treedepth else NA
      max_td <- if (!is.null(diag$max_treedepth)) diag$max_treedepth else NA
      pct_maxd <- if (!is.null(diag$pct_max_treedepth)) diag$pct_max_treedepth else NA
      sampler <- if (!is.null(diag$sampler)) diag$sampler else "unknown"

      # Extract posterior summaries for BYM2 params
      sigma_sp <- NA; rho_sp <- NA
      if ("sigma_spatial" %in% colnames(fit$draws)) {
        sigma_sp <- round(mean(fit$draws[, "sigma_spatial"]), 4)
      }
      if ("rho_spatial" %in% colnames(fit$draws)) {
        rho_sp <- round(mean(fit$draws[, "rho_spatial"]), 4)
      }

      res <- data.frame(
        row     = row_cfg$row,
        name    = row_cfg$name,
        seed    = seed,
        time_s  = round(elapsed, 1),
        div     = n_div,
        avg_td  = round(avg_td, 1),
        pct_maxd = round(pct_maxd * 100, 1),
        sampler = sampler,
        sigma   = sigma_sp,
        rho     = rho_sp,
        stringsAsFactors = FALSE
      )
      results[[length(results) + 1]] <- res

      cat(sprintf("  Time=%.1fs  Div=%d  AvgTD=%.1f  MaxdPct=%.1f%%  Sampler=%s  sigma=%.4f  rho=%.4f\n",
                  elapsed, n_div, avg_td, pct_maxd * 100, sampler, sigma_sp, rho_sp))

    }, error = function(e) {
      cat(sprintf("  ERROR: %s\n", conditionMessage(e)))
      results[[length(results) + 1]] <<- data.frame(
        row = row_cfg$row, name = row_cfg$name, seed = seed,
        time_s = NA, div = NA, avg_td = NA, pct_maxd = NA,
        sampler = "ERROR", sigma = NA, rho = NA,
        stringsAsFactors = FALSE
      )
    })
  }
}

# --- Summary table ---
df_results <- do.call(rbind, results)
cat("\n\n=== BYM2 BENCHMARK RESULTS (OAS + Riebler) ===\n")
cat(sprintf("N=%d, Iter=%d, Warmup=%d, Chains=%d, Seeds=%s\n",
            N_OBS, N_ITER, N_WARMUP, N_CHAINS, paste(SEEDS, collapse=",")))
cat("================================================================\n")
print(df_results, row.names = FALSE)

# Per-row summary
cat("\n\n=== PER-ROW SUMMARY (across seeds) ===\n")
for (r in unique(df_results$row)) {
  sub <- df_results[df_results$row == r, ]
  cat(sprintf("Row %d (%s): time=%.1f-%.1fs  div=%d-%d  avg_td=%.1f-%.1f  rho=%.2f-%.2f\n",
              r, sub$name[1],
              min(sub$time_s, na.rm = TRUE), max(sub$time_s, na.rm = TRUE),
              min(sub$div, na.rm = TRUE), max(sub$div, na.rm = TRUE),
              min(sub$avg_td, na.rm = TRUE), max(sub$avg_td, na.rm = TRUE),
              min(sub$rho, na.rm = TRUE), max(sub$rho, na.rm = TRUE)))
}

cat("\nDone.\n")
