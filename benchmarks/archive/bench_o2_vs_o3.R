#!/usr/bin/env Rscript
# Comprehensive -O2 vs -O3 benchmark across all model code paths
# Run this script twice: once with -O2 in Makevars, once with -O3
# Pass "O2" or "O3" as command-line argument to label the run
args <- commandArgs(trailingOnly = TRUE)
label <- if (length(args) > 0) args[1] else "unknown"

library(numdenom)

set.seed(42)
N <- 500; S <- 50; T_max <- 20

# --- Simulate data ---
site <- rep(1:S, length.out = N)
region <- rep(1:10, each = N/10)
x1 <- rnorm(N)
time_idx <- rep(1:T_max, length.out = N)

# Adjacency for ICAR/BYM2 (chain graph on 50 sites)
adj_mat <- matrix(0L, S, S)
for (i in 1:(S - 1)) { adj_mat[i, i + 1] <- 1L; adj_mat[i + 1, i] <- 1L }

# NB/PG data
lambda_num <- exp(1 + 0.3 * x1)
lambda_denom <- exp(2 + 0.1 * x1)
y_num <- rnbinom(N, mu = lambda_num, size = 5)
y_denom <- rnbinom(N, mu = lambda_denom, size = 5)
y_denom[y_denom == 0] <- 1

# Binomial data
trials <- rpois(N, 20) + 10
successes <- rbinom(N, size = trials, prob = 0.3)

df <- data.frame(
  y_num = y_num, y_denom = y_denom,
  successes = successes, trials = trials,
  x1 = x1, site = factor(site), region = factor(region),
  time = time_idx
)

# --- Model configurations ---
configs <- list(
  # PG family
  list(name = "pg_base",    f = y_num | y_denom ~ x1, fam = "pg", sp = NULL, temp = NULL),
  list(name = "pg_re",      f = y_num | y_denom ~ x1 + (1 | site), fam = "pg", sp = NULL, temp = NULL),
  list(name = "pg_crossed", f = y_num | y_denom ~ x1 + (1 | site) + (1 | region), fam = "pg", sp = NULL, temp = NULL),
  list(name = "pg_icar",    f = y_num | y_denom ~ x1, fam = "pg", sp = "icar", temp = NULL),
  list(name = "pg_bym2",    f = y_num | y_denom ~ x1, fam = "pg", sp = "bym2", temp = NULL),
  list(name = "pg_rw1",     f = y_num | y_denom ~ x1, fam = "pg", sp = NULL, temp = "rw1"),
  list(name = "pg_rw2",     f = y_num | y_denom ~ x1, fam = "pg", sp = NULL, temp = "rw2"),
  list(name = "pg_ar1",     f = y_num | y_denom ~ x1, fam = "pg", sp = NULL, temp = "ar1"),
  list(name = "pg_icar_rw1", f = y_num | y_denom ~ x1, fam = "pg", sp = "icar", temp = "rw1"),

  # NB family
  list(name = "nb_base",    f = y_num | y_denom ~ x1, fam = "nb", sp = NULL, temp = NULL),
  list(name = "nb_re",      f = y_num | y_denom ~ x1 + (1 | site), fam = "nb", sp = NULL, temp = NULL),
  list(name = "nb_crossed", f = y_num | y_denom ~ x1 + (1 | site) + (1 | region), fam = "nb", sp = NULL, temp = NULL),
  list(name = "nb_icar",    f = y_num | y_denom ~ x1, fam = "nb", sp = "icar", temp = NULL),
  list(name = "nb_bym2",    f = y_num | y_denom ~ x1, fam = "nb", sp = "bym2", temp = NULL),
  list(name = "nb_rw1",     f = y_num | y_denom ~ x1, fam = "nb", sp = NULL, temp = "rw1"),
  list(name = "nb_icar_rw1", f = y_num | y_denom ~ x1, fam = "nb", sp = "icar", temp = "rw1"),

  # Binomial family
  list(name = "bin_base",   f = successes | trials ~ x1, fam = "bin", sp = NULL, temp = NULL),
  list(name = "bin_re",     f = successes | trials ~ x1 + (1 | site), fam = "bin", sp = NULL, temp = NULL),
  list(name = "bin_crossed", f = successes | trials ~ x1 + (1 | site) + (1 | region), fam = "bin", sp = NULL, temp = NULL),
  list(name = "bin_icar",   f = successes | trials ~ x1, fam = "bin", sp = "icar", temp = NULL),
  list(name = "bin_rw1",    f = successes | trials ~ x1, fam = "bin", sp = NULL, temp = "rw1"),

  # Random slopes
  list(name = "pg_slopes_uncorr", f = y_num | y_denom ~ x1 + (1 + x1 || site), fam = "pg", sp = NULL, temp = NULL),
  list(name = "pg_slopes_corr",   f = y_num | y_denom ~ x1 + (1 + x1 | site), fam = "pg", sp = NULL, temp = NULL),
  list(name = "nb_slopes_uncorr", f = y_num | y_denom ~ x1 + (1 + x1 || site), fam = "nb", sp = NULL, temp = NULL)
)

get_family <- function(fam) {
  switch(fam,
    pg = ratiod_poisson_gamma(),
    nb = ratiod_negbin_negbin(),
    bin = ratiod_binomial()
  )
}

get_spatial <- function(sp) {
  if (is.null(sp)) return(NULL)
  switch(sp,
    icar = spatial_car(adj = adj_mat, group_var = "site"),
    bym2 = spatial_bym2(adj = adj_mat, group_var = "site")
  )
}

get_temporal <- function(temp) {
  if (is.null(temp)) return(NULL)
  switch(temp,
    rw1 = temporal_rw1(time_var = "time"),
    rw2 = temporal_rw2(time_var = "time"),
    ar1 = temporal_ar1(time_var = "time")
  )
}

cat(sprintf("=== %s Benchmark (N=%d, 500 iter, 1 chain) ===\n\n", label, N))
cat(sprintf("%-20s  %6s\n", "Model", "Time(s)"))
cat(paste(rep("-", 30), collapse = ""), "\n")

results <- data.frame(model = character(), time = numeric(), stringsAsFactors = FALSE)

for (cfg in configs) {
  t_elapsed <- tryCatch({
    system.time({
      fit <- ratiod(cfg$f, data = df,
                    family = get_family(cfg$fam),
                    spatial = get_spatial(cfg$sp),
                    temporal = get_temporal(cfg$temp),
                    mode = "hmc", iter = 500, warmup = 250, chains = 1,
                    gradient_mode = "H", verbose = FALSE)
    })["elapsed"]
  }, error = function(e) {
    cat(sprintf("%-20s  ERROR: %s\n", cfg$name, e$message))
    NA_real_
  })

  if (!is.na(t_elapsed)) {
    cat(sprintf("%-20s  %6.1f\n", cfg$name, t_elapsed))
    results <- rbind(results, data.frame(model = cfg$name, time = t_elapsed))
  }
}

# Save results
outfile <- sprintf("benchmarks/o2_vs_o3_%s.rds", label)
saveRDS(results, outfile)
cat(sprintf("\nResults saved to %s\n", outfile))
