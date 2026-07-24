# =============================================================================
# Stan comparison benchmarks for remaining slow models
# Covers: BYM2, BYM2+RW1, HSGP, slopes+ICAR, temporal GP, ST Type IV
# =============================================================================

suppressPackageStartupMessages({
  library(numdenom)
  library(cmdstanr)
})

N_OBS       <- 500L
N_ITER      <- 500L
N_WARMUP    <- 250L
N_CHAINS    <- 1L
N_SITES     <- 50L
N_TIMES     <- 20L
SEED        <- 123L

set.seed(SEED)

# --- Data generation (matches bench_single_row.R) ---
site <- factor(rep(1:N_SITES, length.out = N_OBS))
time <- rep(1:N_TIMES, length.out = N_OBS)
time_factor <- factor(time)
x <- rnorm(N_OBS)
z <- rnorm(N_OBS)

n_side <- ceiling(sqrt(N_SITES))
grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:N_SITES, ]
site_int <- as.integer(site)
lon_site <- grid$lon[site_int]
lat_site <- grid$lat[site_int]

adj_mat <- matrix(0L, N_SITES, N_SITES)
for (i in 1:N_SITES) {
  for (j in 1:N_SITES) {
    if (i != j) {
      d <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
      if (d <= 1.5) adj_mat[i, j] <- 1L
    }
  }
}

# Edge lists for Stan
edges <- list()
for (i in 1:(N_SITES-1)) {
  for (j in (i+1):N_SITES) {
    if (adj_mat[i, j] == 1L) {
      edges[[length(edges) + 1]] <- c(i, j)
    }
  }
}
n_neighbors <- as.array(as.integer(rowSums(adj_mat)))
n_edges <- length(edges)
edge1 <- as.array(sapply(edges, `[`, 1))
edge2 <- as.array(sapply(edges, `[`, 2))

# BYM2 scale factor
Q <- diag(n_neighbors) - adj_mat
eig <- eigen(Q)$values
scale_factor <- exp(mean(log(eig[eig > 1e-10])))

# --- Family-specific data ---
y_pg_num   <- rpois(N_OBS, exp(2 + 0.5 * x))
y_pg_denom <- rgamma(N_OBS, 10, 1)
df_pg <- data.frame(y = y_pg_num, denom = y_pg_denom, x = x, z = z,
                    site = site, time = time_factor, time_num = time,
                    lon = lon_site, lat = lat_site, spatial_site = site)

y_nb_num   <- rnbinom(N_OBS, mu = exp(2 + 0.3 * x), size = 5)
y_nb_denom <- rnbinom(N_OBS, mu = 100, size = 10)
y_nb_denom[y_nb_denom == 0] <- 1L
df_nb <- data.frame(y = y_nb_num, denom = y_nb_denom, x = x, z = z,
                    site = site, time = time_factor, time_num = time,
                    lon = lon_site, lat = lat_site, spatial_site = site)

trials <- sample(10:50, N_OBS, replace = TRUE)
y_bin  <- rbinom(N_OBS, trials, plogis(0.5 + 0.3 * x))
df_bin <- data.frame(y = y_bin, trials = trials, x = x, z = z,
                     site = site, time = time_factor, time_num = time,
                     lon = lon_site, lat = lat_site, spatial_site = site)

# --- numdenom spatial/temporal objects (correct API) ---
nd_bym2    <- spatial_bym2(adj_mat, level = "group", group_var = "spatial_site")
nd_icar    <- spatial_car(adj_mat, level = "group", group_var = "spatial_site")
nd_hsgp    <- spatial_hsgp(coords = ~ lon + lat)
nd_rw1     <- temporal_rw1("time")
nd_gp_t    <- temporal_gp("time_num")

# --- Helper: run numdenom ---
run_nd <- function(family_call, formula, data, ...) {
  tryCatch({
    t <- system.time({
      fit <- tratio(formula, data = data, family = family_call, ...,
                    control = list(iter = N_ITER, warmup = N_WARMUP, chains = N_CHAINS, gradient_mode = "H", seed = SEED, verbose = FALSE))
    })["elapsed"]
    t
  }, error = function(e) {
    cat("  numdenom ERROR:", conditionMessage(e), "\n")
    NA_real_
  })
}

# --- Helper: run Stan ---
run_stan <- function(stan_file, stan_data, adapt_delta = 0.8) {
  tryCatch({
    model <- cmdstan_model(stan_file)
    t <- system.time({
      fit <- model$sample(
        data = stan_data,
        iter_sampling = N_ITER - N_WARMUP,
        iter_warmup = N_WARMUP,
        chains = N_CHAINS,
        parallel_chains = N_CHAINS,
        refresh = 0,
        show_messages = FALSE,
        adapt_delta = adapt_delta
      )
    })["elapsed"]
    diag <- fit$diagnostic_summary(quiet = TRUE)
    ndiv <- sum(diag$num_divergent)
    cat(sprintf("  Stan: %.1fs, divergences=%d\n", t, ndiv))
    t
  }, error = function(e) {
    cat("  Stan ERROR:", conditionMessage(e), "\n")
    NA_real_
  })
}

results <- data.frame(
  Row = integer(), Model = character(),
  numdenom_s = numeric(), Stan_s = numeric(),
  Ratio = character(), stringsAsFactors = FALSE
)

add_result <- function(row, model, nd_time, stan_time) {
  ratio_str <- if (is.na(nd_time) || is.na(stan_time)) {
    "N/A"
  } else if (stan_time > nd_time) {
    sprintf("%.1fx WIN", stan_time / nd_time)
  } else {
    sprintf("%.1fx LOSS", nd_time / stan_time)
  }
  results[nrow(results) + 1, ] <<- list(row, model, round(nd_time, 1),
                                          round(stan_time, 1), ratio_str)
  cat(sprintf("ROW %d [%s]: nd=%.1fs, stan=%.1fs => %s\n",
              row, model, nd_time, stan_time, ratio_str))
}

stan_dir <- "benchmarks/stan"

# ============================================================
# 1. BYM2 (rows 6, 36, 66)
# ============================================================
cat("\n=== BYM2 ===\n")

bym2_spatial <- list(
  J = N_SITES, n_neighbors = n_neighbors,
  n_edges = n_edges, edge1 = edge1, edge2 = edge2,
  scale_factor = scale_factor
)

# Row 6: PG + BYM2
cat("Row 6: PG + BYM2\n")
t_nd <- run_nd(ratiod_poisson_gamma(), y | denom ~ x + (1|site), df_pg,
               spatial = nd_bym2)
stan_data <- c(list(N = N_OBS, y_num = df_pg$y, y_denom = df_pg$denom, p = 2L,
                    X = cbind(1, df_pg$x), n_groups = N_SITES,
                    group_idx = as.integer(df_pg$site),
                    spatial_idx = as.integer(df_pg$site)), bym2_spatial)
t_stan <- run_stan(file.path(stan_dir, "joint_pg_bym2.stan"), stan_data, adapt_delta = 0.9)
add_result(6, "PG+BYM2", t_nd, t_stan)

# Row 36: NB + BYM2
cat("Row 36: NB + BYM2\n")
t_nd <- run_nd(ratiod_negbin_negbin(), y | denom ~ x + (1|site), df_nb,
               spatial = nd_bym2)
stan_data <- c(list(N = N_OBS, y_num = df_nb$y, y_denom = df_nb$denom, p = 2L,
                    X = cbind(1, df_nb$x), n_groups = N_SITES,
                    group_idx = as.integer(df_nb$site),
                    spatial_idx = as.integer(df_nb$site)), bym2_spatial)
t_stan <- run_stan(file.path(stan_dir, "joint_nb_bym2.stan"), stan_data, adapt_delta = 0.9)
add_result(36, "NB+BYM2", t_nd, t_stan)

# Row 66: Bin + BYM2
cat("Row 66: Bin + BYM2\n")
t_nd <- run_nd(ratiod_binomial(), y | trials ~ x + (1|site), df_bin,
               spatial = nd_bym2)
stan_data <- c(list(N = N_OBS, y = df_bin$y, trials = df_bin$trials, p = 2L,
                    X = cbind(1, df_bin$x), n_groups = N_SITES,
                    group_idx = as.integer(df_bin$site),
                    spatial_idx = as.integer(df_bin$site)), bym2_spatial)
t_stan <- run_stan(file.path(stan_dir, "joint_binom_bym2.stan"), stan_data, adapt_delta = 0.9)
add_result(66, "Bin+BYM2", t_nd, t_stan)

# ============================================================
# 2. BYM2 + RW1 (rows 19, 49, 81)
# ============================================================
cat("\n=== BYM2 + RW1 ===\n")

bym2_rw1_common <- c(bym2_spatial, list(T = N_TIMES))

# Row 19: PG + BYM2 + RW1
cat("Row 19: PG + BYM2 + RW1\n")
t_nd <- run_nd(ratiod_poisson_gamma(), y | denom ~ x + (1|site), df_pg,
               spatial = nd_bym2, temporal = nd_rw1)
stan_data <- c(list(N = N_OBS, y_num = df_pg$y, y_denom = df_pg$denom, p = 2L,
                    X = cbind(1, df_pg$x), n_groups = N_SITES,
                    group_idx = as.integer(df_pg$site),
                    spatial_idx = as.integer(df_pg$site),
                    time_idx = as.integer(df_pg$time)), bym2_rw1_common)
t_stan <- run_stan(file.path(stan_dir, "joint_pg_bym2_rw1.stan"), stan_data, adapt_delta = 0.95)
add_result(19, "PG+BYM2+RW1", t_nd, t_stan)

# Row 49: NB + BYM2 + RW1
cat("Row 49: NB + BYM2 + RW1\n")
t_nd <- run_nd(ratiod_negbin_negbin(), y | denom ~ x + (1|site), df_nb,
               spatial = nd_bym2, temporal = nd_rw1)
stan_data <- c(list(N = N_OBS, y_num = df_nb$y, y_denom = df_nb$denom, p = 2L,
                    X = cbind(1, df_nb$x), n_groups = N_SITES,
                    group_idx = as.integer(df_nb$site),
                    spatial_idx = as.integer(df_nb$site),
                    time_idx = as.integer(df_nb$time)), bym2_rw1_common)
t_stan <- run_stan(file.path(stan_dir, "joint_nb_bym2_rw1.stan"), stan_data, adapt_delta = 0.95)
add_result(49, "NB+BYM2+RW1", t_nd, t_stan)

# Row 81: Bin + BYM2 + RW1
cat("Row 81: Bin + BYM2 + RW1\n")
t_nd <- run_nd(ratiod_binomial(), y | trials ~ x + (1|site), df_bin,
               spatial = nd_bym2, temporal = nd_rw1)
stan_data <- c(list(N = N_OBS, y = df_bin$y, trials = df_bin$trials, p = 2L,
                    X = cbind(1, df_bin$x), n_groups = N_SITES,
                    group_idx = as.integer(df_bin$site),
                    spatial_idx = as.integer(df_bin$site),
                    time_idx = as.integer(df_bin$time)), bym2_rw1_common)
t_stan <- run_stan(file.path(stan_dir, "joint_binom_bym2_rw1.stan"), stan_data, adapt_delta = 0.95)
add_result(81, "Bin+BYM2+RW1", t_nd, t_stan)

# ============================================================
# 3. HSGP (rows 8, 38, 68)
# ============================================================
cat("\n=== HSGP ===\n")

hsgp_common <- list(M = 6L, c = 1.5, sigma2_prior_U = 1.0,
                    sigma2_prior_alpha = 0.01, phi_prior_lower = 0.01,
                    phi_prior_upper = 100.0)

# Row 8: PG + HSGP
cat("Row 8: PG + HSGP\n")
t_nd <- run_nd(ratiod_poisson_gamma(), y | denom ~ x, df_pg,
               spatial = nd_hsgp)
stan_data <- c(list(N = N_OBS, y_num = df_pg$y, y_denom = df_pg$denom,
                    x = df_pg$x, coords = cbind(df_pg$lon, df_pg$lat)), hsgp_common)
t_stan <- run_stan(file.path(stan_dir, "hsgp_pg_joint.stan"), stan_data)
add_result(8, "PG+HSGP", t_nd, t_stan)

# Row 38: NB + HSGP
cat("Row 38: NB + HSGP\n")
t_nd <- run_nd(ratiod_negbin_negbin(), y | denom ~ x, df_nb,
               spatial = nd_hsgp)
stan_data <- c(list(N = N_OBS, y_num = df_nb$y, y_denom = df_nb$denom,
                    x = df_nb$x, coords = cbind(df_nb$lon, df_nb$lat)), hsgp_common)
t_stan <- run_stan(file.path(stan_dir, "hsgp_nb_joint.stan"), stan_data)
add_result(38, "NB+HSGP", t_nd, t_stan)

# Row 68: Bin + HSGP
cat("Row 68: Bin + HSGP\n")
t_nd <- run_nd(ratiod_binomial(), y | trials ~ x, df_bin,
               spatial = nd_hsgp)
stan_data <- c(list(N = N_OBS, y = df_bin$y, trials = df_bin$trials,
                    x = df_bin$x, coords = cbind(df_bin$lon, df_bin$lat)), hsgp_common)
t_stan <- run_stan(file.path(stan_dir, "hsgp_binom_joint.stan"), stan_data)
add_result(68, "Bin+HSGP", t_nd, t_stan)

# ============================================================
# 4. Slopes + ICAR (rows 25, 55, 87)
# ============================================================
cat("\n=== Slopes + ICAR ===\n")

slopes_spatial <- list(
  J = N_SITES, n_neighbors = n_neighbors,
  n_edges = n_edges, edge1 = edge1, edge2 = edge2
)

# Row 25: PG + slopes + ICAR
cat("Row 25: PG + slopes + ICAR\n")
t_nd <- run_nd(ratiod_poisson_gamma(), y | denom ~ x + (1 + z|site), df_pg,
               spatial = nd_icar)
stan_data <- c(list(N = N_OBS, y_num = df_pg$y, y_denom = df_pg$denom, p = 2L,
                    X = cbind(1, df_pg$x), x_slope = df_pg$z,
                    n_groups = N_SITES, group_idx = as.integer(df_pg$site),
                    spatial_idx = as.integer(df_pg$site)), slopes_spatial)
t_stan <- run_stan(file.path(stan_dir, "joint_pg_slopes_icar.stan"), stan_data, adapt_delta = 0.95)
add_result(25, "PG+slopes+ICAR", t_nd, t_stan)

# Row 55: NB + slopes + ICAR
cat("Row 55: NB + slopes + ICAR\n")
t_nd <- run_nd(ratiod_negbin_negbin(), y | denom ~ x + (1 + z|site), df_nb,
               spatial = nd_icar)
stan_data <- c(list(N = N_OBS, y_num = df_nb$y, y_denom = df_nb$denom, p = 2L,
                    X = cbind(1, df_nb$x), x_slope = df_nb$z,
                    n_groups = N_SITES, group_idx = as.integer(df_nb$site),
                    spatial_idx = as.integer(df_nb$site)), slopes_spatial)
t_stan <- run_stan(file.path(stan_dir, "joint_nb_slopes_icar.stan"), stan_data, adapt_delta = 0.95)
add_result(55, "NB+slopes+ICAR", t_nd, t_stan)

# Row 87: Bin + slopes + ICAR
cat("Row 87: Bin + slopes + ICAR\n")
t_nd <- run_nd(ratiod_binomial(), y | trials ~ x + (1 + z|site), df_bin,
               spatial = nd_icar)
stan_data <- c(list(N = N_OBS, y = df_bin$y, trials = df_bin$trials, p = 2L,
                    X = cbind(1, df_bin$x), x_slope = df_bin$z,
                    n_groups = N_SITES, group_idx = as.integer(df_bin$site),
                    spatial_idx = as.integer(df_bin$site)), slopes_spatial)
t_stan <- run_stan(file.path(stan_dir, "joint_binom_slopes_icar.stan"), stan_data, adapt_delta = 0.95)
add_result(87, "Bin+slopes+ICAR", t_nd, t_stan)

# ============================================================
# 5. Temporal GP (rows 14, 44, 74)
# ============================================================
cat("\n=== Temporal GP ===\n")

time_vals <- 1:N_TIMES
time_scaled <- (time_vals - mean(time_vals)) / sd(time_vals)

gp_prior <- list(
  sigma2_prior_U = 1.0, sigma2_prior_alpha = 0.01,
  phi_prior_lower = 0.01, phi_prior_upper = 10.0
)

# Row 14: PG + temporal GP
cat("Row 14: PG + temporal GP\n")
t_nd <- run_nd(ratiod_poisson_gamma(), y | denom ~ x + (1|site), df_pg,
               temporal = nd_gp_t)
t_stan <- run_stan(file.path(stan_dir, "temporal_gp_pg_joint.stan"),
                   c(list(N = N_OBS, `T` = N_TIMES,
                          y_num = df_pg$y, y_denom = df_pg$denom,
                          x = df_pg$x,
                          time_idx = as.integer(df_pg$time),
                          time_values = time_scaled), gp_prior))
add_result(14, "PG+GP_t", t_nd, t_stan)

# Row 44: NB + temporal GP
cat("Row 44: NB + temporal GP\n")
t_nd <- run_nd(ratiod_negbin_negbin(), y | denom ~ x + (1|site), df_nb,
               temporal = nd_gp_t)
t_stan <- run_stan(file.path(stan_dir, "temporal_gp_nb_joint.stan"),
                   c(list(N = N_OBS, `T` = N_TIMES,
                          y_num = df_nb$y, y_denom = as.integer(df_nb$denom),
                          x = df_nb$x,
                          time_idx = as.integer(df_nb$time),
                          time_values = time_scaled), gp_prior))
add_result(44, "NB+GP_t", t_nd, t_stan)

# Row 74: Bin + temporal GP
cat("Row 74: Bin + temporal GP\n")
t_nd <- run_nd(ratiod_binomial(), y | trials ~ x + (1|site), df_bin,
               temporal = nd_gp_t)
t_stan <- run_stan(file.path(stan_dir, "temporal_gp_binomial.stan"),
                   c(list(N = N_OBS, `T` = N_TIMES,
                          y = df_bin$y, trials = df_bin$trials,
                          x = df_bin$x,
                          time_idx = as.integer(df_bin$time),
                          time_values = time_scaled), gp_prior))
add_result(74, "Bin+GP_t", t_nd, t_stan)

# ============================================================
# 6. ST Type IV (rows 29, 59, 91)
# ============================================================
cat("\n=== ST Type IV ===\n")

st_spatial <- list(
  J = N_SITES, n_neighbors = n_neighbors,
  n_edges = n_edges, edge1 = edge1, edge2 = edge2,
  `T` = N_TIMES
)

nd_st_iv <- spatiotemporal(spatial = nd_icar, temporal = nd_rw1, type = "IV")

# Row 29: PG + ST IV
cat("Row 29: PG + ST IV\n")
t_nd <- run_nd(ratiod_poisson_gamma(), y | denom ~ x + (1|site), df_pg,
               spatial = nd_icar, temporal = nd_rw1, spatiotemporal = nd_st_iv)
stan_data <- c(list(N = N_OBS, y_num = df_pg$y, y_denom = df_pg$denom, p = 2L,
                    X = cbind(1, df_pg$x), n_groups = N_SITES,
                    group_idx = as.integer(df_pg$site),
                    spatial_idx = as.integer(df_pg$site),
                    time_idx = as.integer(df_pg$time)), st_spatial)
t_stan <- run_stan(file.path(stan_dir, "joint_pg_st_iv.stan"), stan_data, adapt_delta = 0.95)
add_result(29, "PG+ST_IV", t_nd, t_stan)

# Row 59: NB + ST IV
cat("Row 59: NB + ST IV\n")
t_nd <- run_nd(ratiod_negbin_negbin(), y | denom ~ x + (1|site), df_nb,
               spatial = nd_icar, temporal = nd_rw1, spatiotemporal = nd_st_iv)
stan_data <- c(list(N = N_OBS, y_num = df_nb$y, y_denom = df_nb$denom, p = 2L,
                    X = cbind(1, df_nb$x), n_groups = N_SITES,
                    group_idx = as.integer(df_nb$site),
                    spatial_idx = as.integer(df_nb$site),
                    time_idx = as.integer(df_nb$time)), st_spatial)
t_stan <- run_stan(file.path(stan_dir, "joint_nb_st_iv.stan"), stan_data, adapt_delta = 0.95)
add_result(59, "NB+ST_IV", t_nd, t_stan)

# Row 91: Bin + ST IV
cat("Row 91: Bin + ST IV\n")
t_nd <- run_nd(ratiod_binomial(), y | trials ~ x + (1|site), df_bin,
               spatial = nd_icar, temporal = nd_rw1, spatiotemporal = nd_st_iv)
stan_data <- c(list(N = N_OBS, y = df_bin$y, trials = df_bin$trials, p = 2L,
                    X = cbind(1, df_bin$x), n_groups = N_SITES,
                    group_idx = as.integer(df_bin$site),
                    spatial_idx = as.integer(df_bin$site),
                    time_idx = as.integer(df_bin$time)), st_spatial)
t_stan <- run_stan(file.path(stan_dir, "joint_binom_st_iv.stan"), stan_data, adapt_delta = 0.95)
add_result(91, "Bin+ST_IV", t_nd, t_stan)

# ============================================================
# RESULTS SUMMARY
# ============================================================
cat("\n\n")
cat(strrep("=", 70), "\n")
cat("RESULTS SUMMARY\n")
cat(strrep("=", 70), "\n\n")

cat(sprintf("%-5s %-20s %10s %10s %15s\n", "Row", "Model", "numdenom", "Stan", "Ratio"))
cat(strrep("-", 60), "\n")
for (i in seq_len(nrow(results))) {
  r <- results[i, ]
  cat(sprintf("%-5d %-20s %9.1fs %9.1fs %15s\n",
              r$Row, r$Model, r$numdenom_s, r$Stan_s, r$Ratio))
}
cat("\n")
