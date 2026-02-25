# =============================================================================
# 4-Mode Gradient Benchmark for ALL 107 Model Configurations
# =============================================================================
# Runs each model with H, A_t, A, N gradient modes and records timing.
# Standard publication parameters: N=500, iter=500, warmup=250, chains=1.
#
# Usage:
#   Rscript benchmarks/bench_4modes_all.R                  # all 107 rows
#   Rscript benchmarks/bench_4modes_all.R 1 30             # rows 1-30
#   Rscript benchmarks/bench_4modes_all.R 61 92            # rows 61-92
#   Rscript benchmarks/bench_4modes_all.R --tier fast       # fast tier only
#   Rscript benchmarks/bench_4modes_all.R --tier medium     # medium tier
#   Rscript benchmarks/bench_4modes_all.R --tier slow       # slow tier
#   Rscript benchmarks/bench_4modes_all.R --modes H,A_t     # specific modes
#
# Results saved incrementally to benchmarks/results_4modes_all.rds
# =============================================================================

library(numdenom)

args <- commandArgs(trailingOnly = TRUE)

# Parse arguments
ROW_START <- 1
ROW_END <- 107
MODES <- c("H", "A_t", "A", "N")
TIER_FILTER <- NULL

i <- 1
while (i <= length(args)) {
  if (args[i] == "--tier") {
    TIER_FILTER <- args[i + 1]
    i <- i + 2
  } else if (args[i] == "--modes") {
    MODES <- strsplit(args[i + 1], ",")[[1]]
    i <- i + 2
  } else if (grepl("^[0-9]+$", args[i])) {
    ROW_START <- as.integer(args[i])
    if (i + 1 <= length(args) && grepl("^[0-9]+$", args[i + 1])) {
      ROW_END <- as.integer(args[i + 1])
      i <- i + 2
    } else {
      ROW_END <- ROW_START
      i <- i + 1
    }
  } else {
    i <- i + 1
  }
}

# =============================================================================
# Standard parameters
# =============================================================================
N_OBS       <- 500L
N_ITER      <- 500L
N_WARMUP    <- 250L
N_CHAINS    <- 1L
N_SITES     <- 50L
N_TIMES     <- 20L
TIMEOUT_SEC <- 600   # 10 min timeout per benchmark
SEED        <- 123L

# Latent factor models use smaller N
N_OBS_LATENT <- 50L
N_SITES_LATENT <- 10L

# GP models use smaller N (O(N^3))
N_OBS_GP <- 80L

set.seed(SEED)

# =============================================================================
# Speed tier classification (based on H timing from gradient_methods.md)
# =============================================================================
fast_rows   <- c(1:6, 10:13, 16:20, 22, 24:25, 27,
                 31:36, 40:43, 46:50, 52, 54:55, 57,
                 61:66, 70:73, 76:77, 80:82, 84, 86:87, 89,
                 93:98)
medium_rows <- c(7:8, 14, 21, 37:38, 44, 51, 67:68, 74:75, 78:79,
                 83, 99:102, 103:107)
slow_rows   <- c(9, 15, 23, 26, 28:30, 39, 45, 53, 56, 58:60,
                 69, 85, 88, 90:92)

if (!is.null(TIER_FILTER)) {
  tier_rows <- switch(TIER_FILTER,
    "fast"   = fast_rows,
    "medium" = medium_rows,
    "slow"   = slow_rows,
    stop("Unknown tier: ", TIER_FILTER, ". Use fast/medium/slow.")
  )
  ROW_START <- min(tier_rows)
  ROW_END   <- max(tier_rows)
}

# =============================================================================
# Data generation
# =============================================================================

generate_datasets <- function(n, n_sites, n_times, seed = 123) {
  set.seed(seed)

  # Site/time structure (balanced panel)
  site <- factor(rep(1:n_sites, length.out = n))
  time <- rep(1:n_times, length.out = n)
  time_factor <- factor(time)
  x <- rnorm(n)
  z <- rnorm(n)  # for random slopes

  # Site-level grid coordinates (for ICAR, BYM2, pCAR, HSGP, TVC)
  n_side <- ceiling(sqrt(n_sites))
  grid <- expand.grid(lon = 1:n_side, lat = 1:n_side)[1:n_sites, ]
  site_int <- as.integer(site)
  lon_site <- grid$lon[site_int]
  lat_site <- grid$lat[site_int]

  # Unique observation-level coordinates (for GP, MSGP, SVC)
  lon_obs <- runif(n, 0, 10)
  lat_obs <- runif(n, 0, 10)

  # Adjacency matrix
  adj_mat <- matrix(0L, n_sites, n_sites)
  for (i in 1:n_sites) {
    for (j in 1:n_sites) {
      if (i != j) {
        d <- sqrt((grid$lon[i] - grid$lon[j])^2 + (grid$lat[i] - grid$lat[j])^2)
        if (d <= 1.5) adj_mat[i, j] <- 1L
      }
    }
  }

  # Poisson-Gamma data
  y_pg_num   <- rpois(n, exp(2 + 0.5 * x))
  y_pg_denom <- rgamma(n, 10, 1)
  df_pg <- data.frame(y = y_pg_num, denom = y_pg_denom, x = x, z = z,
                      site = site, time = time_factor,
                      lon = lon_site, lat = lat_site,
                      lon_obs = lon_obs, lat_obs = lat_obs,
                      spatial_site = site)

  # NegBin-NegBin data
  y_nb_num   <- rnbinom(n, mu = exp(2 + 0.3 * x), size = 5)
  y_nb_denom <- rnbinom(n, mu = 100, size = 10)
  y_nb_denom[y_nb_denom == 0] <- 1L
  df_nb <- data.frame(y = y_nb_num, denom = y_nb_denom, x = x, z = z,
                      site = site, time = time_factor,
                      lon = lon_site, lat = lat_site,
                      lon_obs = lon_obs, lat_obs = lat_obs,
                      spatial_site = site)

  # Binomial data
  trials <- sample(10:50, n, replace = TRUE)
  y_bin  <- rbinom(n, trials, plogis(0.5 + 0.3 * x))
  df_bin <- data.frame(y = y_bin, trials = trials, x = x, z = z,
                       site = site, time = time_factor,
                       lon = lon_site, lat = lat_site,
                       lon_obs = lon_obs, lat_obs = lat_obs,
                       spatial_site = site)

  # Gamma-Gamma data
  y_gg_num   <- rgamma(n, shape = 5, rate = 5 / exp(2 + 0.3 * x))
  y_gg_denom <- rgamma(n, shape = 8, rate = 8 / exp(3 + 0.2 * x))
  y_gg_num[y_gg_num <= 0]     <- 0.01
  y_gg_denom[y_gg_denom <= 0] <- 0.01
  df_gg <- data.frame(y = y_gg_num, denom = y_gg_denom, x = x, z = z,
                      site = site, time = time_factor,
                      lon = lon_site, lat = lat_site,
                      spatial_site = site)

  # Lognormal data
  y_ln_num   <- exp(rnorm(n, mean = 2 + 0.3 * x, sd = 0.5))
  y_ln_denom <- exp(rnorm(n, mean = 3 + 0.2 * x, sd = 0.5))
  df_ln <- data.frame(y = y_ln_num, denom = y_ln_denom, x = x, z = z,
                      site = site, time = time_factor,
                      lon = lon_site, lat = lat_site,
                      spatial_site = site)

  # Beta-Binomial data
  mu_bb  <- plogis(0.5 + 0.3 * x)
  phi_bb <- 10
  a_bb   <- mu_bb * phi_bb
  b_bb   <- (1 - mu_bb) * phi_bb
  y_bb   <- rbeta(n, a_bb, b_bb)
  y_bb   <- rbinom(n, trials, y_bb)
  df_bb <- data.frame(y = y_bb, trials = trials, x = x, z = z,
                      site = site, time = time_factor,
                      lon = lon_site, lat = lat_site,
                      spatial_site = site)

  list(
    pg = df_pg, nb = df_nb, bin = df_bin,
    gg = df_gg, ln = df_ln, bb = df_bb,
    adj_mat = adj_mat, grid = grid
  )
}

# Dataset-specific sizes
N_SITES_GP   <- 20L
N_TIMES_GP   <- 10L
N_TIMES_LAT  <- 10L

# Generate standard datasets
cat("Generating standard datasets (N=500)...\n")
data_std <- generate_datasets(N_OBS, N_SITES, N_TIMES, seed = SEED)

# Generate GP datasets (smaller N, O(N^3))
cat("Generating GP datasets (N=80)...\n")
data_gp <- generate_datasets(N_OBS_GP, N_SITES_GP, N_TIMES_GP, seed = SEED)

# Generate latent datasets (smallest N, high dimensionality)
cat("Generating latent datasets (N=50)...\n")
data_lat <- generate_datasets(N_OBS_LATENT, N_SITES_LATENT, N_TIMES_LAT, seed = SEED)

# =============================================================================
# ZI/OI data modifications
# =============================================================================

inject_zeros <- function(df, col = "y", prob = 0.3) {
  df[[col]] <- ifelse(runif(nrow(df)) < prob, 0L, df[[col]])
  df
}

inject_ones_binomial <- function(df, prob_oi = 0.15) {
  is_oi <- rbinom(nrow(df), 1, prob_oi)
  df$y <- ifelse(is_oi == 1, df$trials, df$y)
  df
}

inject_zoib <- function(df, prob_zi = 0.1, prob_oi = 0.1) {
  n <- nrow(df)
  is_zi <- rbinom(n, 1, prob_zi)
  is_oi <- rbinom(n, 1, prob_oi)
  df$y <- ifelse(is_zi == 1, 0L, ifelse(is_oi == 1, df$trials, df$y))
  df
}

# =============================================================================
# Model configuration: all 107 rows
# =============================================================================

# Configuration structure:
#   fam:     family code (pg, nb, bin, gg, ln, bb)
#   re:      "none", "int", "slopes", "crossed"
#   sp:      "none", "icar", "bym2", "gp", "hsgp", "msgp", "pcar", "svc"
#   temp:    "none", "rw1", "rw2", "ar1", "gp_t", "ms_t", "tvc"
#   zi:      "none", "zi", "hurdle", "oi", "zoib"
#   st:      "none", "I", "IV"
#   latent:  TRUE/FALSE
#   use_gp_data: TRUE = use smaller GP dataset

make_row <- function(row, fam, re = "int", sp = "none", temp = "none",
                     zi = "none", st = "none", latent = FALSE,
                     use_gp_data = FALSE) {
  list(row = row, fam = fam, re = re, sp = sp, temp = temp,
       zi = zi, st = st, latent = latent, use_gp_data = use_gp_data)
}

# Define all 107 rows
ROW_CONFIGS <- list(
  # =========================================================================
  # Section 1: poisson_gamma (rows 1-30)
  # =========================================================================
  make_row( 1, "pg", re="none"),
  make_row( 2, "pg", re="int"),
  make_row( 3, "pg", re="slopes"),
  make_row( 4, "pg", re="crossed"),
  make_row( 5, "pg", sp="icar"),
  make_row( 6, "pg", sp="bym2"),
  make_row( 7, "pg", sp="gp", use_gp_data=TRUE),
  make_row( 8, "pg", sp="hsgp"),
  make_row( 9, "pg", sp="msgp", use_gp_data=TRUE),
  make_row(10, "pg", sp="pcar"),
  make_row(11, "pg", temp="rw1"),
  make_row(12, "pg", temp="rw2"),
  make_row(13, "pg", temp="ar1"),
  make_row(14, "pg", temp="gp_t"),
  make_row(15, "pg", temp="ms_t"),
  make_row(16, "pg", zi="zi"),
  make_row(17, "pg", zi="hurdle"),
  make_row(18, "pg", sp="icar", temp="rw1"),
  make_row(19, "pg", sp="bym2", temp="rw1"),
  make_row(20, "pg", sp="icar", temp="ar1"),
  make_row(21, "pg", sp="gp", temp="rw1", use_gp_data=TRUE),
  make_row(22, "pg", sp="hsgp", temp="rw1"),
  make_row(23, "pg", sp="msgp", temp="rw1", use_gp_data=TRUE),
  make_row(24, "pg", sp="icar", zi="zi"),
  make_row(25, "pg", re="slopes", sp="icar"),
  make_row(26, "pg", sp="svc"),
  make_row(27, "pg", temp="tvc"),
  make_row(28, "pg", sp="icar", temp="rw1", st="I"),
  make_row(29, "pg", sp="icar", temp="rw1", st="IV"),
  make_row(30, "pg", latent=TRUE),

  # =========================================================================
  # Section 2: negbin_negbin (rows 31-60)
  # =========================================================================
  make_row(31, "nb", re="none"),
  make_row(32, "nb", re="int"),
  make_row(33, "nb", re="slopes"),
  make_row(34, "nb", re="crossed"),
  make_row(35, "nb", sp="icar"),
  make_row(36, "nb", sp="bym2"),
  make_row(37, "nb", sp="gp", use_gp_data=TRUE),
  make_row(38, "nb", sp="hsgp"),
  make_row(39, "nb", sp="msgp", use_gp_data=TRUE),
  make_row(40, "nb", sp="pcar"),
  make_row(41, "nb", temp="rw1"),
  make_row(42, "nb", temp="rw2"),
  make_row(43, "nb", temp="ar1"),
  make_row(44, "nb", temp="gp_t"),
  make_row(45, "nb", temp="ms_t"),
  make_row(46, "nb", zi="zi"),
  make_row(47, "nb", zi="hurdle"),
  make_row(48, "nb", sp="icar", temp="rw1"),
  make_row(49, "nb", sp="bym2", temp="rw1"),
  make_row(50, "nb", sp="icar", temp="ar1"),
  make_row(51, "nb", sp="gp", temp="rw1", use_gp_data=TRUE),
  make_row(52, "nb", sp="hsgp", temp="rw1"),
  make_row(53, "nb", sp="msgp", temp="rw1", use_gp_data=TRUE),
  make_row(54, "nb", sp="icar", zi="zi"),
  make_row(55, "nb", re="slopes", sp="icar"),
  make_row(56, "nb", sp="svc"),
  make_row(57, "nb", temp="tvc"),
  make_row(58, "nb", sp="icar", temp="rw1", st="I"),
  make_row(59, "nb", sp="icar", temp="rw1", st="IV"),
  make_row(60, "nb", latent=TRUE),

  # =========================================================================
  # Section 3: binomial (rows 61-92)
  # =========================================================================
  make_row(61, "bin", re="none"),
  make_row(62, "bin", re="int"),
  make_row(63, "bin", re="slopes"),
  make_row(64, "bin", re="crossed"),
  make_row(65, "bin", sp="icar"),
  make_row(66, "bin", sp="bym2"),
  make_row(67, "bin", sp="gp", use_gp_data=TRUE),
  make_row(68, "bin", sp="hsgp"),
  make_row(69, "bin", sp="msgp", use_gp_data=TRUE),
  make_row(70, "bin", sp="pcar"),
  make_row(71, "bin", temp="rw1"),
  make_row(72, "bin", temp="rw2"),
  make_row(73, "bin", temp="ar1"),
  make_row(74, "bin", temp="gp_t"),
  make_row(75, "bin", temp="ms_t"),
  make_row(76, "bin", zi="zi"),
  make_row(77, "bin", zi="hurdle"),
  make_row(78, "bin", zi="oi"),
  make_row(79, "bin", zi="zoib"),
  make_row(80, "bin", sp="icar", temp="rw1"),
  make_row(81, "bin", sp="bym2", temp="rw1"),
  make_row(82, "bin", sp="icar", temp="ar1"),
  make_row(83, "bin", sp="gp", temp="rw1", use_gp_data=TRUE),
  make_row(84, "bin", sp="hsgp", temp="rw1"),
  make_row(85, "bin", sp="msgp", temp="rw1", use_gp_data=TRUE),
  make_row(86, "bin", sp="icar", zi="zi"),
  make_row(87, "bin", re="slopes", sp="icar"),
  make_row(88, "bin", sp="svc"),
  make_row(89, "bin", temp="tvc"),
  make_row(90, "bin", sp="icar", temp="rw1", st="I"),
  make_row(91, "bin", sp="icar", temp="rw1", st="IV"),
  make_row(92, "bin", latent=TRUE),

  # =========================================================================
  # Section 4: gamma_gamma (rows 93-97)
  # =========================================================================
  make_row(93, "gg", re="none"),
  make_row(94, "gg", re="int"),
  make_row(95, "gg", sp="icar"),
  make_row(96, "gg", temp="rw1"),
  make_row(97, "gg", sp="icar", temp="rw1"),

  # =========================================================================
  # Section 5: lognormal (rows 98-102)
  # =========================================================================
  make_row( 98, "ln", re="none"),
  make_row( 99, "ln", re="int"),
  make_row(100, "ln", sp="icar"),
  make_row(101, "ln", temp="rw1"),
  make_row(102, "ln", sp="icar", temp="rw1"),

  # =========================================================================
  # Section 6: beta_binomial (rows 103-107)
  # =========================================================================
  make_row(103, "bb", re="none"),
  make_row(104, "bb", re="int"),
  make_row(105, "bb", sp="icar"),
  make_row(106, "bb", temp="rw1"),
  make_row(107, "bb", sp="icar", temp="rw1")
)

# Index configs by row number for quick lookup
config_by_row <- list()
for (cfg in ROW_CONFIGS) {
  config_by_row[[as.character(cfg$row)]] <- cfg
}

# =============================================================================
# Build ratiod() arguments from config
# =============================================================================

build_ratiod_args <- function(cfg) {
  row <- cfg$row
  fam <- cfg$fam
  re  <- cfg$re
  sp  <- cfg$sp
  temp <- cfg$temp
  zi  <- cfg$zi
  st  <- cfg$st
  latent <- cfg$latent
  use_gp <- cfg$use_gp_data

  # Select dataset
  if (latent) {
    ds <- data_lat
  } else if (use_gp) {
    ds <- data_gp
  } else {
    ds <- data_std
  }

  df      <- ds[[fam]]
  adj_mat <- ds$adj_mat

  # Apply ZI/OI data modifications
  if (zi == "zi" && fam %in% c("pg", "nb")) {
    df <- inject_zeros(df, "y", 0.3)
  } else if (zi == "zi" && fam == "bin") {
    df <- inject_zeros(df, "y", 0.3)
  } else if (zi == "hurdle" && fam %in% c("pg", "nb")) {
    df <- inject_zeros(df, "y", 0.3)
  } else if (zi == "hurdle" && fam == "bin") {
    df <- inject_zeros(df, "y", 0.3)
  } else if (zi == "oi") {
    df <- inject_ones_binomial(df, 0.15)
  } else if (zi == "zoib") {
    df <- inject_zoib(df, 0.1, 0.1)
  }

  # Formula
  if (fam %in% c("bin", "bb")) {
    lhs <- "y | trials"
  } else {
    lhs <- "y | denom"
  }

  rhs <- if (re == "none") {
    "x"
  } else if (re == "int") {
    "x + (1 | site)"
  } else if (re == "slopes") {
    "x + (1 + z | site)"
  } else if (re == "crossed") {
    df$site2 <- factor(sample(1:5, nrow(df), replace = TRUE))
    "x + (1 | site) + (1 | site2)"
  }

  form <- as.formula(paste(lhs, "~", rhs))

  # Family
  nd_family <- switch(fam,
    "pg"  = ratiod_poisson_gamma(),
    "nb"  = ratiod_negbin_negbin(),
    "bin" = ratiod_binomial(),
    "gg"  = ratiod_gamma_gamma(),
    "ln"  = ratiod_lognormal(),
    "bb"  = ratiod_beta_binomial()
  )

  # Spatial
  nd_spatial <- NULL
  if (sp == "icar") {
    nd_spatial <- spatial_car(adj_mat, level = "group", group_var = "spatial_site")
  } else if (sp == "bym2") {
    nd_spatial <- spatial_bym2(adj_mat, level = "group", group_var = "spatial_site")
  } else if (sp == "pcar") {
    nd_spatial <- spatial_car(adj_mat, level = "group", group_var = "spatial_site",
                              proper = TRUE)
  } else if (sp == "gp") {
    nd_spatial <- spatial_gp(coords = ~ lon_obs + lat_obs)
  } else if (sp == "hsgp") {
    nd_spatial <- spatial_hsgp(coords = ~ lon + lat)
  } else if (sp == "msgp") {
    nd_spatial <- spatial_multiscale(coords = ~ lon_obs + lat_obs)
  } else if (sp == "svc") {
    nd_spatial <- spatial_svc(coords = ~ lon_obs + lat_obs, terms = 1, nn = 15)
  }

  # Temporal
  nd_temporal <- NULL
  if (temp == "rw1") {
    nd_temporal <- temporal_rw1("time")
  } else if (temp == "rw2") {
    nd_temporal <- temporal_rw2("time")
  } else if (temp == "ar1") {
    nd_temporal <- temporal_ar1("time")
  } else if (temp == "gp_t") {
    nd_temporal <- temporal_gp("time")
  } else if (temp == "ms_t") {
    nd_temporal <- temporal_multiscale("time")
  } else if (temp == "tvc") {
    nd_temporal <- temporal_tvc(time_var = "time", terms = 1)
  }

  # ZI
  nd_zi <- NULL
  if (zi == "zi" && fam %in% c("pg", "nb")) {
    nd_zi <- if (fam == "pg") zi_poisson() else zi_negbin()
  } else if (zi == "zi" && fam == "bin") {
    nd_family <- ratiod_zibinomial()
    nd_zi <- NULL
  } else if (zi == "hurdle" && fam %in% c("pg", "nb")) {
    nd_zi <- if (fam == "pg") hurdle_poisson() else hurdle_negbin()
  } else if (zi == "hurdle" && fam == "bin") {
    nd_family <- ratiod_hurdle_binomial()
    nd_zi <- NULL
  } else if (zi == "oi") {
    nd_family <- ratiod_oibinomial()
    nd_zi <- NULL
  } else if (zi == "zoib") {
    nd_family <- ratiod_zoibinomial()
    nd_zi <- NULL
  }

  # Spatiotemporal: bundles spatial + temporal inside spatiotemporal()
  # When using spatiotemporal, the spatial/temporal are NOT passed separately
  nd_spatiotemporal <- NULL
  if (st != "none") {
    nd_spatiotemporal <- spatiotemporal(
      spatial  = nd_spatial,
      temporal = nd_temporal,
      type     = st
    )
    nd_spatial  <- NULL
    nd_temporal <- NULL
  }

  # Latent
  nd_latent <- NULL
  if (latent) {
    nd_latent <- latent_factor(n_factors = 2)
  }

  # Iteration params
  iter   <- N_ITER
  warmup <- N_WARMUP
  chains <- N_CHAINS

  list(
    formula          = form,
    data             = df,
    family           = nd_family,
    spatial          = nd_spatial,
    temporal         = nd_temporal,
    zi               = nd_zi,
    spatiotemporal   = nd_spatiotemporal,
    latent           = nd_latent,
    iter             = iter,
    warmup           = warmup,
    chains           = chains,
    verbose          = FALSE
  )
}

# =============================================================================
# Row description string
# =============================================================================

describe_row <- function(cfg) {
  parts <- cfg$fam
  if (cfg$re != "int" && cfg$re != "none") parts <- c(parts, cfg$re)
  if (cfg$re == "int") parts <- c(parts, "RE")
  if (cfg$sp != "none") parts <- c(parts, toupper(cfg$sp))
  if (cfg$temp != "none") parts <- c(parts, toupper(cfg$temp))
  if (cfg$zi != "none") parts <- c(parts, toupper(cfg$zi))
  if (cfg$st != "none") parts <- c(parts, paste0("ST-", cfg$st))
  if (cfg$latent) parts <- c(parts, "latent")
  paste(parts, collapse = " + ")
}

# =============================================================================
# Run single benchmark with timeout
# =============================================================================

run_one <- function(ratiod_args, gradient_mode, timeout = TIMEOUT_SEC) {
  ratiod_args$gradient_mode <- gradient_mode

  tryCatch({
    result <- withTimeout(
      system.time(do.call(ratiod, ratiod_args))["elapsed"],
      timeout = timeout
    )
    result
  }, error = function(e) {
    msg <- conditionMessage(e)
    if (grepl("timeout|elapsed", msg, ignore.case = TRUE)) {
      return("TIMEOUT")
    }
    return(paste0("ERROR: ", msg))
  })
}

# Simple timeout wrapper (R.utils not required)
withTimeout <- function(expr, timeout) {
  setTimeLimit(cpu = timeout, elapsed = timeout, transient = TRUE)
  on.exit(setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE))
  expr
}

# =============================================================================
# Load existing results (for incremental runs)
# =============================================================================

results_file <- "benchmarks/results_4modes_all.rds"
if (file.exists(results_file)) {
  all_results <- readRDS(results_file)
  cat(sprintf("Loaded %d existing results from %s\n", nrow(all_results), results_file))
} else {
  all_results <- data.frame(
    row     = integer(),
    desc    = character(),
    mode    = character(),
    time_s  = character(),  # character to allow "TIMEOUT"/"ERROR"
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# Main benchmark loop
# =============================================================================

# Filter to requested rows
rows_to_run <- ROW_START:ROW_END
if (!is.null(TIER_FILTER)) {
  rows_to_run <- intersect(rows_to_run, tier_rows)
}

cat(sprintf("\n=== 4-Mode Benchmark: rows %d-%d, modes: %s ===\n",
            min(rows_to_run), max(rows_to_run), paste(MODES, collapse = ", ")))
cat(sprintf("Standard: N=%d, iter=%d, warmup=%d, chains=%d\n",
            N_OBS, N_ITER, N_WARMUP, N_CHAINS))
cat(sprintf("GP: N=%d | Latent: N=%d | Timeout: %ds\n\n",
            N_OBS_GP, N_OBS_LATENT, TIMEOUT_SEC))

total_benchmarks <- length(rows_to_run) * length(MODES)
completed <- 0
start_time <- Sys.time()

for (row_num in rows_to_run) {
  cfg <- config_by_row[[as.character(row_num)]]
  if (is.null(cfg)) {
    cat(sprintf("Row %d: not defined, skipping\n", row_num))
    next
  }

  desc <- describe_row(cfg)
  cat(sprintf("\n--- Row %3d: %s ---\n", row_num, desc))

  # Build model arguments once per row
  ratiod_args <- tryCatch(
    build_ratiod_args(cfg),
    error = function(e) {
      cat(sprintf("  Setup ERROR: %s\n", conditionMessage(e)))
      NULL
    }
  )

  if (is.null(ratiod_args)) next

  for (mode in MODES) {
    completed <- completed + 1

    # Skip if already computed
    existing <- all_results[all_results$row == row_num & all_results$mode == mode, ]
    if (nrow(existing) > 0) {
      cat(sprintf("  %-3s: %s (cached)\n", mode, existing$time_s[1]))
      next
    }

    cat(sprintf("  %-3s: ", mode))
    flush.console()

    result <- run_one(ratiod_args, mode)

    if (is.numeric(result)) {
      cat(sprintf("%.1fs\n", result))
      time_str <- sprintf("%.1f", result)
    } else {
      cat(sprintf("%s\n", result))
      time_str <- result
    }

    # Append result
    new_row <- data.frame(
      row    = row_num,
      desc   = desc,
      mode   = mode,
      time_s = time_str,
      stringsAsFactors = FALSE
    )
    all_results <- rbind(all_results, new_row)

    # Save incrementally every 4 results (after each complete row)
    if (completed %% length(MODES) == 0 || completed == total_benchmarks) {
      saveRDS(all_results, results_file)
    }
  }
}

# Final save
saveRDS(all_results, results_file)

# =============================================================================
# Summary table
# =============================================================================

cat("\n\n", strrep("=", 80), "\n")
cat("BENCHMARK SUMMARY\n")
cat(strrep("=", 80), "\n\n")

# Reshape to wide format
if (nrow(all_results) > 0) {
  # Get unique rows that were benchmarked
  benchmarked_rows <- sort(unique(all_results$row))
  benchmarked_rows <- benchmarked_rows[benchmarked_rows >= min(rows_to_run) &
                                       benchmarked_rows <= max(rows_to_run)]

  cat(sprintf("%-5s %-35s %10s %10s %10s %10s\n",
              "Row", "Model", "H(s)", "A_t(s)", "A(s)", "N(s)"))
  cat(paste(rep("-", 85), collapse = ""), "\n")

  for (r in benchmarked_rows) {
    desc <- all_results$desc[all_results$row == r][1]
    vals <- list()
    for (m in c("H", "A_t", "A", "N")) {
      v <- all_results$time_s[all_results$row == r & all_results$mode == m]
      vals[[m]] <- if (length(v) > 0) v[1] else "-"
    }
    cat(sprintf("%-5d %-35s %10s %10s %10s %10s\n",
                r, substr(desc, 1, 35),
                vals[["H"]], vals[["A_t"]], vals[["A"]], vals[["N"]]))
  }
}

elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
cat(sprintf("\nTotal time: %.1f minutes\n", elapsed))
cat(sprintf("Results saved to %s\n", results_file))

# =============================================================================
# Generate markdown table for gradient_methods.md
# =============================================================================

cat("\n\n--- Copy-paste for gradient_methods.md ---\n\n")
if (nrow(all_results) > 0) {
  for (r in benchmarked_rows) {
    vals <- list()
    for (m in c("H", "A_t", "A", "N")) {
      v <- all_results$time_s[all_results$row == r & all_results$mode == m]
      if (length(v) > 0 && !grepl("ERROR|TIMEOUT", v[1])) {
        vals[[m]] <- v[1]
      } else {
        vals[[m]] <- ""
      }
    }
    cat(sprintf("| %d | ... | H | %s | %s | %s | %s | ... |\n",
                r, vals[["H"]], vals[["A"]], vals[["A_t"]], vals[["N"]]))
  }
}
