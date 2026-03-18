#' HMC/NUTS Backend for ratiod Models
#'
#' @description
#' Custom Hamiltonian Monte Carlo with NUTS (No-U-Turn Sampler) backend.
#' Provides full MCMC inference for all ratiod families without external
#' dependencies like Stan.
#'
#' @details
#' This backend implements:
#' - Numerical gradients (stable across all model types)
#' - Fixed trajectory HMC with dual averaging step size adaptation
#' - OpenMP parallelization (within-chain for likelihood, across-chain for multi-chain)
#' - ICAR and BYM2 spatial random effects
#' - RW1/RW2/AR1 temporal random effects
#' - Zero-inflation and hurdle models
#'
#' **Supported model families:**
#' - `ratiod_binomial()` - Trial-based proportions
#' - `ratiod_negbin_negbin()` - Two-process count ratios
#' - `ratiod_poisson_gamma()` - Count/effort ratios (CPUE)
#'
#' **Spatial structures:**
#' - `spatial_car()` - ICAR prior for areal data
#' - `spatial_bym2()` - BYM2 prior (structured + unstructured)
#'
#' **Zero-inflation types:**
#' - `zi_poisson()` - Zero-inflated Poisson
#' - `zi_negbin()` - Zero-inflated negative binomial
#' - `hurdle_poisson()` - Hurdle Poisson
#' - `hurdle_negbin()` - Hurdle negative binomial
#'
#' **When to use:**
#' - When Stan is not available or too slow to compile
#' - For portable, self-contained Bayesian inference
#' - When you need full MCMC (not just approximations)
#' - For parallel multi-chain sampling
#'
#' @name hmc_backend
#' @keywords internal
NULL


#' Fit ratiod model using HMC/NUTS
#'
#' @param formula A ratiod_formula object
#' @param data Data frame
#' @param family Model family
#' @param spatial Optional spatial structure (spatial_car or spatial_bym2)
#' @param temporal Optional temporal structure (temporal_rw1, temporal_ar1, etc.)
#' @param zi Optional zero-inflation specification (ratiod_zi object)
#' @param priors Prior specification
#' @param iter Total number of iterations per chain
#' @param warmup Number of warmup iterations per chain
#' @param chains Number of chains to run (can be parallelized)
#' @param cores Number of cores for parallel computation
#' @param L Number of leapfrog steps per iteration
#' @param seed Random seed for reproducibility
#' @param verbose Print progress
#' @param gradient_mode Gradient computation method: "auto", "H", "A_r", "A", "N"
#' @param re_param Random effects parameterization: "centered" or "noncentered"
#' @param metric Mass matrix type: "auto" (default), "diag", "dense", or "block_diag"
#' @param riemannian Use Riemannian (SoftAbs) metric: NULL (auto), TRUE, or FALSE
#'
#' @return A ratiod_fit object
#' @keywords internal
fit_hmc <- function(formula,
                    data,
                    family,
                    spatial = NULL,
                    temporal = NULL,
                    spatiotemporal = NULL,
                    zi = NULL,
                    latent = NULL,
                    priors = NULL,
                    iter = 2000,
                    warmup = floor(iter / 2),
                    chains = 4,
                    cores = NULL,
                    L = 0,
                    adapt_delta = NULL,
                    max_treedepth = NULL,
                    seed = NULL,
                    verbose = TRUE,
                    gradient_mode = "auto",
                    re_param = "noncentered",
                    metric = "auto",
                    riemannian = NULL) {

  # Set cores
  if (is.null(cores)) {
    cores <- min(chains, cpp_get_max_threads())
  }

  # Set seed
  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1)
  }

  # Resolve adapt_delta: NULL → -1.0 (C++ sentinel for auto-selection)
  adapt_delta_value <- if (is.null(adapt_delta)) -1.0 else as.double(adapt_delta)

  # Resolve riemannian: NULL → -1 (auto), TRUE → 1, FALSE → 0
  riemannian_value <- if (is.null(riemannian)) -1L else as.integer(riemannian)

  # Get model type
  model_type <- get_hmc_model_type(family)

  # Extract data
  hmc_data <- prepare_hmc_data(formula, data, family, model_type)

  # Random slopes are now supported - log info if present
  if (isTRUE(hmc_data$has_slopes) && verbose) {
    n_slope_terms <- sum(sapply(hmc_data$re_terms, function(t) length(t$slope_vars) > 0))
    n_corr <- sum(sapply(hmc_data$re_terms, function(t) isTRUE(t$correlated) && length(t$slope_vars) > 0))
    message("  Random slopes: ", n_slope_terms, " term(s) (",
            n_corr, " correlated, ", n_slope_terms - n_corr, " uncorrelated)")
  }

  # Prepare spatial structure
  # Check if GP-based spatial (requires GP sampler) or multiscale temporal
  # without areal spatial (ICAR/BYM2/pCAR can pair with MS_t via regular sampler)
  has_areal_spatial <- !is.null(spatial) && !is.null(spatial$type) &&
    spatial$type %in% c("car", "car_proper", "bym2")
  use_gp_sampler <- is_gp_spatial(spatial) ||
    (is_multiscale_temporal(temporal) && !has_areal_spatial)

  if (use_gp_sampler) {
    # GP spatial uses dedicated sampler
    gp_info <- prepare_gp_for_hmc(spatial, data, hmc_data$N)
    gp_n_units <- if (gp_info$n_unique > 0L) gp_info$n_unique else hmc_data$N
    gp_group <- if (length(gp_info$gp_obs_to_loc) > 0) gp_info$gp_obs_to_loc else seq_len(hmc_data$N)
    # Precompute HSGP basis matrix for HSGP-MSGP ratio reconstruction
    msgp_hsgp_Phi <- NULL
    msgp_hsgp_eigenvalues <- NULL
    if (isTRUE(gp_info$msgp_approx == "hsgp")) {
      hsgp_m <- as.integer(gp_info$hsgp_m %||% 6L)
      c_bnd <- gp_info$hsgp_c %||% 1.5
      coords_mat <- as.matrix(data[, spatial$coord_vars, drop = FALSE])
      if (isTRUE(spatial$scale_coords)) coords_mat <- scale(coords_mat)
      N_obs <- nrow(coords_mat)
      x_range <- diff(range(coords_mat[, 1]))
      y_range <- diff(range(coords_mat[, 2]))
      x_center <- mean(range(coords_mat[, 1]))
      y_center <- mean(range(coords_mat[, 2]))
      L1 <- max(c_bnd * x_range / 2, 0.1)
      L2 <- max(c_bnd * y_range / 2, 0.1)
      m_total <- hsgp_m * hsgp_m
      msgp_hsgp_eigenvalues <- numeric(m_total)
      msgp_hsgp_Phi <- matrix(0, N_obs, m_total)
      for (j1 in seq_len(hsgp_m)) {
        for (j2 in seq_len(hsgp_m)) {
          j_idx <- (j1 - 1) * hsgp_m + j2
          msgp_hsgp_eigenvalues[j_idx] <- (pi * j1 / (2 * L1))^2 + (pi * j2 / (2 * L2))^2
          msgp_hsgp_Phi[, j_idx] <- sin(pi * j1 * (coords_mat[, 1] - x_center + L1) / (2 * L1)) / sqrt(L1) *
            sin(pi * j2 * (coords_mat[, 2] - y_center + L2) / (2 * L2)) / sqrt(L2)
        }
      }
    }

    spatial_info <- list(type = gp_info$gp_type, n_units = gp_n_units,
                         group = gp_group, adj_row_ptr = integer(1),
                         adj_col_idx = integer(0), n_neighbors = integer(0),
                         bym2_scale = 1.0,
                         hsgp_m = gp_info$hsgp_m,  # HSGP: basis functions per dim
                         hsgp_c = gp_info$hsgp_c,
                         msgp_approx = gp_info$msgp_approx,
                         msgp_hsgp_Phi = msgp_hsgp_Phi,
                         msgp_hsgp_eigenvalues = msgp_hsgp_eigenvalues,
                         coord_vars = if (!is.null(spatial)) spatial$coord_vars else NULL,
                         parameterization = gp_info$parameterization,
                         # Range bounds for proper initialization (from prepare_gp_for_hmc)
                         range_local_lower = gp_info$range_local_lower,
                         range_local_upper = gp_info$range_local_upper,
                         range_regional_lower = gp_info$range_regional_lower,
                         range_regional_upper = gp_info$range_regional_upper)
  } else {
    gp_info <- NULL
    spatial_info <- prepare_spatial_for_hmc(spatial, data, hmc_data$N)
  }

  # Auto-cap max_treedepth for GP/MSGP spatial models (many spatial params → deep trees)
  if (is.null(max_treedepth)) {
    gp_type <- if (!is.null(gp_info)) gp_info$gp_type else "none"
    if (gp_type %in% c("gp", "multiscale_gp")) {
      max_treedepth <- 8L
      if (verbose) {
        message("  Auto-capping max_treedepth to 8 for GP spatial model ",
                "(use max_treedepth = 10 to override)")
      }
    } else {
      max_treedepth <- 10L
    }
  }

  # Prepare temporal structure
  temporal_info <- prepare_temporal_for_hmc(temporal, data, hmc_data$N)

  # Prepare zero-inflation structure

  # Check if family itself is ZI/OI/ZOIB (ratiod_zi*, ratiod_hurdle_*, ratiod_oi*, ratiod_zoib*)
  # In this case, extract ZI info from family, not from zi= parameter
  if (is.null(zi) && (isTRUE(family$zero_inflated) || isTRUE(family$one_inflated))) {
    # Determine ZI type string based on family distribution and zi_type
    dist <- family$numerator$distribution
    zi_type <- if (family$zi_type == "hurdle") {
      # Hurdle models
      switch(dist,
        "hurdle_binomial" = "hurdle_binomial",
        "hurdle_poisson" = "hurdle_poisson",
        "hurdle_neg_binomial" = "hurdle_negbin",
        "hurdle_binomial"  # fallback
      )
    } else if (family$zi_type == "one_inflated") {
      # One-inflated models (excess at upper boundary)
      "oi_binomial"
    } else if (family$zi_type == "zoib") {
      # Zero-and-one-inflated binomial
      "zoib"
    } else {
      # Zero-inflated mixture models
      switch(dist,
        "zero_inflated_binomial" = "zi_binomial",
        "zero_inflated_poisson" = "zi_poisson",
        "zero_inflated_neg_binomial" = "zi_negbin",
        "zi_binomial"  # fallback
      )
    }

    # Build ZI/OI info based on model type
    if (family$zi_type == "one_inflated") {
      # OI-binomial: only OI coefficient, no ZI
      zi_info <- list(
        type = zi_type,
        X_zi = matrix(0, nrow = hmc_data$N, ncol = 1),  # Placeholder (not used)
        p_zi = 0L,  # No ZI coefficient for OI-only models
        coef_names = NULL,
        X_oi = matrix(1, nrow = hmc_data$N, ncol = 1),  # Intercept-only
        p_oi = 1L,
        coef_names_oi = "(Intercept)_oi"
      )
    } else if (family$zi_type == "zoib") {
      # ZOIB: both ZI and OI coefficients
      zi_info <- list(
        type = zi_type,
        X_zi = matrix(1, nrow = hmc_data$N, ncol = 1),  # Intercept-only
        p_zi = 1L,
        coef_names = "(Intercept)_zi",
        X_oi = matrix(1, nrow = hmc_data$N, ncol = 1),  # Intercept-only
        p_oi = 1L,
        coef_names_oi = "(Intercept)_oi"
      )
    } else {
      # ZI/Hurdle models: only ZI coefficient
      zi_info <- list(
        type = zi_type,
        X_zi = matrix(1, nrow = hmc_data$N, ncol = 1),  # Intercept-only
        p_zi = 1L,
        coef_names = "(Intercept)_zi",
        X_oi = NULL,
        p_oi = 0L,
        coef_names_oi = NULL
      )
    }
  } else {
    zi_info <- prepare_zi_for_hmc(zi, data, hmc_data$N)
  }

  # Prepare latent factor structure
  latent_info <- prepare_latent_for_hmc(latent, hmc_data$N)

  # Prepare spatiotemporal structure
  spatiotemporal_info <- prepare_spatiotemporal_for_hmc(spatiotemporal, data)

  # Prepare SVC structure (if spatial is an SVC specification)
  svc_info <- prepare_svc_for_hmc(spatial, data, hmc_data$N, hmc_data$X_num)

  # Get prior parameters
  priors <- priors %||% ratiod_priors()
  sigma_beta <- priors$sigma_beta %||% 10.0
  sigma_re_scale <- priors$sigma_re_scale %||% 2.5

  # Family-specific priors for phi parameter
  # - negbin_negbin: phi is overdispersion, use Gamma(2, 0.1) with mode=10
  # - lognormal: phi is sigma (log-scale SD), use Gamma(2, 2) with mode=0.5
  # - gamma_gamma: phi is shape, use Gamma(2, 0.5) with mode=2
  # - poisson_gamma: phi is shape of gamma denominator, use Gamma(2, 0.5) with mode=2
  # - beta_binomial: phi is precision (alpha+beta), use Gamma(2, 0.1) with mode=10
  if (!is.null(priors$phi_shape) && !is.null(priors$phi_rate)) {
    # User explicitly specified priors
    phi_shape <- priors$phi_shape
    phi_rate <- priors$phi_rate
  } else if (model_type == "lognormal") {
    # Lognormal: sigma typically 0.1-2, use Gamma(2, 2) with mode=0.5, mean=1
    phi_shape <- priors$phi_shape %||% 2.0
    phi_rate <- priors$phi_rate %||% 2.0
  } else if (model_type == "gamma_gamma") {
    # Gamma-Gamma: shape typically 1-10, use Gamma(2, 0.5) with mode=2, mean=4
    phi_shape <- priors$phi_shape %||% 2.0
    phi_rate <- priors$phi_rate %||% 0.5
  } else if (model_type == "poisson_gamma") {
    # Poisson-Gamma: shape typically 1-10, use Gamma(2, 0.5) with mode=2, mean=4
    phi_shape <- priors$phi_shape %||% 2.0
    phi_rate <- priors$phi_rate %||% 0.5
  } else {
    # Default (negbin_negbin, beta_binomial): overdispersion/precision, mode=10
    phi_shape <- priors$phi_shape %||% 2.0
    phi_rate <- priors$phi_rate %||% 0.1
  }

  tau_spatial_shape <- priors$tau_spatial_shape %||% 1.0
  tau_spatial_rate <- priors$tau_spatial_rate %||% 0.01


  # Initialize parameters
  q_init <- initialize_hmc_params_full(
    hmc_data, model_type, spatial_info, temporal_info, zi_info, latent_info, svc_info,
    spatiotemporal_info
  )

  if (verbose) {
    sampler_name <- if (L == 0) "NUTS" else paste0("HMC (L=", L, ")")
    message("Running ", sampler_name, " sampler...")
    message("  Parameters: ", length(q_init))
    message("  Iterations: ", iter, " (warmup: ", warmup, ")")
    message("  Chains: ", chains, " (cores: ", cores, ")")
    if (spatial_info$type != "none") {
      message("  Spatial: ", spatial_info$type, " (",
              spatial_info$n_units, " units)")
    }
    if (temporal_info$type != "none") {
      message("  Temporal: ", temporal_info$type, " (",
              temporal_info$n_times, " time points)")
    }
    if (zi_info$type != "none") {
      message("  Zero-inflation: ", zi_info$type, " (",
              zi_info$p_zi, " predictors)")
    }
    if (latent_info$type != "none") {
      message("  Latent factors: ", latent_info$n_factors, " factor(s)")
    }
  }

  # Decide parallelization strategy
  # If chains > 1 and cores > 1: parallelize across chains
  # If chains == 1 and cores > 1: parallelize within chain (likelihood)
  n_threads_within <- ifelse(chains > 1 && cores > 1, 1L, as.integer(cores))

  # Get temporal prior parameters
  tau_temporal_shape <- priors$tau_temporal_shape %||% 1.0
  tau_temporal_rate <- priors$tau_temporal_rate %||% 0.01

  # Prepare multi-term RE data
  n_re_terms <- hmc_data$n_re_terms %||% 0L
  if (n_re_terms > 1 && !is.null(hmc_data$re_group_matrix)) {
    re_group_matrix <- hmc_data$re_group_matrix
    re_n_groups_vec <- sapply(hmc_data$re_terms, function(x) x$n_groups)
  } else if (n_re_terms == 1 && !is.null(hmc_data$re_terms)) {
    # Single RE term: create matrix from first term
    re_group_matrix <- matrix(hmc_data$re_terms[[1]]$group_idx, ncol = 1L)
    re_n_groups_vec <- as.integer(hmc_data$re_terms[[1]]$n_groups)
  } else {
    # No RE: create dummy matrix
    re_group_matrix <- matrix(0L, nrow = hmc_data$N, ncol = 1L)
    re_n_groups_vec <- as.integer(hmc_data$n_re_groups)
  }

  # Prepare slope-related parameters
  has_re_slopes <- isTRUE(hmc_data$has_slopes)
  has_re_correlated_slopes <- isTRUE(hmc_data$has_correlated_slopes)

  if (has_re_slopes && n_re_terms > 0) {
    re_n_coefs_vec <- sapply(hmc_data$re_terms, function(x) x$n_coefs)
    re_correlated_vec <- sapply(hmc_data$re_terms, function(x) isTRUE(x$correlated))
    re_n_chol_vec <- sapply(hmc_data$re_terms, function(x) x$n_chol %||% 0L)
    slope_matrices_list <- hmc_data$slope_matrices
  } else {
    re_n_coefs_vec <- rep(1L, max(1L, n_re_terms))
    re_correlated_vec <- rep(FALSE, max(1L, n_re_terms))
    re_n_chol_vec <- rep(0L, max(1L, n_re_terms))
    slope_matrices_list <- list()
  }

  # Run sampler - branch based on GP vs non-GP spatial
  if (use_gp_sampler) {
    # Prepare multiscale temporal if present
    ms_temporal_info <- prepare_multiscale_temporal_for_hmc(temporal, data, hmc_data$N)

    # Prepare RSR if present
    rsr_info <- prepare_rsr_for_hmc(spatial, data)

    # Bundle GP parameters into list for C++
    gp_params <- list(
      gp_type = gp_info$gp_type,
      coords = gp_info$coords,
      nn_idx = as.integer(gp_info$nn_idx),
      nn_dist = gp_info$nn_dist,
      nn_order = as.integer(gp_info$nn_order),
      nn_order_inv = as.integer(gp_info$nn_order_inv),
      nn_neighbor_dist = gp_info$nn_neighbor_dist,  # Phase 1.3: cached pairwise distances
      nn = as.integer(gp_info$nn),
      cov_type = gp_info$cov_type,
      nu = gp_info$nu %||% 1.5,  # Default nu for non-Matern covariances
      shared = gp_info$shared,
      sigma2_prior_U = priors$gp_sigma2_prior_U %||% 1.0,
      sigma2_prior_alpha = priors$gp_sigma2_prior_alpha %||% 0.01,
      phi_prior_lower = priors$gp_phi_prior_lower %||% 0.01,
      phi_prior_upper = priors$gp_phi_prior_upper %||% 100.0,
      # HSGP parameters
      hsgp_m = as.integer(gp_info$hsgp_m %||% 8L),
      hsgp_c = gp_info$hsgp_c %||% 1.5,
      # GP solver configuration
      solver = gp_info$solver %||% "auto",
      cg_tol = gp_info$cg_tol %||% 1e-6,
      cg_maxiter = as.integer(gp_info$cg_maxiter %||% 100L),
      # Observation-to-location mapping (1-based)
      gp_obs_to_loc = as.integer(gp_info$gp_obs_to_loc),
      n_unique = as.integer(gp_info$n_unique),
      # GP parameterization
      parameterization = gp_info$parameterization %||% "centered",
      # HSGP-MSGP flag
      msgp_approx = gp_info$msgp_approx %||% "nngp"
    )

    # Bundle multiscale GP parameters
    ms_gp_params <- list(
      nn_idx_local = as.integer(gp_info$nn_idx_local),
      nn_dist_local = gp_info$nn_dist_local,
      nn_order_local = as.integer(gp_info$nn_order_local),
      nn_order_inv_local = as.integer(gp_info$nn_order_inv_local),
      nn_local = as.integer(gp_info$nn_local),
      nn_neighbor_dist_local = gp_info$nn_neighbor_dist_local,  # Phase 1.3
      nn_idx_regional = as.integer(gp_info$nn_idx_regional),
      nn_dist_regional = gp_info$nn_dist_regional,
      nn_order_regional = as.integer(gp_info$nn_order_regional),
      nn_order_inv_regional = as.integer(gp_info$nn_order_inv_regional),
      nn_regional = as.integer(gp_info$nn_regional),
      nn_neighbor_dist_regional = gp_info$nn_neighbor_dist_regional,  # Phase 1.3
      range_local_lower = gp_info$range_local_lower,
      range_local_upper = gp_info$range_local_upper,
      range_regional_lower = gp_info$range_regional_lower,
      range_regional_upper = gp_info$range_regional_upper,
      sigma2_local_prior_U = priors$ms_sigma2_local_prior_U %||% 1.0,
      sigma2_local_prior_alpha = priors$ms_sigma2_local_prior_alpha %||% 0.01,
      sigma2_regional_prior_U = priors$ms_sigma2_regional_prior_U %||% 1.0,
      sigma2_regional_prior_alpha = priors$ms_sigma2_regional_prior_alpha %||% 0.01,
      sampler = gp_info$sampler %||% "noncentered"
    )

    # Bundle multiscale temporal parameters
    ms_temporal_params <- list(
      type = ms_temporal_info$ms_temporal_type,
      time_index = as.integer(ms_temporal_info$ms_time_index),
      group_index = as.integer(ms_temporal_info$ms_group_index),
      n_times = as.integer(ms_temporal_info$ms_n_times),
      n_groups = as.integer(ms_temporal_info$ms_n_groups),
      trend_type = ms_temporal_info$trend_type,
      seasonal_period = as.integer(ms_temporal_info$seasonal_period),
      short_term_type = ms_temporal_info$short_term_type,
      shared = ms_temporal_info$shared,
      sigma2_trend_prior_U = priors$ms_sigma2_trend_prior_U %||% 1.0,
      sigma2_trend_prior_alpha = priors$ms_sigma2_trend_prior_alpha %||% 0.01,
      sigma2_seasonal_prior_U = priors$ms_sigma2_seasonal_prior_U %||% 1.0,
      sigma2_seasonal_prior_alpha = priors$ms_sigma2_seasonal_prior_alpha %||% 0.01,
      sigma2_short_prior_U = priors$ms_sigma2_short_prior_U %||% 1.0,
      sigma2_short_prior_alpha = priors$ms_sigma2_short_prior_alpha %||% 0.01
    )

    # Bundle RSR parameters
    rsr_params <- list(
      has_rsr = rsr_info$has_rsr,
      projection = rsr_info$rsr_projection,
      n = as.integer(rsr_info$rsr_n)
    )

    # Bundle regular temporal parameters for GP interface
    # (previously missing — temporal was silently dropped for GP/HSGP models)
    temporal_params_gp <- list(
      type = temporal_info$type,
      time_idx = as.integer(temporal_info$time_index),
      group_idx = as.integer(temporal_info$group_index),
      n_times = as.integer(temporal_info$n_times),
      n_groups = as.integer(temporal_info$n_groups),
      n_params = as.integer(temporal_info$n_temporal_params),
      cyclic = temporal_info$precision_structure$cyclic %||% FALSE,
      shared = temporal_info$shared %||% TRUE,
      tau_shape = tau_temporal_shape,
      tau_rate = tau_temporal_rate
    )

    # Use O2-safe interface with single List parameter
    # This minimizes Rcpp template instantiation at ABI boundary
    fit_raw <- cpp_hmc_fit_gp_v2(list(
      q_init = q_init,
      y_num = as.integer(hmc_data$y_num),
      y_denom = as.integer(hmc_data$y_denom),
      y_denom_cont = hmc_data$y_denom_cont,
      X_num = hmc_data$X_num,
      X_denom = hmc_data$X_denom,
      re_group = as.integer(hmc_data$re_group),
      n_re_groups = as.integer(hmc_data$n_re_groups),
      model_type_str = model_type,
      # Bundled parameter lists
      gp_params = gp_params,
      ms_gp_params = ms_gp_params,
      ms_temporal_params = ms_temporal_params,
      rsr_params = rsr_params,
      temporal_params = temporal_params_gp,
      # Priors
      sigma_beta = sigma_beta,
      sigma_re_scale = sigma_re_scale,
      phi_prior_shape = phi_shape,
      phi_prior_rate = phi_rate,
      # Zero-inflation
      zi_type_str = zi_info$type,
      X_zi = zi_info$X_zi,
      zi_prior_sd = priors$zi_prior_sd %||% 10.0,
      # Sampler
      n_iter = as.integer(iter),
      n_warmup = as.integer(warmup),
      L = as.integer(L),
      n_chains = as.integer(chains),
      seed = as.integer(seed),
      n_threads = n_threads_within,
      verbose = verbose,
      gradient_mode_str = gradient_mode,
      max_treedepth = as.integer(max_treedepth),
      adapt_delta = adapt_delta_value,
      metric_str = metric
    ))
  } else {
    # Bundle parameters into lists to stay under R's 65-argument limit for .Call
    re_params <- list(
      group = as.integer(hmc_data$re_group),
      n_groups = as.integer(hmc_data$n_re_groups),
      n_terms = as.integer(n_re_terms),
      group_matrix = as.matrix(re_group_matrix),
      n_groups_vec = as.integer(re_n_groups_vec),
      has_slopes = has_re_slopes,
      has_correlated_slopes = has_re_correlated_slopes,
      n_coefs_vec = as.integer(re_n_coefs_vec),
      correlated_vec = as.logical(re_correlated_vec),
      n_chol_vec = as.integer(re_n_chol_vec),
      slope_matrices = slope_matrices_list,
      parameterization = as.integer(if (re_param == "centered") 0L else 1L)
    )

    spatial_params <- list(
      type = spatial_info$type,
      group = as.integer(spatial_info$group),
      n_units = spatial_info$n_units,
      adj_row_ptr = as.integer(spatial_info$adj_row_ptr),
      adj_col_idx = as.integer(spatial_info$adj_col_idx),
      n_neighbors = as.integer(spatial_info$n_neighbors),
      bym2_scale = spatial_info$bym2_scale,
      Q_inv = spatial_info$Q_inv,
      L_Q = spatial_info$L_Q,
      parameterization = spatial_info$parameterization %||% "standard"
    )

    temporal_params <- list(
      type = temporal_info$type,
      time_idx = as.integer(temporal_info$time_index),
      group_idx = as.integer(temporal_info$group_index),
      n_times = as.integer(temporal_info$n_times),
      n_groups = as.integer(temporal_info$n_groups),
      n_params = as.integer(temporal_info$n_temporal_params),
      cyclic = temporal_info$precision_structure$cyclic %||% FALSE,
      shared = temporal_info$shared %||% TRUE,
      tau_shape = tau_temporal_shape,
      tau_rate = tau_temporal_rate,
      # GP-specific fields (only used when type = "gp")
      time_values = temporal_info$time_values %||% numeric(0),
      cov_type = temporal_info$cov_type %||% "exponential",
      nu = temporal_info$nu %||% 1.5,
      period = temporal_info$period %||% 1.0,
      gp_sigma2_prior_U = priors$temporal_gp_sigma2_prior_U %||% 1.0,
      gp_sigma2_prior_alpha = priors$temporal_gp_sigma2_prior_alpha %||% 0.01,
      gp_phi_prior_lower = priors$temporal_gp_phi_prior_lower %||% 0.01,
      gp_phi_prior_upper = priors$temporal_gp_phi_prior_upper %||% 10.0,
      gp_parameterization = temporal_info$parameterization %||% "noncentered",
      # Multiscale temporal fields (only used when type = "multiscale")
      trend_type = temporal_info$trend %||% "rw1",
      short_term_type = temporal_info$short_term %||% "ar1",
      seasonal_period = as.integer(temporal_info$precision_structure$seasonal_period %||% 0L),
      sigma2_trend_prior_U = priors$ms_sigma2_trend_prior_U %||% 1.0,
      sigma2_trend_prior_alpha = priors$ms_sigma2_trend_prior_alpha %||% 0.01,
      sigma2_short_prior_U = priors$ms_sigma2_short_prior_U %||% 1.0,
      sigma2_short_prior_alpha = priors$ms_sigma2_short_prior_alpha %||% 0.01
    )

    prior_params <- list(
      sigma_beta = sigma_beta,
      sigma_re_scale = sigma_re_scale,
      phi_shape = phi_shape,
      phi_rate = phi_rate,
      tau_spatial_shape = tau_spatial_shape,
      tau_spatial_rate = tau_spatial_rate
    )

    zi_params <- list(
      type = zi_info$type,
      X = zi_info$X_zi,
      p_zi = zi_info$p_zi %||% as.integer(ncol(zi_info$X_zi)),  # Explicit p_zi for OI-only models
      prior_sd = priors$zi_prior_sd %||% 10.0,
      # OI info for OI-binomial and ZOIB models
      X_oi = zi_info$X_oi,
      p_oi = zi_info$p_oi %||% 0L,
      oi_prior_sd = priors$oi_prior_sd %||% priors$zi_prior_sd %||% 10.0
    )

    latent_params <- list(
      has_latent = latent_info$type != "none",
      n_factors = as.integer(latent_info$n_factors),
      shared = latent_info$shared,
      scale = latent_info$scale %||% TRUE,
      constraint = as.integer(ifelse(latent_info$constraint == "sum_to_zero", 0L, 1L)),
      sigma_prior_rate = latent_info$sigma_prior_rate
    )

    st_is_hsgp <- isTRUE(spatiotemporal_info$spatial_is_hsgp)
    st_params <- list(
      has_spatiotemporal = spatiotemporal_info$has_spatiotemporal %||% FALSE,
      type = spatiotemporal_info$type %||% "none",
      shared = spatiotemporal_info$shared %||% TRUE,
      n_spatial = as.integer(spatiotemporal_info$n_spatial %||% 0L),
      n_times = as.integer(spatiotemporal_info$n_times %||% 0L),
      n_params = as.integer(spatiotemporal_info$n_params %||% 0L),
      s_idx = as.integer(spatiotemporal_info$s_idx %||% integer(0)),
      t_idx = as.integer(spatiotemporal_info$t_idx %||% integer(0)),
      st_flat = as.integer(spatiotemporal_info$st_flat %||% integer(0)),
      temporal_type = spatiotemporal_info$temporal_type %||% "rw1",
      temporal_cyclic = spatiotemporal_info$temporal_cyclic %||% FALSE,
      adj_row_ptr = as.integer(spatiotemporal_info$spatial_Q$adj_row_ptr %||% integer(0)),
      adj_col_idx = as.integer(spatiotemporal_info$spatial_Q$adj_col_idx %||% integer(0)),
      sigma2_prior_U = priors$st_sigma2_prior_U %||% 1.0,
      sigma2_prior_alpha = priors$st_sigma2_prior_alpha %||% 0.01,
      # Kronecker precision mass data
      Qs_inv = spatiotemporal_info$spatial_Q$Q_inv,
      Ls = spatiotemporal_info$spatial_Q$L_Q,
      Qt_inv = spatiotemporal_info$temporal_Q$Q_time_inv,
      Lt = spatiotemporal_info$temporal_Q$L_time,
      # HSGP-ST fields (only used when spatial_is_hsgp = TRUE)
      st_is_hsgp = st_is_hsgp,
      hsgp_m = as.integer(spatiotemporal_info$hsgp_m %||% 0L),
      hsgp_c = spatiotemporal_info$hsgp_c %||% 1.5,
      hsgp_coords = if (st_is_hsgp) as.numeric(t(spatiotemporal_info$hsgp_coords)) else numeric(0),
      hsgp_scale_coords = spatiotemporal_info$hsgp_scale_coords %||% TRUE
    )

    # TVC (Temporally-Varying Coefficients) parameters
    # Check if temporal is a TVC specification
    tvc_info <- prepare_tvc_for_hmc(temporal, data, hmc_data$N, hmc_data$X_num)
    tvc_params <- list(
      has_tvc = tvc_info$has_tvc %||% FALSE,
      n_tvc = as.integer(tvc_info$n_tvc %||% 0L),
      n_times = as.integer(tvc_info$n_times %||% 0L),
      n_groups = as.integer(tvc_info$n_groups %||% 1L),
      structure = tvc_info$structure %||% "rw1",
      shared = tvc_info$shared %||% TRUE,
      cyclic = tvc_info$cyclic %||% FALSE,
      tvc_indices = as.integer(tvc_info$tvc_indices %||% integer(0)),
      time_index = as.integer(tvc_info$time_index %||% integer(0)),
      group_index = as.integer(tvc_info$group_index %||% integer(0)),
      X_tvc = as.numeric(tvc_info$X_tvc %||% numeric(0)),
      tau_shape = priors$tvc_tau_shape %||% 2.0,
      tau_rate = priors$tvc_tau_rate %||% 0.5
    )

    # SVC (Spatially-Varying Coefficients) parameters
    # Check if spatial is an SVC specification
    svc_info <- prepare_svc_for_hmc(spatial, data, hmc_data$N, hmc_data$X_num)
    svc_params <- list(
      has_svc = svc_info$has_svc %||% FALSE,
      n_svc = as.integer(svc_info$n_svc %||% 0L),
      nn = as.integer(svc_info$nn %||% 0L),
      shared = svc_info$shared %||% TRUE,
      cov_type = svc_info$cov_type %||% "exponential",
      coords = as.numeric(svc_info$coords %||% numeric(0)),
      svc_indices = as.integer(svc_info$svc_indices %||% integer(0)),
      X_svc = as.numeric(svc_info$X_svc %||% numeric(0)),
      nn_idx = as.integer(svc_info$nn_idx %||% integer(0)),
      nn_dist = as.numeric(svc_info$nn_dist %||% numeric(0)),
      nn_order = as.integer(svc_info$nn_order %||% integer(0)),
      nn_order_inv = as.integer(svc_info$nn_order_inv %||% integer(0)),
      sigma2_prior_scale = svc_info$sigma2_prior_scale %||% 1.0,
      phi_prior_lower = svc_info$phi_prior_lower %||% 0.01,
      phi_prior_upper = svc_info$phi_prior_upper %||% 10.0,
      tau_shape = priors$svc_tau_shape %||% 1.0,
      tau_rate = priors$svc_tau_rate %||% 0.01,
      svc_approx = svc_info$svc_approx %||% "nngp",
      hsgp_m = as.integer(svc_info$hsgp_m %||% 0L),
      hsgp_c = svc_info$hsgp_c %||% 0.0
    )

    # Populate spatial_info with SVC details for output conversion
    if (svc_info$has_svc) {
      spatial_info$n_svc <- svc_info$n_svc
      spatial_info$svc_names <- svc_info$svc_names
      spatial_info$svc_approx <- svc_info$svc_approx %||% "nngp"
      spatial_info$svc_hsgp_m <- svc_info$hsgp_m %||% 0L
      spatial_info$svc_hsgp_c <- svc_info$hsgp_c %||% 1.5
      spatial_info$svc_coords <- if (!is.null(svc_info$coords)) {
        matrix(svc_info$coords, ncol = 2, byrow = TRUE)
      } else {
        NULL
      }
      spatial_info$X_svc <- if (!is.null(svc_info$X_svc)) {
        matrix(svc_info$X_svc, nrow = hmc_data$N, ncol = svc_info$n_svc)
      } else {
        NULL
      }
    }

    fit_raw <- cpp_hmc_fit(
      q_init = q_init,
      y_num = as.integer(hmc_data$y_num),
      y_denom = as.integer(hmc_data$y_denom),
      y_num_cont = hmc_data$y_num_cont,
      y_denom_cont = hmc_data$y_denom_cont,
      X_num = hmc_data$X_num,
      X_denom = hmc_data$X_denom,
      model_type_str = model_type,
      re_params = re_params,
      spatial_params = spatial_params,
      temporal_params = temporal_params,
      prior_params = prior_params,
      zi_params = zi_params,
      latent_params = latent_params,
      st_params = st_params,
      tvc_params = tvc_params,
      svc_params = svc_params,
      n_iter = as.integer(iter),
      n_warmup = as.integer(warmup),
      L = as.integer(L),
      n_chains = as.integer(chains),
      seed = as.integer(seed),
      n_threads = n_threads_within,
      verbose = verbose,
      gradient_mode_str = gradient_mode,
      max_treedepth = as.integer(max_treedepth),
      metric_str = metric,
      adapt_delta = adapt_delta_value,
      riemannian = riemannian_value
    )
  }

  # Post-fit warnings
  if (verbose && L == 0) {
    # NUTS diagnostics
    n_div <- fit_raw$n_divergent %||%
      sum(unlist(lapply(fit_raw$divergent, function(x) sum(x))))
    if (n_div > 0) {
      warning(n_div, " divergent transition(s) after warmup.",
              " Increase max_treedepth or reparameterize.", call. = FALSE)
    }
  }

  # Convert to ratiod_fit
  fit <- convert_hmc_to_ratiod_fit_full(
    fit_raw = fit_raw,
    hmc_data = hmc_data,
    spatial_info = spatial_info,
    temporal_info = temporal_info,
    spatiotemporal_info = spatiotemporal_info,
    zi_info = zi_info,
    latent_info = latent_info,
    formula = formula,
    data = data,
    family = family,
    model_type = model_type,
    iter = iter,
    warmup = warmup,
    chains = chains,
    re_param = re_param
  )

  return(fit)
}


#' Get HMC model type string
#' @keywords internal
get_hmc_model_type <- function(family) {
  # Check family name first for families that share numerator distribution
  if (identical(family$name, "negbin_gamma")) return("negbin_gamma")

  dist <- family$numerator$distribution
  if (dist == "binomial") return("binomial")
  if (dist %in% c("negbin", "negative_binomial", "neg_binomial_2")) return("negbin_negbin")
  if (dist == "poisson") return("poisson_gamma")
  if (dist == "gamma") return("gamma_gamma")
  if (dist == "lognormal") return("lognormal")
  if (dist == "beta_binomial") return("beta_binomial")

  # ZI/Hurdle/OI/ZOIB variants map to their base model types
  if (dist %in% c("zero_inflated_binomial", "hurdle_binomial",
                  "one_inflated_binomial", "zero_one_inflated_binomial")) return("binomial")
  if (dist %in% c("zero_inflated_poisson", "hurdle_poisson")) return("poisson_gamma")
  if (dist %in% c("zero_inflated_neg_binomial", "hurdle_neg_binomial")) return("negbin_negbin")

  stop("Unsupported family for HMC backend: ", dist)
}


#' Prepare data for HMC sampler
#' @keywords internal
prepare_hmc_data <- function(formula, data, family, model_type) {
  # Extract responses
  y_num <- formula$numerator$response
  y_denom <- formula$denominator$response

  # Design matrices
  X_num <- formula$numerator$X
  X_denom <- formula$denominator$X

  # Ensure X_denom exists
  # For binomial/beta_binomial, denominator is fixed trials - no beta_denom needed
  if (model_type %in% c("binomial", "beta_binomial")) {
    X_denom <- matrix(numeric(0), nrow = nrow(data), ncol = 0)
  } else if (is.null(X_denom) || ncol(X_denom) == 0) {
    X_denom <- matrix(1, nrow = nrow(data), ncol = 1)
    colnames(X_denom) <- "(Intercept)_denom"
  }

  # Random effects
  re_info <- extract_re_for_hmc(formula)

  # Prepare response based on model type
  # Initialize y_num_cont (used for continuous families)
  y_num_cont <- rep(0.0, length(y_num))

  if (model_type == "binomial") {
    y_num <- as.integer(y_num)
    y_denom <- as.integer(y_denom)
    y_denom_cont <- rep(0.0, length(y_num))
  } else if (model_type == "negbin_negbin") {
    y_num <- as.integer(y_num)
    y_denom <- as.integer(y_denom)
    y_denom_cont <- rep(0.0, length(y_num))
  } else if (model_type == "poisson_gamma") {
    y_num <- as.integer(y_num)
    y_denom_int <- rep(1L, length(y_num))
    y_denom_cont <- as.numeric(y_denom)
    # Gamma distribution requires y > 0; clamp zeros with warning
    n_zero <- sum(y_denom_cont <= 0)
    if (n_zero > 0) {
      warning(sprintf(
        "%d denominator value(s) <= 0 (Gamma requires y > 0). Clamping to 1e-6.",
        n_zero), call. = FALSE)
      y_denom_cont[y_denom_cont <= 0] <- 1e-6
    }
    y_denom <- y_denom_int
  } else if (model_type == "negbin_gamma") {
    y_num <- as.integer(y_num)
    y_denom_int <- rep(1L, length(y_num))
    y_denom_cont <- as.numeric(y_denom)
    n_zero <- sum(y_denom_cont <= 0)
    if (n_zero > 0) {
      warning(sprintf(
        "%d denominator value(s) <= 0 (Gamma requires y > 0). Clamping to 1e-6.",
        n_zero), call. = FALSE)
      y_denom_cont[y_denom_cont <= 0] <- 1e-6
    }
    y_denom <- y_denom_int
  } else if (model_type == "gamma_gamma") {
    # Gamma-Gamma: both continuous
    y_num_cont <- as.numeric(y_num)
    y_denom_cont <- as.numeric(y_denom)
    n_zero <- sum(y_num_cont <= 0) + sum(y_denom_cont <= 0)
    if (n_zero > 0) {
      warning(sprintf(
        "%d value(s) <= 0 (Gamma requires y > 0). Clamping to 1e-6.",
        n_zero), call. = FALSE)
      y_num_cont[y_num_cont <= 0] <- 1e-6
      y_denom_cont[y_denom_cont <= 0] <- 1e-6
    }
    y_num <- rep(0L, length(y_num))    # Dummy integer for C++ struct
    y_denom <- rep(0L, length(y_num_cont))
  } else if (model_type == "lognormal") {
    # Lognormal: both continuous
    y_num_cont <- as.numeric(y_num)
    y_denom_cont <- as.numeric(y_denom)
    n_zero <- sum(y_num_cont <= 0) + sum(y_denom_cont <= 0)
    if (n_zero > 0) {
      warning(sprintf(
        "%d value(s) <= 0 (Lognormal requires y > 0). Clamping to 1e-6.",
        n_zero), call. = FALSE)
      y_num_cont[y_num_cont <= 0] <- 1e-6
      y_denom_cont[y_denom_cont <= 0] <- 1e-6
    }
    y_num <- rep(0L, length(y_num))    # Dummy integer for C++ struct
    y_denom <- rep(0L, length(y_num_cont))
  } else if (model_type == "beta_binomial") {
    # Beta-binomial: integer counts
    y_num <- as.integer(y_num)
    y_denom <- as.integer(y_denom)
    y_denom_cont <- rep(0.0, length(y_num))
  }

  # Build slope design matrices for random effects with slopes
  # These are pre-computed during formula parsing with proper expansion
  slope_matrices <- NULL
  if (re_info$has_slopes && re_info$n_re_terms > 0) {
    slope_matrices <- vector("list", re_info$n_re_terms)
    for (t in seq_len(re_info$n_re_terms)) {
      term <- re_info$re_terms[[t]]
      if (length(term$slope_vars) > 0) {
        # Use the pre-computed slope_matrix from formula parsing
        # This handles interactions (x*z), polynomials (poly(x,2)), etc.
        raw_re <- formula$numerator$random_effects[[t]]
        if (!is.null(raw_re$slope_matrix)) {
          slope_matrices[[t]] <- raw_re$slope_matrix
        } else {
          # Fallback: build design matrix manually (for backwards compatibility)
          slope_mat <- matrix(0, nrow = length(y_num), ncol = length(term$slope_vars))
          for (s in seq_along(term$slope_vars)) {
            var_spec <- term$slope_vars[s]
            if (!var_spec %in% names(data)) {
              stop(sprintf("Slope variable '%s' not found in data", var_spec),
                   call. = FALSE)
            }
            slope_mat[, s] <- as.numeric(data[[var_spec]])
          }
          colnames(slope_mat) <- term$slope_vars
          slope_matrices[[t]] <- slope_mat
        }
      } else {
        slope_matrices[[t]] <- NULL
      }
    }
  }

  list(
    y_num = y_num,
    y_denom = y_denom,
    y_num_cont = y_num_cont,
    y_denom_cont = y_denom_cont,
    X_num = X_num,
    X_denom = X_denom,
    # Legacy single-term RE fields (for backwards compatibility)
    re_group = re_info$group_idx,
    n_re_groups = re_info$n_groups,
    re_var = re_info$group_var,
    # New multi-term RE fields
    n_re_terms = re_info$n_re_terms,
    re_terms = re_info$re_terms,
    re_group_matrix = re_info$group_idx_matrix,
    total_re_groups = re_info$total_groups %||% re_info$n_groups,
    total_re_params = re_info$total_re_params %||% re_info$total_groups,
    total_sigma_params = re_info$total_sigma_params %||% re_info$n_re_terms,
    total_chol_params = re_info$total_chol_params %||% 0L,
    has_slopes = re_info$has_slopes %||% FALSE,
    has_correlated_slopes = re_info$has_correlated_slopes %||% FALSE,
    slope_matrices = slope_matrices,
    N = length(y_num),
    p_num = ncol(X_num),
    p_denom = ncol(X_denom)
  )
}


#' Extract RE info for HMC
#'
#' @description
#' Extracts random effects information for the HMC backend.
#' Supports:
#' - Multiple crossed random effects: `(1|site) + (1|year)`
#' - Correlated random slopes: `(1 + x | group)` - estimates intercept-slope correlation
#' - Uncorrelated random slopes: `(1 + x || group)` - no correlation estimated
#'
#' For correlated slopes, the covariance matrix is parameterized via Cholesky decomposition:
#' `Sigma = diag(sigma) * L * L' * diag(sigma)` where L is lower-triangular with unit diagonal
#' (LKJ correlation prior).
#'
#' @param formula A ratiod_formula object
#' @return List with RE structure for HMC backend
#' @keywords internal
extract_re_for_hmc <- function(formula) {
  re_terms <- formula$numerator$random_effects
  n_obs <- length(formula$numerator$response)

  if (is.null(re_terms) || length(re_terms) == 0) {
    return(list(
      # Legacy single-term fields (for backwards compatibility)
      group_idx = rep(0L, n_obs),
      n_groups = 0L,
      group_var = NULL,
      # New multi-term fields
      n_re_terms = 0L,
      re_terms = list(),
      has_slopes = FALSE,
      has_correlated_slopes = FALSE,
      total_re_params = 0L,
      total_sigma_params = 0L,
      total_chol_params = 0L
    ))
  }

  n_re_terms <- length(re_terms)

  # Check for slopes and correlation structure
  has_slopes <- FALSE
  has_correlated_slopes <- FALSE
  for (term in re_terms) {
    if (length(term$slope_vars) > 0) {
      has_slopes <- TRUE
      if (isTRUE(term$correlated)) {
        has_correlated_slopes <- TRUE
      }
    }
  }

  # Process RE terms with full slope support
  re_terms_processed <- vector("list", n_re_terms)
  total_groups <- 0L
  total_re_params <- 0L  # Total RE parameters including slopes
  total_sigma_params <- 0L  # Total variance parameters (one per coefficient type)
  total_chol_params <- 0L  # Total Cholesky lower-triangular parameters (for correlations)

  for (t in seq_len(n_re_terms)) {
    term <- re_terms[[t]]
    n_groups_t <- as.integer(term$n_groups)

    # Number of RE coefficients per group: intercept + slopes (expanded)
    # Use the expanded slope_vars which handles interactions (x*z -> x, z, x:z)
    n_slopes <- if (!is.null(term$slope_vars)) length(term$slope_vars) else 0L
    n_coefs <- (if (isTRUE(term$has_intercept)) 1L else 0L) + n_slopes

    # Total parameters for this term: n_groups * n_coefs
    n_params_t <- n_groups_t * n_coefs

    # Number of Cholesky parameters for correlated slopes
    # For a k×k correlation matrix, we need k*(k-1)/2 off-diagonal elements
    # (diagonal is implicitly 1 for correlation matrix parameterization)
    n_chol_t <- if (n_coefs > 1 && isTRUE(term$correlated)) {
      as.integer(n_coefs * (n_coefs - 1) / 2)
    } else {
      0L
    }

    # Number of sigma parameters for this term
    n_sigma_t <- n_coefs

    re_terms_processed[[t]] <- list(
      group_var = term$group_var,
      group_idx = as.integer(term$group),
      n_groups = n_groups_t,
      has_intercept = isTRUE(term$has_intercept),
      slope_vars = term$slope_vars,  # Expanded slope names from model.matrix
      slope_vars_clean = term$slope_vars_clean,  # Cleaned names for parameter output
      correlated = isTRUE(term$correlated),
      n_coefs = n_coefs,  # Coefficients per group (1 for intercept-only, more for slopes)
      n_sigma = n_sigma_t,  # Number of sigma parameters
      n_chol = n_chol_t,  # Number of Cholesky parameters (0 if uncorrelated)
      re_offset = total_re_params,  # Offset in flattened RE parameter vector
      sigma_offset = total_sigma_params,  # Offset for variance parameters
      chol_offset = total_chol_params  # Offset for Cholesky parameters
    )

    total_groups <- total_groups + n_groups_t
    total_re_params <- total_re_params + n_params_t
    total_sigma_params <- total_sigma_params + n_sigma_t
    total_chol_params <- total_chol_params + n_chol_t
  }

  # Build group index matrix: [n_obs, n_re_terms]
  group_idx_matrix <- matrix(0L, nrow = n_obs, ncol = n_re_terms)
  for (t in seq_len(n_re_terms)) {
    group_idx_matrix[, t] <- as.integer(re_terms[[t]]$group)
  }

  list(
    # Legacy single-term fields (use first term for backwards compatibility)
    group_idx = as.integer(re_terms[[1]]$group),
    n_groups = as.integer(re_terms[[1]]$n_groups),
    group_var = re_terms[[1]]$group_var,
    # New multi-term fields
    n_re_terms = n_re_terms,
    re_terms = re_terms_processed,
    group_idx_matrix = group_idx_matrix,
    total_groups = total_groups,
    total_re_params = total_re_params,  # Total RE effects (intercepts + slopes)
    total_sigma_params = total_sigma_params,  # Total variance parameters
    total_chol_params = total_chol_params,  # Total Cholesky correlation parameters
    has_slopes = has_slopes,
    has_correlated_slopes = has_correlated_slopes
  )
}


#' Prepare spatial structure for HMC
#' @keywords internal
prepare_spatial_for_hmc <- function(spatial, data, N) {
  if (is.null(spatial)) {
    return(list(
      type = "none",
      group = rep(0L, N),
      n_units = 0L,
      adj_row_ptr = integer(1),
      adj_col_idx = integer(0),
      n_neighbors = integer(0),
      bym2_scale = 1.0
    ))
  }

  # Extract spatial type - check $type attribute or class
  spatial_type <- if (inherits(spatial, "ratiod_gp")) {
    "gp"
  } else if (inherits(spatial, "ratiod_svc")) {
    "svc"
  } else if (inherits(spatial, "ratiod_spatial_bym2") ||
             (!is.null(spatial$type) && spatial$type == "bym2")) {
    "bym2"
  } else if (inherits(spatial, "ratiod_spatial_car") ||
             inherits(spatial, "ratiod_spatial_icar") ||
             (!is.null(spatial$type) && spatial$type %in% c("car", "car_proper"))) {
    "icar"
  } else {
    stop("Unknown spatial structure type")
  }

  # Get group mapping
  if (!is.null(spatial$group_var)) {
    spatial_group <- as.integer(factor(data[[spatial$group_var]]))
    n_units <- max(spatial_group)
  } else {
    # Observation-level spatial
    spatial_group <- seq_len(N)
    n_units <- N
  }

  # Convert adjacency matrix to CSR format
  adj <- spatial$adjacency
  if (inherits(adj, "Matrix")) {
    adj <- as.matrix(adj)
  }

  # Build CSR structure
  n_neighbors <- integer(n_units)
  adj_col_idx <- integer(0)
  adj_row_ptr <- integer(n_units + 1)

  for (i in seq_len(n_units)) {
    neighbors <- which(adj[i, ] > 0)
    n_neighbors[i] <- length(neighbors)
    adj_col_idx <- c(adj_col_idx, neighbors - 1L)  # Convert to 0-based for C++
    adj_row_ptr[i + 1] <- adj_row_ptr[i] + n_neighbors[i]
  }

  # BYM2 scale factor (from eigenvalues of precision matrix)
  bym2_scale <- 1.0
  if (spatial_type == "bym2") {
    # Compute scale factor for BYM2
    # This makes the structured component have unit generalized variance
    Q <- diag(n_neighbors) - adj
    # Remove rank deficiency for eigenvalue computation
    Q_scaled <- Q + 0.001 * diag(n_units)
    eig <- eigen(Q_scaled, symmetric = TRUE, only.values = TRUE)
    # Scale factor is based on geometric mean of non-zero eigenvalues
    nonzero_eig <- eig$values[eig$values > 0.01]
    if (length(nonzero_eig) > 0) {
      bym2_scale <- 1 / sqrt(exp(mean(log(nonzero_eig))))
    }
  }

  # Precision mass matrix: compute Q_inv and L_Q for ICAR/BYM2
  # This provides the known correlation structure to the HMC mass matrix,
  # avoiding under-determined OAS estimation for 50+ spatial params.
  Q_inv_flat <- NULL
  L_Q_flat <- NULL
  if (spatial_type %in% c("icar", "bym2", "car", "car_proper") && n_units >= 5) {
    Q <- diag(n_neighbors) - adj
    Q_reg <- Q + 0.01 * diag(n_units)  # Regularize rank-deficient Q
    Q_inv <- solve(Q_reg)
    L_Q <- chol(Q_reg)  # Upper Cholesky: L^T * L = Q_reg
    # Convert to lower Cholesky (column-major): L * L^T = Q_reg
    L_Q_lower <- t(L_Q)
    Q_inv_flat <- as.numeric(Q_inv)      # Column-major flat [S x S]
    L_Q_flat <- as.numeric(L_Q_lower)    # Column-major flat [S x S]
  }

  list(
    type = spatial_type,
    group = spatial_group,
    n_units = n_units,
    adj_row_ptr = adj_row_ptr,
    adj_col_idx = adj_col_idx,
    n_neighbors = n_neighbors,
    bym2_scale = bym2_scale,
    group_var = spatial$group_var,  # Preserve for prediction lookup
    Q_inv = Q_inv_flat,
    L_Q = L_Q_flat,
    parameterization = spatial$parameterization %||% "standard"
  )
}


#' Initialize parameters for HMC with spatial
#' @keywords internal
initialize_hmc_params_spatial <- function(hmc_data, model_type, spatial_info) {
  n_params <- hmc_data$p_num + hmc_data$p_denom

  # Random effects
  if (hmc_data$n_re_groups > 0) {
    n_params <- n_params + 1 + hmc_data$n_re_groups
  }

  # Overdispersion
  if (model_type == "negbin_negbin") {
    n_params <- n_params + 2
  } else if (model_type == "poisson_gamma") {
    n_params <- n_params + 1
  } else if (model_type == "negbin_gamma") {
    n_params <- n_params + 2
  }

  # Spatial
  is_collapsed <- identical(spatial_info$parameterization, "collapsed")
  if (spatial_info$type == "icar") {
    if (is_collapsed) {
      n_params <- n_params + 1  # log_tau only (phi marginalized)
    } else {
      n_params <- n_params + 1 + spatial_info$n_units  # log_tau + phi
    }
  } else if (spatial_info$type == "bym2") {
    if (is_collapsed) {
      n_params <- n_params + 2  # log_sigma_total + logit_rho only
    } else {
      # log_sigma_total + logit_rho + phi_scaled + theta
      n_params <- n_params + 2 + 2 * spatial_info$n_units
    }
  }

  rep(0.0, n_params)
}


#' Convert HMC output to ratiod_fit (with spatial support)
#' @keywords internal
convert_hmc_to_ratiod_fit_spatial <- function(fit_raw, hmc_data, spatial_info,
                                              formula, data, family,
                                              model_type, iter, warmup, chains,
                                              temporal_info = NULL,
                                              spatiotemporal_info = NULL) {
  # Handle multi-chain case
  if (chains > 1) {
    # Combine chains
    all_samples <- do.call(rbind, fit_raw$samples)
    all_log_prob <- unlist(fit_raw$log_prob)
    all_accept <- unlist(fit_raw$accept_prob)
    all_n_leapfrog <- unlist(fit_raw$n_leapfrog)
    all_treedepth <- unlist(fit_raw$treedepth)
    all_divergent <- unlist(fit_raw$divergent)
    epsilon <- mean(fit_raw$epsilon)
  } else {
    all_samples <- fit_raw$samples
    all_log_prob <- fit_raw$log_prob
    all_accept <- fit_raw$accept_prob
    all_n_leapfrog <- fit_raw$n_leapfrog
    all_treedepth <- fit_raw$treedepth
    all_divergent <- fit_raw$divergent
    epsilon <- fit_raw$epsilon
  }

  n_samples <- nrow(all_samples)

  # Build parameter names and extract draws
  draws_list <- build_draws_list_spatial(
    all_samples, hmc_data, spatial_info, model_type, re_param = re_param
  )

  draws <- do.call(cbind, draws_list)
  colnames(draws) <- names(draws_list)

  # Compute ratios
  ratio_draws <- compute_ratio_draws_hmc_spatial(
    all_samples, hmc_data, spatial_info, model_type, re_param = re_param
  )

  # Diagnostics
  n_divergent <- sum(all_divergent)
  avg_accept <- mean(all_accept)

  fit <- list(
    draws = draws,
    ratio_draws = ratio_draws,
    formula = formula,
    data = data,
    family = family,
    spatial = spatial_info,
    temporal = temporal_info,
    spatiotemporal = spatiotemporal_info,
    backend = "hmc",
    algorithm = fit_raw$sampler %||% "HMC",
    re_param = re_param,
    iter = iter,
    warmup = warmup,
    chains = chains,
    n_save = n_samples,
    epsilon = epsilon,
    diagnostics = list(
      algorithm = fit_raw$sampler %||% "HMC",
      n_divergent = n_divergent,
      avg_accept_prob = avg_accept,
      n_leapfrog = all_n_leapfrog,
      treedepth = all_treedepth,
      divergent = all_divergent,
      log_posterior = all_log_prob
    ),
    log_prob = all_log_prob,
    .internal = list(
      samples = all_samples,
      hmc_data = hmc_data,
      model_type = model_type
    )
  )

  class(fit) <- "ratiod_fit"
  return(fit)
}


#' Extract spatial draws from HMC samples into draws_list
#'
#' Shared helper for build_draws_list_spatial and build_draws_list_full.
#' Handles ICAR, BYM2, GP, multiscale GP, and HSGP spatial types,
#' including collapsed parameterization (where phi/theta/w are marginalized out).
#'
#' @param samples Matrix of HMC samples
#' @param idx Current column index into samples
#' @param spatial_info Spatial configuration list
#' @return List with `draws` (named list of draws) and `idx` (updated column index)
#' @keywords internal
extract_spatial_draws <- function(samples, idx, spatial_info) {
  draws <- list()

  if (spatial_info$type == "icar") {
    param <- spatial_info$parameterization %||% "standard"
    draws[["tau_spatial"]] <- exp(samples[, idx])
    idx <- idx + 1
    if (param != "collapsed") {
      for (s in seq_len(spatial_info$n_units)) {
        draws[[paste0("phi_spatial[", s, "]")]] <- samples[, idx]
        idx <- idx + 1
      }
    }
  } else if (spatial_info$type == "bym2") {
    param <- spatial_info$parameterization %||% "standard"
    sigma_total <- exp(samples[, idx])
    idx <- idx + 1
    rho <- 1 / (1 + exp(-samples[, idx]))
    idx <- idx + 1

    sigma_s <- sigma_total * sqrt(rho)
    sigma_u <- sigma_total * sqrt(1 - rho)

    draws[["sigma_spatial"]] <- sigma_total
    draws[["rho_spatial"]] <- rho
    draws[["sigma_s_spatial"]] <- sigma_s
    draws[["sigma_u_spatial"]] <- sigma_u

    if (param != "collapsed") {
      for (s in seq_len(spatial_info$n_units)) {
        draws[[paste0("phi_scaled[", s, "]")]] <- samples[, idx]
        idx <- idx + 1
      }
      for (s in seq_len(spatial_info$n_units)) {
        draws[[paste0("theta[", s, "]")]] <- samples[, idx]
        idx <- idx + 1
      }
    }
  } else if (spatial_info$type == "gp") {
    draws[["sigma2_gp"]] <- exp(samples[, idx])
    idx <- idx + 1
    draws[["phi_gp"]] <- exp(samples[, idx])
    idx <- idx + 1
    param <- spatial_info$parameterization %||% "centered"
    if (param != "collapsed") {
      for (s in seq_len(spatial_info$n_units)) {
        draws[[paste0("gp_w[", s, "]")]] <- samples[, idx]
        idx <- idx + 1
      }
    }
  } else if (spatial_info$type == "multiscale_gp") {
    msgp_approx <- spatial_info$msgp_approx %||% "nngp"
    if (msgp_approx == "hsgp") {
      # HSGP-MSGP: hyperparams + basis coefficients per scale
      hsgp_m <- spatial_info$hsgp_m %||% 6L
      m_total <- hsgp_m * hsgp_m

      draws[["sigma2_hsgp_local"]] <- exp(samples[, idx])
      idx <- idx + 1
      draws[["lengthscale_hsgp_local"]] <- exp(samples[, idx])
      idx <- idx + 1
      for (s in seq_len(m_total)) {
        draws[[paste0("hsgp_beta_local[", s, "]")]] <- samples[, idx]
        idx <- idx + 1
      }
      draws[["sigma2_hsgp_regional"]] <- exp(samples[, idx])
      idx <- idx + 1
      draws[["lengthscale_hsgp_regional"]] <- exp(samples[, idx])
      idx <- idx + 1
      for (s in seq_len(m_total)) {
        draws[[paste0("hsgp_beta_regional[", s, "]")]] <- samples[, idx]
        idx <- idx + 1
      }
    } else {
      # NNGP-MSGP: spatial effects per location
      draws[["sigma2_gp_local"]] <- exp(samples[, idx])
      idx <- idx + 1
      draws[["phi_gp_local"]] <- exp(samples[, idx])
      idx <- idx + 1
      for (s in seq_len(spatial_info$n_units)) {
        draws[[paste0("gp_local_w[", s, "]")]] <- samples[, idx]
        idx <- idx + 1
      }
      draws[["sigma2_gp_regional"]] <- exp(samples[, idx])
      idx <- idx + 1
      draws[["phi_gp_regional"]] <- exp(samples[, idx])
      idx <- idx + 1
      for (s in seq_len(spatial_info$n_units)) {
        draws[[paste0("gp_regional_w[", s, "]")]] <- samples[, idx]
        idx <- idx + 1
      }
    }
  } else if (spatial_info$type == "hsgp") {
    draws[["sigma2_hsgp"]] <- exp(samples[, idx])
    idx <- idx + 1
    draws[["lengthscale_hsgp"]] <- exp(samples[, idx])
    idx <- idx + 1
    hsgp_m <- spatial_info$hsgp_m %||% 8L
    for (s in seq_len(hsgp_m * hsgp_m)) {
      draws[[paste0("hsgp_beta[", s, "]")]] <- samples[, idx]
      idx <- idx + 1
    }
  }

  list(draws = draws, idx = idx)
}


#' Build draws list with spatial parameters
#' @keywords internal
build_draws_list_spatial <- function(samples, hmc_data, spatial_info, model_type,
                                      re_param = "noncentered") {
  draws_list <- list()
  idx <- 1

  # Fixed effects numerator
  for (j in seq_len(hmc_data$p_num)) {
    name <- paste0("beta_num[", j, "]")
    draws_list[[name]] <- samples[, idx]
    idx <- idx + 1
  }

  # Fixed effects denominator
  for (j in seq_len(hmc_data$p_denom)) {
    name <- paste0("beta_denom[", j, "]")
    draws_list[[name]] <- samples[, idx]
    idx <- idx + 1
  }

  # Random effects - transform z to re if noncentered
  if (hmc_data$n_re_groups > 0) {
    sigma_re <- exp(samples[, idx])
    draws_list[["sigma_re"]] <- sigma_re
    idx <- idx + 1
    for (g in seq_len(hmc_data$n_re_groups)) {
      z_or_re <- samples[, idx]
      if (re_param == "noncentered") {
        draws_list[[paste0("re[", g, "]")]] <- sigma_re * z_or_re
      } else {
        draws_list[[paste0("re[", g, "]")]] <- z_or_re
      }
      idx <- idx + 1
    }
  }

  # Overdispersion
  if (model_type == "negbin_negbin" || model_type == "negbin_gamma") {
    draws_list[["phi_num"]] <- exp(samples[, idx])
    idx <- idx + 1
    draws_list[["phi_denom"]] <- exp(samples[, idx])
    idx <- idx + 1
  } else if (model_type == "poisson_gamma") {
    draws_list[["shape"]] <- exp(samples[, idx])
    idx <- idx + 1
  }

  # Spatial
  sp_result <- extract_spatial_draws(samples, idx, spatial_info)
  draws_list <- c(draws_list, sp_result$draws)
  idx <- sp_result$idx

  draws_list
}


#' Compute ratio draws from HMC samples (with spatial)
#' @keywords internal
compute_ratio_draws_hmc_spatial <- function(samples, hmc_data, spatial_info,
                                             model_type, re_param = "noncentered") {
  n_samples <- nrow(samples)
  N <- hmc_data$N

  # Extract fixed effects
  beta_num <- samples[, seq_len(hmc_data$p_num), drop = FALSE]
  beta_denom <- samples[, hmc_data$p_num + seq_len(hmc_data$p_denom), drop = FALSE]

  idx <- hmc_data$p_num + hmc_data$p_denom + 1

  # Random effects - transform z to re if noncentered
  re <- NULL
  if (hmc_data$n_re_groups > 0) {
    sigma_re <- exp(samples[, idx])
    idx <- idx + 1
    z_or_re <- samples[, idx:(idx + hmc_data$n_re_groups - 1), drop = FALSE]
    if (re_param == "noncentered") {
      # Transform: re = sigma * z
      re <- sweep(z_or_re, 1, sigma_re, "*")
    } else {
      re <- z_or_re
    }
    idx <- idx + hmc_data$n_re_groups
  }

  # Overdispersion indices (skip for ratio computation)
  if (model_type == "negbin_negbin" || model_type == "negbin_gamma") {
    idx <- idx + 2
  } else if (model_type == "poisson_gamma") {
    idx <- idx + 1
  }

  # Spatial effects
  spatial_effect <- NULL
  if (spatial_info$type == "icar") {
    icar_param_ratio <- spatial_info$parameterization %||% "standard"
    idx <- idx + 1  # Skip log_tau
    if (icar_param_ratio == "collapsed") {
      # Collapsed: no phi in samples, would need phi_star from C++ (not available in spatial-only path)
      # For now, spatial_effect remains NULL — collapsed should use full path
    } else {
      phi_spatial <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]
      spatial_effect <- phi_spatial
    }
  } else if (spatial_info$type == "bym2") {
    bym2_param_ratio <- spatial_info$parameterization %||% "standard"
    # Riebler parameterization: log_sigma_total, logit_rho
    sigma_total <- exp(samples[, idx])
    idx <- idx + 1
    rho <- 1 / (1 + exp(-samples[, idx]))
    idx <- idx + 1
    sigma_s <- sigma_total * sqrt(rho)
    sigma_u <- sigma_total * sqrt(1 - rho)

    if (bym2_param_ratio == "collapsed") {
      # Collapsed: no phi/theta in samples — use full path
    } else {
      phi_scaled <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]
      idx <- idx + spatial_info$n_units
      theta <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]

      # Compute combined spatial effect: sigma_s * scale * phi + sigma_u * theta
      spatial_effect <- matrix(0, nrow = n_samples, ncol = spatial_info$n_units)
      for (s in seq_len(n_samples)) {
        spatial_effect[s, ] <- sigma_s[s] * phi_scaled[s, ] * spatial_info$bym2_scale +
          sigma_u[s] * theta[s, ]
      }
    }
  } else if (spatial_info$type == "gp") {
    gp_param_ratio <- spatial_info$parameterization %||% "centered"
    idx <- idx + 2  # Skip log_sigma2_gp, log_phi_gp
    if (gp_param_ratio != "collapsed") {
      spatial_effect <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]
      idx <- idx + spatial_info$n_units
    }
  } else if (spatial_info$type == "multiscale_gp") {
    msgp_approx <- spatial_info$msgp_approx %||% "nngp"
    if (msgp_approx == "hsgp") {
      # HSGP-MSGP: skip all params (handled in full ratio computation)
      hsgp_m <- spatial_info$hsgp_m %||% 6L
      m_total <- hsgp_m * hsgp_m
      idx <- idx + 2 + m_total + 2 + m_total  # 2 hyperparams + m^2 beta per scale
    } else {
      idx <- idx + 2  # Skip log_sigma2_local, log_phi_local
      gp_local_w <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]
      idx <- idx + spatial_info$n_units
      idx <- idx + 2  # Skip log_sigma2_regional, log_phi_regional
      gp_regional_w <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]
      idx <- idx + spatial_info$n_units
      spatial_effect <- gp_local_w + gp_regional_w
    }
  }

  # Compute ratios
  ratio_draws <- matrix(NA_real_, nrow = n_samples, ncol = N)

  for (s in seq_len(n_samples)) {
    eta_num <- as.numeric(hmc_data$X_num %*% beta_num[s, ])
    eta_denom <- as.numeric(hmc_data$X_denom %*% beta_denom[s, ])

    # Add RE
    if (!is.null(re)) {
      for (i in seq_len(N)) {
        g <- hmc_data$re_group[i]
        if (g > 0) {
          eta_num[i] <- eta_num[i] + re[s, g]
          eta_denom[i] <- eta_denom[i] + re[s, g]
        }
      }
    }

    # Add spatial (GP/MSGP use spatial_info$group for obs_to_loc mapping)
    if (!is.null(spatial_effect)) {
      for (i in seq_len(N)) {
        sp_g <- spatial_info$group[i]
        if (sp_g > 0) {
          eta_num[i] <- eta_num[i] + spatial_effect[s, sp_g]
          eta_denom[i] <- eta_denom[i] + spatial_effect[s, sp_g]
        }
      }
    }

    # Compute ratio
    if (model_type == "binomial" || model_type == "beta_binomial") {
      ratio_draws[s, ] <- 1 / (1 + exp(-eta_num))  # inv_logit(eta)
    } else {
      ratio_draws[s, ] <- exp(eta_num - eta_denom)  # exp(log(mu_num/mu_denom))
    }
  }

  colnames(ratio_draws) <- paste0("ratio[", seq_len(N), "]")
  ratio_draws
}


#' Prepare temporal structure for HMC
#' @keywords internal
prepare_temporal_for_hmc <- function(temporal, data, N) {
  if (is.null(temporal)) {
    return(list(
      type = "none",
      time_index = rep(0L, N),
      group_index = rep(0L, N),
      n_times = 0L,
      n_groups = 0L,
      n_temporal_params = 0L,
      precision_diag = numeric(0),
      precision_offdiag = numeric(0),
      precision_structure = list(cyclic = FALSE),
      shared = TRUE,
      # TVC fields (not used for regular temporal)
      n_tvc = 0L,
      structure = "none"
    ))
  }

  # Temporal has already been validated by ratiod()
  # Check if it's a TVC specification
  if (inherits(temporal, "ratiod_tvc")) {
    # TVC-specific fields
    return(list(
      type = "tvc",
      time_index = temporal$time_index,
      group_index = temporal$group_index,
      n_times = temporal$n_times,
      n_groups = temporal$n_groups,
      n_temporal_params = temporal$n_temporal_params,
      precision_structure = NULL,
      shared = temporal$shared,
      # TVC-specific
      n_tvc = temporal$n_tvc,
      structure = temporal$structure,
      tvc_indices = temporal$tvc_indices,
      tvc_names = temporal$tvc_names
    ))
  }

  # Check if it's a temporal GP specification
  if (inherits(temporal, "ratiod_temporal_gp")) {
    return(list(
      type = "gp",
      time_index = temporal$time_index,  # Maps obs to unique time (1-based)
      group_index = temporal$group_index,
      n_times = temporal$n_times,
      n_groups = temporal$n_groups,
      n_temporal_params = temporal$n_times * temporal$n_groups,  # One effect per unique time
      precision_structure = list(cyclic = FALSE),
      shared = temporal$shared,
      # GP-specific fields
      time_values = temporal$unique_time_values,  # Unique time values (length n_times)
      cov_type = temporal$cov,
      nu = temporal$nu %||% 1.5,
      period = temporal$period %||% 1.0,
      parameterization = temporal$parameterization %||% "noncentered",
      phi_prior_lower = temporal$phi_prior_lower %||% 0.01,
      phi_prior_upper = temporal$phi_prior_upper %||% 10.0,
      # TVC fields (not used)
      n_tvc = 0L,
      structure = "gp"
    ))
  }

  # Check if it's multiscale temporal
  if (inherits(temporal, "ratiod_temporal_multiscale")) {
    return(list(
      type = "multiscale",
      time_index = temporal$time_index,
      group_index = temporal$group_index,
      n_times = temporal$n_times,
      n_groups = temporal$n_groups,
      n_temporal_params = temporal$n_temporal_params,
      precision_structure = temporal$precision_structure,
      shared = temporal$shared,
      # Multiscale-specific fields needed for parameter initialization
      trend = temporal$trend,
      seasonal = temporal$seasonal,
      short_term = temporal$short_term,
      components = temporal$components,
      # TVC fields (not used)
      n_tvc = 0L,
      structure = "multiscale"
    ))
  }

  # Regular temporal (RW1, RW2, AR1, etc.)
  list(
    type = temporal$type,
    time_index = temporal$time_index,
    group_index = temporal$group_index,
    n_times = temporal$n_times,
    n_groups = temporal$n_groups,
    n_temporal_params = temporal$n_temporal_params,
    precision_structure = temporal$precision_structure,
    shared = temporal$shared,
    # TVC fields (not used for regular temporal)
    n_tvc = 0L,
    structure = temporal$type  # Use type as structure for consistency
  )
}


#' Prepare TVC (Temporally-Varying Coefficients) for HMC
#' @keywords internal
prepare_tvc_for_hmc <- function(temporal, data, N, X_num) {
  # Check if temporal is a TVC specification
  if (is.null(temporal) || !inherits(temporal, "ratiod_tvc")) {
    return(list(
      has_tvc = FALSE,
      n_tvc = 0L,
      n_times = 0L,
      n_groups = 1L,
      structure = "none",
      shared = TRUE,
      cyclic = FALSE,
      tvc_indices = integer(0),
      time_index = integer(0),
      group_index = integer(0),
      X_tvc = numeric(0)
    ))
  }

  # TVC has already been validated by ratiod()
  # Extract TVC-specific fields
  n_tvc <- temporal$n_tvc
  n_times <- temporal$n_times
  n_groups <- temporal$n_groups %||% 1L
  tvc_indices <- temporal$tvc_indices
  time_index <- temporal$time_index
  group_index <- temporal$group_index %||% rep(1L, N)

  # Extract the X_tvc design matrix subset (columns from X_num at tvc_indices)
  # Store as flat vector in row-major order for C++
  X_tvc <- as.numeric(t(X_num[, tvc_indices, drop = FALSE]))

  list(
    has_tvc = TRUE,
    n_tvc = as.integer(n_tvc),
    n_times = as.integer(n_times),
    n_groups = as.integer(n_groups),
    structure = temporal$structure %||% "rw1",
    shared = temporal$shared %||% TRUE,
    cyclic = temporal$cyclic %||% FALSE,
    tvc_indices = as.integer(tvc_indices),
    time_index = as.integer(time_index),
    group_index = as.integer(group_index),
    X_tvc = X_tvc
  )
}


#' Prepare SVC (Spatially-Varying Coefficients) for HMC
#' @keywords internal
prepare_svc_for_hmc <- function(svc, data, N, X_num) {
  # Check if spatial is a SVC specification
  if (is.null(svc) || !inherits(svc, "ratiod_svc")) {
    return(list(
      has_svc = FALSE,
      n_svc = 0L,
      n_obs = as.integer(N),
      nn = 0L,
      shared = TRUE,
      cov_type = "exponential",
      coords = numeric(0),
      svc_indices = integer(0),
      X_svc = numeric(0),
      nn_idx = integer(0),
      nn_dist = numeric(0),
      nn_order = integer(0),
      nn_order_inv = integer(0),
      sigma2_prior_scale = 1.0,
      phi_prior_lower = 0.3,
      phi_prior_upper = 10.0,
      tau_shape = 1.0,
      tau_rate = 0.01
    ))
  }

  # Validate SVC if not already done (coords_matrix will be NULL if not validated)
  if (is.null(svc$coords_matrix)) {
    svc <- validate_svc(svc, data, X_num)
  }

  # Extract SVC-specific fields
  n_svc <- svc$n_svc
  nn <- svc$nn
  svc_indices <- svc$svc_indices
  coords_matrix <- svc$coords_matrix  # N x 2 matrix
  neighbor_info <- svc$neighbor_info

  # Extract the X_svc design matrix subset (columns from X_num at svc_indices)
  # Store as flat vector in row-major order for C++
  X_svc <- as.numeric(t(X_num[, svc_indices, drop = FALSE]))

  # Flatten coords to row-major for C++ (x1, y1, x2, y2, ...)
  coords_flat <- as.numeric(t(coords_matrix))

  approx <- svc$approx %||% "nngp"

  if (approx == "hsgp") {
    list(
      has_svc = TRUE,
      svc_approx = "hsgp",
      n_svc = as.integer(n_svc),
      n_obs = as.integer(N),
      nn = 0L,
      shared = svc$shared %||% TRUE,
      cov_type = "exponential",
      coords = coords_flat,
      svc_indices = as.integer(svc_indices),
      X_svc = X_svc,
      nn_idx = integer(0),
      nn_dist = numeric(0),
      nn_order = integer(0),
      nn_order_inv = integer(0),
      sigma2_prior_scale = 1.0,
      phi_prior_lower = 0.3,
      phi_prior_upper = 30.0,
      tau_shape = 1.0,
      tau_rate = 0.01,
      hsgp_m = as.integer(svc$m %||% 6L),
      hsgp_c = svc$c_boundary %||% 1.5
    )
  } else {
    # Flatten neighbor indices and distances (N x nn matrices)
    nn_idx_flat <- as.integer(t(neighbor_info$nn_idx))
    nn_dist_flat <- as.numeric(t(neighbor_info$nn_dist))

    list(
      has_svc = TRUE,
      svc_approx = "nngp",
      n_svc = as.integer(n_svc),
      n_obs = as.integer(N),
      nn = as.integer(nn),
      shared = svc$shared %||% TRUE,
      cov_type = svc$cov %||% "exponential",
      coords = coords_flat,
      svc_indices = as.integer(svc_indices),
      X_svc = X_svc,
      nn_idx = nn_idx_flat,
      nn_dist = nn_dist_flat,
      nn_order = as.integer(neighbor_info$nn_order - 1L),
      nn_order_inv = as.integer(neighbor_info$nn_order_inv - 1L),
      sigma2_prior_scale = 1.0,
      phi_prior_lower = 0.3,
      phi_prior_upper = 30.0,
      tau_shape = 1.0,
      tau_rate = 0.01,
      hsgp_m = 0L,
      hsgp_c = 0.0
    )
  }
}


#' Initialize parameters for HMC with spatial and temporal
#' @keywords internal
initialize_hmc_params_spatial_temporal <- function(hmc_data, model_type,
                                                   spatial_info, temporal_info) {
  n_params <- hmc_data$p_num + hmc_data$p_denom

  # Random effects
  if (hmc_data$n_re_groups > 0) {
    n_params <- n_params + 1 + hmc_data$n_re_groups
  }

  # Overdispersion
  if (model_type == "negbin_negbin") {
    n_params <- n_params + 2
  } else if (model_type == "poisson_gamma") {
    n_params <- n_params + 1
  } else if (model_type == "negbin_gamma") {
    n_params <- n_params + 2
  }

  # Spatial
  if (spatial_info$type == "icar") {
    icar_param_init <- spatial_info$parameterization %||% "standard"
    if (icar_param_init == "collapsed") {
      n_params <- n_params + 1  # log_tau only
    } else {
      n_params <- n_params + 1 + spatial_info$n_units  # log_tau + phi
    }
  } else if (spatial_info$type == "bym2") {
    bym2_param_init <- spatial_info$parameterization %||% "standard"
    if (bym2_param_init == "collapsed") {
      n_params <- n_params + 2  # log_sigma_total + logit_rho only
    } else {
      # log_sigma_total + logit_rho + phi_scaled + theta
      n_params <- n_params + 2 + 2 * spatial_info$n_units
    }
  }

  # Temporal
  if (temporal_info$type != "none") {
    # log_tau_temporal + temporal effects
    n_params <- n_params + 1 + temporal_info$n_temporal_params
    # AR1 also has rho parameter
    if (temporal_info$type == "ar1") {
      n_params <- n_params + 1  # logit_rho_ar1
    }
  }

  rep(0.0, n_params)
}


#' Check if HMC backend can be used
#' @keywords internal
can_use_hmc_backend <- function(family) {
  # HMC works for all families

  TRUE
}


#' Get maximum number of threads available
#' @keywords internal
get_max_threads <- function() {
  cpp_get_max_threads()
}


#' Prepare zero-inflation structure for HMC
#' @keywords internal
prepare_zi_for_hmc <- function(zi, data, N) {
  if (is.null(zi)) {
    return(list(
      type = "none",
      X_zi = matrix(0, nrow = N, ncol = 1),
      p_zi = 1L,
      coef_names = NULL,
      X_oi = NULL,
      p_oi = 0L,
      coef_names_oi = NULL
    ))
  }

  # ZI is a ratiod_zi object with formula and type
  zi_type <- zi$type  # "zi_poisson", "zi_negbin", "hurdle_poisson", "hurdle_negbin"

  # Build design matrix from ZI formula
  if (is.null(zi$formula) || zi$formula == ~ 1) {
    # Intercept-only ZI model
    X_zi <- matrix(1, nrow = N, ncol = 1)
    colnames(X_zi) <- "(Intercept)_zi"
  } else {
    # Build design matrix from formula
    X_zi <- model.matrix(zi$formula, data = data)
  }

  list(
    type = zi_type,
    X_zi = X_zi,
    p_zi = ncol(X_zi),
    coef_names = colnames(X_zi),
    # OI info (not used for standard ZI, but needed for consistent interface)
    X_oi = NULL,
    p_oi = 0L,
    coef_names_oi = NULL
  )
}


#' Initialize parameters for HMC with full feature support
#' @keywords internal
initialize_hmc_params_full <- function(hmc_data, model_type, spatial_info,
                                        temporal_info, zi_info, latent_info = NULL,
                                        svc_info = NULL,
                                        spatiotemporal_info = NULL) {
  # Start building parameter vector with data-driven initialization
  q_init <- numeric(0)
  idx <- 1

  # Fixed effects: initialize intercept based on data for better starting point
  # For log-link families (gamma, lognormal, poisson, negbin), set intercept to log(mean(y))
  # This prevents huge gradients from starting at mu=1 when data has mean >> 1

  # Numerator fixed effects
  p_num <- hmc_data$p_num
  if (p_num > 0) {
    beta_num_init <- rep(0.0, p_num)
    # Data-driven intercept for continuous families (log link)
    if (model_type %in% c("gamma_gamma", "lognormal", "poisson_gamma", "negbin_negbin", "negbin_gamma")) {
      if (!is.null(hmc_data$y_num_cont) && any(hmc_data$y_num_cont > 0)) {
        # Use y_num_cont for continuous families
        y_mean <- mean(hmc_data$y_num_cont[hmc_data$y_num_cont > 0])
        if (is.finite(y_mean) && y_mean > 0) {
          beta_num_init[1] <- log(y_mean)
        }
      } else if (!is.null(hmc_data$y_num) && any(hmc_data$y_num > 0)) {
        # Use y_num for count families
        y_mean <- mean(hmc_data$y_num[hmc_data$y_num > 0])
        if (is.finite(y_mean) && y_mean > 0) {
          beta_num_init[1] <- log(y_mean)
        }
      }
    } else if (model_type %in% c("binomial", "beta_binomial")) {
      # Binomial/beta-binomial: logit link, initialize to logit(p_hat)
      if (!is.null(hmc_data$y_num) && !is.null(hmc_data$y_denom) &&
          any(hmc_data$y_denom > 0)) {
        # p_hat = sum(successes) / sum(trials)
        total_successes <- sum(hmc_data$y_num)
        total_trials <- sum(hmc_data$y_denom)
        if (total_trials > 0) {
          p_hat <- total_successes / total_trials
          # Bound away from 0 and 1 to avoid infinite logit
          p_hat <- max(0.01, min(0.99, p_hat))
          beta_num_init[1] <- qlogis(p_hat)  # logit(p_hat)
        }
      }
    }
    q_init <- c(q_init, beta_num_init)
  }

  # Denominator fixed effects
  p_denom <- hmc_data$p_denom
  if (p_denom > 0) {
    beta_denom_init <- rep(0.0, p_denom)
    if (model_type %in% c("gamma_gamma", "lognormal", "poisson_gamma", "negbin_negbin", "negbin_gamma")) {
      if (!is.null(hmc_data$y_denom_cont) && any(hmc_data$y_denom_cont > 0)) {
        y_mean <- mean(hmc_data$y_denom_cont[hmc_data$y_denom_cont > 0])
        if (is.finite(y_mean) && y_mean > 0) {
          beta_denom_init[1] <- log(y_mean)
        }
      } else if (!is.null(hmc_data$y_denom) && any(hmc_data$y_denom > 0)) {
        y_mean <- mean(hmc_data$y_denom[hmc_data$y_denom > 0])
        if (is.finite(y_mean) && y_mean > 0) {
          beta_denom_init[1] <- log(y_mean)
        }
      }
    }
    q_init <- c(q_init, beta_denom_init)
  }

  # Random effects (supports multi-term RE with slopes and correlations)
  n_re_terms <- hmc_data$n_re_terms %||% 0L
  has_slopes <- hmc_data$has_slopes %||% FALSE
  has_correlated_slopes <- hmc_data$has_correlated_slopes %||% FALSE

  if (n_re_terms > 0) {
    if (has_slopes) {
      q_init <- c(q_init, rep(0.0, hmc_data$total_sigma_params + hmc_data$total_re_params))
      if (has_correlated_slopes) {
        q_init <- c(q_init, rep(0.0, hmc_data$total_chol_params))
      }
    } else if (n_re_terms > 1) {
      q_init <- c(q_init, rep(0.0, n_re_terms + hmc_data$total_re_groups))
    } else if (hmc_data$n_re_groups > 0) {
      q_init <- c(q_init, rep(0.0, 1 + hmc_data$n_re_groups))
    }
  } else if (hmc_data$n_re_groups > 0) {
    # Single-term legacy path (used by GP interface where n_re_terms = 0)
    q_init <- c(q_init, rep(0.0, 1 + hmc_data$n_re_groups))
  }

  # Overdispersion / shape / sigma parameters
  # Initialize log(phi) = log(2) ~ 0.69 for moderate overdispersion/shape
  if (model_type == "negbin_negbin") {
    q_init <- c(q_init, log(2), log(2))  # log_phi_num, log_phi_denom
  } else if (model_type == "poisson_gamma") {
    q_init <- c(q_init, log(2))  # log_shape
  } else if (model_type == "negbin_gamma") {
    q_init <- c(q_init, log(2), log(2))  # log_phi_num (NB), log_phi_denom (Gamma shape)
  } else if (model_type == "gamma_gamma") {
    q_init <- c(q_init, log(2), log(2))  # log_shape_num, log_shape_denom
  } else if (model_type == "lognormal") {
    q_init <- c(q_init, log(0.5), log(0.5))  # log_sigma_num, log_sigma_denom
  } else if (model_type == "beta_binomial") {
    q_init <- c(q_init, log(10))  # log_phi (precision)
  }

  # Spatial
  if (spatial_info$type == "icar") {
    icar_param <- spatial_info$parameterization %||% "standard"
    if (icar_param == "collapsed") {
      q_init <- c(q_init, rep(0.0, 1))  # log_tau only
    } else {
      q_init <- c(q_init, rep(0.0, 1 + spatial_info$n_units))
    }
  } else if (spatial_info$type == "bym2") {
    bym2_param <- spatial_info$parameterization %||% "standard"
    if (bym2_param == "collapsed") {
      q_init <- c(q_init, rep(0.0, 2))  # log_sigma_total, logit_rho only
    } else {
      q_init <- c(q_init, rep(0.0, 2 + 2 * spatial_info$n_units))
    }
  } else if (spatial_info$type == "gp") {
    # Layout: log_sigma2, log_phi, w[n_units] (or just log_sigma2, log_phi for collapsed)
    gp_param <- spatial_info$parameterization %||% "centered"
    if (gp_param == "collapsed") {
      # Collapsed: only hyperparams, GP effects marginalized
      q_init <- c(q_init, rep(0.0, 2))
    } else {
      q_init <- c(q_init, rep(0.0, 2 + spatial_info$n_units))
    }
  } else if (spatial_info$type == "multiscale_gp") {
    # Initialize phi/lengthscale to geometric mean of range bounds (on log scale)
    range_local_lower <- spatial_info$range_local_lower %||% 0.01
    range_local_upper <- spatial_info$range_local_upper %||% 1.0
    range_regional_lower <- spatial_info$range_regional_lower %||% 1.0
    range_regional_upper <- spatial_info$range_regional_upper %||% 10.0
    log_phi_local_init <- log(sqrt(range_local_lower * range_local_upper))
    log_phi_regional_init <- log(sqrt(range_regional_lower * range_regional_upper))

    msgp_approx <- spatial_info$msgp_approx %||% "nngp"
    if (msgp_approx == "hsgp") {
      # HSGP-MSGP: m^2 basis coefficients per scale
      hsgp_m <- spatial_info$hsgp_m %||% 6L
      n_per_scale <- hsgp_m * hsgp_m
    } else {
      # NNGP-MSGP: N spatial effects per scale
      n_per_scale <- spatial_info$n_units
    }

    # Layout: log_sigma2_local, log_ls_local, beta_local[n], log_sigma2_regional, log_ls_regional, beta_regional[n]
    q_init <- c(q_init,
                0.0, log_phi_local_init, rep(0.0, n_per_scale),
                0.0, log_phi_regional_init, rep(0.0, n_per_scale))
  } else if (spatial_info$type == "hsgp") {
    hsgp_m <- spatial_info$hsgp_m %||% 8L
    q_init <- c(q_init, rep(0.0, 2 + hsgp_m * hsgp_m))
  }

  # Temporal (regular temporal effects, not TVC or multiscale)
  # Note: multiscale temporal is handled by the GP sampler path which has its own param layout
  if (temporal_info$type != "none" && temporal_info$type != "tvc" &&
      temporal_info$type != "multiscale") {
    if (temporal_info$type == "gp") {
      # Temporal GP: log_sigma2 + log_phi + N effects
      n_temporal_params <- 2 + temporal_info$n_temporal_params
    } else {
      # RW1/RW2/AR1: log_tau + effects (+ rho for AR1)
      n_temporal_params <- 1 + temporal_info$n_temporal_params
      if (temporal_info$type == "ar1") {
        n_temporal_params <- n_temporal_params + 1
      }
    }
    q_init <- c(q_init, rep(0.0, n_temporal_params))
  }

  # Multiscale temporal: handled by GP sampler path
  # Layout: [log_sigma2_trend, trend[n_times], log_sigma2_seasonal, seasonal[period],
  #          log_sigma2_short, (logit_rho if AR1), short[n_times]]
  if (temporal_info$type == "multiscale") {
    n_times <- temporal_info$n_times %||% 0L
    trend_type <- temporal_info$trend %||% "none"
    seasonal_period <- temporal_info$seasonal %||% 0L
    short_type <- temporal_info$short_term %||% "none"

    n_ms_params <- 0L
    # Trend component
    if (trend_type != "none") {
      n_ms_params <- n_ms_params + 1 + n_times  # log_sigma2_trend + effects
    }
    # Seasonal component
    if (!is.null(seasonal_period) && seasonal_period >= 2) {
      n_ms_params <- n_ms_params + 1 + seasonal_period  # log_sigma2_seasonal + effects
    }
    # Short-term component
    if (short_type != "none") {
      n_ms_params <- n_ms_params + 1 + n_times  # log_sigma2_short + effects
      if (short_type == "ar1") {
        n_ms_params <- n_ms_params + 1  # logit_rho
      }
    }
    q_init <- c(q_init, rep(0.0, n_ms_params))
  }

  # TVC (Temporally-Varying Coefficients)
  if (!is.null(temporal_info$type) && temporal_info$type == "tvc") {
    n_tvc <- temporal_info$n_tvc %||% 0L
    n_times <- temporal_info$n_times %||% 0L
    n_groups <- temporal_info$n_groups %||% 1L
    structure <- temporal_info$structure %||% "rw1"

    if (n_tvc > 0) {
      n_tvc_params <- n_tvc
      if (structure == "ar1") {
        n_tvc_params <- n_tvc_params + 1
      }
      n_tvc_params <- n_tvc_params + n_times * n_tvc * n_groups
      q_init <- c(q_init, rep(0.0, n_tvc_params))
    }
  }

  # Zero-inflation
  if (zi_info$type != "none") {
    q_init <- c(q_init, rep(0.0, zi_info$p_zi))
    if (!is.null(zi_info$p_oi) && zi_info$p_oi > 0) {
      q_init <- c(q_init, rep(0.0, zi_info$p_oi))
    }
  }

  # Latent factors
  if (!is.null(latent_info) && latent_info$type != "none") {
    K <- latent_info$n_factors
    N <- latent_info$n_obs
    q_init <- c(q_init, rep(0.0, K + N * K))
  }

  # Spatiotemporal interaction (Knorr-Held Type I-IV)
  if (!is.null(spatiotemporal_info) &&
      isTRUE(spatiotemporal_info$has_spatiotemporal)) {
    st_type <- spatiotemporal_info$type %||% "none"
    if (st_type != "none") {
      n_st_spatial <- spatiotemporal_info$n_spatial %||% 0L
      n_st_times <- spatiotemporal_info$n_times %||% 0L
      n_st_delta <- spatiotemporal_info$n_params %||% (n_st_spatial * n_st_times)

      # log_tau_st (precision for interaction)
      q_init <- c(q_init, 0.0)

      # tau_st2 removed — single tau for all ST types

      # logit_rho_st for AR1 temporal in ST
      st_temporal_type <- spatiotemporal_info$temporal_type %||% "rw1"
      if (st_temporal_type == "ar1") {
        q_init <- c(q_init, 0.0)
      }

      # GP range parameters for separable/non-separable GP
      if (st_type %in% c("separable", "nonsep_gp")) {
        q_init <- c(q_init, 0.0, 0.0)  # log_phi_space, log_phi_time
      }

      # ST interaction effects delta[S*T]
      q_init <- c(q_init, rep(0.0, n_st_delta))
    }
  }

  # SVC (Spatially-Varying Coefficients)
  if (!is.null(svc_info) && isTRUE(svc_info$has_svc)) {
    n_svc <- svc_info$n_svc %||% 0L
    n_obs <- svc_info$n_obs %||% hmc_data$N
    if (n_svc > 0) {
      svc_approx <- svc_info$svc_approx %||% "nngp"
      if (svc_approx == "hsgp") {
        m_total <- as.integer(svc_info$hsgp_m)^2
        # log_sigma2_svc (n_svc) + log_lengthscale_svc (n_svc) + beta (n_svc * m^2)
        q_init <- c(q_init, rep(0.0, 2 * n_svc + n_svc * m_total))
      } else {
        # log_sigma2_svc (n_svc) + log_phi_svc (n_svc) + svc_w (n_svc * n_obs)
        q_init <- c(q_init, rep(0.0, 2 * n_svc + n_svc * n_obs))
      }
    }
  }

  q_init
}


#' Convert HMC output to ratiod_fit (full feature support)
#' @keywords internal
convert_hmc_to_ratiod_fit_full <- function(fit_raw, hmc_data, spatial_info,
                                           temporal_info,
                                           spatiotemporal_info = NULL,
                                           zi_info,
                                           latent_info = NULL,
                                           formula, data, family,
                                           model_type, iter, warmup, chains,
                                           re_param = "noncentered") {
  # Handle multi-chain case
  if (chains > 1) {
    # Combine chains
    all_samples <- do.call(rbind, fit_raw$samples)
    all_log_prob <- unlist(fit_raw$log_prob)
    all_accept <- unlist(fit_raw$accept_prob)
    all_n_leapfrog <- unlist(fit_raw$n_leapfrog)
    all_treedepth <- unlist(fit_raw$treedepth)
    all_divergent <- unlist(fit_raw$divergent)
    epsilon <- mean(fit_raw$epsilon)
  } else {
    all_samples <- fit_raw$samples
    all_log_prob <- fit_raw$log_prob
    all_accept <- fit_raw$accept_prob
    all_n_leapfrog <- fit_raw$n_leapfrog
    all_treedepth <- fit_raw$treedepth
    all_divergent <- fit_raw$divergent
    epsilon <- fit_raw$epsilon
  }

  n_samples <- nrow(all_samples)

  # Build parameter names and extract draws
  draws_list <- build_draws_list_full(
    all_samples, hmc_data, spatial_info, temporal_info, zi_info, model_type, latent_info,
    re_param = re_param
  )

  draws <- do.call(cbind, draws_list)
  colnames(draws) <- names(draws_list)

  # Compute ratios
  gp_w_star <- fit_raw$gp_w_star  # NULL if not collapsed GP
  icar_phi_star <- fit_raw$icar_phi_star  # NULL if not collapsed ICAR/BYM2
  bym2_theta_star <- fit_raw$bym2_theta_star  # NULL if not collapsed BYM2
  ratio_draws <- compute_ratio_draws_hmc_full(
    all_samples, hmc_data, spatial_info, temporal_info, zi_info, model_type,
    re_param = re_param, gp_w_star = gp_w_star,
    icar_phi_star = icar_phi_star, bym2_theta_star = bym2_theta_star
  )

  # Diagnostics
  n_divergent <- sum(all_divergent)
  avg_accept <- mean(all_accept)

  # Create samples list (one matrix per chain) for compatibility
  # Each matrix has named columns for parameters
  if (chains > 1) {
    samples_list <- lapply(seq_len(chains), function(c) {
      chain_samples <- fit_raw$samples[[c]]
      chain_draws <- build_draws_list_full(
        chain_samples, hmc_data, spatial_info, temporal_info, zi_info, model_type, latent_info,
        re_param = re_param
      )
      chain_mat <- do.call(cbind, chain_draws)
      colnames(chain_mat) <- names(chain_draws)
      chain_mat
    })
  } else {
    samples_list <- list(draws)
  }

  fit <- list(
    draws = draws,
    samples = samples_list,
    ratio_draws = ratio_draws,
    formula = formula,
    data = data,
    family = family,
    spatial = spatial_info,
    temporal = temporal_info,
    spatiotemporal = spatiotemporal_info,
    zi = zi_info,
    latent = latent_info,
    backend = "hmc",
    algorithm = fit_raw$sampler %||% "HMC",
    re_param = re_param,
    iter = iter,
    warmup = warmup,
    chains = chains,
    n_save = n_samples,
    epsilon = epsilon,
    diagnostics = list(
      algorithm = fit_raw$sampler %||% "HMC",
      n_divergent = n_divergent,
      avg_accept_prob = avg_accept,
      n_leapfrog = all_n_leapfrog,
      treedepth = all_treedepth,
      divergent = all_divergent,
      log_posterior = all_log_prob
    ),
    log_prob = all_log_prob,
    .internal = list(
      samples = all_samples,
      hmc_data = hmc_data,
      model_type = model_type,
      latent_info = latent_info
    )
  )

  class(fit) <- "ratiod_fit"
  return(fit)
}


#' Build draws list with full feature support
#' @keywords internal
build_draws_list_full <- function(samples, hmc_data, spatial_info, temporal_info,
                                   zi_info, model_type, latent_info = NULL,
                                   re_param = "noncentered") {
  draws_list <- list()
  idx <- 1

  # Fixed effects numerator
  for (j in seq_len(hmc_data$p_num)) {
    name <- paste0("beta_num[", j, "]")
    draws_list[[name]] <- samples[, idx]
    idx <- idx + 1
  }

  # Fixed effects denominator
  for (j in seq_len(hmc_data$p_denom)) {
    name <- paste0("beta_denom[", j, "]")
    draws_list[[name]] <- samples[, idx]
    idx <- idx + 1
  }

  # Random effects (supports multi-term RE with slopes)
  # When re_param == "noncentered", samples contain z values that need transformation to actual RE
  # When re_param == "centered", samples contain actual RE values
  n_re_terms <- hmc_data$n_re_terms %||% 0L
  has_slopes <- hmc_data$has_slopes %||% FALSE
  has_correlated_slopes <- hmc_data$has_correlated_slopes %||% FALSE

  if (n_re_terms > 0 && has_slopes) {
    # With random slopes: each term has n_coefs sigma parameters and n_groups * n_coefs RE params
    # Layout: [sigmas for all terms] [chol params for correlated terms] [RE for all terms]

    # Store sigma values for later transformation (if non-centered)
    sigma_values <- list()
    chol_values <- list()

    # First extract all sigma parameters
    for (t in seq_len(n_re_terms)) {
      term <- hmc_data$re_terms[[t]]
      n_coefs <- term$n_coefs
      # Use cleaned slope names for output if available
      slope_names <- term$slope_vars_clean %||% term$slope_vars

      if (n_coefs == 1) {
        sigma_t <- exp(samples[, idx])
        draws_list[[paste0("sigma_re[", t, "]")]] <- sigma_t
        sigma_values[[t]] <- list(sigma_t)
        idx <- idx + 1
      } else {
        # Multiple sigmas: intercept + slopes
        sigma_values[[t]] <- list()
        coef_idx <- 1
        if (isTRUE(term$has_intercept)) {
          sigma_t <- exp(samples[, idx])
          draws_list[[paste0("sigma_re[", t, ",intercept]")]] <- sigma_t
          sigma_values[[t]][[coef_idx]] <- sigma_t
          idx <- idx + 1
          coef_idx <- coef_idx + 1
        }
        for (s in seq_along(slope_names)) {
          sigma_t <- exp(samples[, idx])
          draws_list[[paste0("sigma_re[", t, ",", slope_names[s], "]")]] <- sigma_t
          sigma_values[[t]][[coef_idx]] <- sigma_t
          idx <- idx + 1
          coef_idx <- coef_idx + 1
        }
      }
    }

    # Extract Cholesky parameters for correlated terms
    for (t in seq_len(n_re_terms)) {
      term <- hmc_data$re_terms[[t]]
      n_chol <- term$n_chol %||% 0L
      if (n_chol > 0 && isTRUE(term$correlated)) {
        chol_values[[t]] <- list()
        # Store raw Cholesky parameters (off-diagonal elements)
        for (c in seq_len(n_chol)) {
          chol_val <- samples[, idx]
          draws_list[[paste0("L_chol[", t, ",", c, "]")]] <- chol_val
          chol_values[[t]][[c]] <- chol_val
          idx <- idx + 1
        }
      }
    }

    # Then extract all RE effects, transforming z to re if non-centered
    for (t in seq_len(n_re_terms)) {
      term <- hmc_data$re_terms[[t]]
      n_groups_t <- term$n_groups
      n_coefs <- term$n_coefs
      # Use cleaned slope names for output if available
      slope_names <- term$slope_vars_clean %||% term$slope_vars

      for (g in seq_len(n_groups_t)) {
        if (n_coefs == 1) {
          z_or_re <- samples[, idx]
          if (re_param == "noncentered") {
            # Transform: re = sigma * z
            draws_list[[paste0("re[", t, ",", g, "]")]] <- sigma_values[[t]][[1]] * z_or_re
          } else {
            draws_list[[paste0("re[", t, ",", g, "]")]] <- z_or_re
          }
          idx <- idx + 1
        } else {
          # Multiple coefficients per group - need matrix transformation for correlated
          # Extract all z values for this group first
          z_vals <- list()
          re_indices <- list()
          coef_idx <- 1
          if (isTRUE(term$has_intercept)) {
            z_vals[[coef_idx]] <- samples[, idx]
            re_indices[[coef_idx]] <- list(name = paste0("re[", t, ",", g, ",intercept]"), idx = idx)
            idx <- idx + 1
            coef_idx <- coef_idx + 1
          }
          for (s in seq_along(slope_names)) {
            z_vals[[coef_idx]] <- samples[, idx]
            re_indices[[coef_idx]] <- list(name = paste0("re[", t, ",", g, ",", slope_names[s], "]"), idx = idx)
            idx <- idx + 1
            coef_idx <- coef_idx + 1
          }

          if (re_param == "noncentered" && isTRUE(term$correlated) && n_coefs == 2) {
            # For 2x2 correlated case: re = diag(sigma) * L * z
            # L = [[1, 0], [L21, sqrt(1-L21^2)]]
            sigma1 <- sigma_values[[t]][[1]]
            sigma2 <- sigma_values[[t]][[2]]
            L21 <- tanh(chol_values[[t]][[1]])  # tanh parameterization: raw -> L
            L22 <- sqrt(1 - L21^2)
            z1 <- z_vals[[1]]
            z2 <- z_vals[[2]]
            # re1 = sigma1 * (1 * z1 + 0 * z2) = sigma1 * z1
            # re2 = sigma2 * (L21 * z1 + L22 * z2)
            draws_list[[re_indices[[1]]$name]] <- sigma1 * z1
            draws_list[[re_indices[[2]]$name]] <- sigma2 * (L21 * z1 + L22 * z2)
          } else if (re_param == "noncentered") {
            # Uncorrelated or >2 coefs: re_j = sigma_j * z_j
            for (j in seq_along(z_vals)) {
              draws_list[[re_indices[[j]]$name]] <- sigma_values[[t]][[j]] * z_vals[[j]]
            }
          } else {
            # Centered: store as-is
            for (j in seq_along(z_vals)) {
              draws_list[[re_indices[[j]]$name]] <- z_vals[[j]]
            }
          }
        }
      }
    }
  } else if (n_re_terms > 1) {
    # Multiple RE terms (intercept only): extract sigma_re for each term
    sigma_values <- list()
    for (t in seq_len(n_re_terms)) {
      sigma_t <- exp(samples[, idx])
      draws_list[[paste0("sigma_re[", t, "]")]] <- sigma_t
      sigma_values[[t]] <- sigma_t
      idx <- idx + 1
    }
    # Extract RE effects for each term
    for (t in seq_len(n_re_terms)) {
      n_groups_t <- hmc_data$re_terms[[t]]$n_groups
      for (g in seq_len(n_groups_t)) {
        z_or_re <- samples[, idx]
        if (re_param == "noncentered") {
          draws_list[[paste0("re[", t, ",", g, "]")]] <- sigma_values[[t]] * z_or_re
        } else {
          draws_list[[paste0("re[", t, ",", g, "]")]] <- z_or_re
        }
        idx <- idx + 1
      }
    }
  } else if (hmc_data$n_re_groups > 0) {
    # Single RE term (intercept only)
    sigma_re <- exp(samples[, idx])
    draws_list[["sigma_re"]] <- sigma_re
    idx <- idx + 1
    for (g in seq_len(hmc_data$n_re_groups)) {
      z_or_re <- samples[, idx]
      if (re_param == "noncentered") {
        draws_list[[paste0("re[", g, "]")]] <- sigma_re * z_or_re
      } else {
        draws_list[[paste0("re[", g, "]")]] <- z_or_re
      }
      idx <- idx + 1
    }
  }

  # Overdispersion
  if (model_type == "negbin_negbin" || model_type == "negbin_gamma") {
    draws_list[["phi_num"]] <- exp(samples[, idx])
    idx <- idx + 1
    draws_list[["phi_denom"]] <- exp(samples[, idx])
    idx <- idx + 1
  } else if (model_type == "poisson_gamma") {
    draws_list[["shape"]] <- exp(samples[, idx])
    idx <- idx + 1
  } else if (model_type == "gamma_gamma") {
    draws_list[["shape_num"]] <- exp(samples[, idx])
    idx <- idx + 1
    draws_list[["shape_denom"]] <- exp(samples[, idx])
    idx <- idx + 1
  } else if (model_type == "lognormal") {
    draws_list[["sigma_num"]] <- exp(samples[, idx])
    idx <- idx + 1
    draws_list[["sigma_denom"]] <- exp(samples[, idx])
    idx <- idx + 1
  } else if (model_type == "beta_binomial") {
    draws_list[["phi"]] <- exp(samples[, idx])  # precision parameter
    idx <- idx + 1
  }

  # SVC (Spatially Varying Coefficients) - must come before other spatial types
  # C++ order: fixed -> RE -> overdispersion -> SVC -> spatial -> temporal
  if (spatial_info$type == "svc") {
    n_svc <- spatial_info$n_svc %||% 0L
    N_obs <- spatial_info$n_units %||% 0L
    svc_names <- spatial_info$svc_names
    svc_approx <- spatial_info$svc_approx %||% "nngp"

    if (n_svc > 0 && N_obs > 0) {
      # Extract log_sigma2 for each SVC term (variance parameter)
      for (j in seq_len(n_svc)) {
        name <- if (!is.null(svc_names) && j <= length(svc_names)) {
          paste0("sigma2_svc[", svc_names[j], "]")
        } else {
          paste0("sigma2_svc[", j, "]")
        }
        draws_list[[name]] <- exp(samples[, idx])
        idx <- idx + 1
      }

      # Extract log_phi/log_lengthscale for each SVC term
      for (j in seq_len(n_svc)) {
        param_name <- if (svc_approx == "hsgp") "lengthscale_svc" else "phi_svc"
        name <- if (!is.null(svc_names) && j <= length(svc_names)) {
          paste0(param_name, "[", svc_names[j], "]")
        } else {
          paste0(param_name, "[", j, "]")
        }
        draws_list[[name]] <- exp(samples[, idx])
        idx <- idx + 1
      }

      if (svc_approx == "hsgp") {
        # HSGP-SVC: extract basis coefficients (m^2 per term)
        m_total <- as.integer(spatial_info$svc_hsgp_m)^2
        for (j in seq_len(n_svc)) {
          term_name <- if (!is.null(svc_names) && j <= length(svc_names)) {
            svc_names[j]
          } else {
            as.character(j)
          }
          for (k in seq_len(m_total)) {
            draws_list[[paste0("svc_beta[", term_name, ",", k, "]")]] <- samples[, idx]
            idx <- idx + 1
          }
        }
      } else {
        # NNGP-SVC: extract spatial coefficients (N per term)
        for (j in seq_len(n_svc)) {
          term_name <- if (!is.null(svc_names) && j <= length(svc_names)) {
            svc_names[j]
          } else {
            as.character(j)
          }
          for (i in seq_len(N_obs)) {
            draws_list[[paste0("svc[", term_name, ",", i, "]")]] <- samples[, idx]
            idx <- idx + 1
          }
        }
      }
    }
  }

  # Spatial (ICAR, BYM2, GP, multiscale GP, HSGP)
  sp_result <- extract_spatial_draws(samples, idx, spatial_info)
  draws_list <- c(draws_list, sp_result$draws)
  idx <- sp_result$idx

  # Temporal
  if (temporal_info$type != "none" && temporal_info$type != "tvc") {
    if (temporal_info$type == "gp") {
      # Temporal GP: sigma2 and phi instead of tau
      draws_list[["sigma2_temporal_gp"]] <- exp(samples[, idx])
      idx <- idx + 1
      # Logit-bounded phi: phi = lower + range * sigmoid(logit_phi)
      phi_lower <- temporal_info$phi_prior_lower %||% 0.01
      phi_upper <- temporal_info$phi_prior_upper %||% 10.0
      phi_range <- phi_upper - phi_lower
      draws_list[["phi_temporal_gp"]] <- phi_lower + phi_range / (1 + exp(-samples[, idx]))
      idx <- idx + 1

      # Temporal GP effects — extract z params, transform to f if NC
      z_start_idx <- idx
      z_samples <- matrix(NA_real_, nrow = nrow(samples),
                          ncol = temporal_info$n_temporal_params)
      for (i in seq_len(temporal_info$n_temporal_params)) {
        z_samples[, i] <- samples[, idx]
        idx <- idx + 1
      }

      gp_param <- temporal_info$parameterization %||% "noncentered"
      if (gp_param == "noncentered") {
        # Transform z -> f using state-space forward pass (vectorized over MCMC samples)
        sigma2_draws <- draws_list[["sigma2_temporal_gp"]]
        phi_draws <- draws_list[["phi_temporal_gp"]]
        T_times <- temporal_info$n_times
        n_groups <- temporal_info$n_groups
        time_vals <- temporal_info$time_values

        f_samples <- z_samples  # Same dimensions, will overwrite
        for (s in seq_len(nrow(samples))) {
          sigma_s <- sqrt(sigma2_draws[s])
          phi_s <- phi_draws[s]

          for (g in seq_len(n_groups)) {
            off <- (g - 1L) * T_times
            # f[0] = sigma * z[0]
            f_samples[s, off + 1L] <- sigma_s * z_samples[s, off + 1L]
            for (tt in seq(2L, T_times)) {
              dt <- time_vals[tt] - time_vals[tt - 1L]
              rho_t <- exp(-dt / phi_s)
              one_minus_rho2 <- max(1.0 - rho_t^2, 1e-10)
              a_t <- sigma_s * sqrt(one_minus_rho2)
              f_samples[s, off + tt] <- rho_t * f_samples[s, off + tt - 1L] +
                                        a_t * z_samples[s, off + tt]
            }
          }
        }
        # Store f (actual temporal effects) in draws
        for (i in seq_len(temporal_info$n_temporal_params)) {
          draws_list[[paste0("temporal_gp[", i, "]")]] <- f_samples[, i]
        }
      } else {
        # Centered: z == f, store directly
        for (i in seq_len(temporal_info$n_temporal_params)) {
          draws_list[[paste0("temporal_gp[", i, "]")]] <- z_samples[, i]
        }
      }
    } else {
      # Regular temporal (RW1, RW2, AR1)
      draws_list[["tau_temporal"]] <- exp(samples[, idx])
      idx <- idx + 1

      # AR1 rho parameter
      if (temporal_info$type == "ar1") {
        logit_rho_ar1 <- samples[, idx]
        draws_list[["rho_ar1"]] <- 1 / (1 + exp(-logit_rho_ar1))
        idx <- idx + 1
      }

      # Temporal effects (all types: extract directly)
      for (t in seq_len(temporal_info$n_temporal_params)) {
        draws_list[[paste0("temporal[", t, "]")]] <- samples[, idx]
        idx <- idx + 1
      }
    }
  }

  # TVC (Temporally-Varying Coefficients)
  if (temporal_info$type == "tvc") {
    n_tvc <- temporal_info$n_tvc %||% 0L
    n_times <- temporal_info$n_times %||% 0L
    n_groups <- temporal_info$n_groups %||% 1L
    structure <- temporal_info$structure %||% "rw1"
    tvc_names <- temporal_info$tvc_names

    # TVC precision parameters (one per TVC term)
    for (j in seq_len(n_tvc)) {
      name <- if (!is.null(tvc_names) && j <= length(tvc_names)) {
        paste0("tau_tvc[", tvc_names[j], "]")
      } else {
        paste0("tau_tvc[", j, "]")
      }
      draws_list[[name]] <- exp(samples[, idx])
      idx <- idx + 1
    }

    # AR1 rho for TVC
    if (structure == "ar1") {
      logit_rho <- samples[, idx]
      draws_list[["rho_tvc"]] <- 1 / (1 + exp(-logit_rho))
      idx <- idx + 1
    }

    # TVC values: w[g, j, t]
    for (g in seq_len(n_groups)) {
      for (j in seq_len(n_tvc)) {
        term_name <- if (!is.null(tvc_names) && j <= length(tvc_names)) {
          tvc_names[j]
        } else {
          as.character(j)
        }
        for (t in seq_len(n_times)) {
          if (n_groups > 1) {
            draws_list[[paste0("tvc[", term_name, ",g", g, ",t", t, "]")]] <- samples[, idx]
          } else {
            draws_list[[paste0("tvc[", term_name, ",t", t, "]")]] <- samples[, idx]
          }
          idx <- idx + 1
        }
      }
    }
  }

  # Zero-inflation coefficients
  if (zi_info$type != "none") {
    for (j in seq_len(zi_info$p_zi)) {
      coef_name <- if (!is.null(zi_info$coef_names)) {
        zi_info$coef_names[j]
      } else {
        paste0("beta_zi[", j, "]")
      }
      draws_list[[paste0("beta_zi[", j, "]")]] <- samples[, idx]
      idx <- idx + 1
    }
  }

  # One-inflation coefficients (for OI-binomial and ZOIB models)
  if (!is.null(zi_info$p_oi) && zi_info$p_oi > 0) {
    for (j in seq_len(zi_info$p_oi)) {
      coef_name <- if (!is.null(zi_info$coef_names_oi)) {
        zi_info$coef_names_oi[j]
      } else {
        paste0("beta_oi[", j, "]")
      }
      draws_list[[paste0("beta_oi[", j, "]")]] <- samples[, idx]
      idx <- idx + 1
    }
  }

  # Latent factors
  if (!is.null(latent_info) && latent_info$type != "none" && latent_info$n_factors > 0) {
    n_factors <- latent_info$n_factors
    n_obs <- latent_info$n_obs

    # Extract log_sigma for each factor (transform to sigma)
    for (k in seq_len(n_factors)) {
      draws_list[[paste0("sigma_latent[", k, "]")]] <- exp(samples[, idx])
      idx <- idx + 1
    }

    # Extract factor scores (N x K)
    # Store only a summary (mean per factor) to avoid huge output
    # The raw factors are high-dimensional (N x K parameters)
    for (k in seq_len(n_factors)) {
      # Compute mean factor score across observations for each draw
      factor_cols <- idx:(idx + n_obs - 1)
      factor_mat <- samples[, factor_cols, drop = FALSE]
      draws_list[[paste0("latent_mean[", k, "]")]] <- rowMeans(factor_mat)
      idx <- idx + n_obs
    }
  }

  draws_list
}


#' Compute ratio draws from HMC samples (full feature support)
#' @keywords internal
compute_ratio_draws_hmc_full <- function(samples, hmc_data, spatial_info,
                                          temporal_info, zi_info, model_type,
                                          re_param = "noncentered",
                                          gp_w_star = NULL,
                                          icar_phi_star = NULL,
                                          bym2_theta_star = NULL) {
  n_samples <- nrow(samples)
  N <- hmc_data$N

  # Extract fixed effects
  beta_num <- samples[, seq_len(hmc_data$p_num), drop = FALSE]
  beta_denom <- samples[, hmc_data$p_num + seq_len(hmc_data$p_denom), drop = FALSE]

  idx <- hmc_data$p_num + hmc_data$p_denom + 1

  # Random effects (supports multi-term RE with slopes)
  # When re_param == "noncentered", samples contain z values that need transformation
  re <- NULL
  re_multi <- NULL
  re_slopes_multi <- NULL
  n_re_terms <- hmc_data$n_re_terms %||% 0L
  has_slopes <- hmc_data$has_slopes %||% FALSE
  has_correlated_slopes <- hmc_data$has_correlated_slopes %||% FALSE

  if (n_re_terms > 0 && has_slopes) {
    # With random slopes - extract sigma, cholesky, then RE and transform

    # Extract all sigma_re parameters (one per coefficient type per term)
    sigma_values <- list()
    for (t in seq_len(n_re_terms)) {
      term <- hmc_data$re_terms[[t]]
      n_coefs <- term$n_coefs
      sigma_values[[t]] <- list()
      for (c in seq_len(n_coefs)) {
        sigma_values[[t]][[c]] <- exp(samples[, idx])
        idx <- idx + 1
      }
    }

    # Extract Cholesky parameters for correlated terms
    chol_values <- list()
    if (has_correlated_slopes) {
      for (t in seq_len(n_re_terms)) {
        term <- hmc_data$re_terms[[t]]
        n_chol <- term$n_chol %||% 0L
        if (n_chol > 0 && isTRUE(term$correlated)) {
          chol_values[[t]] <- list()
          for (c in seq_len(n_chol)) {
            chol_values[[t]][[c]] <- samples[, idx]
            idx <- idx + 1
          }
        }
      }
    }

    # Extract RE for all terms (including slopes) and transform if noncentered
    re_multi <- list()
    re_slopes_multi <- list()

    for (t in seq_len(n_re_terms)) {
      term <- hmc_data$re_terms[[t]]
      n_groups_t <- term$n_groups
      n_coefs <- term$n_coefs

      # For each term: extract [intercept, slope1, slope2, ...] for each group
      # Total params = n_groups * n_coefs
      n_params_t <- n_groups_t * n_coefs
      re_params <- samples[, idx:(idx + n_params_t - 1), drop = FALSE]
      idx <- idx + n_params_t

      # Reshape to [samples, groups, coefs]
      # Store intercepts separately for backward compatibility
      re_multi[[t]] <- matrix(0, nrow = n_samples, ncol = n_groups_t)
      if (n_coefs > 1) {
        re_slopes_multi[[t]] <- array(0, dim = c(n_samples, n_groups_t, n_coefs - 1))
      }

      for (g in seq_len(n_groups_t)) {
        base_col <- (g - 1) * n_coefs + 1

        if (re_param == "noncentered") {
          # Transform z to actual RE
          if (n_coefs == 1) {
            # Simple case: re = sigma * z
            re_multi[[t]][, g] <- sigma_values[[t]][[1]] * re_params[, base_col]
          } else if (n_coefs == 2 && isTRUE(term$correlated) && !is.null(chol_values[[t]])) {
            # Correlated 2x2 case: re = diag(sigma) * L * z
            # L = [[1, 0], [L21, sqrt(1-L21^2)]]
            z1 <- re_params[, base_col]
            z2 <- re_params[, base_col + 1]
            sigma1 <- sigma_values[[t]][[1]]
            sigma2 <- sigma_values[[t]][[2]]
            L21 <- tanh(chol_values[[t]][[1]])  # tanh parameterization: raw -> L
            L22 <- sqrt(pmax(0, 1 - L21^2))  # pmax to avoid numerical issues
            re_multi[[t]][, g] <- sigma1 * z1  # Intercept
            re_slopes_multi[[t]][, g, 1] <- sigma2 * (L21 * z1 + L22 * z2)  # Slope
          } else {
            # Uncorrelated or >2 coefs: re_j = sigma_j * z_j
            re_multi[[t]][, g] <- sigma_values[[t]][[1]] * re_params[, base_col]
            if (n_coefs > 1) {
              for (s in seq_len(n_coefs - 1)) {
                re_slopes_multi[[t]][, g, s] <- sigma_values[[t]][[s + 1]] * re_params[, base_col + s]
              }
            }
          }
        } else {
          # Centered: use values as-is
          re_multi[[t]][, g] <- re_params[, base_col]  # Intercept
          if (n_coefs > 1) {
            for (s in seq_len(n_coefs - 1)) {
              re_slopes_multi[[t]][, g, s] <- re_params[, base_col + s]
            }
          }
        }
      }
    }
  } else if (n_re_terms > 1) {
    # Multiple RE terms (intercept only): extract sigma_re for each term
    sigma_values <- list()
    for (t in seq_len(n_re_terms)) {
      sigma_values[[t]] <- exp(samples[, idx])
      idx <- idx + 1
    }
    # Extract RE for all terms
    re_multi <- list()
    for (t in seq_len(n_re_terms)) {
      n_groups_t <- hmc_data$re_terms[[t]]$n_groups
      z_or_re <- samples[, idx:(idx + n_groups_t - 1), drop = FALSE]
      if (re_param == "noncentered") {
        # Transform: re = sigma * z
        re_multi[[t]] <- sweep(z_or_re, 1, sigma_values[[t]], "*")
      } else {
        re_multi[[t]] <- z_or_re
      }
      idx <- idx + n_groups_t
    }
  } else if (hmc_data$n_re_groups > 0) {
    # Single RE term (intercept only)
    sigma_re <- exp(samples[, idx])
    idx <- idx + 1
    z_or_re <- samples[, idx:(idx + hmc_data$n_re_groups - 1), drop = FALSE]
    if (re_param == "noncentered") {
      # Transform: re = sigma * z
      re <- sweep(z_or_re, 1, sigma_re, "*")
    } else {
      re <- z_or_re
    }
    idx <- idx + hmc_data$n_re_groups
  }

  # Overdispersion indices (skip for ratio computation)
  if (model_type == "negbin_negbin" || model_type == "negbin_gamma") {
    idx <- idx + 2
  } else if (model_type == "poisson_gamma") {
    idx <- idx + 1
  } else if (model_type == "gamma_gamma") {
    idx <- idx + 2
  } else if (model_type == "lognormal") {
    idx <- idx + 2
  } else if (model_type == "beta_binomial") {
    idx <- idx + 1
  }

  # SVC (Spatially Varying Coefficients) - must come before other spatial types
  # C++ order: fixed -> RE -> overdispersion -> SVC -> spatial -> temporal
  svc_effect <- NULL
  if (spatial_info$type == "svc") {
    n_svc <- spatial_info$n_svc %||% 0L
    N_obs <- spatial_info$n_units %||% 0L
    svc_approx <- spatial_info$svc_approx %||% "nngp"

    if (n_svc > 0 && N_obs > 0) {
      # Extract hyperparameters
      log_sigma2_svc <- samples[, idx:(idx + n_svc - 1), drop = FALSE]
      idx <- idx + n_svc  # log_sigma2
      log_ls_svc <- samples[, idx:(idx + n_svc - 1), drop = FALSE]
      idx <- idx + n_svc  # log_phi / log_lengthscale

      if (svc_approx == "hsgp") {
        # HSGP-SVC: extract basis coefficients
        m_total <- as.integer(spatial_info$svc_hsgp_m)^2
        n_svc_params <- n_svc * m_total
      } else {
        # NNGP-SVC: extract w values
        n_svc_params <- n_svc * N_obs
      }
      svc_w <- samples[, idx:(idx + n_svc_params - 1), drop = FALSE]
      idx <- idx + n_svc_params

      svc_effect <- list(
        w = svc_w,
        log_sigma2 = log_sigma2_svc,
        log_ls = log_ls_svc,
        n_svc = n_svc,
        N_obs = N_obs,
        X_svc = spatial_info$X_svc,
        approx = svc_approx,
        hsgp_m = spatial_info$svc_hsgp_m %||% 0L,
        coords = spatial_info$svc_coords
      )
    }
  }

  # Spatial effects (ICAR, BYM2)
  spatial_effect <- NULL
  if (spatial_info$type == "icar") {
    icar_param_ratio <- spatial_info$parameterization %||% "standard"
    idx <- idx + 1  # Skip log_tau
    if (icar_param_ratio == "collapsed") {
      # Use phi* from inner Laplace optimization (stored during sampling)
      if (!is.null(icar_phi_star)) {
        spatial_effect <- icar_phi_star
      }
    } else {
      phi_spatial <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]
      spatial_effect <- phi_spatial
      idx <- idx + spatial_info$n_units
    }
  } else if (spatial_info$type == "bym2") {
    bym2_param_ratio <- spatial_info$parameterization %||% "standard"
    # Riebler parameterization: log_sigma_total, logit_rho
    sigma_total <- exp(samples[, idx])
    idx <- idx + 1
    rho <- 1 / (1 + exp(-samples[, idx]))
    idx <- idx + 1
    sigma_s <- sigma_total * sqrt(rho)
    sigma_u <- sigma_total * sqrt(1 - rho)

    if (bym2_param_ratio == "collapsed") {
      # Use phi* and theta* from inner Laplace optimization
      if (!is.null(icar_phi_star) && !is.null(bym2_theta_star)) {
        spatial_effect <- matrix(0, nrow = n_samples, ncol = spatial_info$n_units)
        for (s in seq_len(n_samples)) {
          spatial_effect[s, ] <- sigma_s[s] * icar_phi_star[s, ] * spatial_info$bym2_scale +
            sigma_u[s] * bym2_theta_star[s, ]
        }
      }
    } else {
      phi_scaled <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]
      idx <- idx + spatial_info$n_units
      theta <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]
      idx <- idx + spatial_info$n_units

      # Compute combined spatial effect: sigma_s * scale * phi + sigma_u * theta
      spatial_effect <- matrix(0, nrow = n_samples, ncol = spatial_info$n_units)
      for (s in seq_len(n_samples)) {
        spatial_effect[s, ] <- sigma_s[s] * phi_scaled[s, ] * spatial_info$bym2_scale +
          sigma_u[s] * theta[s, ]
      }
    }
  } else if (spatial_info$type == "gp") {
    gp_param_ratio <- spatial_info$parameterization %||% "centered"
    idx <- idx + 2  # Skip log_sigma2_gp, log_phi_gp
    if (gp_param_ratio == "collapsed") {
      # Use w* from inner Laplace optimization (stored during sampling)
      if (!is.null(gp_w_star)) {
        spatial_effect <- gp_w_star
      }
    } else {
      spatial_effect <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]
      idx <- idx + spatial_info$n_units
    }
  } else if (spatial_info$type == "multiscale_gp") {
    msgp_approx <- spatial_info$msgp_approx %||% "nngp"
    if (msgp_approx == "hsgp") {
      # HSGP-MSGP: reconstruct spatial field from basis coefficients
      hsgp_m <- spatial_info$hsgp_m %||% 6L
      m_total <- hsgp_m * hsgp_m
      Phi <- spatial_info$msgp_hsgp_Phi
      eigenvalues <- spatial_info$msgp_hsgp_eigenvalues
      N <- nrow(Phi)

      sigma2_local <- exp(samples[, idx])
      idx <- idx + 1
      ls_local <- exp(samples[, idx])
      idx <- idx + 1
      beta_local <- samples[, idx:(idx + m_total - 1), drop = FALSE]
      idx <- idx + m_total

      sigma2_regional <- exp(samples[, idx])
      idx <- idx + 1
      ls_regional <- exp(samples[, idx])
      idx <- idx + 1
      beta_regional <- samples[, idx:(idx + m_total - 1), drop = FALSE]
      idx <- idx + m_total

      spatial_effect <- matrix(0, n_samples, N)
      for (s in seq_len(n_samples)) {
        S_local <- sigma2_local[s] * sqrt(2 * pi) * ls_local[s] *
          exp(-0.5 * ls_local[s]^2 * eigenvalues)
        S_regional <- sigma2_regional[s] * sqrt(2 * pi) * ls_regional[s] *
          exp(-0.5 * ls_regional[s]^2 * eigenvalues)
        spatial_effect[s, ] <- Phi %*% (sqrt(S_local) * beta_local[s, ]) +
          Phi %*% (sqrt(S_regional) * beta_regional[s, ])
      }
    } else {
      idx <- idx + 2  # Skip log_sigma2_local, log_phi_local
      gp_local_w <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]
      idx <- idx + spatial_info$n_units
      idx <- idx + 2  # Skip log_sigma2_regional, log_phi_regional
      gp_regional_w <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]
      idx <- idx + spatial_info$n_units
      spatial_effect <- gp_local_w + gp_regional_w
    }
  }

  # Temporal effects
  temporal_effect <- NULL
  if (temporal_info$type != "none" && temporal_info$type != "tvc") {
    if (temporal_info$type == "gp") {
      idx <- idx + 2  # Skip log_sigma2_temporal_gp, logit_phi_temporal_gp
    } else {
      idx <- idx + 1  # Skip log_tau_temporal
      if (temporal_info$type == "ar1") {
        idx <- idx + 1  # Skip logit_rho_ar1
      }
    }

    temporal_effect <- samples[, idx:(idx + temporal_info$n_temporal_params - 1), drop = FALSE]
    idx <- idx + temporal_info$n_temporal_params
  }

  # ZI coefficients (skip for ratio computation - they affect likelihood, not ratio)
  # ZI parameters are used for P(Y=0) vs P(Y>0), not the mean ratio

  # Precompute HSGP-SVC basis matrix (if needed)
  svc_hsgp_basis <- NULL
  if (!is.null(svc_effect) && (svc_effect$approx %||% "nngp") == "hsgp") {
    m_per_dim <- as.integer(svc_effect$hsgp_m)
    m_total <- m_per_dim^2
    coords <- svc_effect$coords
    c_val <- spatial_info$svc_hsgp_c %||% 1.5
    L1 <- c_val * (max(coords[, 1]) - min(coords[, 1])) / 2
    L2 <- c_val * (max(coords[, 2]) - min(coords[, 2])) / 2
    center1 <- (max(coords[, 1]) + min(coords[, 1])) / 2
    center2 <- (max(coords[, 2]) + min(coords[, 2])) / 2
    eigenvals <- numeric(m_total)
    phi_basis <- matrix(0, N, m_total)
    k <- 0
    for (m1 in seq_len(m_per_dim)) {
      for (m2 in seq_len(m_per_dim)) {
        k <- k + 1
        lam1 <- (pi * m1 / (2 * L1))^2
        lam2 <- (pi * m2 / (2 * L2))^2
        eigenvals[k] <- lam1 + lam2
        phi1 <- sin(pi * m1 * (coords[, 1] - center1 + L1) / (2 * L1)) / sqrt(L1)
        phi2 <- sin(pi * m2 * (coords[, 2] - center2 + L2) / (2 * L2)) / sqrt(L2)
        phi_basis[, k] <- phi1 * phi2
      }
    }
    svc_hsgp_basis <- list(phi = phi_basis, eigenvals = eigenvals, m_total = m_total)
  }

  # Compute ratios
  ratio_draws <- matrix(NA_real_, nrow = n_samples, ncol = N)

  for (s in seq_len(n_samples)) {
    eta_num <- as.numeric(hmc_data$X_num %*% beta_num[s, ])
    eta_denom <- as.numeric(hmc_data$X_denom %*% beta_denom[s, ])

    # Add RE (supports multi-term with slopes)
    if (!is.null(re_multi)) {
      # Multi-term RE (possibly with slopes)
      for (i in seq_len(N)) {
        for (t in seq_along(re_multi)) {
          g <- hmc_data$re_group_matrix[i, t]
          if (g > 0) {
            # Add intercept contribution
            re_contrib <- re_multi[[t]][s, g]

            # Add slope contributions if present
            if (!is.null(re_slopes_multi) && !is.null(re_slopes_multi[[t]])) {
              term <- hmc_data$re_terms[[t]]
              n_slopes <- length(term$slope_vars)
              if (n_slopes > 0 && !is.null(hmc_data$slope_matrices[[t]])) {
                for (slope_idx in seq_len(n_slopes)) {
                  x_slope <- hmc_data$slope_matrices[[t]][i, slope_idx]
                  re_slope <- re_slopes_multi[[t]][s, g, slope_idx]
                  re_contrib <- re_contrib + re_slope * x_slope
                }
              }
            }

            eta_num[i] <- eta_num[i] + re_contrib
            eta_denom[i] <- eta_denom[i] + re_contrib
          }
        }
      }
    } else if (!is.null(re)) {
      # Single-term RE (no slopes)
      for (i in seq_len(N)) {
        g <- hmc_data$re_group[i]
        if (g > 0) {
          eta_num[i] <- eta_num[i] + re[s, g]
          eta_denom[i] <- eta_denom[i] + re[s, g]
        }
      }
    }

    # Add spatial
    if (!is.null(spatial_effect)) {
      for (i in seq_len(N)) {
        sp_g <- spatial_info$group[i]
        if (sp_g > 0) {
          eta_num[i] <- eta_num[i] + spatial_effect[s, sp_g]
          eta_denom[i] <- eta_denom[i] + spatial_effect[s, sp_g]
        }
      }
    }

    # Add temporal
    if (!is.null(temporal_effect) && temporal_info$shared) {
      for (i in seq_len(N)) {
        t_idx <- temporal_info$time_index[i]
        g_idx <- temporal_info$group_index[i]
        if (t_idx > 0) {
          # Map to flat temporal parameter index
          param_idx <- if (temporal_info$n_groups > 1) {
            (g_idx - 1) * temporal_info$n_times + t_idx
          } else {
            t_idx
          }
          if (param_idx <= ncol(temporal_effect)) {
            eta_num[i] <- eta_num[i] + temporal_effect[s, param_idx]
            eta_denom[i] <- eta_denom[i] + temporal_effect[s, param_idx]
          }
        }
      }
    }

    # Add SVC effect: sum_j(w_j[i] * x_j[i])
    if (!is.null(svc_effect)) {
      n_svc <- svc_effect$n_svc
      N_obs <- svc_effect$N_obs
      X_svc <- svc_effect$X_svc
      svc_w <- svc_effect$w
      svc_approx_local <- svc_effect$approx %||% "nngp"

      if (svc_approx_local == "hsgp" && !is.null(svc_hsgp_basis)) {
        # HSGP-SVC: reconstruct spatial field from basis coefficients
        m_total <- svc_hsgp_basis$m_total
        for (j in seq_len(n_svc)) {
          sigma2_j <- exp(svc_effect$log_sigma2[s, j])
          ls_j <- exp(svc_effect$log_ls[s, j])
          beta_j <- svc_w[s, ((j - 1) * m_total + 1):(j * m_total)]
          # Spectral density: S(w) = sigma2 * sqrt(2*pi) * ls * exp(-0.5*ls^2*w)
          S_k <- sigma2_j * sqrt(2 * pi) * ls_j *
                 exp(-0.5 * ls_j^2 * svc_hsgp_basis$eigenvals)
          sqrt_S_k <- sqrt(pmax(S_k, 0))
          # f_j = Phi %*% (sqrt_S_k * beta_j)
          f_j <- svc_hsgp_basis$phi %*% (sqrt_S_k * beta_j)
          x_j <- if (!is.null(X_svc)) X_svc[, j] else rep(1.0, N)
          eta_num <- eta_num + as.numeric(f_j) * x_j
          eta_denom <- eta_denom + as.numeric(f_j) * x_j
        }
      } else {
        # NNGP-SVC: direct w values
        for (i in seq_len(N)) {
          svc_contrib <- 0
          for (j in seq_len(n_svc)) {
            w_ji <- svc_w[s, (j - 1) * N_obs + i]
            x_ji <- if (!is.null(X_svc)) X_svc[i, j] else 1.0
            svc_contrib <- svc_contrib + w_ji * x_ji
          }
          eta_num[i] <- eta_num[i] + svc_contrib
          eta_denom[i] <- eta_denom[i] + svc_contrib
        }
      }
    }

    # Compute ratio
    if (model_type == "binomial" || model_type == "beta_binomial") {
      ratio_draws[s, ] <- 1 / (1 + exp(-eta_num))  # inv_logit(eta)
    } else {
      ratio_draws[s, ] <- exp(eta_num - eta_denom)  # exp(log(mu_num/mu_denom))
    }
  }

  colnames(ratio_draws) <- paste0("ratio[", seq_len(N), "]")
  ratio_draws
}


# =============================================================================
# GP Spatial Support (v0.9.4+)
# =============================================================================

#' Check if spatial specification is GP-based
#' @keywords internal
is_gp_spatial <- function(spatial) {
  if (is.null(spatial)) return(FALSE)
  inherits(spatial, "ratiod_gp") ||
    inherits(spatial, "ratiod_multiscale") ||
    inherits(spatial, "ratiod_hsgp")
}

#' Check if temporal specification is multiscale
#' @keywords internal
is_multiscale_temporal <- function(temporal) {
  if (is.null(temporal)) return(FALSE)
  inherits(temporal, "ratiod_temporal_multiscale")
}

#' Prepare GP spatial structure for HMC
#'
#' Computes nearest neighbor structure for NNGP approximation.
#'
#' @param gp A ratiod_gp or ratiod_multiscale spatial specification
#' @param data Data frame with coordinate columns
#' @param N Number of observations
#' @return List of GP parameters for cpp_hmc_fit_gp
#' @keywords internal
prepare_gp_for_hmc <- function(gp, data, N) {
  if (is.null(gp)) {
    return(list(
      gp_type = "none",
      coords = numeric(0),
      nn_idx = integer(0),
      nn_dist = numeric(0),
      nn_order = integer(0),
      nn_order_inv = integer(0),
      nn_neighbor_dist = numeric(0),  # Phase 1.3
      nn = 0L,
      nn_idx_local = integer(0),
      nn_dist_local = numeric(0),
      nn_order_local = integer(0),
      nn_order_inv_local = integer(0),
      nn_local = 0L,
      nn_neighbor_dist_local = numeric(0),  # Phase 1.3
      nn_idx_regional = integer(0),
      nn_dist_regional = numeric(0),
      nn_order_regional = integer(0),
      nn_order_inv_regional = integer(0),
      nn_regional = 0L,
      nn_neighbor_dist_regional = numeric(0),  # Phase 1.3
      range_local_lower = 0,
      range_local_upper = 1,
      range_regional_lower = 1,
      range_regional_upper = 10,
      cov_type = "exponential",
      nu = 1.5,
      shared = TRUE,
      sampler = "noncentered",
      # Solver config (defaults)
      solver = "auto",
      cg_tol = 1e-6,
      cg_maxiter = 100L,
      gp_obs_to_loc = integer(0),
      n_unique = 0L
    ))
  }

  # Handle HSGP specially (no neighbor computation needed)
  if (inherits(gp, "ratiod_hsgp")) {
    validated <- validate_hsgp(gp, data)
    coords_mat <- validated$coords_matrix
    coords_flat <- as.vector(t(coords_mat))

    return(list(
      gp_type = "hsgp",
      coords = coords_flat,
      hsgp_m = as.integer(gp$m),
      hsgp_c = gp$c,
      shared = gp$shared,
      # Empty placeholders for standard GP fields
      nn_idx = integer(0),
      nn_dist = numeric(0),
      nn_order = integer(0),
      nn_order_inv = integer(0),
      nn_neighbor_dist = numeric(0),  # Phase 1.3
      nn = 0L,
      nn_idx_local = integer(0),
      nn_dist_local = numeric(0),
      nn_order_local = integer(0),
      nn_order_inv_local = integer(0),
      nn_local = 0L,
      nn_neighbor_dist_local = numeric(0),  # Phase 1.3
      nn_idx_regional = integer(0),
      nn_dist_regional = numeric(0),
      nn_order_regional = integer(0),
      nn_order_inv_regional = integer(0),
      nn_regional = 0L,
      nn_neighbor_dist_regional = numeric(0),  # Phase 1.3
      range_local_lower = 0,
      range_local_upper = 1,
      range_regional_lower = 1,
      range_regional_upper = 10,
      cov_type = "exponential",
      nu = 1.5,
      sampler = "noncentered",
      # Solver config (HSGP uses Cholesky internally, but keep consistent interface)
      solver = "cholesky",
      cg_tol = 1e-6,
      cg_maxiter = 100L,
      gp_obs_to_loc = integer(0),
      n_unique = 0L
    ))
  }

  # Validate GP specification (computes neighbor structure)
  validated <- validate_gp(gp, data)

  # Extract unique coordinates (NNGP computed on unique locations only)
  coords_mat <- validated$unique_coords
  coords_flat <- as.vector(t(coords_mat))  # Row-major flatten

  if (inherits(gp, "ratiod_multiscale") && isTRUE(gp$approx == "hsgp")) {
    # HSGP-MSGP: two HSGP evaluations with shared basis (no NNGP needed)
    validated_hsgp <- validate_hsgp_multiscale(gp, data)
    coords_flat_hsgp <- as.vector(t(validated_hsgp$coords_matrix))

    list(
      gp_type = "multiscale_gp",
      msgp_approx = "hsgp",
      hsgp_m = as.integer(gp$m),
      hsgp_c = gp$c_boundary,
      coords = coords_flat_hsgp,
      # Range constraints (used for lengthscale prior means)
      range_local_lower = gp$range_local[1],
      range_local_upper = gp$range_local[2],
      range_regional_lower = gp$range_regional[1],
      range_regional_upper = gp$range_regional[2],
      # Common params
      cov_type = gp$cov %||% "exponential",
      nu = gp$nu %||% 1.5,
      shared = gp$shared,
      sampler = gp$sampler %||% "noncentered",
      parameterization = "noncentered",  # HSGP always NC
      solver = "cholesky",
      cg_tol = 1e-6,
      cg_maxiter = 100L,
      # Empty NNGP placeholders
      nn_idx = integer(0), nn_dist = numeric(0),
      nn_order = integer(0), nn_order_inv = integer(0),
      nn_neighbor_dist = numeric(0), nn = 0L,
      nn_idx_local = integer(0), nn_dist_local = numeric(0),
      nn_order_local = integer(0), nn_order_inv_local = integer(0),
      nn_local = 0L, nn_neighbor_dist_local = numeric(0),
      nn_idx_regional = integer(0), nn_dist_regional = numeric(0),
      nn_order_regional = integer(0), nn_order_inv_regional = integer(0),
      nn_regional = 0L, nn_neighbor_dist_regional = numeric(0),
      gp_obs_to_loc = integer(0),
      n_unique = 0L
    )
  } else if (inherits(gp, "ratiod_multiscale")) {
    # NNGP-MSGP: standard neighbor-based multi-scale GP
    local_info <- validated$neighbor_info_local
    regional_info <- validated$neighbor_info_regional

    # Phase 1.3: Flatten nn_neighbor_dist arrays for C++ row-major access
    nn_neighbor_dist_local_flat <- as.vector(aperm(local_info$nn_neighbor_dist, c(3, 2, 1)))
    nn_neighbor_dist_regional_flat <- as.vector(aperm(regional_info$nn_neighbor_dist, c(3, 2, 1)))

    list(
      gp_type = "multiscale_gp",
      coords = coords_flat,
      # Single-scale params (not used for multiscale)
      nn_idx = integer(0),
      nn_dist = numeric(0),
      nn_order = integer(0),
      nn_order_inv = integer(0),
      nn_neighbor_dist = numeric(0),  # Phase 1.3
      nn = 0L,
      # Local scale
      nn_idx_local = as.integer(as.vector(t(local_info$nn_idx))),
      nn_dist_local = as.vector(t(local_info$nn_dist)),
      nn_order_local = as.integer(local_info$nn_order),
      nn_order_inv_local = as.integer(local_info$nn_order_inv),
      nn_local = as.integer(validated$nn_local),
      nn_neighbor_dist_local = nn_neighbor_dist_local_flat,  # Phase 1.3
      # Regional scale
      nn_idx_regional = as.integer(as.vector(t(regional_info$nn_idx))),
      nn_dist_regional = as.vector(t(regional_info$nn_dist)),
      nn_order_regional = as.integer(regional_info$nn_order),
      nn_order_inv_regional = as.integer(regional_info$nn_order_inv),
      nn_regional = as.integer(validated$nn_regional),
      nn_neighbor_dist_regional = nn_neighbor_dist_regional_flat,  # Phase 1.3
      # Range constraints
      range_local_lower = gp$range_local[1],
      range_local_upper = gp$range_local[2],
      range_regional_lower = gp$range_regional[1],
      range_regional_upper = gp$range_regional[2],
      # Common params
      cov_type = gp$cov,
      nu = gp$nu,
      shared = gp$shared,
      sampler = gp$sampler %||% "noncentered",
      parameterization = gp$parameterization %||% "centered",
      # Solver config (multiscale uses Cholesky by default)
      solver = "cholesky",
      cg_tol = 1e-6,
      cg_maxiter = 100L,
      # Observation-to-location mapping (1-based)
      gp_obs_to_loc = as.integer(validated$obs_to_loc),
      n_unique = validated$n_unique
    )
  } else {
    # Single-scale GP
    nn_info <- validated$neighbor_info

    # Phase 1.3: Flatten nn_neighbor_dist from N x nn x nn for C++ row-major access
    # C++ accesses as: i * nn * nn + j1 * nn + j2 (0-indexed, i slowest, j2 fastest)
    # R stores column-major: arr[i, j1, j2] at index (j2-1)*N*nn + (j1-1)*N + i (1-indexed)
    # Use C++ helper for fast flattening
    nn <- validated$nn
    N_gp <- nrow(nn_info$nn_idx)
    nn_neighbor_dist_flat <- cpp_flatten_3d_rowmajor(
      as.vector(nn_info$nn_neighbor_dist), N_gp, nn, nn
    )

    list(
      gp_type = "gp",
      coords = coords_flat,
      nn_idx = as.integer(as.vector(t(nn_info$nn_idx))),
      nn_dist = as.vector(t(nn_info$nn_dist)),
      nn_order = as.integer(nn_info$nn_order),
      nn_order_inv = as.integer(nn_info$nn_order_inv),
      nn_neighbor_dist = nn_neighbor_dist_flat,  # Phase 1.3: cached pairwise distances
      nn = as.integer(validated$nn),
      # Multi-scale params (not used for single-scale)
      nn_idx_local = integer(0),
      nn_dist_local = numeric(0),
      nn_order_local = integer(0),
      nn_order_inv_local = integer(0),
      nn_local = 0L,
      nn_neighbor_dist_local = numeric(0),  # Phase 1.3
      nn_idx_regional = integer(0),
      nn_dist_regional = numeric(0),
      nn_order_regional = integer(0),
      nn_order_inv_regional = integer(0),
      nn_regional = 0L,
      nn_neighbor_dist_regional = numeric(0),  # Phase 1.3
      range_local_lower = 0,
      range_local_upper = 1,
      range_regional_lower = 1,
      range_regional_upper = 10,
      # Common params
      cov_type = gp$cov,
      nu = gp$nu,
      shared = gp$shared,
      parameterization = gp$parameterization %||% "centered",
      # Solver config
      solver = gp$solver %||% "auto",
      cg_tol = gp$cg_tol %||% 1e-6,
      cg_maxiter = as.integer(gp$cg_maxiter %||% 100L),
      # Observation-to-location mapping (1-based)
      gp_obs_to_loc = as.integer(validated$obs_to_loc),
      n_unique = validated$n_unique
    )
  }
}

#' Prepare multiscale temporal structure for HMC
#' @keywords internal
prepare_multiscale_temporal_for_hmc <- function(temporal, data, N) {
  if (is.null(temporal) || !inherits(temporal, "ratiod_temporal_multiscale")) {
    return(list(
      ms_temporal_type = "none",
      ms_time_index = integer(N),
      ms_group_index = integer(N),
      ms_n_times = 0L,
      ms_n_groups = 1L,
      trend_type = "none",
      seasonal_period = 0L,
      short_term_type = "none",
      shared = TRUE
    ))
  }

  # Validate temporal specification
  validated <- validate_temporal_multiscale(temporal, data)

  list(
    ms_temporal_type = "multiscale",
    ms_time_index = as.integer(validated$time_index),
    ms_group_index = as.integer(validated$group_index),
    ms_n_times = as.integer(validated$n_times),
    ms_n_groups = as.integer(validated$n_groups),
    trend_type = temporal$trend,
    seasonal_period = as.integer(temporal$seasonal %||% 0L),
    short_term_type = temporal$short_term,
    shared = temporal$shared
  )
}

#' Prepare RSR structure for HMC
#' @keywords internal
prepare_rsr_for_hmc <- function(spatial, data) {
  if (is.null(spatial) || !isTRUE(spatial$rsr)) {
    return(list(
      has_rsr = FALSE,
      rsr_projection = numeric(0),
      rsr_n = 0L
    ))
  }

  # Validate RSR and compute projection
  validated <- validate_rsr(spatial, data)

  if (is.null(validated$rsr_projection)) {
    return(list(
      has_rsr = FALSE,
      rsr_projection = numeric(0),
      rsr_n = 0L
    ))
  }

  n <- nrow(validated$rsr_projection)
  list(
    has_rsr = TRUE,
    rsr_projection = as.vector(validated$rsr_projection),  # Row-major flatten
    rsr_n = as.integer(n)
  )
}
