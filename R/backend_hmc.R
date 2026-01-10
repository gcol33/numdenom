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
#'
#' @return A ratiod_fit object
#' @keywords internal
fit_hmc <- function(formula,
                    data,
                    family,
                    spatial = NULL,
                    temporal = NULL,
                    zi = NULL,
                    latent = NULL,
                    priors = NULL,
                    iter = 2000,
                    warmup = floor(iter / 2),
                    chains = 4,
                    cores = NULL,
                    L = 20,
                    seed = NULL,
                    verbose = TRUE) {

  # Set cores
  if (is.null(cores)) {
    cores <- min(chains, cpp_get_max_threads())
  }

  # Set seed
  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1)
  }

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
  spatial_info <- prepare_spatial_for_hmc(spatial, data, hmc_data$N)

  # Prepare temporal structure
  temporal_info <- prepare_temporal_for_hmc(temporal, data, hmc_data$N)

  # Prepare zero-inflation structure
  zi_info <- prepare_zi_for_hmc(zi, data, hmc_data$N)

  # Prepare latent factor structure
  latent_info <- prepare_latent_for_hmc(latent, hmc_data$N)

  # Get prior parameters
  priors <- priors %||% ratiod_priors()
  sigma_beta <- priors$sigma_beta %||% 10.0
  sigma_re_scale <- priors$sigma_re_scale %||% 2.5
  phi_shape <- priors$phi_shape %||% 2.0
  phi_rate <- priors$phi_rate %||% 0.1
  tau_spatial_shape <- priors$tau_spatial_shape %||% 1.0
  tau_spatial_rate <- priors$tau_spatial_rate %||% 0.01


  # Initialize parameters
  q_init <- initialize_hmc_params_full(
    hmc_data, model_type, spatial_info, temporal_info, zi_info, latent_info
  )

  if (verbose) {
    message("Running HMC sampler...")
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

  # Run sampler
  fit_raw <- cpp_hmc_fit(
    q_init = q_init,
    y_num = as.integer(hmc_data$y_num),
    y_denom = as.integer(hmc_data$y_denom),
    y_denom_cont = hmc_data$y_denom_cont,
    X_num = hmc_data$X_num,
    X_denom = hmc_data$X_denom,
    re_group = as.integer(hmc_data$re_group),
    n_re_groups = as.integer(hmc_data$n_re_groups),
    n_re_terms = as.integer(n_re_terms),
    re_group_matrix = as.matrix(re_group_matrix),
    re_n_groups_vec = as.integer(re_n_groups_vec),
    # Random slopes
    has_re_slopes = has_re_slopes,
    has_re_correlated_slopes = has_re_correlated_slopes,
    re_n_coefs_vec = as.integer(re_n_coefs_vec),
    re_correlated_vec = as.logical(re_correlated_vec),
    re_n_chol_vec = as.integer(re_n_chol_vec),
    slope_matrices_list = slope_matrices_list,
    model_type_str = model_type,
    # Spatial
    spatial_type_str = spatial_info$type,
    spatial_group = as.integer(spatial_info$group),
    n_spatial_units = spatial_info$n_units,
    adj_row_ptr = as.integer(spatial_info$adj_row_ptr),
    adj_col_idx = as.integer(spatial_info$adj_col_idx),
    n_neighbors = as.integer(spatial_info$n_neighbors),
    bym2_scale_factor = spatial_info$bym2_scale,
    # Temporal
    temporal_type_str = temporal_info$type,
    temporal_time_idx = as.integer(temporal_info$time_index),
    temporal_group_idx = as.integer(temporal_info$group_index),
    n_times = as.integer(temporal_info$n_times),
    n_temporal_groups = as.integer(temporal_info$n_groups),
    n_temporal_params = as.integer(temporal_info$n_temporal_params),
    temporal_cyclic = temporal_info$precision_structure$cyclic %||% FALSE,
    temporal_shared = temporal_info$shared %||% TRUE,
    tau_temporal_shape = tau_temporal_shape,
    tau_temporal_rate = tau_temporal_rate,
    # Priors
    sigma_beta = sigma_beta,
    sigma_re_scale = sigma_re_scale,
    phi_prior_shape = phi_shape,
    phi_prior_rate = phi_rate,
    tau_spatial_shape = tau_spatial_shape,
    tau_spatial_rate = tau_spatial_rate,
    # Zero-inflation
    zi_type_str = zi_info$type,
    X_zi = zi_info$X_zi,
    zi_prior_sd = priors$zi_prior_sd %||% 10.0,
    # Latent factors
    has_latent = latent_info$type != "none",
    latent_n_factors = as.integer(latent_info$n_factors),
    latent_shared = latent_info$shared,
    latent_scale = latent_info$scale %||% TRUE,
    latent_constraint = as.integer(ifelse(latent_info$constraint == "sum_to_zero", 0L, 1L)),
    latent_sigma_prior_rate = latent_info$sigma_prior_rate,
    # Sampler
    n_iter = as.integer(iter),
    n_warmup = as.integer(warmup),
    L = as.integer(L),
    n_chains = as.integer(chains),
    seed = as.integer(seed),
    n_threads = n_threads_within,
    verbose = verbose
  )

  # Convert to ratiod_fit
  fit <- convert_hmc_to_ratiod_fit_full(
    fit_raw = fit_raw,
    hmc_data = hmc_data,
    spatial_info = spatial_info,
    temporal_info = temporal_info,
    zi_info = zi_info,
    latent_info = latent_info,
    formula = formula,
    data = data,
    family = family,
    model_type = model_type,
    iter = iter,
    warmup = warmup,
    chains = chains
  )

  return(fit)
}


#' Get HMC model type string
#' @keywords internal
get_hmc_model_type <- function(family) {
  dist <- family$numerator$distribution
  if (dist == "binomial") return("binomial")
  if (dist %in% c("negbin", "negative_binomial", "neg_binomial_2")) return("negbin_negbin")
  if (dist == "poisson") return("poisson_gamma")
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
  if (is.null(X_denom) || ncol(X_denom) == 0) {
    X_denom <- matrix(1, nrow = nrow(data), ncol = 1)
    colnames(X_denom) <- "(Intercept)_denom"
  }

  # Random effects
  re_info <- extract_re_for_hmc(formula)

  # Prepare response based on model type
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
    y_denom <- y_denom_int
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

  # Extract spatial type
  spatial_type <- if (inherits(spatial, "ratiod_spatial_bym2")) {
    "bym2"
  } else if (inherits(spatial, "ratiod_spatial_car") ||
             inherits(spatial, "ratiod_spatial_icar")) {
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
    adj_col_idx <- c(adj_col_idx, neighbors)
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

  list(
    type = spatial_type,
    group = spatial_group,
    n_units = n_units,
    adj_row_ptr = adj_row_ptr,
    adj_col_idx = adj_col_idx,
    n_neighbors = n_neighbors,
    bym2_scale = bym2_scale
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
  }

  # Spatial
  if (spatial_info$type == "icar") {
    n_params <- n_params + 1 + spatial_info$n_units  # log_tau + phi
  } else if (spatial_info$type == "bym2") {
    # log_sigma + logit_rho + phi_scaled + theta
    n_params <- n_params + 2 + 2 * spatial_info$n_units
  }

  rep(0.0, n_params)
}


#' Convert HMC output to ratiod_fit (with spatial support)
#' @keywords internal
convert_hmc_to_ratiod_fit_spatial <- function(fit_raw, hmc_data, spatial_info,
                                              formula, data, family,
                                              model_type, iter, warmup, chains) {
  # Handle multi-chain case
  if (chains > 1) {
    # Combine chains
    all_samples <- do.call(rbind, fit_raw$samples)
    all_log_prob <- unlist(fit_raw$log_prob)
    all_accept <- unlist(fit_raw$accept_prob)
    all_divergent <- unlist(fit_raw$divergent)
    epsilon <- mean(fit_raw$epsilon)
  } else {
    all_samples <- fit_raw$samples
    all_log_prob <- fit_raw$log_prob
    all_accept <- fit_raw$accept_prob
    all_divergent <- fit_raw$divergent
    epsilon <- fit_raw$epsilon
  }

  n_samples <- nrow(all_samples)

  # Build parameter names and extract draws
  draws_list <- build_draws_list_spatial(
    all_samples, hmc_data, spatial_info, model_type
  )

  draws <- do.call(cbind, draws_list)
  colnames(draws) <- names(draws_list)

  # Compute ratios
  ratio_draws <- compute_ratio_draws_hmc_spatial(
    all_samples, hmc_data, spatial_info, model_type
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
    backend = "hmc",
    algorithm = "HMC",
    iter = iter,
    warmup = warmup,
    chains = chains,
    n_save = n_samples,
    epsilon = epsilon,
    diagnostics = list(
      n_divergent = n_divergent,
      avg_accept_prob = avg_accept,
      divergent = all_divergent
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


#' Build draws list with spatial parameters
#' @keywords internal
build_draws_list_spatial <- function(samples, hmc_data, spatial_info, model_type) {
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

  # Random effects
  if (hmc_data$n_re_groups > 0) {
    draws_list[["sigma_re"]] <- exp(samples[, idx])
    idx <- idx + 1
    for (g in seq_len(hmc_data$n_re_groups)) {
      draws_list[[paste0("re[", g, "]")]] <- samples[, idx]
      idx <- idx + 1
    }
  }

  # Overdispersion
  if (model_type == "negbin_negbin") {
    draws_list[["phi_num"]] <- exp(samples[, idx])
    idx <- idx + 1
    draws_list[["phi_denom"]] <- exp(samples[, idx])
    idx <- idx + 1
  } else if (model_type == "poisson_gamma") {
    draws_list[["shape"]] <- exp(samples[, idx])
    idx <- idx + 1
  }

  # Spatial
  if (spatial_info$type == "icar") {
    draws_list[["tau_spatial"]] <- exp(samples[, idx])
    idx <- idx + 1
    for (s in seq_len(spatial_info$n_units)) {
      draws_list[[paste0("phi_spatial[", s, "]")]] <- samples[, idx]
      idx <- idx + 1
    }
  } else if (spatial_info$type == "bym2") {
    draws_list[["sigma_spatial"]] <- exp(samples[, idx])
    idx <- idx + 1
    logit_rho <- samples[, idx]
    draws_list[["rho_spatial"]] <- 1 / (1 + exp(-logit_rho))
    idx <- idx + 1

    # phi_scaled
    for (s in seq_len(spatial_info$n_units)) {
      draws_list[[paste0("phi_scaled[", s, "]")]] <- samples[, idx]
      idx <- idx + 1
    }
    # theta
    for (s in seq_len(spatial_info$n_units)) {
      draws_list[[paste0("theta[", s, "]")]] <- samples[, idx]
      idx <- idx + 1
    }
  }

  draws_list
}


#' Compute ratio draws from HMC samples (with spatial)
#' @keywords internal
compute_ratio_draws_hmc_spatial <- function(samples, hmc_data, spatial_info,
                                             model_type) {
  n_samples <- nrow(samples)
  N <- hmc_data$N

  # Extract fixed effects
  beta_num <- samples[, seq_len(hmc_data$p_num), drop = FALSE]
  beta_denom <- samples[, hmc_data$p_num + seq_len(hmc_data$p_denom), drop = FALSE]

  idx <- hmc_data$p_num + hmc_data$p_denom + 1

  # Random effects
  re <- NULL
  if (hmc_data$n_re_groups > 0) {
    idx <- idx + 1  # Skip log_sigma_re
    re <- samples[, idx:(idx + hmc_data$n_re_groups - 1), drop = FALSE]
    idx <- idx + hmc_data$n_re_groups
  }

  # Overdispersion indices (skip for ratio computation)
  if (model_type == "negbin_negbin") {
    idx <- idx + 2
  } else if (model_type == "poisson_gamma") {
    idx <- idx + 1
  }

  # Spatial effects
  spatial_effect <- NULL
  if (spatial_info$type == "icar") {
    idx <- idx + 1  # Skip log_tau
    phi_spatial <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]
    spatial_effect <- phi_spatial
  } else if (spatial_info$type == "bym2") {
    sigma_spatial <- exp(samples[, idx])
    idx <- idx + 1
    logit_rho <- samples[, idx]
    rho <- 1 / (1 + exp(-logit_rho))
    idx <- idx + 1

    phi_scaled <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]
    idx <- idx + spatial_info$n_units
    theta <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]

    # Compute combined spatial effect
    spatial_effect <- matrix(0, nrow = n_samples, ncol = spatial_info$n_units)
    for (s in seq_len(n_samples)) {
      spatial_effect[s, ] <- sigma_spatial[s] * (
        sqrt(rho[s]) * phi_scaled[s, ] * spatial_info$bym2_scale +
        sqrt(1 - rho[s]) * theta[s, ]
      )
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

    # Compute ratio
    if (model_type == "binomial") {
      ratio_draws[s, ] <- 1 / (1 + exp(-eta_num))
    } else {
      ratio_draws[s, ] <- exp(eta_num - eta_denom)
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
      shared = TRUE
    ))
  }

  # Temporal has already been validated by ratiod()
  list(
    type = temporal$type,
    time_index = temporal$time_index,
    group_index = temporal$group_index,
    n_times = temporal$n_times,
    n_groups = temporal$n_groups,
    n_temporal_params = temporal$n_temporal_params,
    precision_structure = temporal$precision_structure,
    shared = temporal$shared
  )
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
  }

  # Spatial
  if (spatial_info$type == "icar") {
    n_params <- n_params + 1 + spatial_info$n_units  # log_tau + phi
  } else if (spatial_info$type == "bym2") {
    # log_sigma + logit_rho + phi_scaled + theta
    n_params <- n_params + 2 + 2 * spatial_info$n_units
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
      coef_names = NULL
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
    coef_names = colnames(X_zi)
  )
}


#' Initialize parameters for HMC with full feature support
#' @keywords internal
initialize_hmc_params_full <- function(hmc_data, model_type, spatial_info,
                                        temporal_info, zi_info, latent_info = NULL) {
  n_params <- hmc_data$p_num + hmc_data$p_denom

  # Random effects (supports multi-term RE with slopes and correlations)
  n_re_terms <- hmc_data$n_re_terms %||% 0L
  has_slopes <- hmc_data$has_slopes %||% FALSE
  has_correlated_slopes <- hmc_data$has_correlated_slopes %||% FALSE

  if (n_re_terms > 0) {
    if (has_slopes) {
      # With slopes: total_sigma_params (log scale) + total_chol_params (if correlated) + total_re_params RE effects
      n_params <- n_params + hmc_data$total_sigma_params + hmc_data$total_re_params
      if (has_correlated_slopes) {
        # Add Cholesky parameters for correlated slopes
        n_params <- n_params + hmc_data$total_chol_params
      }
    } else if (n_re_terms > 1) {
      # Multiple RE terms (intercept only): n_re_terms sigma + total_re_groups RE
      n_params <- n_params + n_re_terms + hmc_data$total_re_groups
    } else if (hmc_data$n_re_groups > 0) {
      # Single RE term (intercept only): 1 sigma + n_re_groups RE params
      n_params <- n_params + 1 + hmc_data$n_re_groups
    }
  }

  # Overdispersion
  if (model_type == "negbin_negbin") {
    n_params <- n_params + 2
  } else if (model_type == "poisson_gamma") {
    n_params <- n_params + 1
  }

  # Spatial
  if (spatial_info$type == "icar") {
    n_params <- n_params + 1 + spatial_info$n_units  # log_tau + phi
  } else if (spatial_info$type == "bym2") {
    # log_sigma + logit_rho + phi_scaled + theta
    n_params <- n_params + 2 + 2 * spatial_info$n_units
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

  # Zero-inflation
  if (zi_info$type != "none") {
    n_params <- n_params + zi_info$p_zi  # ZI regression coefficients
  }

  # Latent factors
  if (!is.null(latent_info) && latent_info$type != "none") {
    K <- latent_info$n_factors
    N <- latent_info$n_obs
    # K log_sigma params + N*K factor score params
    n_params <- n_params + K + N * K
  }

  rep(0.0, n_params)
}


#' Convert HMC output to ratiod_fit (full feature support)
#' @keywords internal
convert_hmc_to_ratiod_fit_full <- function(fit_raw, hmc_data, spatial_info,
                                           temporal_info, zi_info,
                                           latent_info = NULL,
                                           formula, data, family,
                                           model_type, iter, warmup, chains) {
  # Handle multi-chain case
  if (chains > 1) {
    # Combine chains
    all_samples <- do.call(rbind, fit_raw$samples)
    all_log_prob <- unlist(fit_raw$log_prob)
    all_accept <- unlist(fit_raw$accept_prob)
    all_divergent <- unlist(fit_raw$divergent)
    epsilon <- mean(fit_raw$epsilon)
  } else {
    all_samples <- fit_raw$samples
    all_log_prob <- fit_raw$log_prob
    all_accept <- fit_raw$accept_prob
    all_divergent <- fit_raw$divergent
    epsilon <- fit_raw$epsilon
  }

  n_samples <- nrow(all_samples)

  # Build parameter names and extract draws
  draws_list <- build_draws_list_full(
    all_samples, hmc_data, spatial_info, temporal_info, zi_info, model_type
  )

  draws <- do.call(cbind, draws_list)
  colnames(draws) <- names(draws_list)

  # Compute ratios
  ratio_draws <- compute_ratio_draws_hmc_full(
    all_samples, hmc_data, spatial_info, temporal_info, zi_info, model_type
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
        chain_samples, hmc_data, spatial_info, temporal_info, zi_info, model_type
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
    zi = zi_info,
    latent = latent_info,
    backend = "hmc",
    algorithm = "HMC",
    iter = iter,
    warmup = warmup,
    chains = chains,
    n_save = n_samples,
    epsilon = epsilon,
    diagnostics = list(
      n_divergent = n_divergent,
      avg_accept_prob = avg_accept,
      divergent = all_divergent
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
                                   zi_info, model_type) {
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
  n_re_terms <- hmc_data$n_re_terms %||% 0L
  has_slopes <- hmc_data$has_slopes %||% FALSE
  has_correlated_slopes <- hmc_data$has_correlated_slopes %||% FALSE

  if (n_re_terms > 0 && has_slopes) {
    # With random slopes: each term has n_coefs sigma parameters and n_groups * n_coefs RE params
    # Layout: [sigmas for all terms] [chol params for correlated terms] [RE for all terms]

    # First extract all sigma parameters
    for (t in seq_len(n_re_terms)) {
      term <- hmc_data$re_terms[[t]]
      n_coefs <- term$n_coefs
      # Use cleaned slope names for output if available
      slope_names <- term$slope_vars_clean %||% term$slope_vars

      if (n_coefs == 1) {
        draws_list[[paste0("sigma_re[", t, "]")]] <- exp(samples[, idx])
        idx <- idx + 1
      } else {
        # Multiple sigmas: intercept + slopes
        coef_idx <- 1
        if (isTRUE(term$has_intercept)) {
          draws_list[[paste0("sigma_re[", t, ",intercept]")]] <- exp(samples[, idx])
          idx <- idx + 1
          coef_idx <- coef_idx + 1
        }
        for (s in seq_along(slope_names)) {
          draws_list[[paste0("sigma_re[", t, ",", slope_names[s], "]")]] <- exp(samples[, idx])
          idx <- idx + 1
        }
      }
    }

    # Extract Cholesky parameters for correlated terms
    for (t in seq_len(n_re_terms)) {
      term <- hmc_data$re_terms[[t]]
      n_chol <- term$n_chol %||% 0L
      if (n_chol > 0 && isTRUE(term$correlated)) {
        # Store raw Cholesky parameters (off-diagonal elements)
        for (c in seq_len(n_chol)) {
          draws_list[[paste0("L_chol[", t, ",", c, "]")]] <- samples[, idx]
          idx <- idx + 1
        }
      }
    }

    # Then extract all RE effects
    for (t in seq_len(n_re_terms)) {
      term <- hmc_data$re_terms[[t]]
      n_groups_t <- term$n_groups
      n_coefs <- term$n_coefs
      # Use cleaned slope names for output if available
      slope_names <- term$slope_vars_clean %||% term$slope_vars

      for (g in seq_len(n_groups_t)) {
        if (n_coefs == 1) {
          draws_list[[paste0("re[", t, ",", g, "]")]] <- samples[, idx]
          idx <- idx + 1
        } else {
          # Multiple coefficients per group
          coef_idx <- 1
          if (isTRUE(term$has_intercept)) {
            draws_list[[paste0("re[", t, ",", g, ",intercept]")]] <- samples[, idx]
            idx <- idx + 1
            coef_idx <- coef_idx + 1
          }
          for (s in seq_along(slope_names)) {
            draws_list[[paste0("re[", t, ",", g, ",", slope_names[s], "]")]] <- samples[, idx]
            idx <- idx + 1
          }
        }
      }
    }
  } else if (n_re_terms > 1) {
    # Multiple RE terms (intercept only): extract sigma_re for each term
    for (t in seq_len(n_re_terms)) {
      draws_list[[paste0("sigma_re[", t, "]")]] <- exp(samples[, idx])
      idx <- idx + 1
    }
    # Extract RE effects for each term
    for (t in seq_len(n_re_terms)) {
      n_groups_t <- hmc_data$re_terms[[t]]$n_groups
      for (g in seq_len(n_groups_t)) {
        draws_list[[paste0("re[", t, ",", g, "]")]] <- samples[, idx]
        idx <- idx + 1
      }
    }
  } else if (hmc_data$n_re_groups > 0) {
    # Single RE term (intercept only)
    draws_list[["sigma_re"]] <- exp(samples[, idx])
    idx <- idx + 1
    for (g in seq_len(hmc_data$n_re_groups)) {
      draws_list[[paste0("re[", g, "]")]] <- samples[, idx]
      idx <- idx + 1
    }
  }

  # Overdispersion
  if (model_type == "negbin_negbin") {
    draws_list[["phi_num"]] <- exp(samples[, idx])
    idx <- idx + 1
    draws_list[["phi_denom"]] <- exp(samples[, idx])
    idx <- idx + 1
  } else if (model_type == "poisson_gamma") {
    draws_list[["shape"]] <- exp(samples[, idx])
    idx <- idx + 1
  }

  # Spatial
  if (spatial_info$type == "icar") {
    draws_list[["tau_spatial"]] <- exp(samples[, idx])
    idx <- idx + 1
    for (s in seq_len(spatial_info$n_units)) {
      draws_list[[paste0("phi_spatial[", s, "]")]] <- samples[, idx]
      idx <- idx + 1
    }
  } else if (spatial_info$type == "bym2") {
    draws_list[["sigma_spatial"]] <- exp(samples[, idx])
    idx <- idx + 1
    logit_rho <- samples[, idx]
    draws_list[["rho_spatial"]] <- 1 / (1 + exp(-logit_rho))
    idx <- idx + 1

    # phi_scaled
    for (s in seq_len(spatial_info$n_units)) {
      draws_list[[paste0("phi_scaled[", s, "]")]] <- samples[, idx]
      idx <- idx + 1
    }
    # theta
    for (s in seq_len(spatial_info$n_units)) {
      draws_list[[paste0("theta[", s, "]")]] <- samples[, idx]
      idx <- idx + 1
    }
  }

  # Temporal
  if (temporal_info$type != "none") {
    draws_list[["tau_temporal"]] <- exp(samples[, idx])
    idx <- idx + 1

    # AR1 rho parameter
    if (temporal_info$type == "ar1") {
      logit_rho_ar1 <- samples[, idx]
      draws_list[["rho_ar1"]] <- 1 / (1 + exp(-logit_rho_ar1))
      idx <- idx + 1
    }

    # Temporal effects
    for (t in seq_len(temporal_info$n_temporal_params)) {
      draws_list[[paste0("temporal[", t, "]")]] <- samples[, idx]
      idx <- idx + 1
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

  draws_list
}


#' Compute ratio draws from HMC samples (full feature support)
#' @keywords internal
compute_ratio_draws_hmc_full <- function(samples, hmc_data, spatial_info,
                                          temporal_info, zi_info, model_type) {
  n_samples <- nrow(samples)
  N <- hmc_data$N

  # Extract fixed effects
  beta_num <- samples[, seq_len(hmc_data$p_num), drop = FALSE]
  beta_denom <- samples[, hmc_data$p_num + seq_len(hmc_data$p_denom), drop = FALSE]

  idx <- hmc_data$p_num + hmc_data$p_denom + 1

  # Random effects (supports multi-term RE with slopes)
  re <- NULL
  re_multi <- NULL
  re_slopes_multi <- NULL
  n_re_terms <- hmc_data$n_re_terms %||% 0L
  has_slopes <- hmc_data$has_slopes %||% FALSE
  has_correlated_slopes <- hmc_data$has_correlated_slopes %||% FALSE

  if (n_re_terms > 0 && has_slopes) {
    # With random slopes - need to skip sigma, cholesky, then extract RE

    # Skip all sigma_re parameters (one per coefficient type per term)
    for (t in seq_len(n_re_terms)) {
      n_coefs <- hmc_data$re_terms[[t]]$n_coefs
      idx <- idx + n_coefs
    }

    # Skip Cholesky parameters for correlated terms
    if (has_correlated_slopes) {
      for (t in seq_len(n_re_terms)) {
        n_chol <- hmc_data$re_terms[[t]]$n_chol %||% 0L
        idx <- idx + n_chol
      }
    }

    # Extract RE for all terms (including slopes)
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
        re_multi[[t]][, g] <- re_params[, base_col]  # Intercept
        if (n_coefs > 1) {
          for (s in seq_len(n_coefs - 1)) {
            re_slopes_multi[[t]][, g, s] <- re_params[, base_col + s]
          }
        }
      }
    }
  } else if (n_re_terms > 1) {
    # Multiple RE terms (intercept only): skip sigma_re for each term
    idx <- idx + n_re_terms
    # Extract RE for all terms
    re_multi <- list()
    for (t in seq_len(n_re_terms)) {
      n_groups_t <- hmc_data$re_terms[[t]]$n_groups
      re_multi[[t]] <- samples[, idx:(idx + n_groups_t - 1), drop = FALSE]
      idx <- idx + n_groups_t
    }
  } else if (hmc_data$n_re_groups > 0) {
    idx <- idx + 1  # Skip log_sigma_re
    re <- samples[, idx:(idx + hmc_data$n_re_groups - 1), drop = FALSE]
    idx <- idx + hmc_data$n_re_groups
  }

  # Overdispersion indices (skip for ratio computation)
  if (model_type == "negbin_negbin") {
    idx <- idx + 2
  } else if (model_type == "poisson_gamma") {
    idx <- idx + 1
  }

  # Spatial effects
  spatial_effect <- NULL
  if (spatial_info$type == "icar") {
    idx <- idx + 1  # Skip log_tau
    phi_spatial <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]
    spatial_effect <- phi_spatial
    idx <- idx + spatial_info$n_units
  } else if (spatial_info$type == "bym2") {
    sigma_spatial <- exp(samples[, idx])
    idx <- idx + 1
    logit_rho <- samples[, idx]
    rho <- 1 / (1 + exp(-logit_rho))
    idx <- idx + 1

    phi_scaled <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]
    idx <- idx + spatial_info$n_units
    theta <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]
    idx <- idx + spatial_info$n_units

    # Compute combined spatial effect
    spatial_effect <- matrix(0, nrow = n_samples, ncol = spatial_info$n_units)
    for (s in seq_len(n_samples)) {
      spatial_effect[s, ] <- sigma_spatial[s] * (
        sqrt(rho[s]) * phi_scaled[s, ] * spatial_info$bym2_scale +
        sqrt(1 - rho[s]) * theta[s, ]
      )
    }
  }

  # Temporal effects
  temporal_effect <- NULL
  if (temporal_info$type != "none") {
    idx <- idx + 1  # Skip log_tau_temporal
    if (temporal_info$type == "ar1") {
      idx <- idx + 1  # Skip logit_rho_ar1
    }
    temporal_effect <- samples[, idx:(idx + temporal_info$n_temporal_params - 1), drop = FALSE]
    idx <- idx + temporal_info$n_temporal_params
  }

  # ZI coefficients (skip for ratio computation - they affect likelihood, not ratio)
  # ZI parameters are used for P(Y=0) vs P(Y>0), not the mean ratio

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

    # Compute ratio
    if (model_type == "binomial") {
      ratio_draws[s, ] <- 1 / (1 + exp(-eta_num))
    } else {
      ratio_draws[s, ] <- exp(eta_num - eta_denom)
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
    inherits(spatial, "ratiod_multiscale")
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
      nn = 0L,
      cov_type = "exponential",
      nu = 1.5,
      shared = TRUE
    ))
  }

  # Validate GP specification
  validated <- validate_gp(gp, data)

  # Extract coordinates
  coords_mat <- validated$coords
  coords_flat <- as.vector(t(coords_mat))  # Row-major flatten

  if (inherits(gp, "ratiod_multiscale")) {
    # Multi-scale GP
    list(
      gp_type = "multiscale_gp",
      coords = coords_flat,
      # Single-scale params (not used for multiscale)
      nn_idx = integer(0),
      nn_dist = numeric(0),
      nn_order = integer(0),
      nn_order_inv = integer(0),
      nn = 0L,
      # Local scale
      nn_idx_local = as.integer(as.vector(t(validated$neighbors_local))),
      nn_dist_local = as.vector(t(validated$distances_local)),
      nn_order_local = as.integer(validated$order_local),
      nn_order_inv_local = as.integer(validated$order_inv_local),
      nn_local = as.integer(gp$nn_local),
      # Regional scale
      nn_idx_regional = as.integer(as.vector(t(validated$neighbors_regional))),
      nn_dist_regional = as.vector(t(validated$distances_regional)),
      nn_order_regional = as.integer(validated$order_regional),
      nn_order_inv_regional = as.integer(validated$order_inv_regional),
      nn_regional = as.integer(gp$nn_regional),
      # Range constraints
      range_local_lower = gp$range_local[1],
      range_local_upper = gp$range_local[2],
      range_regional_lower = gp$range_regional[1],
      range_regional_upper = gp$range_regional[2],
      # Common params
      cov_type = gp$cov,
      nu = gp$nu,
      shared = gp$shared
    )
  } else {
    # Single-scale GP
    list(
      gp_type = "gp",
      coords = coords_flat,
      nn_idx = as.integer(as.vector(t(validated$neighbors))),
      nn_dist = as.vector(t(validated$distances)),
      nn_order = as.integer(validated$order),
      nn_order_inv = as.integer(validated$order_inv),
      nn = as.integer(gp$nn),
      # Multi-scale params (not used for single-scale)
      nn_idx_local = integer(0),
      nn_dist_local = numeric(0),
      nn_order_local = integer(0),
      nn_order_inv_local = integer(0),
      nn_local = 0L,
      nn_idx_regional = integer(0),
      nn_dist_regional = numeric(0),
      nn_order_regional = integer(0),
      nn_order_inv_regional = integer(0),
      nn_regional = 0L,
      range_local_lower = 0,
      range_local_upper = 1,
      range_regional_lower = 1,
      range_regional_upper = 10,
      # Common params
      cov_type = gp$cov,
      nu = gp$nu,
      shared = gp$shared
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
