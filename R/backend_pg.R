#' Pólya-Gamma Backend for Binomial Models
#'
#' @description
#' Fast Gibbs sampling for binomial models using Pólya-Gamma data augmentation.
#' This backend provides efficient inference for `quotr_binomial()` family models
#' with random effects and spatial structure (ICAR or BYM2).
#'
#' @details
#' The Pólya-Gamma method (Polson, Scott & Windle, 2013) enables efficient
#' Gibbs sampling for binomial regression by introducing auxiliary variables
#' that make the full conditionals Gaussian.
#'
#' **Supported model structures:**
#' - Fixed effects (intercept, covariates)
#' - Group-level random intercepts
#' - ICAR (Intrinsic CAR) spatial effects
#' - BYM2 spatial effects (structured + unstructured components)
#'
#' **When to use this backend:**
#' - Binomial family models with moderate to large data

#' - When Stan sampling is too slow
#' - When you need fast approximate inference
#'
#' @references
#' Polson, N.G., Scott, J.G., and Windle, J. (2013).
#' "Bayesian Inference for Logistic Models Using Pólya-Gamma Latent Variables."
#' Journal of the American Statistical Association, 108(504), 1339-1349.
#'
#' @name pg_backend
#' @keywords internal
NULL


#' Fit binomial model using Pólya-Gamma Gibbs sampling
#'
#' @param formula A quotr_formula object
#' @param data Data frame
#' @param family Must be quotr_binomial()
#' @param spatial Optional spatial structure from spatial_car() or spatial_bym2()
#' @param iter Total iterations per chain
#' @param warmup Warmup iterations per chain
#' @param chains Number of chains to run
#' @param thin Thinning interval
#' @param cores Number of cores for within-chain parallelism
#' @param seed Random seed for reproducibility
#' @param prior_beta_sd Prior SD for fixed effects (default 10)
#' @param prior_sigma_scale Half-Cauchy scale for RE SD (default 2.5)
#' @param prior_tau_shape Gamma shape for spatial precision (default 1)
#' @param prior_tau_rate Gamma rate for spatial precision (default 0.01)
#' @param verbose Print progress
#'
#' @return A quotr_fit object
#' @keywords internal
fit_pg_binomial <- function(formula,
                            data,
                            family,
                            spatial = NULL,
                            iter = 2000,
                            warmup = floor(iter / 2),
                            chains = 4,
                            thin = 1,
                            cores = NULL,
                            seed = NULL,
                            prior_beta_sd = 10,
                            prior_sigma_scale = 2.5,
                            prior_tau_shape = 1,
                            prior_tau_rate = 0.01,
                            verbose = TRUE) {

  # Set cores for within-chain parallelism
  if (is.null(cores)) {
    cores <- cpp_pg_get_max_threads()
  }

  # Set base seed for reproducibility
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Extract response (numerator = successes, denominator = trials)
  y <- formula$numerator$response
  trials <- formula$denominator$response

  if (is.null(trials)) {
    stop("Binomial models require trials (denominator). ",
         "Use formula: successes | trials ~ predictors")
  }

  # Use the design matrix already built by quotr_formula
  X <- formula$numerator$X

  # Extract random effects structure
  re_info <- extract_re_from_data(formula, data)

  # Check for spatial structure
  has_spatial <- !is.null(spatial)

  # Prepare arguments
  n_iter <- as.integer(iter)
  n_warmup <- as.integer(warmup)

  # Prepare spatial info if needed (do once, not per chain)
  spatial_info <- NULL
  is_bym2 <- FALSE
  if (has_spatial) {
    spatial_info <- prepare_spatial_for_pg(spatial, data, formula)
    is_bym2 <- !is.null(spatial$type) && tolower(spatial$type) == "bym2"
  }

  # Run multiple chains sequentially (R's RNG is not thread-safe)
  chain_results <- vector("list", chains)

  for (chain in seq_len(chains)) {
    if (verbose && chains > 1) {
      cat(sprintf("\n=== Chain %d/%d ===\n", chain, chains))
    }

    # Set different seed for each chain
    if (!is.null(seed)) {
      set.seed(seed + chain - 1)
    }

    if (has_spatial) {
      if (is_bym2) {
        # BYM2 spatial model
        scale_factor <- spatial$scale_factor %||% 1.0
        prior_rho_alpha <- spatial$prior_rho_alpha %||% 0.5
        prior_rho_beta <- spatial$prior_rho_beta %||% 0.5

        chain_results[[chain]] <- cpp_pg_binomial_gibbs_bym2(
          y = as.integer(y),
          n = as.integer(trials),
          X = X,
          re_group = re_info$group_idx,
          n_re_groups = re_info$n_groups,
          spatial_group = spatial_info$group_idx,
          n_spatial_units = spatial_info$n_units,
          adj_list = spatial_info$adj_list,
          n_neighbors = spatial_info$n_neighbors,
          scale_factor = scale_factor,
          n_iter = n_iter,
          n_warmup = n_warmup,
          thin = as.integer(thin),
          prior_beta_sd = prior_beta_sd,
          prior_sigma_re_scale = prior_sigma_scale,
          prior_sigma_spatial_scale = prior_sigma_scale,
          prior_rho_alpha = prior_rho_alpha,
          prior_rho_beta = prior_rho_beta,
          store_eta = TRUE,
          verbose = verbose,
          n_threads = as.integer(cores)
        )
      } else {
        # ICAR spatial model
        chain_results[[chain]] <- cpp_pg_binomial_gibbs_spatial(
          y = as.integer(y),
          n = as.integer(trials),
          X = X,
          re_group = re_info$group_idx,
          n_re_groups = re_info$n_groups,
          spatial_group = spatial_info$group_idx,
          n_spatial_units = spatial_info$n_units,
          adj_list = spatial_info$adj_list,
          n_neighbors = spatial_info$n_neighbors,
          n_iter = n_iter,
          n_warmup = n_warmup,
          thin = as.integer(thin),
          prior_beta_sd = prior_beta_sd,
          prior_sigma_re_scale = prior_sigma_scale,
          prior_tau_shape = prior_tau_shape,
          prior_tau_rate = prior_tau_rate,
          store_eta = TRUE,
          verbose = verbose,
          n_threads = as.integer(cores)
        )
      }

    } else if (re_info$n_groups > 0) {
      # Fit with random effects only
      chain_results[[chain]] <- cpp_pg_binomial_gibbs(
        y = as.integer(y),
        n = as.integer(trials),
        X = X,
        group = re_info$group_idx,
        n_groups = re_info$n_groups,
        n_iter = n_iter,
        n_warmup = n_warmup,
        thin = as.integer(thin),
        prior_beta_sd = prior_beta_sd,
        prior_sigma_scale = prior_sigma_scale,
        store_eta = TRUE,
        verbose = verbose,
        n_threads = as.integer(cores)
      )

    } else {
      # No random effects - add dummy group
      chain_results[[chain]] <- cpp_pg_binomial_gibbs(
        y = as.integer(y),
        n = as.integer(trials),
        X = X,
        group = as.integer(rep(1L, length(y))),
        n_groups = 0L,
        n_iter = n_iter,
        n_warmup = n_warmup,
        thin = as.integer(thin),
        prior_beta_sd = prior_beta_sd,
        prior_sigma_scale = prior_sigma_scale,
        store_eta = TRUE,
        verbose = verbose,
        n_threads = as.integer(cores)
      )
    }
  }

  # Convert to quotr_fit format with multi-chain support
  fit <- convert_pg_to_quotr_fit(
    fit_raw = chain_results,
    formula = formula,
    data = data,
    family = family,
    spatial = spatial,
    X = X,
    re_info = re_info,
    iter = iter,
    warmup = warmup,
    thin = thin,
    chains = chains
  )

  return(fit)
}


#' Extract random effects information from data
#'
#' @keywords internal
extract_re_from_data <- function(formula, data) {
  # Check if there are random effects in the formula
  # quotr_formula stores RE in $numerator$random_effects
  re_terms <- formula$numerator$random_effects

  if (is.null(re_terms) || length(re_terms) == 0) {
    return(list(
      group_idx = as.integer(rep(1L, nrow(data))),
      n_groups = 0L,
      group_var = NULL,
      group_levels = NULL
    ))
  }

  # Get the first RE grouping variable
  # For now, support single random intercept
  re_info <- re_terms[[1]]
  re_var <- re_info$group_var

  # The group indices are already computed in quotr_formula
  group_idx <- as.integer(re_info$group)
  n_groups <- re_info$n_groups

  return(list(
    group_idx = group_idx,
    n_groups = n_groups,
    group_var = re_var,
    group_levels = NULL  # Not stored but not needed for PG
  ))
}


#' Prepare spatial structure for PG backend
#'
#' @keywords internal
prepare_spatial_for_pg <- function(spatial, data, formula) {
  # Get spatial group variable
  if (spatial$level == "group") {
    spatial_var <- spatial$group_var
    if (is.null(spatial_var)) {
      stop("spatial_car() with level='group' requires group_var")
    }
  } else {
    # Observation level - use row indices
    spatial_var <- NULL
  }

  if (!is.null(spatial_var)) {
    group_factor <- as.factor(data[[spatial_var]])
    group_idx <- as.integer(group_factor)
    n_units <- nlevels(group_factor)
  } else {
    # Observation level
    group_idx <- seq_len(nrow(data))
    n_units <- nrow(data)
  }

  # Convert adjacency matrix to list format
  adj_matrix <- spatial$adjacency
  adj_list <- vector("list", n_units)
  n_neighbors <- integer(n_units)

  for (i in seq_len(n_units)) {
    neighbors <- which(adj_matrix[i, ] > 0)
    adj_list[[i]] <- as.integer(neighbors)
    n_neighbors[i] <- length(neighbors)
  }

  return(list(
    group_idx = as.integer(group_idx),
    n_units = as.integer(n_units),
    adj_list = adj_list,
    n_neighbors = as.integer(n_neighbors)
  ))
}


#' Convert PG fit to quotr_fit object
#'
#' @param fit_raw List of chain results (each element is a single chain's output)
#' @param chains Number of chains
#' @keywords internal
convert_pg_to_quotr_fit <- function(fit_raw, formula, data, family,
                                    spatial, X, re_info, iter, warmup, thin,
                                    chains = 1) {

  # fit_raw is now a list of chain results
  n_chains <- length(fit_raw)
  n_save_per_chain <- nrow(fit_raw[[1]]$beta)
  n_save_total <- n_save_per_chain * n_chains

  # Create draws in format compatible with posterior package
  beta_names <- colnames(X)
  if (is.null(beta_names)) {
    beta_names <- paste0("beta[", seq_len(ncol(X)), "]")
  }

  # Determine if spatial model
  is_bym2 <- !is.null(spatial) && !is.null(spatial$type) &&
             tolower(spatial$type) == "bym2"

  # Build draws array with chain dimension: [iteration, chain, parameter]
  # For compatibility with posterior package
  n_params <- ncol(fit_raw[[1]]$beta)
  if (re_info$n_groups > 0) n_params <- n_params + 1  # sigma_re
  if (!is.null(spatial)) {
    if (is_bym2) {
      n_params <- n_params + 2  # sigma_spatial, rho
    } else {
      n_params <- n_params + 1  # tau_spatial
    }
  }

  # Create 3D array for draws: [iteration, chain, variable]
  draws_array <- array(NA_real_,
                       dim = c(n_save_per_chain, n_chains, n_params))

  # Build parameter names
  param_names <- beta_names
  if (re_info$n_groups > 0) {
    param_names <- c(param_names, "sigma_re")
  }
  if (!is.null(spatial)) {
    if (is_bym2) {
      param_names <- c(param_names, "sigma_spatial", "rho")
    } else {
      param_names <- c(param_names, "tau_spatial")
    }
  }
  dimnames(draws_array) <- list(
    iteration = seq_len(n_save_per_chain),
    chain = seq_len(n_chains),
    variable = param_names
  )

  # Fill in the draws from each chain
  for (c in seq_len(n_chains)) {
    chain_fit <- fit_raw[[c]]
    col_idx <- 1

    # Fixed effects
    for (j in seq_len(ncol(chain_fit$beta))) {
      draws_array[, c, col_idx] <- chain_fit$beta[, j]
      col_idx <- col_idx + 1
    }

    # Random effects SD
    if (re_info$n_groups > 0) {
      draws_array[, c, col_idx] <- chain_fit$sigma_re
      col_idx <- col_idx + 1
    }

    # Spatial parameters
    if (!is.null(spatial)) {
      if (is_bym2) {
        draws_array[, c, col_idx] <- chain_fit$sigma_spatial
        col_idx <- col_idx + 1
        draws_array[, c, col_idx] <- chain_fit$rho
        col_idx <- col_idx + 1
      } else {
        draws_array[, c, col_idx] <- chain_fit$tau
        col_idx <- col_idx + 1
      }
    }
  }

  # Also create combined eta array for ratio computation: [iteration, chain, obs]
  n_obs <- ncol(fit_raw[[1]]$eta)
  eta_array <- array(NA_real_,
                     dim = c(n_save_per_chain, n_chains, n_obs))
  for (c in seq_len(n_chains)) {
    eta_array[, c, ] <- fit_raw[[c]]$eta
  }

  # Combine internal components from all chains for backwards compatibility
  # Stack matrices row-wise for methods that expect 2D format
  combined_beta <- do.call(rbind, lapply(fit_raw, `[[`, "beta"))
  combined_re <- do.call(rbind, lapply(fit_raw, `[[`, "re"))
  combined_eta <- do.call(rbind, lapply(fit_raw, `[[`, "eta"))
  combined_spatial <- if (!is.null(fit_raw[[1]]$spatial)) {
    do.call(rbind, lapply(fit_raw, `[[`, "spatial"))
  } else NULL

  # Create quotr_fit object
  fit <- list(
    draws = draws_array,
    formula = formula,
    data = data,
    family = family,
    spatial = spatial,
    backend = "pg",
    iter = iter,
    warmup = warmup,
    thin = thin,
    chains = n_chains,
    n_save = n_save_total,
    n_save_per_chain = n_save_per_chain,
    # Store raw components for ratio computation
    .internal = list(
      eta = combined_eta,
      eta_array = eta_array,  # 3D array version
      beta = combined_beta,
      re = combined_re,
      spatial = combined_spatial,
      phi_scaled = if (!is.null(fit_raw[[1]]$phi_scaled)) {
        do.call(rbind, lapply(fit_raw, `[[`, "phi_scaled"))
      } else NULL,
      theta = if (!is.null(fit_raw[[1]]$theta)) {
        do.call(rbind, lapply(fit_raw, `[[`, "theta"))
      } else NULL,
      X = X,
      re_info = re_info,
      chain_results = fit_raw  # Keep original chain results for diagnostics
    )
  )

  class(fit) <- "quotr_fit"
  return(fit)
}


#' Check if PG backend is appropriate
#'
#' @param family A quotr family object
#' @return Logical indicating if PG backend can be used
#' @keywords internal
can_use_pg_backend <- function(family) {
  # Check if this is a binomial family
  # The family structure uses $numerator$distribution
  isTRUE(family$numerator$distribution == "binomial")
}
