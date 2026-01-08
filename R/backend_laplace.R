#' Laplace Approximation Backend
#'
#' @description
#' Fast approximate inference using nested Laplace approximation.
#' Suitable for large areal datasets where full MCMC is too slow.
#'
#' @details
#' The Laplace backend finds the mode of the posterior and uses a Gaussian
#' approximation centered at the mode. This is much faster than MCMC but
#' provides approximate inference.
#'
#' **Supported model structures:**
#' - Fixed effects
#' - Group-level random intercepts
#' - All three families (binomial, negbin, poisson_gamma)
#'
#' **When to use:**
#' - Large datasets (N > 10,000)
#' - Screening models before full MCMC
#' - When approximate posteriors are acceptable
#'
#' @name laplace_backend
#' @keywords internal
NULL


#' Fit model using Laplace approximation
#'
#' @param formula A quotr_formula object
#' @param data Data frame
#' @param family Model family
#' @param spatial Optional spatial structure
#' @param priors Prior specification
#' @param n_samples Number of posterior samples to draw
#' @param cores Number of cores for parallel computation
#' @param verbose Print progress
#'
#' @return A quotr_fit object
#' @keywords internal
fit_laplace <- function(formula,
                        data,
                        family,
                        spatial = NULL,
                        priors = NULL,
                        n_samples = 1000,
                        cores = NULL,
                        verbose = TRUE) {

  # Set cores
  if (is.null(cores)) {
    cores <- cpp_laplace_get_max_threads()
  }

  # Determine which family/likelihood to use
  family_type <- get_laplace_family(family)

  # Extract response based on family
  if (family_type == "binomial") {
    y <- formula$numerator$response
    n_trials <- formula$denominator$response
    phi <- 1.0  # Not used for binomial
  } else if (family_type == "negbin") {
    y <- formula$numerator$response
    n_trials <- rep(1L, length(y))  # Placeholder
    phi <- priors$phi %||% 1.0
  } else {
    y <- formula$numerator$response
    n_trials <- rep(1L, length(y))
    phi <- 1.0
  }

  # Design matrix
  X <- formula$numerator$X

  # Random effects
  re_info <- extract_re_for_laplace(formula)

  # Hyperparameters
  sigma_re <- priors$sigma_re %||% 1.0
  tau_spatial <- priors$tau_spatial %||% 1.0

  # Check if spatial model
  has_spatial <- !is.null(spatial)

  if (has_spatial) {
    # Dispatch to spatial Laplace
    return(fit_laplace_spatial(
      formula = formula,
      data = data,
      family = family,
      family_type = family_type,
      spatial = spatial,
      y = y,
      n_trials = n_trials,
      X = X,
      re_info = re_info,
      phi = phi,
      sigma_re = sigma_re,
      tau_spatial = tau_spatial,
      n_samples = n_samples,
      cores = cores,
      verbose = verbose
    ))
  }

  # For Laplace, we optimize sigma_re (empirical Bayes)
  if (verbose) {
    message("Finding mode of posterior...")
  }

  # First pass: find mode with initial sigma_re
  result <- optimize_laplace_hyperparams(
    y = y,
    n_trials = n_trials,
    X = X,
    re_idx = re_info$group_idx,
    n_re_groups = re_info$n_groups,
    family = family_type,
    phi = phi,
    sigma_re_init = sigma_re,
    n_threads = cores,
    verbose = verbose
  )

  # Now sample from Laplace approximation
  if (verbose) {
    message("Sampling from Laplace approximation...")
  }

  # Get the Hessian at the mode for sampling
  hess_result <- compute_hessian_at_mode(
    y = y,
    n_trials = n_trials,
    X = X,
    re_idx = re_info$group_idx,
    n_re_groups = re_info$n_groups,
    mode = result$mode,
    family = family_type,
    phi = phi,
    sigma_re = result$sigma_re_opt
  )

  # Sample from N(mode, H^{-1})
  samples <- cpp_laplace_sample(
    mode = result$mode,
    H = hess_result$H,
    n_samples = as.integer(n_samples)
  )

  # Convert to quotr_fit format
  fit <- convert_laplace_to_quotr_fit(
    samples = samples,
    result = result,
    formula = formula,
    data = data,
    family = family,
    X = X,
    re_info = re_info,
    n_samples = n_samples
  )

  return(fit)
}


#' Get Laplace family string
#' @keywords internal
get_laplace_family <- function(family) {
  dist <- family$numerator$distribution
  if (dist == "binomial") return("binomial")
  if (dist == "negbin" || dist == "negative_binomial") return("negbin")
  if (dist == "poisson") return("poisson")
  stop("Unsupported family for Laplace backend: ", dist)
}


#' Extract RE info for Laplace
#' @keywords internal
extract_re_for_laplace <- function(formula) {
  re_terms <- formula$numerator$random_effects

  if (is.null(re_terms) || length(re_terms) == 0) {
    n_obs <- length(formula$numerator$response)
    return(list(
      group_idx = as.numeric(rep(1, n_obs)),
      n_groups = 0L,
      group_var = NULL
    ))
  }

  re_info <- re_terms[[1]]
  return(list(
    group_idx = as.numeric(re_info$group),
    n_groups = as.integer(re_info$n_groups),
    group_var = re_info$group_var
  ))
}


#' Optimize hyperparameters via marginal likelihood
#' @keywords internal
optimize_laplace_hyperparams <- function(y, n_trials, X, re_idx, n_re_groups,
                                         family, phi, sigma_re_init, n_threads,
                                         verbose) {

  if (n_re_groups == 0) {
    # No random effects - just find mode
    result <- cpp_laplace_fit(
      y = as.integer(y),
      n = as.integer(n_trials),
      X = X,
      re_idx = re_idx,
      n_re_groups = 0L,
      sigma_re = 1.0,
      family = family,
      phi = phi,
      n_threads = as.integer(n_threads)
    )

    return(list(
      mode = result$mode,
      log_marginal = result$log_marginal,
      sigma_re_opt = 1.0,
      converged = result$converged
    ))
  }

  # Optimize sigma_re by maximizing marginal likelihood
  obj_fn <- function(log_sigma) {
    sigma <- exp(log_sigma)
    result <- cpp_laplace_fit(
      y = as.integer(y),
      n = as.integer(n_trials),
      X = X,
      re_idx = re_idx,
      n_re_groups = n_re_groups,
      sigma_re = sigma,
      family = family,
      phi = phi,
      n_threads = as.integer(n_threads)
    )
    # Return negative log marginal (for minimization)
    if (!result$converged) return(1e10)
    -result$log_marginal
  }

  # Optimize
  opt <- tryCatch({
    stats::optim(
      par = log(sigma_re_init),
      fn = obj_fn,
      method = "Brent",
      lower = log(0.01),
      upper = log(10)
    )
  }, error = function(e) {
    list(par = log(sigma_re_init), convergence = 1)
  })

  sigma_re_opt <- exp(opt$par)

  if (verbose) {
    message(sprintf("  Optimal sigma_re: %.3f", sigma_re_opt))
  }

  # Final fit at optimal sigma_re
  result <- cpp_laplace_fit(
    y = as.integer(y),
    n = as.integer(n_trials),
    X = X,
    re_idx = re_idx,
    n_re_groups = n_re_groups,
    sigma_re = sigma_re_opt,
    family = family,
    phi = phi,
    n_threads = as.integer(n_threads)
  )

  return(list(
    mode = result$mode,
    log_marginal = result$log_marginal,
    sigma_re_opt = sigma_re_opt,
    converged = result$converged
  ))
}


#' Compute Hessian at mode for sampling
#' @keywords internal
compute_hessian_at_mode <- function(y, n_trials, X, re_idx, n_re_groups,
                                    mode, family, phi, sigma_re) {
  N <- length(y)
  p <- ncol(X)
  n_x <- p + n_re_groups

  # Compute eta at mode
  eta <- as.numeric(X %*% mode[1:p])
  if (n_re_groups > 0) {
    for (i in seq_len(N)) {
      g <- as.integer(re_idx[i])
      if (g > 0 && g <= n_re_groups) {
        eta[i] <- eta[i] + mode[p + g]
      }
    }
  }

  # Build Hessian
  H <- matrix(0, n_x, n_x)

  # Likelihood contributions
  for (i in seq_len(N)) {
    if (family == "binomial") {
      p_i <- 1 / (1 + exp(-eta[i]))
      h_i <- n_trials[i] * p_i * (1 - p_i)
    } else if (family == "negbin") {
      mu_i <- exp(eta[i])
      h_i <- (y[i] + phi) * mu_i * phi / (mu_i + phi)^2
    } else {
      h_i <- exp(eta[i])
    }

    # Fixed effects block
    for (j in seq_len(p)) {
      for (k in seq_len(p)) {
        H[j, k] <- H[j, k] + h_i * X[i, j] * X[i, k]
      }
    }

    # Random effects
    if (n_re_groups > 0) {
      g <- as.integer(re_idx[i])
      if (g > 0 && g <= n_re_groups) {
        H[p + g, p + g] <- H[p + g, p + g] + h_i

        for (j in seq_len(p)) {
          H[j, p + g] <- H[j, p + g] + h_i * X[i, j]
          H[p + g, j] <- H[p + g, j] + h_i * X[i, j]
        }
      }
    }
  }

  # Add prior precision
  tau_re <- 1 / (sigma_re^2 + 1e-10)
  for (g in seq_len(n_re_groups)) {
    H[p + g, p + g] <- H[p + g, p + g] + tau_re
  }

  # Small regularization for fixed effects
  for (j in seq_len(p)) {
    H[j, j] <- H[j, j] + 1e-4
  }

  return(list(H = H))
}


#' Convert Laplace results to quotr_fit
#' @keywords internal
convert_laplace_to_quotr_fit <- function(samples, result, formula, data,
                                         family, X, re_info, n_samples) {

  p <- ncol(X)
  n_re <- re_info$n_groups

  # Extract fixed effects samples
  beta_names <- colnames(X)
  if (is.null(beta_names)) {
    beta_names <- paste0("beta[", seq_len(p), "]")
  }

  draws_list <- list()
  for (j in seq_len(p)) {
    draws_list[[beta_names[j]]] <- samples[, j]
  }

  # Add sigma_re as point estimate (could improve by sampling)
  if (n_re > 0) {
    draws_list[["sigma_re"]] <- rep(result$sigma_re_opt, n_samples)
  }

  draws <- do.call(cbind, draws_list)
  colnames(draws) <- names(draws_list)

  fit <- list(
    draws = draws,
    formula = formula,
    data = data,
    family = family,
    backend = "laplace",
    n_save = n_samples,
    laplace_result = result,
    .internal = list(
      mode = result$mode,
      sigma_re = result$sigma_re_opt,
      log_marginal = result$log_marginal,
      X = X,
      re_info = re_info,
      samples = samples
    )
  )

  class(fit) <- "quotr_fit"
  return(fit)
}


#' Check if Laplace backend can be used
#' @keywords internal
can_use_laplace_backend <- function(family) {
  # Laplace works for all families (including spatial now)
  TRUE
}


#' Fit spatial model using Laplace approximation
#'
#' @param formula A quotr_formula object
#' @param data Data frame
#' @param family Model family
#' @param family_type String family type for C++
#' @param spatial Spatial structure specification
#' @param y Response vector
#' @param n_trials Trials vector
#' @param X Design matrix
#' @param re_info Random effects info
#' @param phi Overdispersion parameter
#' @param sigma_re RE standard deviation
#' @param tau_spatial Spatial precision
#' @param n_samples Number of posterior samples
#' @param cores Number of cores
#' @param verbose Print progress
#'
#' @return A quotr_fit object
#' @keywords internal
fit_laplace_spatial <- function(formula,
                                 data,
                                 family,
                                 family_type,
                                 spatial,
                                 y,
                                 n_trials,
                                 X,
                                 re_info,
                                 phi,
                                 sigma_re,
                                 tau_spatial,
                                 n_samples,
                                 cores,
                                 verbose) {

  if (verbose) {
    message("Fitting spatial model with Laplace approximation...")
  }

  # Prepare spatial structure
  spatial_info <- prepare_spatial_for_laplace(spatial, data, formula)

  if (verbose) {
    message(sprintf("  Spatial units: %d", spatial_info$n_units))
    message(sprintf("  Spatial type: %s", spatial$type %||% "ICAR"))
  }

  # Check if BYM2 model
  is_bym2 <- !is.null(spatial$type) && tolower(spatial$type) == "bym2"

  if (is_bym2) {
    return(fit_laplace_bym2(
      formula = formula,
      data = data,
      family = family,
      family_type = family_type,
      spatial = spatial,
      spatial_info = spatial_info,
      y = y,
      n_trials = n_trials,
      X = X,
      re_info = re_info,
      phi = phi,
      sigma_re = sigma_re,
      n_samples = n_samples,
      cores = cores,
      verbose = verbose
    ))
  }

  # Fit with fixed hyperparameters first (ICAR)
  if (verbose) {
    message("Finding mode of posterior...")
  }

  result <- cpp_laplace_fit_spatial(
    y = as.integer(y),
    n = as.integer(n_trials),
    X = X,
    re_idx = re_info$group_idx,
    n_re_groups = re_info$n_groups,
    sigma_re = sigma_re,
    spatial_idx = spatial_info$group_idx,
    n_spatial_units = spatial_info$n_units,
    adj_row_ptr = spatial_info$adj_row_ptr,
    adj_col_idx = spatial_info$adj_col_idx,
    n_neighbors = spatial_info$n_neighbors,
    tau_spatial = tau_spatial,
    family = family_type,
    phi = phi,
    n_threads = as.integer(cores)
  )

  if (!result$converged) {
    warning("Laplace approximation did not converge")
  }

  if (verbose) {
    message(sprintf("  Converged in %d iterations", result$n_iter))
    message("Sampling from Laplace approximation...")
  }

  # Compute Hessian at mode for sampling
  hess_result <- compute_hessian_spatial(
    y = y,
    n_trials = n_trials,
    X = X,
    re_idx = re_info$group_idx,
    n_re_groups = re_info$n_groups,
    spatial_idx = spatial_info$group_idx,
    n_spatial_units = spatial_info$n_units,
    adj_row_ptr = spatial_info$adj_row_ptr,
    adj_col_idx = spatial_info$adj_col_idx,
    n_neighbors = spatial_info$n_neighbors,
    mode = result$mode,
    family = family_type,
    phi = phi,
    sigma_re = sigma_re,
    tau_spatial = tau_spatial
  )

  # Sample from Laplace approximation
  samples <- cpp_laplace_sample(
    mode = result$mode,
    H = hess_result$H,
    n_samples = as.integer(n_samples)
  )

  # Convert to quotr_fit
  fit <- convert_laplace_spatial_to_quotr_fit(
    samples = samples,
    result = result,
    formula = formula,
    data = data,
    family = family,
    X = X,
    re_info = re_info,
    spatial_info = spatial_info,
    n_samples = n_samples
  )

  return(fit)
}


#' Prepare spatial structure for Laplace
#' @keywords internal
prepare_spatial_for_laplace <- function(spatial, data, formula) {
  # Get the spatial grouping variable
  group_var <- spatial$group_var
  if (is.null(group_var)) {
    stop("Spatial structure must specify group_var")
  }

  # Get group indices
  if (!group_var %in% names(data)) {
    stop(sprintf("Spatial group variable '%s' not found in data", group_var))
  }

  group_factor <- as.factor(data[[group_var]])
  group_idx <- as.integer(group_factor)
  n_units <- nlevels(group_factor)

  # Get adjacency structure
  adj_matrix <- spatial$adj_matrix
  if (is.null(adj_matrix)) {
    stop("Spatial structure must include adj_matrix")
  }

  # Convert adjacency matrix to CSR format
  n_neighbors <- integer(n_units)
  adj_row_ptr <- integer(n_units + 1)
  adj_col_idx <- integer(0)

  adj_row_ptr[1] <- 1L  # 1-based for R
  for (i in seq_len(n_units)) {
    neighbors <- which(adj_matrix[i, ] != 0)
    n_neighbors[i] <- length(neighbors)
    adj_col_idx <- c(adj_col_idx, neighbors)
    adj_row_ptr[i + 1] <- adj_row_ptr[i] + n_neighbors[i]
  }

  list(
    group_idx = group_idx,
    n_units = as.integer(n_units),
    n_neighbors = as.integer(n_neighbors),
    adj_row_ptr = as.integer(adj_row_ptr),
    adj_col_idx = as.integer(adj_col_idx)
  )
}


#' Compute Hessian at mode for spatial model
#' @keywords internal
compute_hessian_spatial <- function(y, n_trials, X, re_idx, n_re_groups,
                                     spatial_idx, n_spatial_units,
                                     adj_row_ptr, adj_col_idx, n_neighbors,
                                     mode, family, phi, sigma_re, tau_spatial) {
  N <- length(y)
  p <- ncol(X)
  n_x <- p + n_re_groups + n_spatial_units
  spatial_start <- p + n_re_groups

  # Compute eta at mode
  eta <- as.numeric(X %*% mode[1:p])
  if (n_re_groups > 0) {
    for (i in seq_len(N)) {
      g <- as.integer(re_idx[i])
      if (g > 0 && g <= n_re_groups) {
        eta[i] <- eta[i] + mode[p + g]
      }
    }
  }
  if (n_spatial_units > 0) {
    for (i in seq_len(N)) {
      s <- as.integer(spatial_idx[i])
      if (s > 0 && s <= n_spatial_units) {
        eta[i] <- eta[i] + mode[spatial_start + s]
      }
    }
  }

  # Build Hessian
  H <- matrix(0, n_x, n_x)
  tau_re <- 1 / (sigma_re^2 + 1e-10)

  # Likelihood contributions
  for (i in seq_len(N)) {
    if (family == "binomial") {
      p_i <- 1 / (1 + exp(-eta[i]))
      h_i <- n_trials[i] * p_i * (1 - p_i)
    } else if (family == "negbin") {
      mu_i <- exp(eta[i])
      h_i <- (y[i] + phi) * mu_i * phi / (mu_i + phi)^2
    } else {
      h_i <- exp(eta[i])
    }

    # Fixed effects block
    for (j in seq_len(p)) {
      for (k in seq_len(p)) {
        H[j, k] <- H[j, k] + h_i * X[i, j] * X[i, k]
      }
    }

    # Random effects
    if (n_re_groups > 0) {
      g <- as.integer(re_idx[i])
      if (g > 0 && g <= n_re_groups) {
        H[p + g, p + g] <- H[p + g, p + g] + h_i
        for (j in seq_len(p)) {
          H[j, p + g] <- H[j, p + g] + h_i * X[i, j]
          H[p + g, j] <- H[p + g, j] + h_i * X[i, j]
        }
      }
    }

    # Spatial effects
    if (n_spatial_units > 0) {
      s <- as.integer(spatial_idx[i])
      if (s > 0 && s <= n_spatial_units) {
        sp_idx <- spatial_start + s
        H[sp_idx, sp_idx] <- H[sp_idx, sp_idx] + h_i
        for (j in seq_len(p)) {
          H[j, sp_idx] <- H[j, sp_idx] + h_i * X[i, j]
          H[sp_idx, j] <- H[sp_idx, j] + h_i * X[i, j]
        }
        if (n_re_groups > 0) {
          g <- as.integer(re_idx[i])
          if (g > 0 && g <= n_re_groups) {
            H[p + g, sp_idx] <- H[p + g, sp_idx] + h_i
            H[sp_idx, p + g] <- H[sp_idx, p + g] + h_i
          }
        }
      }
    }
  }

  # Add prior precision for RE
  for (g in seq_len(n_re_groups)) {
    H[p + g, p + g] <- H[p + g, p + g] + tau_re
  }

  # Add ICAR precision for spatial
  for (s in seq_len(n_spatial_units)) {
    sp_idx <- spatial_start + s
    H[sp_idx, sp_idx] <- H[sp_idx, sp_idx] + tau_spatial * n_neighbors[s]
    for (k in adj_row_ptr[s]:(adj_row_ptr[s + 1] - 1)) {
      neighbor <- adj_col_idx[k]
      nb_idx <- spatial_start + neighbor
      H[sp_idx, nb_idx] <- H[sp_idx, nb_idx] - tau_spatial
    }
  }

  # Regularization for fixed effects
  for (j in seq_len(p)) {
    H[j, j] <- H[j, j] + 1e-4
  }

  list(H = H)
}


#' Convert spatial Laplace results to quotr_fit
#' @keywords internal
convert_laplace_spatial_to_quotr_fit <- function(samples, result, formula, data,
                                                   family, X, re_info, spatial_info,
                                                   n_samples) {
  p <- ncol(X)
  n_re <- re_info$n_groups
  n_spatial <- spatial_info$n_units
  spatial_start <- p + n_re

  # Extract fixed effects samples
  beta_names <- colnames(X)
  if (is.null(beta_names)) {
    beta_names <- paste0("beta[", seq_len(p), "]")
  }

  draws_list <- list()
  for (j in seq_len(p)) {
    draws_list[[beta_names[j]]] <- samples[, j]
  }

  # Add spatial effects
  if (n_spatial > 0) {
    for (s in seq_len(n_spatial)) {
      draws_list[[paste0("spatial[", s, "]")]] <- samples[, spatial_start + s]
    }
  }

  draws <- do.call(cbind, draws_list)
  colnames(draws) <- names(draws_list)

  fit <- list(
    draws = draws,
    formula = formula,
    data = data,
    family = family,
    backend = "laplace",
    n_save = n_samples,
    laplace_result = result,
    .internal = list(
      mode = result$mode,
      log_marginal = result$log_marginal,
      X = X,
      re_info = re_info,
      spatial_info = spatial_info,
      samples = samples
    )
  )

  class(fit) <- "quotr_fit"
  return(fit)
}


#' Fit BYM2 spatial model using Laplace approximation
#'
#' @param formula A quotr_formula object
#' @param data Data frame
#' @param family Model family
#' @param family_type String family type for C++
#' @param spatial Spatial structure specification
#' @param spatial_info Prepared spatial info
#' @param y Response vector
#' @param n_trials Trials vector
#' @param X Design matrix
#' @param re_info Random effects info
#' @param phi Overdispersion parameter
#' @param sigma_re RE standard deviation
#' @param n_samples Number of posterior samples
#' @param cores Number of cores
#' @param verbose Print progress
#'
#' @return A quotr_fit object
#' @keywords internal
fit_laplace_bym2 <- function(formula,
                              data,
                              family,
                              family_type,
                              spatial,
                              spatial_info,
                              y,
                              n_trials,
                              X,
                              re_info,
                              phi,
                              sigma_re,
                              n_samples,
                              cores,
                              verbose) {

  # Extract BYM2 hyperparameters
  sigma_spatial <- spatial$sigma %||% 1.0
  rho <- spatial$rho %||% 0.5
  scale_factor <- spatial$scale_factor %||% 1.0

  if (verbose) {
    message("Fitting BYM2 spatial model with Laplace approximation...")
    message(sprintf("  sigma_spatial: %.3f, rho: %.3f", sigma_spatial, rho))
  }

  # Fit with fixed hyperparameters
  if (verbose) {
    message("Finding mode of posterior...")
  }

  result <- cpp_laplace_fit_bym2(
    y = as.integer(y),
    n = as.integer(n_trials),
    X = X,
    re_idx = re_info$group_idx,
    n_re_groups = re_info$n_groups,
    sigma_re = sigma_re,
    spatial_idx = spatial_info$group_idx,
    n_spatial_units = spatial_info$n_units,
    adj_row_ptr = spatial_info$adj_row_ptr,
    adj_col_idx = spatial_info$adj_col_idx,
    n_neighbors = spatial_info$n_neighbors,
    sigma_spatial = sigma_spatial,
    rho = rho,
    scale_factor = scale_factor,
    family = family_type,
    phi = phi,
    n_threads = as.integer(cores)
  )

  if (!result$converged) {
    warning("Laplace approximation did not converge")
  }

  if (verbose) {
    message(sprintf("  Converged in %d iterations", result$n_iter))
    message("Sampling from Laplace approximation...")
  }

  # Compute Hessian at mode for sampling
  hess_result <- compute_hessian_bym2(
    y = y,
    n_trials = n_trials,
    X = X,
    re_idx = re_info$group_idx,
    n_re_groups = re_info$n_groups,
    spatial_idx = spatial_info$group_idx,
    n_spatial_units = spatial_info$n_units,
    adj_row_ptr = spatial_info$adj_row_ptr,
    adj_col_idx = spatial_info$adj_col_idx,
    n_neighbors = spatial_info$n_neighbors,
    mode = result$mode,
    family = family_type,
    phi = phi,
    sigma_re = sigma_re,
    sigma_spatial = sigma_spatial,
    rho = rho,
    scale_factor = scale_factor
  )

  # Sample from Laplace approximation
  samples <- cpp_laplace_sample(
    mode = result$mode,
    H = hess_result$H,
    n_samples = as.integer(n_samples)
  )

  # Convert to quotr_fit
  fit <- convert_laplace_bym2_to_quotr_fit(
    samples = samples,
    result = result,
    formula = formula,
    data = data,
    family = family,
    X = X,
    re_info = re_info,
    spatial_info = spatial_info,
    sigma_spatial = sigma_spatial,
    rho = rho,
    scale_factor = scale_factor,
    n_samples = n_samples
  )

  return(fit)
}


#' Compute Hessian at mode for BYM2 model
#' @keywords internal
compute_hessian_bym2 <- function(y, n_trials, X, re_idx, n_re_groups,
                                  spatial_idx, n_spatial_units,
                                  adj_row_ptr, adj_col_idx, n_neighbors,
                                  mode, family, phi, sigma_re,
                                  sigma_spatial, rho, scale_factor) {
  N <- length(y)
  p <- ncol(X)
  # Parameters: fixed effects + RE + phi_scaled + theta
  n_x <- p + n_re_groups + 2 * n_spatial_units
  phi_start <- p + n_re_groups
  theta_start <- phi_start + n_spatial_units

  sqrt_rho <- sqrt(rho + 1e-10)
  sqrt_1_rho <- sqrt(1.0 - rho + 1e-10)

  # Compute eta at mode
  eta <- as.numeric(X %*% mode[1:p])
  if (n_re_groups > 0) {
    for (i in seq_len(N)) {
      g <- as.integer(re_idx[i])
      if (g > 0 && g <= n_re_groups) {
        eta[i] <- eta[i] + mode[p + g]
      }
    }
  }
  if (n_spatial_units > 0) {
    for (i in seq_len(N)) {
      s <- as.integer(spatial_idx[i])
      if (s > 0 && s <= n_spatial_units) {
        phi_s <- mode[phi_start + s]
        theta_s <- mode[theta_start + s]
        spatial_effect <- sigma_spatial * (sqrt_rho * phi_s * scale_factor + sqrt_1_rho * theta_s)
        eta[i] <- eta[i] + spatial_effect
      }
    }
  }

  # Build Hessian
  H <- matrix(0, n_x, n_x)
  tau_re <- 1 / (sigma_re^2 + 1e-10)

  # Likelihood contributions
  for (i in seq_len(N)) {
    if (family == "binomial") {
      p_i <- 1 / (1 + exp(-eta[i]))
      h_i <- n_trials[i] * p_i * (1 - p_i)
    } else if (family == "negbin") {
      mu_i <- exp(eta[i])
      h_i <- (y[i] + phi) * mu_i * phi / (mu_i + phi)^2
    } else {
      h_i <- exp(eta[i])
    }

    # Fixed effects block
    for (j in seq_len(p)) {
      for (k in seq_len(p)) {
        H[j, k] <- H[j, k] + h_i * X[i, j] * X[i, k]
      }
    }

    # Random effects
    if (n_re_groups > 0) {
      g <- as.integer(re_idx[i])
      if (g > 0 && g <= n_re_groups) {
        H[p + g, p + g] <- H[p + g, p + g] + h_i
        for (j in seq_len(p)) {
          H[j, p + g] <- H[j, p + g] + h_i * X[i, j]
          H[p + g, j] <- H[p + g, j] + h_i * X[i, j]
        }
      }
    }

    # BYM2 spatial effects
    if (n_spatial_units > 0) {
      s <- as.integer(spatial_idx[i])
      if (s > 0 && s <= n_spatial_units) {
        phi_idx <- phi_start + s
        theta_idx <- theta_start + s

        coef_phi <- sigma_spatial * sqrt_rho * scale_factor
        coef_theta <- sigma_spatial * sqrt_1_rho

        # phi_scaled contributions
        H[phi_idx, phi_idx] <- H[phi_idx, phi_idx] + h_i * coef_phi^2
        # theta contributions
        H[theta_idx, theta_idx] <- H[theta_idx, theta_idx] + h_i * coef_theta^2
        # phi-theta cross
        H[phi_idx, theta_idx] <- H[phi_idx, theta_idx] + h_i * coef_phi * coef_theta
        H[theta_idx, phi_idx] <- H[theta_idx, phi_idx] + h_i * coef_phi * coef_theta

        # Cross with fixed effects
        for (j in seq_len(p)) {
          H[j, phi_idx] <- H[j, phi_idx] + h_i * X[i, j] * coef_phi
          H[phi_idx, j] <- H[phi_idx, j] + h_i * X[i, j] * coef_phi
          H[j, theta_idx] <- H[j, theta_idx] + h_i * X[i, j] * coef_theta
          H[theta_idx, j] <- H[theta_idx, j] + h_i * X[i, j] * coef_theta
        }
      }
    }
  }

  # Add prior precision for RE
  for (g in seq_len(n_re_groups)) {
    H[p + g, p + g] <- H[p + g, p + g] + tau_re
  }

  # Add ICAR precision for phi_scaled
  for (s in seq_len(n_spatial_units)) {
    phi_idx <- phi_start + s
    H[phi_idx, phi_idx] <- H[phi_idx, phi_idx] + n_neighbors[s]
    for (k in adj_row_ptr[s]:(adj_row_ptr[s + 1] - 1)) {
      neighbor <- adj_col_idx[k]
      nb_idx <- phi_start + neighbor
      H[phi_idx, nb_idx] <- H[phi_idx, nb_idx] - 1.0
    }
  }

  # Add N(0,1) prior precision for theta
  for (s in seq_len(n_spatial_units)) {
    theta_idx <- theta_start + s
    H[theta_idx, theta_idx] <- H[theta_idx, theta_idx] + 1.0
  }

  # Regularization for fixed effects
  for (j in seq_len(p)) {
    H[j, j] <- H[j, j] + 1e-4
  }

  list(H = H)
}


#' Convert BYM2 Laplace results to quotr_fit
#' @keywords internal
convert_laplace_bym2_to_quotr_fit <- function(samples, result, formula, data,
                                                family, X, re_info, spatial_info,
                                                sigma_spatial, rho, scale_factor,
                                                n_samples) {
  p <- ncol(X)
  n_re <- re_info$n_groups
  n_spatial <- spatial_info$n_units
  phi_start <- p + n_re
  theta_start <- phi_start + n_spatial

  sqrt_rho <- sqrt(rho + 1e-10)
  sqrt_1_rho <- sqrt(1.0 - rho + 1e-10)

  # Extract fixed effects samples
  beta_names <- colnames(X)
  if (is.null(beta_names)) {
    beta_names <- paste0("beta[", seq_len(p), "]")
  }

  draws_list <- list()
  for (j in seq_len(p)) {
    draws_list[[beta_names[j]]] <- samples[, j]
  }

  # Compute combined spatial effects u = sigma * (sqrt(rho) * phi * scale + sqrt(1-rho) * theta)
  if (n_spatial > 0) {
    for (s in seq_len(n_spatial)) {
      phi_s <- samples[, phi_start + s]
      theta_s <- samples[, theta_start + s]
      u_s <- sigma_spatial * (sqrt_rho * phi_s * scale_factor + sqrt_1_rho * theta_s)
      draws_list[[paste0("spatial[", s, "]")]] <- u_s
      draws_list[[paste0("phi_scaled[", s, "]")]] <- phi_s
      draws_list[[paste0("theta[", s, "]")]] <- theta_s
    }
  }

  # Add BYM2 hyperparameters as point estimates
  draws_list[["sigma_spatial"]] <- rep(sigma_spatial, n_samples)
  draws_list[["rho"]] <- rep(rho, n_samples)

  draws <- do.call(cbind, draws_list)
  colnames(draws) <- names(draws_list)

  fit <- list(
    draws = draws,
    formula = formula,
    data = data,
    family = family,
    backend = "laplace",
    n_save = n_samples,
    laplace_result = result,
    spatial_type = "bym2",
    .internal = list(
      mode = result$mode,
      log_marginal = result$log_marginal,
      X = X,
      re_info = re_info,
      spatial_info = spatial_info,
      sigma_spatial = sigma_spatial,
      rho = rho,
      scale_factor = scale_factor,
      samples = samples
    )
  )

  class(fit) <- "quotr_fit"
  return(fit)
}
