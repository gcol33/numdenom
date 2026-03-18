#' Gibbs Sampling Backend for Spatial Models
#'
#' @description
#' Component-wise Gibbs sampler for ICAR and BYM2 spatial models. Uses univariate
#' MH updates for spatial effects with the ICAR conditional prior as proposal,
#' conjugate Gamma update for spatial precision, and block MH for fixed effects.
#'
#' @details
#' This backend is designed for models where spatial effects dominate the
#' parameter space (large S). Unlike HMC, the Gibbs sampler updates spatial
#' effects one at a time, avoiding the curse of dimensionality.
#'
#' **Supported families:**
#' - `ratiod_poisson_gamma()` (Poisson numerator, Gamma denominator)
#' - `ratiod_negbin_negbin()` (NegBin numerator, NegBin denominator)
#' - `ratiod_binomial()` (Binomial)
#' - `ratiod_negbin_gamma()` (NegBin numerator, Gamma denominator)
#'
#' **Supported spatial structures:**
#' - ICAR (intrinsic CAR) via `spatial_car()`
#' - BYM2 via `spatial_bym2()`
#'
#' @name gibbs_backend
#' @keywords internal
NULL


#' Fit ratiod model using Gibbs sampling
#'
#' @param formula A ratiod_formula object
#' @param data Data frame
#' @param family Model family
#' @param spatial Spatial structure from spatial_car()
#' @param iter Total iterations
#' @param warmup Warmup iterations
#' @param thin Thinning interval
#' @param seed Random seed
#' @param verbose Print progress
#'
#' @return A ratiod_fit object
#' @keywords internal
fit_gibbs <- function(formula,
                      data,
                      family,
                      spatial,
                      temporal = NULL,
                      iter = 2000,
                      warmup = floor(iter / 2),
                      thin = 1,
                      seed = NULL,
                      verbose = TRUE) {

  if (is.null(spatial)) {
    stop("Gibbs backend requires spatial structure. Use spatial_car().",
         call. = FALSE)
  }

  if (!is.null(spatial$type) && !spatial$type %in% c("icar", "car", "bym2")) {
    stop(sprintf("Gibbs backend supports ICAR and BYM2 spatial only, got '%s'.",
                 spatial$type), call. = FALSE)
  }

  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1)
  }

  # Map family to Gibbs family string
  family_str <- get_gibbs_family_string(family)

  # Get spatial group mapping
  spatial_var <- spatial$group_var
  if (is.null(spatial_var)) {
    stop("Gibbs backend requires group-level spatial effects (group_var).",
         call. = FALSE)
  }

  group_factor <- as.factor(data[[spatial_var]])
  spatial_group <- as.integer(group_factor)  # 1-based

  S <- nlevels(group_factor)
  N <- nrow(data)

  # Build design matrices from formula
  X_num <- formula$numerator$X
  p_num <- ncol(X_num)

  is_binomial <- (family_str == "binomial")
  if (is_binomial) {
    X_denom <- matrix(0, nrow = 0, ncol = 0)
    p_denom <- 0L
  } else {
    X_denom <- formula$denominator$X
    p_denom <- ncol(X_denom)
  }

  # Extract responses
  y_num <- as.integer(formula$numerator$response)

  # Denominator response depends on family
  y_denom <- NULL
  y_denom_cont <- NULL

  if (family_str == "binomial") {
    y_denom <- as.integer(formula$denominator$response)
  } else if (family_str == "negbin_negbin") {
    y_denom <- as.integer(formula$denominator$response)
  } else {
    # poisson_gamma, negbin_gamma: continuous denominator
    y_denom_cont <- as.numeric(formula$denominator$response)
  }

  # Build adjacency in CSR format (1-based for R, C++ converts to 0-based)
  adj_matrix <- spatial$adjacency
  adj_row_ptr <- integer(S + 1)
  adj_col_idx <- integer(0)
  n_neighbors <- integer(S)

  for (i in seq_len(S)) {
    neighbors <- which(adj_matrix[i, ] > 0)
    n_neighbors[i] <- length(neighbors)
    adj_col_idx <- c(adj_col_idx, neighbors)  # 1-based
    adj_row_ptr[i + 1] <- adj_row_ptr[i] + length(neighbors)
  }

  # Build data list for C++
  data_list <- list(
    y_num = y_num,
    X_num = as.numeric(t(X_num)),  # Row-major flat vector
    spatial_group = spatial_group,  # 1-based (C++ converts)
    adj_row_ptr = adj_row_ptr,     # Already 0-indexed start
    adj_col_idx = as.integer(adj_col_idx),  # 1-based (C++ converts)
    n_neighbors = n_neighbors,
    N = as.integer(N),
    S = as.integer(S),
    p_num = as.integer(p_num),
    p_denom = as.integer(p_denom),
    n_iter = as.integer(iter),
    n_warmup = as.integer(warmup),
    thin = as.integer(thin),
    seed = as.integer(seed %% .Machine$integer.max),
    verbose = verbose,
    family = family_str
  )

  # Add denominator data
  if (!is_binomial) {
    data_list$X_denom <- as.numeric(t(X_denom))  # Row-major flat vector
  }

  if (family_str == "negbin_negbin" || family_str == "binomial") {
    data_list$y_denom <- as.integer(y_denom)
  } else {
    data_list$y_denom_cont <- as.numeric(y_denom_cont)
  }

  # BYM2 settings
  is_bym2 <- !is.null(spatial$type) && tolower(spatial$type) == "bym2"
  data_list$is_bym2 <- is_bym2
  if (is_bym2) {
    data_list$bym2_scale <- spatial$scale_factor %||% 1.0
  }

  # TVC settings
  has_tvc <- !is.null(temporal) && inherits(temporal, "ratiod_tvc")
  data_list$has_tvc <- has_tvc
  if (has_tvc) {
    tvc <- validate_tvc(temporal, data, X_num)
    data_list$tvc_time_index <- as.integer(tvc$time_index)
    data_list$tvc_group_index <- as.integer(tvc$group_index)
    data_list$tvc_X <- as.numeric(t(tvc$X_tvc))  # Row-major
    data_list$tvc_n_times <- as.integer(tvc$n_times)
    data_list$tvc_n_groups <- as.integer(tvc$n_groups)
    data_list$tvc_n_terms <- as.integer(tvc$n_tvc)
    data_list$tvc_shared <- tvc$shared %||% TRUE
    data_list$tvc_structure <- tvc$structure %||% "rw1"
  }

  # Call C++
  cpp_result <- cpp_gibbs_spatial(data_list)

  # Build ratiod_fit object
  n_save <- cpp_result$n_save
  n_params <- cpp_result$n_params

  # Draws matrix: C++ stores row-major but Rcpp::NumericMatrix fills column-major
  # Reconstruct with byrow=TRUE to fix the layout
  draws_mat <- matrix(as.numeric(cpp_result$draws),
                      nrow = n_save, ncol = n_params, byrow = TRUE)
  param_names <- as.character(cpp_result$param_names)
  colnames(draws_mat) <- param_names

  # Phi draws: same row-major issue
  phi_draws <- matrix(as.numeric(cpp_result$phi_draws),
                      nrow = n_save, ncol = cpp_result$S, byrow = TRUE)

  # Build 3D draws array [iteration, chain, variable] for posterior compatibility
  draws_array <- array(draws_mat,
                       dim = c(n_save, 1L, n_params),
                       dimnames = list(
                         iteration = seq_len(n_save),
                         chain = 1L,
                         variable = param_names
                       ))

  # Compute eta for ratio extraction
  # eta_num[i,draw] = phi[spatial_group[i],draw] + X_num[i,] %*% beta_num[draw,]
  # For each draw, compute eta_num and eta_denom per observation
  beta_num_cols <- seq_len(p_num)
  beta_denom_cols <- if (p_denom > 0) (p_num + 1):(p_num + p_denom) else integer(0)

  eta_num <- matrix(NA_real_, nrow = n_save, ncol = N)
  eta_denom <- matrix(NA_real_, nrow = n_save, ncol = N)

  for (draw in seq_len(n_save)) {
    beta_n <- draws_mat[draw, beta_num_cols]
    beta_d <- if (p_denom > 0) draws_mat[draw, beta_denom_cols] else numeric(0)
    phi_vec <- phi_draws[draw, ]

    for (i in seq_len(N)) {
      s <- spatial_group[i]
      eta_num[draw, i] <- phi_vec[s] + sum(X_num[i, ] * beta_n)
      if (is_binomial) {
        eta_denom[draw, i] <- 0.0
      } else {
        eta_denom[draw, i] <- phi_vec[s] + sum(X_denom[i, ] * beta_d)
      }
    }
  }

  # Build ratiod_fit
  fit <- list(
    draws = draws_array,
    formula = formula,
    data = data,
    family = family,
    spatial = spatial,
    backend = "gibbs",
    iter = iter,
    warmup = warmup,
    thin = thin,
    chains = 1L,
    n_save = n_save,
    n_save_per_chain = n_save,
    .internal = list(
      eta_num = eta_num,
      eta_denom = eta_denom,
      eta = eta_num,  # For methods expecting single eta
      eta_array = array(eta_num, dim = c(n_save, 1L, N)),
      phi_draws = phi_draws,
      X = X_num,
      accept_phi = cpp_result$accept_phi,
      accept_beta = cpp_result$accept_beta,
      accept_disp = cpp_result$accept_disp
    )
  )

  class(fit) <- "ratiod_fit"
  return(fit)
}


#' Map family to Gibbs family string
#' @keywords internal
get_gibbs_family_string <- function(family) {
  if (identical(family$name, "negbin_gamma")) return("negbin_gamma")

  dist <- family$numerator$distribution
  if (dist == "binomial") return("binomial")
  if (dist %in% c("negbin", "negative_binomial", "neg_binomial_2")) return("negbin_negbin")
  if (dist == "poisson") return("poisson_gamma")

  stop(sprintf("Gibbs backend does not support family '%s'. Use mode='hmc'.",
               dist), call. = FALSE)
}
