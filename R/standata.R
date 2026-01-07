#' Prepare data for Stan model
#'
#' @description
#' Convert a quotr_formula object and family specification into the
#' data list required by the Stan model.
#'
#' @param formula A quotr_formula object
#' @param family A quotr_family object
#' @param data Original data frame
#' @param spatial Optional spatial specification
#' @param priors Prior specification
#'
#' @return A named list suitable for Stan
#' @keywords internal
make_standata <- function(formula, family, data, spatial = NULL, priors = NULL) {

  # Use default priors if not specified
  if (is.null(priors)) {
    priors <- quotr_priors()
  }

  # Basic dimensions
  N <- nrow(data)

  # Fixed effects
  X_num <- formula$numerator$X
  X_denom <- formula$denominator$X
  K_num <- ncol(X_num)
  K_denom <- ncol(X_denom)

  # Responses
  y_num <- formula$numerator$response
  y_denom <- formula$denominator$response

  # Validate responses for family
  validate_response(y_num, family$numerator$distribution, "numerator")
  validate_response(y_denom, family$denominator$distribution, "denominator")

  # Build random effects structure
  re_data <- build_re_structure(formula, data)

  # Build spatial structure
  spatial_data <- build_spatial_structure(spatial, data)

  # Combine into Stan data list
  stan_data <- list(
    N = N,
    K_num = K_num,
    K_denom = K_denom,
    y_num = as.integer(y_num),
    y_denom = as.integer(y_denom),
    X_num = X_num,
    X_denom = X_denom,

    # Random effects
    n_re_groups = re_data$n_groups,
    n_re_total = re_data$n_total,
    re_idx = re_data$idx,
    re_group_size = re_data$group_size,
    re_group_start = re_data$group_start,
    re_shared = re_data$shared,

    # Spatial
    use_spatial = spatial_data$use_spatial,
    n_spatial = spatial_data$n_spatial,
    n_edges = spatial_data$n_edges,
    node1 = spatial_data$node1,
    node2 = spatial_data$node2,
    spatial_idx = spatial_data$spatial_idx,

    # Priors
    prior_sigma_U = priors$sigma_U,
    prior_sigma_alpha = priors$sigma_alpha,
    prior_phi_U = priors$phi_U,
    prior_phi_alpha = priors$phi_alpha
  )

  stan_data
}

#' Validate response variable for distribution
#'
#' @param y Response vector
#' @param dist Distribution name
#' @param name Name for error messages
#' @keywords internal
validate_response <- function(y, dist, name) {
  if (dist %in% c("poisson", "neg_binomial_2", "binomial")) {
    # Must be non-negative integers
    if (!is.numeric(y)) {
      stop(sprintf("%s response must be numeric", name), call. = FALSE)
    }
    if (any(y < 0, na.rm = TRUE)) {
      stop(sprintf("%s response must be non-negative", name), call. = FALSE)
    }
    if (any(y != floor(y), na.rm = TRUE)) {
      stop(sprintf("%s response must be integer counts", name), call. = FALSE)
    }
  } else if (dist == "gamma") {
    # Must be positive
    if (!is.numeric(y)) {
      stop(sprintf("%s response must be numeric", name), call. = FALSE)
    }
    if (any(y <= 0, na.rm = TRUE)) {
      stop(sprintf("%s response must be positive", name), call. = FALSE)
    }
  }
}

#' Build random effects data structure for Stan
#'
#' @param formula quotr_formula object
#' @param data Data frame
#'
#' @return List with RE indexing for Stan
#' @keywords internal
build_re_structure <- function(formula, data) {
  N <- nrow(data)

  # Collect all random effects from numerator, denominator, and shared
  all_re <- list()
  shared_flags <- integer()

  # Shared random effects take priority
  if (length(formula$shared$random_effects) > 0) {
    for (re in formula$shared$random_effects) {
      all_re[[length(all_re) + 1]] <- re
      shared_flags <- c(shared_flags, 1L)
    }
  }

  # Add process-specific random effects (not already in shared)
  shared_groups <- vapply(
    formula$shared$random_effects,
    `[[`, character(1), "group_var"
  )

  for (re in formula$numerator$random_effects) {
    if (!(re$group_var %in% shared_groups)) {
      all_re[[length(all_re) + 1]] <- re
      shared_flags <- c(shared_flags, 0L)
    }
  }

  for (re in formula$denominator$random_effects) {
    if (!(re$group_var %in% shared_groups) &&
        !(re$group_var %in% vapply(all_re, `[[`, character(1), "group_var"))) {
      all_re[[length(all_re) + 1]] <- re
      shared_flags <- c(shared_flags, 0L)
    }
  }

  n_groups <- length(all_re)

  if (n_groups == 0) {
    # No random effects
    return(list(
      n_groups = 0L,
      n_total = 0L,
      idx = matrix(0L, nrow = N, ncol = 1),
      group_size = integer(0),
      group_start = integer(0),
      shared = integer(0)
    ))
  }

  # Build index matrix
  group_sizes <- vapply(all_re, `[[`, integer(1), "n_groups")
  group_starts <- c(1L, cumsum(group_sizes[-length(group_sizes)]) + 1L)
  n_total <- sum(group_sizes)

  idx_matrix <- matrix(0L, nrow = N, ncol = n_groups)

  for (g in seq_len(n_groups)) {
    re <- all_re[[g]]
    idx_matrix[, g] <- group_starts[g] + re$group - 1L
  }

  list(
    n_groups = n_groups,
    n_total = n_total,
    idx = idx_matrix,
    group_size = group_sizes,
    group_start = group_starts,
    shared = shared_flags
  )
}

#' Build spatial data structure for Stan
#'
#' @param spatial Spatial specification or NULL
#' @param data Data frame
#'
#' @return List with spatial data for Stan
#' @keywords internal
build_spatial_structure <- function(spatial, data) {
  if (is.null(spatial)) {
    return(list(
      use_spatial = 0L,
      n_spatial = 0L,
      n_edges = 0L,
      node1 = integer(0),
      node2 = integer(0),
      spatial_idx = integer(0)
    ))
  }

  # Extract adjacency matrix
  adj <- spatial$adjacency
  n_spatial <- nrow(adj)

  # Convert to edge list for Stan ICAR
  edge_list <- which(adj > 0, arr.ind = TRUE)
  edge_list <- edge_list[edge_list[, 1] < edge_list[, 2], , drop = FALSE]
  n_edges <- nrow(edge_list)

  # Get spatial index for each observation
  if (spatial$level == "group") {
    # Spatial at group level - need group variable
    group_var <- spatial$group_var
    if (is.null(group_var) || !(group_var %in% names(data))) {
      stop("spatial$group_var must be specified for group-level spatial", call. = FALSE)
    }
    spatial_idx <- as.integer(as.factor(data[[group_var]]))
  } else {
    # Observation-level spatial
    spatial_idx <- seq_len(nrow(data))
  }

  list(
    use_spatial = 1L,
    n_spatial = n_spatial,
    n_edges = n_edges,
    node1 = as.integer(edge_list[, 1]),
    node2 = as.integer(edge_list[, 2]),
    spatial_idx = spatial_idx
  )
}
