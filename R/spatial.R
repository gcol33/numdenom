#' Spatial structure specifications for quotr
#'
#' @description
#' Functions to specify spatial random effects for quotr models.
#' Spatial effects are shared between numerator and denominator by default,
#' which helps prevent spurious ratio effects from spatially-structured
#' unmeasured confounders.
#'
#' @name quotr_spatial
NULL

#' CAR spatial structure
#'
#' @description
#' Specify a conditional autoregressive (CAR) spatial random effect.
#' Uses the intrinsic CAR (ICAR) prior, which assumes neighboring areas
#' have similar values.
#'
#' @param adjacency Adjacency matrix (sparse or dense). A symmetric matrix
#'   where entry (i,j) is 1 if areas i and j are neighbors, 0 otherwise.
#' @param level Level at which spatial structure applies:
#'   - "group": Spatial effect at the grouping variable level (e.g., sites).
#'     Requires `group_var` to be specified.
#'   - "obs": Spatial effect at the observation level.
#' @param group_var Name of the grouping variable in data (required if
#'   `level = "group"`).
#' @param shared Logical; if TRUE (default), spatial effect enters both
#'   numerator and denominator linear predictors identically.
#'
#' @return A `quotr_spatial` object
#'
#' @examples
#' \dontrun{
#' # Create adjacency matrix for 10 sites
#' adj <- matrix(0, 10, 10)
#' adj[1, 2] <- adj[2, 1] <- 1
#' adj[2, 3] <- adj[3, 2] <- 1
#' # ... etc
#'
#' # Group-level spatial (sites with replication within)
#' fit <- quotr(
#'   numerator = count ~ x + (1 | site),
#'   denominator = effort ~ (1 | site),
#'   shared = ~ (1 | site),
#'   spatial = spatial_car(adj, level = "group", group_var = "site"),
#'   data = df,
#'   family = quotr_negbin_negbin()
#' )
#' }
#'
#' @export
spatial_car <- function(adjacency, level = c("group", "obs"),
                        group_var = NULL, shared = TRUE) {

  level <- match.arg(level)

  # Validate adjacency matrix
  if (!is.matrix(adjacency) && !inherits(adjacency, "Matrix")) {
    stop("`adjacency` must be a matrix", call. = FALSE)
  }

  if (nrow(adjacency) != ncol(adjacency)) {
    stop("`adjacency` must be square", call. = FALSE)
  }

  # Check symmetry
  if (!isSymmetric(unname(as.matrix(adjacency)))) {
    stop("`adjacency` must be symmetric", call. = FALSE)
  }

  # Check for group_var if level = "group"
  if (level == "group" && is.null(group_var)) {
    stop("`group_var` is required when level = 'group'", call. = FALSE)
  }

  structure(
    list(
      type = "car",
      adjacency = adjacency,
      level = level,
      group_var = group_var,
      shared = shared,
      n_spatial = nrow(adjacency)
    ),
    class = c("quotr_spatial", "list")
  )
}

#' BYM2 spatial structure
#'
#' @description
#' Specify a Besag-York-Mollie 2 (BYM2) spatial random effect.
#' BYM2 decomposes the spatial effect into a structured (ICAR) component
#' and an unstructured (IID) component, with a mixing parameter controlling
#' the proportion of variance attributable to spatial structure.
#'
#' BYM2 is preferred over plain CAR when you want to:
#' - Distinguish structured vs unstructured spatial variation
#' - Have an interpretable spatial fraction parameter
#' - Use the scaling from Riebler et al. (2016)
#'
#' @inheritParams spatial_car
#' @param scale_factor Scaling factor for the ICAR component. If NULL
#'   (default), computed from the adjacency matrix following Riebler et al.
#'
#' @return A `quotr_spatial` object
#'
#' @references
#' Riebler, A., Sorbye, S. H., Simpson, D., & Rue, H. (2016). An intuitive
#' Bayesian spatial model for disease mapping that accounts for scaling.
#' Statistical Methods in Medical Research, 25(4), 1145-1165.
#'
#' @examples
#' \dontrun{
#' fit <- quotr(
#'   numerator = cases ~ age + (1 | region),
#'   denominator = population ~ (1 | region),
#'   shared = ~ (1 | region),
#'   spatial = spatial_bym2(adj, level = "group", group_var = "region"),
#'   data = epi_data,
#'   family = quotr_binomial()
#' )
#' }
#'
#' @export
spatial_bym2 <- function(adjacency, level = c("group", "obs"),
                         group_var = NULL, shared = TRUE,
                         scale_factor = NULL) {

  level <- match.arg(level)

  # Validate adjacency matrix
  if (!is.matrix(adjacency) && !inherits(adjacency, "Matrix")) {
    stop("`adjacency` must be a matrix", call. = FALSE)
  }

  if (nrow(adjacency) != ncol(adjacency)) {
    stop("`adjacency` must be square", call. = FALSE)
  }

  if (!isSymmetric(unname(as.matrix(adjacency)))) {
    stop("`adjacency` must be symmetric", call. = FALSE)
  }

  if (level == "group" && is.null(group_var)) {
    stop("`group_var` is required when level = 'group'", call. = FALSE)
  }

  # Compute scale factor if not provided
  if (is.null(scale_factor)) {
    scale_factor <- compute_bym2_scale(adjacency)
  }

  structure(
    list(
      type = "bym2",
      adjacency = adjacency,
      level = level,
      group_var = group_var,
      shared = shared,
      n_spatial = nrow(adjacency),
      scale_factor = scale_factor
    ),
    class = c("quotr_spatial", "list")
  )
}

#' Compute BYM2 scaling factor
#'
#' @description
#' Compute the scaling factor for BYM2 following Riebler et al. (2016).
#' This makes the spatial fraction parameter interpretable.
#'
#' @param adjacency Adjacency matrix
#'
#' @return Scaling factor (scalar)
#' @keywords internal
compute_bym2_scale <- function(adjacency) {
  # Build precision matrix Q for ICAR
  n <- nrow(adjacency)
  adj <- as.matrix(adjacency)
  diag(adj) <- 0

  # Number of neighbors for each area
  n_neighbors <- rowSums(adj)

  # ICAR precision matrix: Q_ii = n_neighbors[i], Q_ij = -1 if neighbors
  Q <- diag(n_neighbors) - adj

  # Compute generalized inverse (Q is rank-deficient)
  # Use eigendecomposition
  eig <- eigen(Q, symmetric = TRUE)

  # Remove the zero eigenvalue (rank deficiency)
  non_zero <- abs(eig$values) > 1e-10
  lambda <- eig$values[non_zero]

  # Geometric mean of non-zero eigenvalues
  # This is the scaling factor following INLA convention
  scale <- exp(mean(log(lambda)))

  scale
}

#' Print method for quotr_spatial
#'
#' @param x A quotr_spatial object
#' @param ... Ignored
#'
#' @export
print.quotr_spatial <- function(x, ...) {
  cat("quotr spatial specification\n")
  cat("===========================\n\n")
  cat("Type:", toupper(x$type), "\n")
  cat("Level:", x$level, "\n")
  cat("Spatial units:", x$n_spatial, "\n")
  cat("Shared:", if (x$shared) "Yes (enters both processes)" else "No", "\n")
  if (!is.null(x$group_var)) {
    cat("Group variable:", x$group_var, "\n")
  }
  if (x$type == "bym2") {
    cat("Scale factor:", round(x$scale_factor, 4), "\n")
  }
  invisible(x)
}

#' Check if adjacency matrix is connected
#'
#' @description
#' Check if the spatial graph defined by the adjacency matrix is fully
#' connected. A disconnected graph can cause identifiability issues.
#'
#' @param adjacency Adjacency matrix
#'
#' @return Logical; TRUE if connected
#' @keywords internal
is_connected <- function(adjacency) {
  n <- nrow(adjacency)
  if (n == 0) return(TRUE)

  adj <- as.matrix(adjacency)
  diag(adj) <- 0

  # BFS to check connectivity
  visited <- logical(n)
  queue <- 1L
  visited[1] <- TRUE

  while (length(queue) > 0) {
    current <- queue[1]
    queue <- queue[-1]

    neighbors <- which(adj[current, ] > 0)
    new_neighbors <- neighbors[!visited[neighbors]]

    visited[new_neighbors] <- TRUE
    queue <- c(queue, new_neighbors)
  }

  all(visited)
}

#' Validate spatial specification against data
#'
#' @param spatial quotr_spatial object
#' @param data Data frame
#'
#' @return NULL (invisibly); errors if validation fails
#' @keywords internal
validate_spatial <- function(spatial, data) {
  if (is.null(spatial)) return(invisible(NULL))

  # Check group variable exists
  if (spatial$level == "group") {
    if (!(spatial$group_var %in% names(data))) {
      stop(sprintf("Spatial group variable '%s' not found in data",
                   spatial$group_var), call. = FALSE)
    }

    # Check number of groups matches adjacency matrix
    n_groups <- length(unique(data[[spatial$group_var]]))
    if (n_groups != spatial$n_spatial) {
      stop(sprintf(
        "Number of groups in data (%d) does not match adjacency matrix (%d)",
        n_groups, spatial$n_spatial
      ), call. = FALSE)
    }
  } else {
    # Observation-level spatial
    if (nrow(data) != spatial$n_spatial) {
      stop(sprintf(
        "Number of observations (%d) does not match adjacency matrix (%d)",
        nrow(data), spatial$n_spatial
      ), call. = FALSE)
    }
  }

  # Check connectivity
  if (!is_connected(spatial$adjacency)) {
    warning(
      "Spatial adjacency graph is not fully connected.\n",
      "This may cause identifiability issues. Consider:\n",
      "  - Adding edges to connect isolated components\n",
      "  - Fitting separate models for each connected component",
      call. = FALSE
    )
  }

  invisible(NULL)
}
