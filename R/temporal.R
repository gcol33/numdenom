#' Temporal structure specifications for quotr
#'
#' @description
#' Functions to specify temporal random effects for quotr models.
#' Temporal effects are shared between numerator and denominator by default,
#' which helps prevent spurious ratio effects from temporally-structured
#' unmeasured confounders.
#'
#' @name quotr_temporal
NULL

#' RW1 temporal structure (First-order Random Walk)
#'
#' @description
#' Specify a first-order random walk temporal random effect.
#' RW1 assumes that the difference between consecutive time points is
#' normally distributed: `phi[t] - phi[t-1] ~ N(0, sigma^2)`.
#'
#' This is an intrinsic model (improper prior) that captures smooth
#' temporal trends. It is equivalent to an ICAR model on a 1D chain.
#'
#' @param time_var Name of the time variable in data (character).
#' @param group_var Optional name of grouping variable for panel data.
#'   If provided, separate random walks are estimated for each group.
#' @param cyclic Logical; if TRUE, the random walk wraps around (first and
#'   last time points are neighbors). Useful for seasonal data.
#' @param shared Logical; if TRUE (default), temporal effect enters both
#'   numerator and denominator linear predictors identically.
#'
#' @return A `quotr_temporal` object
#'
#' @details
#' The RW1 precision matrix Q has the form (for T time points):
#' \deqn{Q[t,t] = 1 \text{ for } t = 1, T}
#' \deqn{Q[t,t] = 2 \text{ for } 1 < t < T}
#' \deqn{Q[t,t-1] = Q[t-1,t] = -1}
#'
#' This is a rank-deficient matrix (rank T-1), so a sum-to-zero constraint
#' is applied for identifiability.
#'
#' @examples
#' \dontrun{
#' # Simple temporal trend
#' fit <- quotr(
#'   count | effort ~ x + temporal_rw1(year),
#'   data = df,
#'   family = quotr_poisson_gamma()
#' )
#'
#' # Panel data: separate trends per site
#' fit <- quotr(
#'   count | effort ~ x + (1 | site) + temporal_rw1(year, group = site),
#'   data = df,
#'   family = quotr_poisson_gamma()
#' )
#'
#' # Cyclic for monthly seasonal data
#' fit <- quotr(
#'   count | effort ~ x + temporal_rw1(month, cyclic = TRUE),
#'   data = df,
#'   family = quotr_poisson_gamma()
#' )
#' }
#'
#' @export
temporal_rw1 <- function(time_var, group_var = NULL, cyclic = FALSE,
                         shared = TRUE) {

 if (!is.character(time_var) || length(time_var) != 1) {
   stop("`time_var` must be a single character string", call. = FALSE)
 }

 if (!is.null(group_var)) {
   if (!is.character(group_var) || length(group_var) != 1) {
     stop("`group_var` must be a single character string", call. = FALSE)
   }
 }

 structure(
   list(
     type = "rw1",
     time_var = time_var,
     group_var = group_var,
     cyclic = cyclic,
     shared = shared,
     # These will be filled in when validated against data
     n_times = NULL,
     n_groups = NULL,
     time_index = NULL,
     group_index = NULL,
     precision_structure = NULL
   ),
   class = c("quotr_temporal", "list")
 )
}


#' RW2 temporal structure (Second-order Random Walk)
#'
#' @description
#' Specify a second-order random walk temporal random effect.
#' RW2 penalizes deviations from linearity, resulting in smoother trends
#' than RW1: `phi[t] - 2*phi[t-1] + phi[t-2] ~ N(0, sigma^2)`.
#'
#' @inheritParams temporal_rw1
#'
#' @return A `quotr_temporal` object
#'
#' @details
#' RW2 produces smoother curves than RW1 because it penalizes the second
#' derivative (curvature) rather than the first derivative (slope).
#' It requires at least 3 time points.
#'
#' The precision matrix is rank T-2 (two constraints needed).
#'
#' @examples
#' \dontrun{
#' # Smooth temporal trend
#' fit <- quotr(
#'   count | effort ~ x + temporal_rw2(year),
#'   data = df,
#'   family = quotr_poisson_gamma()
#' )
#' }
#'
#' @export
temporal_rw2 <- function(time_var, group_var = NULL, cyclic = FALSE,
                         shared = TRUE) {

 if (!is.character(time_var) || length(time_var) != 1) {
   stop("`time_var` must be a single character string", call. = FALSE)
 }

 if (!is.null(group_var)) {
   if (!is.character(group_var) || length(group_var) != 1) {
     stop("`group_var` must be a single character string", call. = FALSE)
   }
 }

 structure(
   list(
     type = "rw2",
     time_var = time_var,
     group_var = group_var,
     cyclic = cyclic,
     shared = shared,
     n_times = NULL,
     n_groups = NULL,
     time_index = NULL,
     group_index = NULL,
     precision_structure = NULL
   ),
   class = c("quotr_temporal", "list")
 )
}


#' AR1 temporal structure (First-order Autoregressive)
#'
#' @description
#' Specify a first-order autoregressive temporal random effect.
#' AR1 models temporal correlation where each time point depends on
#' the previous one: `phi[t] = rho * phi[t-1] + epsilon[t]`.
#'
#' Unlike RW1/RW2, AR1 is a proper (stationary) model with an estimated
#' autocorrelation parameter rho.
#'
#' @inheritParams temporal_rw1
#' @param rho_prior Prior for the autocorrelation parameter. Default is
#'   `NULL` which uses a Uniform(-1, 1) prior. Can specify a Beta prior
#'   on (rho+1)/2 for more informative priors.
#'
#' @return A `quotr_temporal` object
#'
#' @details
#' The AR1 process has marginal variance sigma^2 / (1 - rho^2) and
#' correlation between time points t and s of rho^|t-s|.
#'
#' The precision matrix is tridiagonal and full rank, so no constraints
#' are needed.
#'
#' @examples
#' \dontrun{
#' # AR1 temporal correlation
#' fit <- quotr(
#'   count | effort ~ x + temporal_ar1(year),
#'   data = df,
#'   family = quotr_poisson_gamma()
#' )
#'
#' # Panel data with group-specific AR1
#' fit <- quotr(
#'   count | effort ~ x + temporal_ar1(year, group = site),
#'   data = df,
#'   family = quotr_poisson_gamma()
#' )
#' }
#'
#' @export
temporal_ar1 <- function(time_var, group_var = NULL, shared = TRUE,
                         rho_prior = NULL) {

 if (!is.character(time_var) || length(time_var) != 1) {
   stop("`time_var` must be a single character string", call. = FALSE)
 }

 if (!is.null(group_var)) {
   if (!is.character(group_var) || length(group_var) != 1) {
     stop("`group_var` must be a single character string", call. = FALSE)
   }
 }

 structure(
   list(
     type = "ar1",
     time_var = time_var,
     group_var = group_var,
     cyclic = FALSE,  # AR1 is not cyclic
     shared = shared,
     rho_prior = rho_prior,
     n_times = NULL,
     n_groups = NULL,
     time_index = NULL,
     group_index = NULL,
     precision_structure = NULL
   ),
   class = c("quotr_temporal", "list")
 )
}


#' Print method for quotr_temporal
#'
#' @param x A quotr_temporal object
#' @param ... Ignored
#'
#' @export
print.quotr_temporal <- function(x, ...) {
 cat("quotr temporal specification\n")
 cat("============================\n\n")

 type_name <- switch(x$type,
   rw1 = "RW1 (First-order Random Walk)",
   rw2 = "RW2 (Second-order Random Walk)",
   ar1 = "AR1 (First-order Autoregressive)",
   x$type
 )

 cat("Type:", type_name, "\n")
 cat("Time variable:", x$time_var, "\n")

 if (!is.null(x$group_var)) {
   cat("Group variable:", x$group_var, "\n")
 }

 if (x$type %in% c("rw1", "rw2") && x$cyclic) {
   cat("Cyclic: Yes\n")
 }

 cat("Shared:", if (x$shared) "Yes (enters both processes)" else "No", "\n")

 if (!is.null(x$n_times)) {
   cat("Time points:", x$n_times, "\n")
 }
 if (!is.null(x$n_groups) && x$n_groups > 1) {
   cat("Groups:", x$n_groups, "\n")
 }

 invisible(x)
}


#' Validate temporal specification against data
#'
#' @param temporal quotr_temporal object
#' @param data Data frame
#'
#' @return Updated quotr_temporal object with indices computed
#' @keywords internal
validate_temporal <- function(temporal, data) {
 if (is.null(temporal)) return(NULL)

 # Check time variable exists
 if (!(temporal$time_var %in% names(data))) {
   stop(sprintf("Temporal variable '%s' not found in data",
                temporal$time_var), call. = FALSE)
 }

 # Check group variable exists if specified
 if (!is.null(temporal$group_var)) {
   if (!(temporal$group_var %in% names(data))) {
     stop(sprintf("Temporal group variable '%s' not found in data",
                  temporal$group_var), call. = FALSE)
   }
 }

 # Get time values and create indices
 time_vals <- data[[temporal$time_var]]

 # Convert to factor to get consistent indexing
 if (is.factor(time_vals)) {
   time_factor <- time_vals
 } else {
   # Sort unique values to ensure temporal ordering
   unique_times <- sort(unique(time_vals))
   time_factor <- factor(time_vals, levels = unique_times)
 }

 temporal$n_times <- nlevels(time_factor)
 temporal$time_index <- as.integer(time_factor)
 temporal$time_levels <- levels(time_factor)

 # Check minimum time points for RW2
 if (temporal$type == "rw2" && temporal$n_times < 3) {
   stop("RW2 requires at least 3 time points", call. = FALSE)
 }

 # Handle grouping
 if (!is.null(temporal$group_var)) {
   group_vals <- data[[temporal$group_var]]
   group_factor <- as.factor(group_vals)
   temporal$n_groups <- nlevels(group_factor)
   temporal$group_index <- as.integer(group_factor)
   temporal$group_levels <- levels(group_factor)

   # Total temporal parameters = n_times * n_groups
   temporal$n_temporal_params <- temporal$n_times * temporal$n_groups
 } else {
   temporal$n_groups <- 1L
   temporal$group_index <- rep(1L, nrow(data))
   temporal$n_temporal_params <- temporal$n_times
 }

 # Build precision structure
 temporal$precision_structure <- build_temporal_precision_structure(temporal)

 temporal
}


#' Build temporal precision matrix structure
#'
#' @param temporal Validated quotr_temporal object
#'
#' @return List with precision matrix information
#' @keywords internal
build_temporal_precision_structure <- function(temporal) {
 T <- temporal$n_times
 cyclic <- temporal$cyclic

 if (temporal$type == "rw1") {
   # RW1 precision matrix (tridiagonal)
   # Q[t,t] = 2 (interior), 1 (boundary)
   # Q[t,t-1] = Q[t,t+1] = -1

   if (cyclic) {
     # Cyclic: all diagonal = 2, wrap around
     diag_vals <- rep(2, T)
     off_diag <- rep(-1, T)  # Including wrap-around
   } else {
     # Non-cyclic: boundary diagonal = 1
     diag_vals <- c(1, rep(2, T - 2), 1)
     off_diag <- rep(-1, T - 1)
   }

   list(
     type = "rw1",
     T = T,
     cyclic = cyclic,
     diag = diag_vals,
     off_diag = off_diag,
     rank_deficiency = if (cyclic) 1 else 1  # Always rank T-1
   )

 } else if (temporal$type == "rw2") {
   # RW2 precision matrix (pentadiagonal)
   # Second differences

   if (cyclic) {
     diag_vals <- rep(6, T)
     off_diag_1 <- rep(-4, T)  # First off-diagonal
     off_diag_2 <- rep(1, T)   # Second off-diagonal
   } else {
     # Non-cyclic boundary handling
     diag_vals <- c(1, 5, rep(6, T - 4), 5, 1)
     if (T == 3) diag_vals <- c(1, 2, 1)
     if (T == 4) diag_vals <- c(1, 5, 5, 1)
     off_diag_1 <- c(-2, rep(-4, T - 3), -2)
     if (T == 3) off_diag_1 <- c(-2, -2)
     off_diag_2 <- rep(1, T - 2)
   }

   list(
     type = "rw2",
     T = T,
     cyclic = cyclic,
     diag = diag_vals,
     off_diag_1 = off_diag_1,
     off_diag_2 = off_diag_2,
     rank_deficiency = if (cyclic) 2 else 2  # Rank T-2
   )

 } else if (temporal$type == "ar1") {
   # AR1 precision matrix (tridiagonal, full rank)
   # Depends on rho, so just store structure

   list(
     type = "ar1",
     T = T,
     # Full precision matrix constructed at runtime with rho
     rank_deficiency = 0  # Full rank
   )
 }
}


#' Build RW1 precision matrix
#'
#' @param T Number of time points
#' @param tau Precision parameter (1/sigma^2)
#' @param cyclic Whether to use cyclic boundary
#'
#' @return Sparse precision matrix
#' @keywords internal
build_rw1_precision <- function(T, tau = 1, cyclic = FALSE) {
 if (T < 2) {
   stop("RW1 requires at least 2 time points", call. = FALSE)
 }

 # Build as dense then convert (small matrices anyway)
 Q <- matrix(0, T, T)

 if (cyclic) {
   for (t in 1:T) {
     Q[t, t] <- 2
     next_t <- if (t == T) 1 else t + 1
     prev_t <- if (t == 1) T else t - 1
     Q[t, next_t] <- -1
     Q[t, prev_t] <- -1
   }
 } else {
   # Non-cyclic
   Q[1, 1] <- 1
   Q[1, 2] <- -1
   Q[T, T] <- 1
   Q[T, T - 1] <- -1

   if (T > 2) {
     for (t in 2:(T - 1)) {
       Q[t, t] <- 2
       Q[t, t - 1] <- -1
       Q[t, t + 1] <- -1
     }
   }
 }

 Q * tau
}


#' Build AR1 precision matrix
#'
#' @param T Number of time points
#' @param rho Autocorrelation parameter (-1 < rho < 1)
#' @param tau Marginal precision (1/sigma^2)
#'
#' @return Precision matrix
#' @keywords internal
build_ar1_precision <- function(T, rho, tau = 1) {
 if (T < 2) {
   stop("AR1 requires at least 2 time points", call. = FALSE)
 }

 if (abs(rho) >= 1) {
   stop("rho must be in (-1, 1) for stationarity", call. = FALSE)
 }

 # AR1 precision matrix
 # Q = tau * (1 - rho^2)^(-1) * tridiag structure
 # Diagonal: 1 + rho^2 (interior), 1 (boundary)
 # Off-diagonal: -rho

 Q <- matrix(0, T, T)

 # Boundary
 Q[1, 1] <- 1
 Q[T, T] <- 1

 # Interior
 if (T > 2) {
   for (t in 2:(T - 1)) {
     Q[t, t] <- 1 + rho^2
   }
 }

 # Off-diagonal
 for (t in 1:(T - 1)) {
   Q[t, t + 1] <- -rho
   Q[t + 1, t] <- -rho
 }

 # Scale by precision / (1 - rho^2)
 # The conditional variance is sigma^2 * (1 - rho^2) for interior
 # So precision matrix should give marginal precision tau
 Q * tau / (1 - rho^2)
}
