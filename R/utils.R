#' Null-coalescing operator
#'
#' @description
#' Returns `a` unless it is `NULL`, in which case it returns `b`. Used
#' throughout the package to fall back on a default when an optional list
#' element is absent. Defined here because the base R version arrived in
#' 4.4.0 and the package supports R (>= 4.1.0).
#'
#' @param a Value to return when it is not `NULL`.
#' @param b Fallback returned when `a` is `NULL`.
#'
#' @return `a` if it is not `NULL`, otherwise `b`.
#'
#' @keywords internal
#' @noRd
`%||%` <- function(a, b) if (is.null(a)) b else a
