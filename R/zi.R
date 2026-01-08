#' Zero-Inflation and Hurdle Model Specifications
#'
#' @description
#' Functions to specify zero-inflation or hurdle model components for quotr models.
#' These handle excess zeros in count data where the standard count distribution
#' (Poisson, negative binomial) doesn't adequately capture the zero proportion.
#'
#' @details
#' **Zero-Inflated Models:**
#' Zero-inflated models assume two data-generating processes:
#' 1. A process that generates "structural zeros" with probability \eqn{\pi}
#' 2. A count process that generates counts (including zeros) with probability \eqn{1-\pi}
#'
#' The probability of observing y is:
#' \deqn{P(Y=0) = \pi + (1-\pi) \cdot P_{count}(0)}
#' \deqn{P(Y=y|y>0) = (1-\pi) \cdot P_{count}(y)}
#'
#' **Hurdle Models:**
#' Hurdle models treat the process as:
#' 1. A binary process determining zero vs non-zero
#' 2. A truncated count process (conditional on y > 0)
#'
#' \deqn{P(Y=0) = 1 - \theta}
#' \deqn{P(Y=y|y>0) = \theta \cdot P_{count}(y|y>0)}
#'
#' **When to use:**
#' - `zi_poisson()`: Excess zeros with equidispersed positive counts
#' - `zi_negbin()`: Excess zeros with overdispersed positive counts
#' - `hurdle_poisson()`: Zero hurdle with equidispersed positive counts
#' - `hurdle_negbin()`: Zero hurdle with overdispersed positive counts
#'
#' @name zi_specification
#' @keywords models
NULL


#' Create a zero-inflated Poisson specification
#'
#' @param formula Optional formula for the zero-inflation probability.
#'   If NULL, uses intercept-only model for P(structural zero).
#'
#' @return A `quotr_zi` object
#'
#' @examples
#' # Intercept-only ZI
#' zi_poisson()
#'
#' # ZI probability depends on a covariate
#' zi_poisson(~ habitat)
#'
#' @export
zi_poisson <- function(formula = NULL) {
  structure(
    list(
      type = "zi_poisson",
      formula = formula,
      distribution = "poisson"
    ),
    class = "quotr_zi"
  )
}


#' Create a zero-inflated negative binomial specification
#'
#' @param formula Optional formula for the zero-inflation probability.
#'   If NULL, uses intercept-only model for P(structural zero).
#'
#' @return A `quotr_zi` object
#'
#' @examples
#' # Intercept-only ZI
#' zi_negbin()
#'
#' # ZI probability depends on multiple covariates
#' zi_negbin(~ habitat + season)
#'
#' @export
zi_negbin <- function(formula = NULL) {
  structure(
    list(
      type = "zi_negbin",
      formula = formula,
      distribution = "negbin"
    ),
    class = "quotr_zi"
  )
}


#' Create a hurdle Poisson specification
#'
#' @param formula Optional formula for the hurdle probability (P(Y > 0)).
#'   If NULL, uses intercept-only model.
#'
#' @return A `quotr_zi` object
#'
#' @examples
#' # Intercept-only hurdle
#' hurdle_poisson()
#'
#' # Hurdle probability depends on effort
#' hurdle_poisson(~ effort)
#'
#' @export
hurdle_poisson <- function(formula = NULL) {
  structure(
    list(
      type = "hurdle_poisson",
      formula = formula,
      distribution = "poisson"
    ),
    class = "quotr_zi"
  )
}


#' Create a hurdle negative binomial specification
#'
#' @param formula Optional formula for the hurdle probability (P(Y > 0)).
#'   If NULL, uses intercept-only model.
#'
#' @return A `quotr_zi` object
#'
#' @examples
#' # Intercept-only hurdle
#' hurdle_negbin()
#'
#' # Hurdle probability depends on sampling effort
#' hurdle_negbin(~ log_effort)
#'
#' @export
hurdle_negbin <- function(formula = NULL) {
  structure(
    list(
      type = "hurdle_negbin",
      formula = formula,
      distribution = "negbin"
    ),
    class = "quotr_zi"
  )
}


#' Print method for quotr_zi objects
#'
#' @param x A quotr_zi object
#' @param ... Additional arguments (ignored)
#'
#' @return Invisibly returns the input
#' @export
print.quotr_zi <- function(x, ...) {
  type_label <- switch(x$type,
    "zi_poisson" = "Zero-Inflated Poisson",
    "zi_negbin" = "Zero-Inflated Negative Binomial",
    "hurdle_poisson" = "Hurdle Poisson",
    "hurdle_negbin" = "Hurdle Negative Binomial",
    x$type
  )

  cat("quotr Zero-Inflation Specification\n")
  cat("==================================\n")
  cat("Type:", type_label, "\n")

  if (is.null(x$formula)) {
    cat("Formula: ~ 1 (intercept only)\n")
  } else {
    cat("Formula:", deparse(x$formula), "\n")
  }

  invisible(x)
}


#' Validate ZI specification
#'
#' @param zi A quotr_zi object or NULL
#' @param data Data frame for checking formula variables
#'
#' @return Validated quotr_zi object or NULL
#' @keywords internal
validate_zi <- function(zi, data = NULL) {
  if (is.null(zi)) {
    return(NULL)
  }

  if (!inherits(zi, "quotr_zi")) {
    stop("zi must be a quotr_zi object (created by zi_poisson(), zi_negbin(), ",
         "hurdle_poisson(), or hurdle_negbin())", call. = FALSE)
  }

  valid_types <- c("zi_poisson", "zi_negbin", "hurdle_poisson", "hurdle_negbin")
  if (!zi$type %in% valid_types) {
    stop("Unknown ZI type: ", zi$type, call. = FALSE)
  }

  # Validate formula variables exist in data
  if (!is.null(zi$formula) && !is.null(data)) {
    formula_vars <- all.vars(zi$formula)
    missing_vars <- setdiff(formula_vars, names(data))
    if (length(missing_vars) > 0) {
      stop("ZI formula contains variables not in data: ",
           paste(missing_vars, collapse = ", "), call. = FALSE)
    }
  }

  zi
}
