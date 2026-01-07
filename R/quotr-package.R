#' @keywords internal
"_PACKAGE"

#' quotr: Bayesian Hierarchical Models for Ratios, Rates, and Proportions
#'
#' @description
#' An opinionated framework for Bayesian hierarchical modelling of ratios,
#' rates, and proportions. The core premise is non-negotiable:
#'
#' **Ratios are not data. They are derived quantities.**
#'
#' All inference is performed on the latent processes generating the numerator

#' and the denominator, never on their quotient. Ratios, rates, and proportions
#' are computed post hoc with full uncertainty propagation.
#'
#' @section Model families:
#' \describe{
#'   \item{quotr_binomial()}{Trial-based ratios (successes/trials)}
#'   \item{quotr_negbin_negbin()}{Two-process ratios (count/count)}
#'   \item{quotr_poisson_gamma()}{Count/effort ratios (e.g., CPUE)}
#' }
#'
#' @section Key features:
#' \itemize{
#'   \item Shared random effects between numerator and denominator (default)
#'   \item Spatial structure via CAR/BYM2
#'   \item Full posterior inference on derived ratios
#'   \item Explicit rejection of offset-based approaches
#' }
#'
#' @name quotr-package
#' @aliases quotr-package
NULL

# Suppress R CMD check notes about global variables
utils::globalVariables(c(".", "value", "variable"))
