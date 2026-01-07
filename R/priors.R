#' Prior specification for quotr models
#'
#' @description
#' Specify priors for variance components using penalized complexity (PC) priors.
#' PC priors shrink toward simpler models (smaller variance = simpler).
#'
#' The prior is specified via: P(sigma > U) = alpha
#' - U is the upper bound you consider "large"
#' - alpha is the probability of exceeding it (typically 0.01 or 0.05)
#'
#' @param sigma_U Upper bound for random effect SD. Default 1.0 means
#'   P(sigma > 1) = sigma_alpha.
#' @param sigma_alpha Tail probability. Default 0.01 means 1% chance
#'   of sigma exceeding sigma_U.
#' @param phi_U Upper bound for overdispersion parameter.
#' @param phi_alpha Tail probability for overdispersion.
#' @param beta_sd SD for normal prior on fixed effects. Default 2.5.
#'
#' @return A `quotr_priors` object
#'
#' @details
#' PC priors (Simpson et al., 2017) provide principled regularization that:
#' - Favors simpler models (smaller variance components)
#' - Has interpretable parameters (tail probabilities)
#' - Prevents overfitting with sparse data
#'
#' @references
#' Simpson, D., Rue, H., Riebler, A., Martins, T. G., & Sorbye, S. H. (2017).
#' Penalising model component complexity: A principled, practical approach to
#' constructing priors. Statistical Science, 32(1), 1-28.
#'
#' @examples
#' # Default priors
#' quotr_priors()
#'
#' # More informative: expect smaller random effects
#' quotr_priors(sigma_U = 0.5, sigma_alpha = 0.01)
#'
#' # Less informative: allow larger random effects
#' quotr_priors(sigma_U = 2.0, sigma_alpha = 0.05)
#'
#' @export
quotr_priors <- function(
    sigma_U = 1.0,
    sigma_alpha = 0.01,
    phi_U = 10.0,
    phi_alpha = 0.01,
    beta_sd = 2.5
) {

  # Validate
  if (sigma_U <= 0) stop("sigma_U must be positive", call. = FALSE)
  if (sigma_alpha <= 0 || sigma_alpha >= 1) {
    stop("sigma_alpha must be in (0, 1)", call. = FALSE)
  }
  if (phi_U <= 0) stop("phi_U must be positive", call. = FALSE)
  if (phi_alpha <= 0 || phi_alpha >= 1) {
    stop("phi_alpha must be in (0, 1)", call. = FALSE)
  }
  if (beta_sd <= 0) stop("beta_sd must be positive", call. = FALSE)

  structure(
    list(
      sigma_U = sigma_U,
      sigma_alpha = sigma_alpha,
      phi_U = phi_U,
      phi_alpha = phi_alpha,
      beta_sd = beta_sd
    ),
    class = "quotr_priors"
  )
}

#' Print method for quotr_priors
#'
#' @param x A quotr_priors object
#' @param ... Ignored
#'
#' @export
print.quotr_priors <- function(x, ...) {
  cat("quotr prior specification (PC priors)\n")
  cat("=====================================\n\n")
  cat("Random effect SD:\n")
  cat(sprintf("  P(sigma > %.2f) = %.3f\n", x$sigma_U, x$sigma_alpha))
  cat(sprintf("  => exponential rate: %.3f\n\n", -log(x$sigma_alpha) / x$sigma_U))
  cat("Overdispersion:\n")
  cat(sprintf("  P(phi > %.2f) = %.3f\n", x$phi_U, x$phi_alpha))
  cat(sprintf("  => exponential rate: %.3f\n\n", -log(x$phi_alpha) / x$phi_U))
  cat("Fixed effects:\n")
  cat(sprintf("  Normal(0, %.2f)\n", x$beta_sd))
  invisible(x)
}
