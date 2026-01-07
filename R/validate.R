#' Posterior predictive checks for quotr models
#'
#' @description
#' Visual and numerical checks comparing observed data to posterior
#' predictive distributions. Essential for assessing model fit.
#'
#' @name quotr_validate
NULL

#' Posterior predictive check
#'
#' @description
#' Generate posterior predictive checks for a fitted quotr model.
#' Compares observed data to replicated data from the posterior predictive
#' distribution.
#'
#' @param object A `quotr_fit` object
#' @param type Type of check: "dens_overlay", "scatter", "intervals", "stat"
#' @param component Which component: "numerator", "denominator", or "both"
#' @param stat Function for "stat" type (default: mean)
#' @param ndraws Number of posterior draws to use (default: 50 for plots)
#' @param ... Additional arguments passed to bayesplot functions
#'
#' @return A ggplot object
#'
#' @examples
#' \dontrun{
#' # Density overlay
#' pp_check(fit, type = "dens_overlay")
#'
#' # Check both components
#' pp_check(fit, type = "dens_overlay", component = "both")
#'
#' # Predictive intervals
#' pp_check(fit, type = "intervals")
#' }
#'
#' @export
pp_check <- function(object, ...) {
  UseMethod("pp_check")
}

#' @export
#' @rdname pp_check
pp_check.quotr_fit <- function(object, type = c("dens_overlay", "scatter", "intervals", "stat"),
                                component = c("numerator", "denominator", "both"),
                                stat = mean, ndraws = 50, ...) {

  type <- match.arg(type)
  component <- match.arg(component)

  if (!requireNamespace("bayesplot", quietly = TRUE)) {
    stop("Package 'bayesplot' is required for pp_check. Install with:\n",
         "  install.packages('bayesplot')", call. = FALSE)
  }

  # Extract posterior predictive draws
  y_rep_num <- object$stanfit$draws("y_num_rep", format = "matrix")
  y_rep_denom <- object$stanfit$draws("y_denom_rep", format = "matrix")

  # Observed data
  y_num <- object$stan_data$y_num
  y_denom <- object$stan_data$y_denom

  # Select which to plot
  if (component == "numerator") {
    y <- y_num
    yrep <- y_rep_num
    title_suffix <- " (numerator)"
  } else if (component == "denominator") {
    y <- y_denom
    yrep <- y_rep_denom
    title_suffix <- " (denominator)"
  } else {
    # Both - create side-by-side plots
    if (!requireNamespace("patchwork", quietly = TRUE)) {
      message("Install 'patchwork' for side-by-side plots. Showing numerator only.")
      y <- y_num
      yrep <- y_rep_num
      title_suffix <- " (numerator)"
    } else {
      p1 <- pp_check_single(y_num, y_rep_num, type, stat, ndraws, " (numerator)", ...)
      p2 <- pp_check_single(y_denom, y_rep_denom, type, stat, ndraws, " (denominator)", ...)
      return(patchwork::wrap_plots(p1, p2, ncol = 2))
    }
  }

  pp_check_single(y, yrep, type, stat, ndraws, title_suffix, ...)
}

#' Single component pp_check
#'
#' @keywords internal
pp_check_single <- function(y, yrep, type, stat, ndraws, title_suffix, ...) {

  # Subsample draws for visualization
  if (nrow(yrep) > ndraws) {
    idx <- sample(nrow(yrep), ndraws)
    yrep <- yrep[idx, , drop = FALSE]
  }

  # Create plot based on type
  if (type == "dens_overlay") {
    p <- bayesplot::ppc_dens_overlay(y, yrep, ...)
  } else if (type == "scatter") {
    # Use mean of yrep for scatter
    yrep_mean <- colMeans(yrep)
    p <- bayesplot::ppc_scatter_avg(y, yrep, ...)
  } else if (type == "intervals") {
    p <- bayesplot::ppc_intervals(y, yrep, ...)
  } else if (type == "stat") {
    p <- bayesplot::ppc_stat(y, yrep, stat = stat, ...)
  }

  p + ggplot2::ggtitle(paste0("Posterior predictive check", title_suffix))
}

#' Prior predictive simulation
#'
#' @description
#' Simulate data from the prior predictive distribution to check
#' whether priors are reasonable.
#'
#' @param formula A quotr_formula object or formula components
#' @param family A quotr_family object
#' @param data Data frame (structure used for dimensions)
#' @param priors Prior specification
#' @param n Number of prior predictive datasets to simulate
#'
#' @return A list with simulated datasets
#'
#' @examples
#' \dontrun{
#' # Check if default priors are reasonable
#' prior_sims <- prior_predict(
#'   numerator = count ~ x + (1 | site),
#'   denominator = effort ~ (1 | site),
#'   data = my_data,
#'   family = quotr_negbin_negbin(),
#'   n = 100
#' )
#'
#' # Examine range of simulated values
#' hist(sapply(prior_sims$y_num, mean))
#' }
#'
#' @export
prior_predict <- function(formula, family, data, priors = NULL, n = 100) {
  stop("prior_predict not yet implemented", call. = FALSE)
}

#' LOO cross-validation
#'
#' @description
#' Compute leave-one-out cross-validation using Pareto-smoothed
#' importance sampling (PSIS-LOO).
#'
#' @param x A `quotr_fit` object
#' @param ... Additional arguments passed to loo::loo
#'
#' @return A loo object
#'
#' @export
loo.quotr_fit <- function(x, ...) {
  if (!requireNamespace("loo", quietly = TRUE)) {
    stop("Package 'loo' is required. Install with:\n",
         "  install.packages('loo')", call. = FALSE)
  }

  # Try combined log_lik first (all models should have this)
  ll <- tryCatch(
    x$stanfit$draws("log_lik", format = "matrix"),
    error = function(e) NULL
  )

  if (!is.null(ll)) {
    return(loo::loo(ll, ...))
  }

  # Fall back to separate components for two-process models
  ll_num <- tryCatch(
    x$stanfit$draws("log_lik_num", format = "matrix"),
    error = function(e) NULL
  )
  ll_denom <- tryCatch(
    x$stanfit$draws("log_lik_denom", format = "matrix"),
    error = function(e) NULL
  )

  if (is.null(ll_num) && is.null(ll_denom)) {
    stop("Log-likelihood not found in model output.", call. = FALSE)
  }

  # Sum components that exist
  if (!is.null(ll_num) && !is.null(ll_denom)) {
    ll <- ll_num + ll_denom
  } else if (!is.null(ll_num)) {
    ll <- ll_num
  } else {
    ll <- ll_denom
  }

  loo::loo(ll, ...)
}

#' WAIC computation
#'
#' @description
#' Compute Widely Applicable Information Criterion (WAIC).
#'
#' @param x A `quotr_fit` object
#' @param ... Additional arguments passed to loo::waic
#'
#' @return A waic object
#'
#' @export
waic.quotr_fit <- function(x, ...) {
  if (!requireNamespace("loo", quietly = TRUE)) {
    stop("Package 'loo' is required. Install with:\n",
         "  install.packages('loo')", call. = FALSE)
  }

  # Try combined log_lik first
  ll <- tryCatch(
    x$stanfit$draws("log_lik", format = "matrix"),
    error = function(e) NULL
  )

  if (!is.null(ll)) {
    return(loo::waic(ll, ...))
  }

  # Fall back to separate components
  ll_num <- tryCatch(
    x$stanfit$draws("log_lik_num", format = "matrix"),
    error = function(e) NULL
  )
  ll_denom <- tryCatch(
    x$stanfit$draws("log_lik_denom", format = "matrix"),
    error = function(e) NULL
  )

  if (is.null(ll_num) && is.null(ll_denom)) {
    stop("Log-likelihood not found in model output.", call. = FALSE)
  }

  if (!is.null(ll_num) && !is.null(ll_denom)) {
    ll <- ll_num + ll_denom
  } else if (!is.null(ll_num)) {
    ll <- ll_num
  } else {
    ll <- ll_denom
  }

  loo::waic(ll, ...)
}

#' Compare quotr models
#'
#' @description
#' Compare multiple quotr models using LOO-CV or WAIC.
#'
#' @param ... Multiple `quotr_fit` objects to compare
#' @param criterion Comparison criterion: "loo" (default) or "waic"
#'
#' @return A loo_compare object
#'
#' @examples
#' \dontrun{
#' # Fit models with different structures
#' fit1 <- quotr(y_num ~ x, y_denom ~ 1, data = df, family = quotr_negbin_negbin())
#' fit2 <- quotr(y_num ~ x + z, y_denom ~ 1, data = df, family = quotr_negbin_negbin())
#'
#' # Compare
#' quotr_compare(fit1, fit2)
#' }
#'
#' @export
quotr_compare <- function(..., criterion = c("loo", "waic")) {
  criterion <- match.arg(criterion)

  if (!requireNamespace("loo", quietly = TRUE)) {
    stop("Package 'loo' is required. Install with:\n",
         "  install.packages('loo')", call. = FALSE)
  }

  models <- list(...)

  if (length(models) < 2) {
    stop("At least two models required for comparison", call. = FALSE)
  }

  # Check all are quotr_fit objects
  if (!all(vapply(models, inherits, logical(1), "quotr_fit"))) {
    stop("All objects must be quotr_fit models", call. = FALSE)
  }

  # Compute criterion for each model
  if (criterion == "loo") {
    results <- lapply(models, loo.quotr_fit)
  } else {
    results <- lapply(models, waic.quotr_fit)
  }

  # Get model names from call
  model_names <- as.character(substitute(list(...)))[-1]
  names(results) <- model_names

  # Compare
  loo::loo_compare(results)
}
