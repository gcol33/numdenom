#' Fit a quotr model
#'
#' @description
#' Fit a Bayesian hierarchical model for ratios, rates, or proportions.
#' The model jointly estimates the numerator and denominator processes
#' with optional shared latent structure.
#'
#' **Core principle:** Ratios are derived quantities, computed post hoc
#' from the posterior of the joint model. The ratio is never modelled directly.
#'
#' @param numerator Formula for the numerator process. Must include a response
#'   variable on the left-hand side.
#' @param denominator Formula for the denominator process.
#' @param shared Formula for shared random effects between numerator and
#'   denominator. Default NULL infers shared structure from matching random
#'   effects in both formulas. Use `~ 0` for independence (triggers warning).
#' @param data Data frame containing all variables.
#' @param family A quotr family object specifying the distributions.
#'   See [quotr_negbin_negbin()], [quotr_binomial()], [quotr_poisson_gamma()].
#' @param spatial Optional spatial structure specification.
#'   See [spatial_car()], [spatial_bym2()].
#' @param priors Prior specification. See [quotr_priors()].
#' @param chains Number of MCMC chains (default 4).
#' @param iter Total iterations per chain including warmup (default 2000).
#' @param warmup Warmup iterations per chain (default 1000).
#' @param thin Thinning interval (default 1).
#' @param cores Number of cores for parallel chains (default: all available).
#' @param seed Random seed for reproducibility.
#' @param ... Additional arguments passed to `cmdstanr::sample()`.
#'
#' @return A `quotr_fit` object containing:
#' \describe{
#'   \item{stanfit}{The cmdstanr fit object}
#'   \item{formula}{The parsed formula specification}
#'   \item{family}{The model family}
#'   \item{data}{Original data}
#'   \item{stan_data}{Data passed to Stan}
#' }
#'
#' @examples
#' \dontrun{
#' # CPUE example: catch per unit effort
#' fit <- quotr(
#'   numerator = catch ~ depth + season + (1 | site),
#'   denominator = effort ~ season + (1 | site),
#'   shared = ~ (1 | site),
#'   data = trawl_data,
#'   family = quotr_negbin_negbin()
#' )
#'
#' # Extract ratio posteriors
#' cpue <- ratio(fit)
#' summary(cpue)
#' }
#'
#' @export
quotr <- function(
    numerator,
    denominator,
    shared = NULL,
    data,
    family = quotr_negbin_negbin(),
    spatial = NULL,
    priors = NULL,
    chains = 4,
    iter = 2000,
    warmup = 1000,
    thin = 1,
    cores = parallel::detectCores(),
    seed = NULL,
    ...
) {

  # Check cmdstanr is available
 if (!requireNamespace("cmdstanr", quietly = TRUE)) {
    stop(
      "Package 'cmdstanr' is required. Install from:\n",
      "  https://mc-stan.org/cmdstanr/",
      call. = FALSE
    )
  }

  # Validate inputs
  if (missing(numerator)) stop("`numerator` formula is required", call. = FALSE)
  if (missing(denominator)) stop("`denominator` formula is required", call. = FALSE)
  if (missing(data)) stop("`data` is required", call. = FALSE)

  if (!inherits(family, "quotr_family")) {
    stop("`family` must be a quotr_family object. See ?quotr_negbin_negbin",
         call. = FALSE)
  }

  # Parse formulas
  formula_spec <- quotr_formula(
    numerator = numerator,
    denominator = denominator,
    shared = shared,
    data = data
  )

  # Use default priors if not specified
  if (is.null(priors)) {
    priors <- quotr_priors()
  }

  # Prepare Stan data
  stan_data <- make_standata(
    formula = formula_spec,
    family = family,
    data = data,
    spatial = spatial,
    priors = priors
  )

  # Get Stan model
  stan_model <- quotr_stan_model(family$stan_model)

  # Fit model
  message("Fitting quotr model...")
  message(sprintf("  Family: %s", family$name))
  message(sprintf("  Observations: %d", nrow(data)))
  message(sprintf("  Chains: %d, Iterations: %d (warmup: %d)",
                  chains, iter, warmup))

  fit <- stan_model$sample(
    data = stan_data,
    chains = chains,
    iter_warmup = warmup,
    iter_sampling = iter - warmup,
    thin = thin,
    parallel_chains = min(chains, cores),
    seed = seed,
    refresh = max(100, (iter - warmup) %/% 10),
    ...
  )

  # Check diagnostics
  check_diagnostics(fit)

  # Create result object
  result <- structure(
    list(
      stanfit = fit,
      formula = formula_spec,
      family = family,
      data = data,
      stan_data = stan_data,
      priors = priors,
      spatial = spatial
    ),
    class = "quotr_fit"
  )

  result
}

#' Check MCMC diagnostics and warn if issues detected
#'
#' @param fit cmdstanr fit object
#' @keywords internal
check_diagnostics <- function(fit) {
  diag <- fit$diagnostic_summary()

  # Check for divergences
  n_div <- sum(diag$num_divergent)
  if (n_div > 0) {
    warning(
      sprintf("%d divergent transitions detected.\n", n_div),
      "This may indicate model misspecification or difficult posterior geometry.\n",
      "Consider:\n",
      "  - Increasing adapt_delta (e.g., adapt_delta = 0.95)\
n",
      "  - Reparameterizing the model\n",
      "  - Using more informative priors",
      call. = FALSE
    )
  }

  # Check for low E-BFMI
  ebfmi <- diag$ebfmi
  if (any(ebfmi < 0.2)) {
    warning(
      "Low E-BFMI detected in some chains.\n",
      "This may indicate poor exploration of the posterior.\n",
      "Consider increasing the number of warmup iterations.",
      call. = FALSE
    )
  }

  # Check for high Rhat
  tryCatch({
    summ <- fit$summary()
    max_rhat <- max(summ$rhat, na.rm = TRUE)
    if (max_rhat > 1.05) {
      warning(
        sprintf("High Rhat detected (max = %.3f).\n", max_rhat),
        "Chains may not have converged. Consider:\n",
        "  - Running more iterations\n",
        "  - Checking for multimodality",
        call. = FALSE
      )
    }
  }, error = function(e) NULL)
}

#' Print method for quotr_fit
#'
#' @param x A quotr_fit object
#' @param ... Ignored
#'
#' @export
print.quotr_fit <- function(x, ...) {
  cat("quotr model fit\n")
  cat("===============\n\n")

  cat("Family:", x$family$name, "\n")
  cat(x$family$description, "\n\n")

  cat("Data:\n")
  cat("  Observations:", nrow(x$data), "\n")
  cat("  Numerator response:", x$formula$numerator$response_var, "\n")
  cat("  Denominator response:", x$formula$denominator$response_var, "\n\n")

  cat("Structure:\n")
  cat("  Numerator fixed effects:", ncol(x$formula$numerator$X), "\n")
  cat("  Denominator fixed effects:", ncol(x$formula$denominator$X), "\n")
  cat("  Random effect groups:", x$stan_data$n_re_groups, "\n")
  cat("  Spatial:", if (x$stan_data$use_spatial) "Yes" else "No", "\n\n")

  cat("Use summary() for parameter estimates\n")
  cat("Use ratio() to extract ratio posteriors\n")

  invisible(x)
}

#' Summary method for quotr_fit
#'
#' @param object A quotr_fit object
#' @param ... Ignored
#'
#' @export
summary.quotr_fit <- function(object, ...) {
  cat("quotr model summary\n")
  cat("===================\n\n")

  # Get parameter summary
  summ <- object$stanfit$summary(
    variables = c("beta_num", "beta_denom", "sigma_shared", "phi_num", "phi_denom")
  )

  cat("Fixed effects (numerator):\n")
  num_summ <- summ[grepl("^beta_num", summ$variable), ]
  if (nrow(num_summ) > 0) {
    num_summ$variable <- colnames(object$formula$numerator$X)
    print(num_summ[, c("variable", "mean", "sd", "q5", "q95", "rhat")], row.names = FALSE)
  }
  cat("\n")

  cat("Fixed effects (denominator):\n")
  denom_summ <- summ[grepl("^beta_denom", summ$variable), ]
  if (nrow(denom_summ) > 0) {
    denom_summ$variable <- colnames(object$formula$denominator$X)
    print(denom_summ[, c("variable", "mean", "sd", "q5", "q95", "rhat")], row.names = FALSE)
  }
  cat("\n")

  cat("Variance components:\n")
  var_summ <- summ[grepl("^(sigma|phi)", summ$variable), ]
  if (nrow(var_summ) > 0) {
    print(var_summ[, c("variable", "mean", "sd", "q5", "q95", "rhat")], row.names = FALSE)
  }
  cat("\n")

  # Diagnostics summary
  diag <- object$stanfit$diagnostic_summary()
  cat("Diagnostics:\n")
  cat("  Divergences:", sum(diag$num_divergent), "\n")
  cat("  Max treedepth:", sum(diag$num_max_treedepth), "\n")
  cat("  E-BFMI (min):", round(min(diag$ebfmi), 3), "\n")

  invisible(object)
}
