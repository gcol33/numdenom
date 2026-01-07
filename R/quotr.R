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
#' @param formula Model formula specifying the response and predictors.
#'   Two syntax options:
#'   - **Combined** (recommended): `num | denom ~ predictors + (1|group)`
#'     Both processes share the same predictors and random effects.
#'   - **Separate**: `num ~ predictors` with `formula_denom` argument.
#' @param data Data frame containing all variables.
#' @param family A quotr family object specifying the distributions.
#'   Options:
#'   - [quotr_negbin_negbin()]: Both count processes (default)
#'   - [quotr_binomial()]: Successes/trials
#'   - [quotr_poisson_gamma()]: Count/continuous effort (CPUE)
#' @param formula_num Optional one-sided formula for additional numerator
#'   predictors: `~ extra_terms`. Added to the main formula.
#' @param formula_denom Optional formula for denominator. Required if main
#'   formula has single response. Can be `denom ~ predictors`.
#' @param shared Formula specifying shared random effects structure.
#'   - `NULL` (default): Infer from matching random effects in both processes.
#'   - `~ (1 | group)`: Explicit shared structure.
#'   - `~ 0`: Independence assumption (triggers warning).
#' @param spatial Optional spatial structure specification.
#'   See [spatial_car()], [spatial_bym2()].
#' @param priors Prior specification. See [quotr_priors()].
#' @param chains Number of MCMC chains (default 4).
#' @param iter Total iterations per chain (default 2000).
#' @param warmup Warmup iterations per chain (default `iter/2`).
#' @param thin Thinning interval (default 1).
#' @param cores Number of cores for parallel chains (default: chains).
#' @param seed Random seed for reproducibility.
#' @param backend Stan backend: "cmdstanr" (default) or "rstan".
#' @param refresh Progress update frequency (default: iter/10).
#' @param ... Additional arguments passed to the sampler.
#'
#' @return A `quotr_fit` object containing:
#' \describe{
#'   \item{fit}{The Stan fit object}
#'   \item{formula}{Parsed formula specification}
#'   \item{family}{Model family}
#'   \item{data}{Original data}
#'   \item{stan_data}{Data passed to Stan}
#' }
#'
#' @examples
#' \dontrun{
#' # CPUE example with combined formula
#' fit <- quotr(
#'   catch | effort ~ depth + season + (1 | site),
#'   data = trawl_data,
#'   family = quotr_poisson_gamma()
#' )
#'
#' # Different predictors for each process
#' fit <- quotr(
#'   catch | effort ~ (1 | site),
#'   formula_num = ~ depth + season,
#'   formula_denom = ~ weather,
#'   data = trawl_data,
#'   family = quotr_poisson_gamma()
#' )
#'
#' # Extract ratio posteriors
#' cpue <- ratio(fit)
#' summary(cpue)
#'
#' # Compare ratios between groups
#' ratio_contrast(fit, ~ season)
#' }
#'
#' @seealso
#' - [ratio()] for extracting ratio posteriors
#' - [ratio_contrast()] for comparing ratios between groups
#' - [pp_check()] for posterior predictive checks
#' - [loo()] and [waic()] for model comparison
#'
#' @export
quotr <- function(formula,
                  data,
                  family = quotr_negbin_negbin(),
                  formula_num = NULL,
                  formula_denom = NULL,
                  shared = NULL,
                  spatial = NULL,
                  priors = NULL,
                  chains = 4,
                  iter = 2000,
                  warmup = floor(iter / 2),
                  thin = 1,
                  cores = getOption("mc.cores", chains),
                  seed = NULL,
                  backend = c("cmdstanr", "rstan"),
                  refresh = NULL,
                  ...) {

 backend <- match.arg(backend)

  # Check backend availability
  if (backend == "cmdstanr") {
    if (!requireNamespace("cmdstanr", quietly = TRUE)) {
      stop(
        "Package 'cmdstanr' is required. Install from:\n",
        "  install.packages('cmdstanr', repos = c('https://stan-dev.r-universe.dev', getOption('repos')))\n",
        "  cmdstanr::install_cmdstan()",
        call. = FALSE
      )
    }
  }

  # Validate inputs
  if (missing(formula)) {
    stop("`formula` is required. Use: num | denom ~ predictors", call. = FALSE)
  }
  if (missing(data)) {
    stop("`data` is required", call. = FALSE)
  }
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame", call. = FALSE)
  }
  if (!inherits(family, "quotr_family")) {
    stop(
      "`family` must be a quotr_family object.\n",
      "Options: quotr_negbin_negbin(), quotr_binomial(), quotr_poisson_gamma()",
      call. = FALSE
    )
  }

  # Parse formula
  formula_spec <- quotr_formula(
    formula = formula,
    formula_num = formula_num,
    formula_denom = formula_denom,
    shared = shared,
    data = data
  )

  # Set default priors
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
  stan_model <- get_stan_model(family$stan_model, backend = backend)

  # Set refresh rate
  if (is.null(refresh)) {
    refresh <- max(100, (iter - warmup) %/% 10)
  }

  # Fit message
  message("Fitting quotr model...")
  message(sprintf("  Family: %s", family$name))
  message(sprintf("  Observations: %d", nrow(data)))
  message(sprintf("  Chains: %d, Iter: %d (warmup: %d)", chains, iter, warmup))
  if (formula_spec$shared$type %in% c("inferred", "explicit")) {
    n_shared <- length(formula_spec$shared$random_effects)
    message(sprintf("  Shared RE groups: %d", n_shared))
  }

  # Sample
  if (backend == "cmdstanr") {
    fit <- stan_model$sample(
      data = stan_data,
      chains = chains,
      iter_warmup = warmup,
      iter_sampling = iter - warmup,
      thin = thin,
      parallel_chains = min(chains, cores),
      seed = seed,
      refresh = refresh,
      ...
    )
  } else {
    stop("rstan backend not yet implemented", call. = FALSE)
  }

  # Check diagnostics
  check_diagnostics(fit)

  # Build result object
  result <- structure(
    list(
      fit = fit,
      formula = formula_spec,
      family = family,
      data = data,
      stan_data = stan_data,
      priors = priors,
      spatial = spatial,
      backend = backend
    ),
    class = "quotr_fit"
  )

  result
}

#' Get compiled Stan model
#'
#' @param model_name Name of Stan model (without extension)
#' @param backend "cmdstanr" or "rstan"
#' @return Compiled Stan model object
#' @keywords internal
get_stan_model <- function(model_name, backend = "cmdstanr") {
  stan_file <- system.file(
    "stan", paste0(model_name, ".stan"),
    package = "quotr"
  )

  if (stan_file == "") {
    stop(sprintf("Stan model '%s' not found", model_name), call. = FALSE)
  }

  if (backend == "cmdstanr") {
    cmdstanr::cmdstan_model(stan_file)
  } else {
    stop("rstan backend not yet implemented", call. = FALSE)
  }
}

#' Check MCMC diagnostics
#'
#' @param fit Stan fit object
#' @keywords internal
check_diagnostics <- function(fit) {
  diag <- fit$diagnostic_summary(quiet = TRUE)

  # Check divergences
  n_div <- sum(diag$num_divergent)
  if (n_div > 0) {
    warning(
      sprintf("%d divergent transitions detected.\n", n_div),
      "Consider increasing adapt_delta or using more informative priors.",
      call. = FALSE
    )
  }

  # Check low E-BFMI
  if (any(diag$ebfmi < 0.2)) {
    warning(
      "Low E-BFMI detected. Consider increasing warmup iterations.",
      call. = FALSE
    )
  }

  # Check high Rhat
  tryCatch({
    summ <- fit$summary()
    max_rhat <- max(summ$rhat, na.rm = TRUE)
    if (!is.na(max_rhat) && max_rhat > 1.05) {
      warning(
        sprintf("High Rhat (%.3f). Chains may not have converged.", max_rhat),
        call. = FALSE
      )
    }
  }, error = function(e) NULL)
}

#' Print method for quotr_fit
#'
#' @param x A quotr_fit object
#' @param ... Ignored
#' @export
print.quotr_fit <- function(x, ...) {
  cat("quotr model fit\n")
  cat("===============\n\n")

  cat("Family:", x$family$name, "\n")
  cat("  ", x$family$description, "\n\n")

  cat("Data:\n")
  cat("  Observations:", nrow(x$data), "\n")
  cat("  Numerator:", x$formula$numerator$response_var, "\n")
  cat("  Denominator:", x$formula$denominator$response_var, "\n\n")

  cat("Structure:\n")
  cat("  Num fixed effects:", ncol(x$formula$numerator$X), "\n")
  cat("  Denom fixed effects:", ncol(x$formula$denominator$X), "\n")

  n_re <- x$stan_data$n_re_groups
  if (n_re > 0) {
    cat("  Random effect groups:", n_re, "\n")
    if (x$formula$shared$type %in% c("inferred", "explicit")) {
      cat("  Shared structure:", x$formula$shared$type, "\n")
    }
  }

  if (x$stan_data$use_spatial) {
    cat("  Spatial:", "Yes\n")
  }

  cat("\nUse summary() for parameter estimates\n")
  cat("Use ratio() to extract ratio posteriors\n")

  invisible(x)
}

#' Summary method for quotr_fit
#'
#' @param object A quotr_fit object
#' @param prob Probability mass for credible intervals (default 0.95)
#' @param ... Ignored
#' @export
summary.quotr_fit <- function(object, prob = 0.95, ...) {
  cat("quotr model summary\n")
  cat("===================\n\n")

  # Calculate quantile probabilities
  alpha <- (1 - prob) / 2
  probs <- c(alpha, 0.5, 1 - alpha)

  # Get parameter names based on family
  if (object$family$name == "binomial_fixed") {
    params <- c("beta")
    if (object$stan_data$n_re_groups > 0) {
      params <- c(params, "sigma_re")
    }
  } else {
    params <- c("beta_num", "beta_denom")
    if (object$stan_data$n_re_groups > 0) {
      params <- c(params, "sigma_shared", "sigma_num", "sigma_denom")
    }
    if (object$family$name == "negbin_negbin") {
      params <- c(params, "phi_num", "phi_denom")
    } else if (object$family$name == "poisson_gamma") {
      params <- c(params, "shape_denom")
    }
  }

  if (object$stan_data$use_spatial) {
    params <- c(params, "sigma_spatial")
  }

  # Get summary
  summ <- object$fit$summary(variables = params)

  # Print fixed effects
  cat("Fixed effects (numerator):\n")
  if (object$family$name == "binomial_fixed") {
    beta_summ <- summ[grepl("^beta\\[", summ$variable), ]
    beta_summ$variable <- colnames(object$formula$numerator$X)
  } else {
    beta_summ <- summ[grepl("^beta_num", summ$variable), ]
    if (nrow(beta_summ) > 0) {
      beta_summ$variable <- colnames(object$formula$numerator$X)
    }
  }
  if (nrow(beta_summ) > 0) {
    print(beta_summ[, c("variable", "mean", "sd", "q5", "q95", "rhat")],
          row.names = FALSE)
  }
  cat("\n")

  if (object$family$name != "binomial_fixed") {
    cat("Fixed effects (denominator):\n")
    beta_d_summ <- summ[grepl("^beta_denom", summ$variable), ]
    if (nrow(beta_d_summ) > 0) {
      beta_d_summ$variable <- colnames(object$formula$denominator$X)
      print(beta_d_summ[, c("variable", "mean", "sd", "q5", "q95", "rhat")],
            row.names = FALSE)
    }
    cat("\n")
  }

  # Variance components
  cat("Variance components:\n")
  var_summ <- summ[grepl("^(sigma|phi|shape)", summ$variable), ]
  if (nrow(var_summ) > 0) {
    print(var_summ[, c("variable", "mean", "sd", "q5", "q95", "rhat")],
          row.names = FALSE)
  }
  cat("\n")

  # Diagnostics
  diag <- object$fit$diagnostic_summary(quiet = TRUE)
  cat("Diagnostics:\n")
  cat("  Divergences:", sum(diag$num_divergent), "\n")
  cat("  Max treedepth:", sum(diag$num_max_treedepth), "\n")
  cat("  E-BFMI (min):", round(min(diag$ebfmi), 3), "\n")

  invisible(object)
}

#' Extract posterior draws from quotr_fit
#'
#' @param object A quotr_fit object
#' @param variable Variable name or pattern
#' @param ... Passed to posterior::as_draws_df
#' @return A draws_df object
#' @export
as.data.frame.quotr_fit <- function(x, ...) {
  posterior::as_draws_df(x$fit$draws(...))
}
