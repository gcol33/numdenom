#' Fit a ratiod model
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
#'   Two syntax options: **Combined** (recommended) `num | denom ~ predictors + (1|group)`
#'   where both processes share the same predictors and random effects;
#'   **Separate** `num ~ predictors` with the `formula_denom` argument.
#' @param data Data frame containing all variables.
#' @param family A ratiod family object specifying the distributions:
#'   [ratiod_negbin_negbin()] for both count processes (default),
#'   [ratiod_binomial()] for successes/trials,
#'   [ratiod_poisson_gamma()] for count/continuous effort (CPUE).
#' @param formula_num Optional one-sided formula for additional numerator
#'   predictors: `~ extra_terms`. Added to the main formula.
#' @param formula_denom Optional formula for denominator. Required if main
#'   formula has single response. Can be `denom ~ predictors`.
#' @param shared Formula specifying shared random effects structure:
#'   `NULL` (default) infers from matching random effects in both processes;
#'   `~ (1 | group)` for explicit shared structure;
#'   `~ 0` for independence assumption (triggers warning).
#' @param spatial Optional spatial structure specification.
#'   See [spatial_car()], [spatial_bym2()].
#' @param temporal Optional temporal structure specification.
#'   See [temporal_rw1()], [temporal_rw2()], [temporal_ar1()].
#' @param zi Optional zero-inflation specification.
#'   See [zi_poisson()], [zi_negbin()], [hurdle_poisson()], [hurdle_negbin()].
#' @param latent Optional latent factor specification for unmeasured confounders.
#'   See [latent_factor()].
#' @param priors Prior specification. See [ratiod_priors()].
#' @param chains Number of MCMC chains (default 4).
#' @param iter Total iterations per chain (default 2000).
#' @param warmup Warmup iterations per chain (default `iter/2`).
#' @param thin Thinning interval (default 1).
#' @param cores Number of cores for parallel chains (default: chains).
#' @param seed Random seed for reproducibility.
#' @param backend Inference backend: `"auto"` (default) automatically selects
#'   optimal backend; `"hmc"` full MCMC via native HMC/NUTS sampler;
#'   `"pg"` Polya-Gamma Gibbs sampling (binomial only, experimental);
#'   `"laplace"` Laplace approximation (fastest, approximate inference).
#' @param refresh Progress update frequency (default: iter/10).
#' @param ... Additional arguments passed to the sampler.
#'
#' @return A `ratiod_fit` object containing:
#' \describe{
#'   \item{draws}{Posterior draws matrix}
#'   \item{formula}{Parsed formula specification}
#'   \item{family}{Model family}
#'   \item{data}{Original data}
#'   \item{backend}{Inference backend used}
#'   \item{diagnostics}{MCMC diagnostics (if applicable)}
#' }
#'
#' @examples
#' # Quick example: create model object (no fitting)
#' set.seed(123)
#' df <- data.frame(
#'   catch = rpois(50, 10),
#'   effort = rgamma(50, 2, 0.5),
#'   depth = rnorm(50),
#'   season = factor(sample(c("spring", "summer"), 50, replace = TRUE)),
#'   site = factor(sample(1:5, 50, replace = TRUE))
#' )
#'
#' \donttest{
#' # CPUE example with combined formula
#' fit <- ratiod(
#'   catch | effort ~ depth + season + (1 | site),
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   iter = 200, warmup = 100, chains = 1
#' )
#'
#' # Different predictors for each process
#' fit2 <- ratiod(
#'   catch | effort ~ (1 | site),
#'   formula_num = ~ depth + season,
#'   formula_denom = ~ season,
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   iter = 200, warmup = 100, chains = 1
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
#' - [loo::loo()] and [loo::waic()] for model comparison
#'
#' @export
ratiod <- function(formula,
                  data,
                  family = ratiod_negbin_negbin(),
                  formula_num = NULL,
                  formula_denom = NULL,
                  shared = NULL,
                  spatial = NULL,
                  temporal = NULL,
                  zi = NULL,
                  latent = NULL,
                  priors = NULL,
                  chains = 4,
                  iter = 2000,
                  warmup = floor(iter / 2),
                  thin = 1,
                  cores = getOption("mc.cores", chains),
                  seed = NULL,
                  backend = c("auto", "hmc", "pg", "laplace"),
                  refresh = NULL,
                  ...) {

  backend <- match.arg(backend)

  # Validate temporal specification
  if (!is.null(temporal)) {
    if (!inherits(temporal, "ratiod_temporal")) {
      stop(
        "`temporal` must be a ratiod_temporal object.\n",
        "Options: temporal_rw1(), temporal_rw2(), temporal_ar1()",
        call. = FALSE
      )
    }
    temporal <- validate_temporal(temporal, data)
  }

  # Validate zero-inflation specification
  if (!is.null(zi)) {
    zi <- validate_zi(zi, data)
  }

  # Validate latent factor specification
  if (!is.null(latent)) {
    if (!inherits(latent, "ratiod_latent")) {
      stop(
        "`latent` must be a ratiod_latent object from latent_factor()",
        call. = FALSE
      )
    }
    latent <- validate_latent(latent, nrow(data))
  }

  # Auto-select backend based on model characteristics
  if (backend == "auto") {
    backend <- select_backend(
      family = family,
      n_obs = nrow(data),
      has_spatial = !is.null(spatial),
      has_temporal = !is.null(temporal)
    )
    message(sprintf("Auto-selected backend: %s", backend))
  }

  # PG backend only works for binomial family (currently experimental)
  if (backend == "pg") {
    if (!can_use_pg_backend(family)) {
      stop(
        "PG backend only supports ratiod_binomial() family.\n",
        "For other families, use backend = 'hmc'.",
        call. = FALSE
      )
    }
    warning("PG backend is experimental. Consider using backend = 'hmc' for production.", call. = FALSE)
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
  if (!inherits(family, "ratiod_family")) {
    stop(
      "`family` must be a ratiod_family object.\n",
      "Options: ratiod_negbin_negbin(), ratiod_binomial(), ratiod_poisson_gamma()",
      call. = FALSE
    )
  }

  # Suggest alternative backends based on data characteristics
  suggest_backend(
    n_obs = nrow(data),
    family = family,
    spatial = spatial,
    temporal = temporal,
    current_backend = backend
  )

  # Parse formula
  formula_spec <- ratiod_formula(
    formula = formula,
    formula_num = formula_num,
    formula_denom = formula_denom,
    shared = shared,
    data = data
  )

  # Set default priors
  if (is.null(priors)) {
    priors <- ratiod_priors()
  }

  # -------------------------------------------------------------------------
  # Dispatch to PG backend for binomial models
  # -------------------------------------------------------------------------
  if (backend == "pg") {
    message("Fitting ratiod model with Polya-Gamma backend...")
    message(sprintf("  Family: %s", family$name))
    message(sprintf("  Observations: %d", nrow(data)))
    message(sprintf("  Iterations: %d (warmup: %d)", iter, warmup))
    message(sprintf("  Cores: %d", cores))

    result <- fit_pg_binomial(
      formula = formula_spec,
      data = data,
      family = family,
      spatial = spatial,
      iter = iter,
      warmup = warmup,
      thin = thin,
      cores = cores,
      prior_beta_sd = priors$beta_sd %||% 10,
      prior_sigma_scale = priors$sigma_scale %||% 2.5,
      prior_tau_shape = priors$tau_shape %||% 1,
      prior_tau_rate = priors$tau_rate %||% 0.01,
      verbose = !is.null(refresh) && refresh > 0
    )

    return(result)
  }

  # -------------------------------------------------------------------------
  # Dispatch to Laplace backend for fast approximate inference
  # -------------------------------------------------------------------------
  if (backend == "laplace") {
    message("Fitting ratiod model with Laplace approximation...")
    message(sprintf("  Family: %s", family$name))
    message(sprintf("  Observations: %d", nrow(data)))
    message(sprintf("  Posterior samples: %d", iter - warmup))
    message(sprintf("  Cores: %d", cores))

    result <- fit_laplace(
      formula = formula_spec,
      data = data,
      family = family,
      spatial = spatial,
      priors = priors,
      n_samples = iter - warmup,
      cores = cores,
      verbose = !is.null(refresh) && refresh > 0
    )

    return(result)
  }

  # -------------------------------------------------------------------------
  # Dispatch to HMC/NUTS backend for full MCMC without Stan
  # -------------------------------------------------------------------------
  if (backend == "hmc") {
    message("Fitting ratiod model with HMC/NUTS backend...")
    message(sprintf("  Family: %s", family$name))
    message(sprintf("  Observations: %d", nrow(data)))
    message(sprintf("  Iterations: %d (warmup: %d)", iter, warmup))
    if (!is.null(temporal)) {
      message(sprintf("  Temporal: %s (%d time points)", temporal$type, temporal$n_times))
    }

    result <- fit_hmc(
      formula = formula_spec,
      data = data,
      family = family,
      spatial = spatial,
      temporal = temporal,
      zi = zi,
      latent = latent,
      priors = priors,
      iter = iter,
      warmup = warmup,
      chains = chains,
      cores = cores,
      seed = seed,
      verbose = is.null(refresh) || refresh > 0
    )

    return(result)
  }

  # If we get here, no valid backend was selected
  stop(
    sprintf("Unknown backend: '%s'. Use one of: 'auto', 'hmc', 'pg', 'laplace'", backend),
    call. = FALSE
  )
}

# Null coalescing operator
`%||%` <- function(a, b) if (is.null(a)) b else a

#' Suggest alternative backends based on data characteristics
#'
#' @param n_obs Number of observations
#' @param family ratiod family object
#' @param spatial Spatial specification (or NULL)
#' @param temporal Temporal specification (or NULL)
#' @param current_backend Currently selected backend
#' @keywords internal
suggest_backend <- function(n_obs, family, spatial, temporal = NULL, current_backend) {

  # Thresholds for suggestions
  LARGE_DATA_THRESHOLD <- 10000
  VERY_LARGE_DATA_THRESHOLD <- 50000

  suggestions <- character(0)

  # Large dataset warnings for HMC
  if (n_obs >= VERY_LARGE_DATA_THRESHOLD && current_backend == "hmc") {
    if (!is.null(spatial)) {
      suggestions <- c(suggestions,
        sprintf("Large spatial dataset (%s observations).", format(n_obs, big.mark = ",")),
        "Consider backend = 'laplace' for faster approximate inference."
      )
    } else {
      suggestions <- c(suggestions,
        sprintf("Large dataset (%s observations).", format(n_obs, big.mark = ",")),
        "HMC may be slow. Consider:",
        "  - Reducing data via stratified sampling",
        "  - backend = 'laplace' for approximate inference"
      )
    }
  } else if (n_obs >= LARGE_DATA_THRESHOLD && current_backend == "hmc") {
    suggestions <- c(suggestions,
      sprintf("Moderately large dataset (%s observations).", format(n_obs, big.mark = ",")),
      "If fitting is slow, consider:",
      "  - Fewer iterations (iter = 1000 for initial exploration)",
      "  - backend = 'laplace' for approximate inference"
    )
  }

  # Print suggestions as a message (not warning - it's informational)
  if (length(suggestions) > 0) {
    message("\n", cli_rule("Backend suggestion"))
    for (s in suggestions) {
      message("  ", s)
    }
    message(cli_rule(), "\n")
  }

  invisible(NULL)
}

#' Simple CLI rule for messages
#' @keywords internal
cli_rule <- function(title = NULL) {
  width <- getOption("width", 80)
  if (is.null(title)) {
    paste(rep("-", min(width, 60)), collapse = "")
  } else {
    title_len <- nchar(title) + 2
    side_len <- (min(width, 60) - title_len) / 2
    paste0(
      paste(rep("-", floor(side_len)), collapse = ""),
      " ", title, " ",
      paste(rep("-", ceiling(side_len)), collapse = "")
    )
  }
}

#' Select optimal backend based on model characteristics
#'
#' @description
#' Automatically chooses the best backend for a given model based on
#' the family, data size, and presence of spatial/temporal effects.
#'
#' @param family ratiod family object
#' @param n_obs Number of observations
#' @param has_spatial Whether spatial effects are specified
#' @param has_temporal Whether temporal effects are specified
#'
#' @return Character string: "pg", "hmc", or "laplace"
#'
#' @details
#' Selection logic:
#' - **Binomial family**: Use PG (Polya-Gamma) - fastest for binomial
#' - **Spatial/temporal models**: Use HMC (only HMC supports all structure types)
#' - **Very large data (N > 50,000)**: Use Laplace (approximate but fast)
#' - **Default**: Use HMC (full MCMC, works for all families)
#'
#' @keywords internal
select_backend <- function(family, n_obs, has_spatial = FALSE, has_temporal = FALSE) {

  # Thresholds
  VERY_LARGE <- 50000
  LARGE <- 10000

  family_name <- family$name

  # Spatial/temporal models: HMC is most complete
  if (has_spatial || has_temporal) {
    return("hmc")
  }

  # Very large data: Laplace for speed (approximate but fast)
  if (n_obs > VERY_LARGE) {
    return("laplace")
  }

  # Default: HMC (full MCMC, works for everything)
  # Note: PG backend exists for binomial but is experimental
  return("hmc")
}


#' Print method for ratiod_fit
#'
#' @param x A ratiod_fit object
#' @param ... Ignored
#' @export
print.ratiod_fit <- function(x, ...) {
  cat("ratiod model fit\n")
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

#' Summary method for ratiod_fit
#'
#' @description
#' Provides a summary of the fitted model including parameter estimates,
#' credible intervals, and MCMC diagnostics. Works with all backends
#' (HMC, PG, Laplace).
#'
#' @param object A ratiod_fit object
#' @param prob Probability mass for credible intervals (default 0.95)
#' @param ... Ignored
#' @export
summary.ratiod_fit <- function(object, prob = 0.95, ...) {
  cat("ratiod model summary\n")
  cat("===================\n\n")

  backend <- object$backend %||% "unknown"
  cat("Backend:", backend, "\n")
  cat("Family:", object$family$name, "\n\n")

  # Calculate quantile probabilities
  alpha <- (1 - prob) / 2
  probs <- c(alpha, 0.5, 1 - alpha)
  ci_names <- paste0(c(alpha * 100, 50, (1 - alpha) * 100), "%")

  # Get draws
  draws <- object$draws
  if (is.null(draws)) {
    cat("No draws available\n")
    return(invisible(object))
  }

  # Compute summary statistics
  summ <- compute_param_summary(draws, probs)

  # Add diagnostics for MCMC backends
  if (backend %in% c("hmc", "pg")) {
    diag <- tryCatch(
      mcmc_diagnostics(object),
      error = function(e) NULL
    )
    if (!is.null(diag)) {
      summ <- merge(summ, diag[, c("parameter", "rhat", "ess_bulk")],
                    by = "parameter", all.x = TRUE)
    }
  }

  # Select main parameters (exclude high-dimensional)
  main_pars <- select_main_params(summ$parameter)
  summ_main <- summ[summ$parameter %in% main_pars, ]

  # Categorize and print parameters
  print_param_summary(summ_main, object$family$name, ci_names)

  # Print diagnostics summary
  print_diagnostics_summary(object)

  invisible(object)
}


#' Compute parameter summary statistics
#' @keywords internal
compute_param_summary <- function(draws, probs) {
  par_names <- colnames(draws)
  n_pars <- ncol(draws)

  means <- colMeans(draws)
  sds <- apply(draws, 2, stats::sd)
  quants <- apply(draws, 2, stats::quantile, probs = probs)

  data.frame(
    parameter = par_names,
    mean = means,
    sd = sds,
    q_lower = quants[1, ],
    q_median = quants[2, ],
    q_upper = quants[3, ],
    stringsAsFactors = FALSE
  )
}


#' Print categorized parameter summary
#' @keywords internal
print_param_summary <- function(summ, family_name, ci_names) {
  # Format columns
  summ$mean <- round(summ$mean, 3)
  summ$sd <- round(summ$sd, 3)
  summ$q_lower <- round(summ$q_lower, 3)
  summ$q_upper <- round(summ$q_upper, 3)

  if ("rhat" %in% names(summ)) {
    summ$rhat <- round(summ$rhat, 3)
    summ$ess_bulk <- round(summ$ess_bulk, 0)
  }

  # Rename CI columns
  names(summ)[names(summ) == "q_lower"] <- ci_names[1]
  names(summ)[names(summ) == "q_upper"] <- ci_names[3]

  # Select columns for display
  display_cols <- c("parameter", "mean", "sd", ci_names[1], ci_names[3])
  if ("rhat" %in% names(summ)) {
    display_cols <- c(display_cols, "rhat", "ess_bulk")
  }

  # Fixed effects numerator
  beta_num <- summ[grepl("^beta_num", summ$parameter), ]
  if (nrow(beta_num) > 0) {
    cat("Fixed effects (numerator):\n")
    print(beta_num[, display_cols, drop = FALSE], row.names = FALSE)
    cat("\n")
  }

  # Fixed effects denominator
  beta_denom <- summ[grepl("^beta_denom", summ$parameter), ]
  if (nrow(beta_denom) > 0) {
    cat("Fixed effects (denominator):\n")
    print(beta_denom[, display_cols, drop = FALSE], row.names = FALSE)
    cat("\n")
  }

  # Fixed effects (binomial - single process)
  beta_single <- summ[grepl("^\\(Intercept\\)|^beta\\[", summ$parameter), ]
  if (nrow(beta_single) > 0 && nrow(beta_num) == 0) {
    cat("Fixed effects:\n")
    print(beta_single[, display_cols, drop = FALSE], row.names = FALSE)
    cat("\n")
  }

  # Variance components
  var_pars <- summ[grepl("^(sigma|phi_num|phi_denom|tau|shape|rho)", summ$parameter), ]
  if (nrow(var_pars) > 0) {
    cat("Variance components:\n")
    print(var_pars[, display_cols, drop = FALSE], row.names = FALSE)
    cat("\n")
  }
}


#' Print diagnostics summary
#' @keywords internal
print_diagnostics_summary <- function(object) {
  backend <- object$backend %||% "unknown"

  cat("Diagnostics:\n")

  if (backend == "hmc") {
    # HMC-specific diagnostics
    diag <- object$diagnostics

    if (!is.null(diag)) {
      n_div <- diag$n_divergent %||% 0
      avg_accept <- diag$avg_accept_prob %||% NA

      cat("  Divergences:", n_div, "\n")
      if (!is.na(avg_accept)) {
        cat("  Avg. acceptance:", round(avg_accept, 3), "\n")
      }

      if (n_div > 0) {
        cat("  Warning: Divergent transitions detected.\n")
        cat("  Consider: reparameterization or increasing adapt_delta.\n")
      }
    }

    cat("  Chains:", object$chains %||% 1, "\n")
    cat("  Iterations:", object$iter %||% NA, "\n")
    cat("  Warmup:", object$warmup %||% NA, "\n")

  } else if (backend == "pg") {
    # Polya-Gamma diagnostics
    cat("  Iterations:", object$iter %||% NA, "\n")
    cat("  Warmup:", object$warmup %||% NA, "\n")
    cat("  Thin:", object$thin %||% 1, "\n")

  } else if (backend == "laplace") {
    # Laplace diagnostics
    result <- object$laplace_result %||% object$.internal
    if (!is.null(result)) {
      converged <- result$converged %||% NA
      log_marg <- result$log_marginal %||% NA

      cat("  Converged:", converged, "\n")
      if (!is.na(log_marg)) {
        cat("  Log marginal:", round(log_marg, 2), "\n")
      }
    }
    cat("  Posterior samples:", object$n_save %||% NA, "\n")
    cat("  Note: Laplace provides approximate inference.\n")
  }
}

#' Extract posterior draws from ratiod_fit
#'
#' @param x A ratiod_fit object
#' @param ... Passed to posterior::as_draws_df
#' @return A draws_df object
#' @export
as.data.frame.ratiod_fit <- function(x, ...) {
  posterior::as_draws_df(x$fit$draws(...))
}


#' Plot method for ratiod_fit
#'
#' @description
#' Creates diagnostic plots for ratiod model fits including trace plots
#' and optional density plots. Works with all backends (HMC, PG, Laplace).
#'
#' @param x A ratiod_fit object
#' @param pars Character vector of parameter names to plot. If NULL (default),
#'   plots main parameters (fixed effects, variance components).
#' @param type Type of plot: "trace" (default), "dens" (density), or "both".
#' @param n_col Number of columns in the plot grid (default: auto).
#' @param ... Additional arguments (currently ignored).
#'
#' @return Invisibly returns the plot object (base graphics or ggplot2).
#'
#' @examples
#' # Create simple dataset
#' set.seed(456)
#' df <- data.frame(
#'   count = rpois(40, 8),
#'   effort = rgamma(40, 2, 0.5),
#'   depth = rnorm(40),
#'   site = factor(sample(1:4, 40, replace = TRUE))
#' )
#'
#' \donttest{
#' fit <- ratiod(
#'   count | effort ~ depth + (1|site),
#'   data = df,
#'   family = ratiod_poisson_gamma(),
#'   iter = 200, warmup = 100, chains = 1
#' )
#' plot(fit)
#' plot(fit, pars = "beta_num", type = "both")
#' }
#'
#' @export
plot.ratiod_fit <- function(x, pars = NULL, type = c("trace", "dens", "both"),
                           n_col = NULL, ...) {
  type <- match.arg(type)

  # Get draws in array format for multi-chain support
  draws_info <- get_draws_array(x)
  draws_array <- draws_info$draws
  n_chains <- draws_info$n_chains

  # Select parameters to plot
  all_pars <- dimnames(draws_array)[[3]]
  if (is.null(pars)) {
    # Default: main parameters (exclude high-dimensional RE and spatial)
    pars <- select_main_params(all_pars)
  } else {
    # Handle regex-like selection
    pars <- grep_params(pars, all_pars)
  }

  if (length(pars) == 0) {
    stop("No parameters selected for plotting", call. = FALSE)
  }

  # Subset draws
  draws_subset <- draws_array[, , pars, drop = FALSE]

  # Use bayesplot if available, otherwise base R
  if (requireNamespace("bayesplot", quietly = TRUE)) {
    plot_with_bayesplot(draws_subset, type, n_chains)
  } else {
    plot_with_base(draws_subset, type, n_chains, n_col)
  }

  invisible(x)
}


#' Get draws in array format
#'
#' Converts draws to a 3D array with dimensions iterations x chains x parameters.
#'
#' @param fit A ratiod_fit object
#' @return A list with draws array and number of chains
#' @keywords internal
get_draws_array <- function(fit) {
  backend <- fit$backend %||% "unknown"
  chains <- fit$chains %||% 1

  if (backend == "hmc" && chains > 1) {
    # Multi-chain HMC: need to reconstruct chain structure
    # The raw samples per chain are stored in .internal or we split evenly
    draws <- fit$draws
    n_total <- nrow(draws)
    n_per_chain <- n_total / chains
    n_pars <- ncol(draws)

    draws_array <- array(
      dim = c(n_per_chain, chains, n_pars),
      dimnames = list(
        iteration = seq_len(n_per_chain),
        chain = seq_len(chains),
        parameter = colnames(draws)
      )
    )

    for (c in seq_len(chains)) {
      start_idx <- (c - 1) * n_per_chain + 1
      end_idx <- c * n_per_chain
      draws_array[, c, ] <- as.matrix(draws[start_idx:end_idx, ])
    }

    return(list(draws = draws_array, n_chains = chains))

  } else {
    # Single chain or non-HMC: treat as one chain
    draws <- fit$draws
    n_samples <- nrow(draws)
    n_pars <- ncol(draws)

    draws_array <- array(
      as.matrix(draws),
      dim = c(n_samples, 1, n_pars),
      dimnames = list(
        iteration = seq_len(n_samples),
        chain = 1,
        parameter = colnames(draws)
      )
    )

    return(list(draws = draws_array, n_chains = 1))
  }
}


#' Select main parameters (exclude high-dimensional)
#' @keywords internal
select_main_params <- function(all_pars) {
  # Keep: beta, sigma, phi, tau, rho
  # Exclude: re[...], spatial[...], phi_spatial[...], theta[...], phi_scaled[...]
  pattern <- "^(re|spatial|phi_spatial|theta|phi_scaled|ratio)\\["
  is_high_dim <- grepl(pattern, all_pars)
  main_pars <- all_pars[!is_high_dim]

  # Limit to reasonable number

  if (length(main_pars) > 12) {
    main_pars <- main_pars[1:12]
  }

  main_pars
}


#' Grep-style parameter selection
#' @keywords internal
grep_params <- function(pars, all_pars) {
  selected <- character(0)
  for (p in pars) {
    if (p %in% all_pars) {
      selected <- c(selected, p)
    } else {
      # Try as regex
      matches <- grep(p, all_pars, value = TRUE)
      selected <- c(selected, matches)
    }
  }
  unique(selected)
}


#' Plot using bayesplot
#' @keywords internal
plot_with_bayesplot <- function(draws_array, type, n_chains) {
  if (type == "trace") {
    print(bayesplot::mcmc_trace(draws_array))
  } else if (type == "dens") {
    if (n_chains > 1) {
      print(bayesplot::mcmc_dens_overlay(draws_array))
    } else {
      print(bayesplot::mcmc_dens(draws_array))
    }
  } else {
    # "both" - trace and density side by side
    print(bayesplot::mcmc_combo(draws_array, combo = c("trace", "dens_overlay")))
  }
}


#' Plot using base R graphics
#' @keywords internal
plot_with_base <- function(draws_array, type, n_chains, n_col = NULL) {
  n_pars <- dim(draws_array)[3]
  par_names <- dimnames(draws_array)[[3]]

  # Determine layout
  if (type == "both") {
    n_plots <- n_pars * 2
    if (is.null(n_col)) n_col <- 2
    n_row <- ceiling(n_pars / 1)
    graphics::par(mfrow = c(n_pars, 2), mar = c(2, 3, 2, 1), oma = c(0, 0, 2, 0))
  } else {
    n_plots <- n_pars
    if (is.null(n_col)) n_col <- min(3, ceiling(sqrt(n_pars)))
    n_row <- ceiling(n_pars / n_col)
    graphics::par(mfrow = c(n_row, n_col), mar = c(2, 3, 2, 1), oma = c(0, 0, 2, 0))
  }

  # Chain colors
  chain_cols <- c("#1b9e77", "#d95f02", "#7570b3", "#e7298a",
                  "#66a61e", "#e6ab02", "#a6761d", "#666666")

  for (i in seq_len(n_pars)) {
    par_name <- par_names[i]

    if (type %in% c("trace", "both")) {
      # Trace plot
      y_range <- range(draws_array[, , i])
      graphics::plot(
        NULL, NULL,
        xlim = c(1, dim(draws_array)[1]),
        ylim = y_range,
        xlab = "", ylab = "",
        main = par_name,
        type = "n"
      )
      for (c in seq_len(n_chains)) {
        graphics::lines(
          draws_array[, c, i],
          col = grDevices::adjustcolor(chain_cols[((c - 1) %% 8) + 1], alpha.f = 0.7),
          lwd = 0.5
        )
      }
    }

    if (type %in% c("dens", "both")) {
      # Density plot
      all_vals <- as.vector(draws_array[, , i])
      d <- stats::density(all_vals)

      if (n_chains > 1) {
        # Overlay densities per chain
        graphics::plot(d, main = par_name, xlab = "", ylab = "",
                       col = "gray", lwd = 2, type = "n")
        for (c in seq_len(n_chains)) {
          dc <- stats::density(draws_array[, c, i])
          graphics::lines(dc, col = chain_cols[((c - 1) %% 8) + 1], lwd = 1.5)
        }
      } else {
        graphics::plot(d, main = par_name, xlab = "", ylab = "",
                       col = chain_cols[1], lwd = 2)
        graphics::polygon(d, col = grDevices::adjustcolor(chain_cols[1], alpha.f = 0.3),
                          border = NA)
      }
    }
  }

  backend_label <- "ratiod MCMC diagnostics"
  graphics::mtext(backend_label, outer = TRUE, cex = 1.1)
}


#' Compute MCMC diagnostics (Rhat, ESS) for ratiod_fit
#'
#' @description
#' Computes split-Rhat and effective sample size (ESS) for all parameters
#' using the posterior package. Works with all backends.
#'
#' @param fit A ratiod_fit object
#' @param pars Character vector of parameter names (default: all).
#'
#' @return A data frame with columns: parameter, rhat, ess_bulk, ess_tail.
#'
#' @details
#' For single-chain fits, Rhat is computed using split-Rhat (splitting the

#' chain in half). Multi-chain fits use the standard multi-chain Rhat.
#'
#' @export
mcmc_diagnostics <- function(fit, pars = NULL) {
  if (!inherits(fit, "ratiod_fit")) {
    stop("fit must be a ratiod_fit object", call. = FALSE)
  }

  # Get draws in array format
  draws_info <- get_draws_array(fit)
  draws_array <- draws_info$draws

  # Select parameters
  all_pars <- dimnames(draws_array)[[3]]
  if (!is.null(pars)) {
    pars <- grep_params(pars, all_pars)
    draws_array <- draws_array[, , pars, drop = FALSE]
  }

  # Convert to posterior::draws_array for diagnostics
  if (requireNamespace("posterior", quietly = TRUE)) {
    draws_post <- posterior::as_draws_array(draws_array)

    # Compute diagnostics
    rhat_vals <- posterior::rhat(draws_post)
    ess_bulk_vals <- posterior::ess_bulk(draws_post)
    ess_tail_vals <- posterior::ess_tail(draws_post)

    # Get parameter names from the draws array
    par_names <- dimnames(draws_array)[[3]]

    result <- data.frame(
      parameter = par_names,
      rhat = as.numeric(rhat_vals),
      ess_bulk = as.numeric(ess_bulk_vals),
      ess_tail = as.numeric(ess_tail_vals),
      stringsAsFactors = FALSE
    )
  } else {
    # Fallback: basic Rhat computation without posterior package
    result <- compute_diagnostics_basic(draws_array)
  }

  class(result) <- c("ratiod_diagnostics", "data.frame")
  return(result)
}


#' Basic Rhat computation (fallback when posterior package unavailable)
#' @keywords internal
compute_diagnostics_basic <- function(draws_array) {
  n_iter <- dim(draws_array)[1]
  n_chains <- dim(draws_array)[2]
  n_pars <- dim(draws_array)[3]
  par_names <- dimnames(draws_array)[[3]]

  rhat_vals <- numeric(n_pars)
  ess_vals <- numeric(n_pars)

  for (p in seq_len(n_pars)) {
    if (n_chains == 1) {
      # Split-Rhat for single chain
      half <- floor(n_iter / 2)
      chain1 <- draws_array[1:half, 1, p]
      chain2 <- draws_array[(half + 1):(2 * half), 1, p]
      chains_split <- cbind(chain1, chain2)
      rhat_vals[p] <- compute_split_rhat(chains_split)
      ess_vals[p] <- compute_ess_basic(draws_array[, 1, p])
    } else {
      # Multi-chain Rhat
      chains_mat <- matrix(draws_array[, , p], nrow = n_iter, ncol = n_chains)
      rhat_vals[p] <- compute_split_rhat(chains_mat)
      ess_vals[p] <- compute_ess_basic(as.vector(draws_array[, , p]))
    }
  }

  data.frame(
    parameter = par_names,
    rhat = rhat_vals,
    ess_bulk = ess_vals,
    ess_tail = ess_vals,  # Simplified - same as bulk
    stringsAsFactors = FALSE
  )
}


#' Compute split-Rhat
#' @keywords internal
compute_split_rhat <- function(chains_mat) {
  # chains_mat: [iterations, chains]
  n <- nrow(chains_mat)
  m <- ncol(chains_mat)

  # Split each chain in half
  half <- floor(n / 2)
  split_chains <- matrix(NA_real_, nrow = half, ncol = 2 * m)
  for (j in seq_len(m)) {
    split_chains[, 2 * j - 1] <- chains_mat[1:half, j]
    split_chains[, 2 * j] <- chains_mat[(half + 1):(2 * half), j]
  }

  m_split <- 2 * m
  n_split <- half

  # Chain means and variances
  chain_means <- colMeans(split_chains)
  chain_vars <- apply(split_chains, 2, stats::var)

  # Between-chain variance
  B <- n_split * stats::var(chain_means)

  # Within-chain variance
  W <- mean(chain_vars)

  # Estimated variance
  var_plus <- ((n_split - 1) / n_split) * W + (1 / n_split) * B

  # Rhat
  rhat <- sqrt(var_plus / W)

  return(rhat)
}


#' Basic ESS computation
#' @keywords internal
compute_ess_basic <- function(x) {
  n <- length(x)
  if (n < 10) return(n)

  # Autocorrelation
  acf_vals <- stats::acf(x, lag.max = min(n - 1, 100), plot = FALSE)$acf[-1]

  # Find first negative autocorrelation
  first_neg <- which(acf_vals < 0)[1]
  if (is.na(first_neg)) first_neg <- length(acf_vals)

  # Sum of autocorrelations (Geyer's method simplified)
  tau <- 1 + 2 * sum(acf_vals[1:min(first_neg, length(acf_vals))])
  tau <- max(tau, 1)

  ess <- n / tau
  return(ess)
}


#' Print method for ratiod_diagnostics
#'
#' @param x A ratiod_diagnostics object
#' @param ... Ignored
#'
#' @export
print.ratiod_diagnostics <- function(x, ...) {
  cat("MCMC Diagnostics\n")
  cat("================\n\n")

  # Format for printing
  x$rhat <- round(x$rhat, 3)
  x$ess_bulk <- round(x$ess_bulk, 0)
  x$ess_tail <- round(x$ess_tail, 0)

  print.data.frame(x, row.names = FALSE)

  # Warnings
  bad_rhat <- x$rhat > 1.01
  low_ess <- x$ess_bulk < 400 | x$ess_tail < 400

  if (any(bad_rhat, na.rm = TRUE)) {
    cat("\nWarning: ", sum(bad_rhat, na.rm = TRUE),
        " parameter(s) have Rhat > 1.01\n")
  }
  if (any(low_ess, na.rm = TRUE)) {
    cat("Warning: ", sum(low_ess, na.rm = TRUE),
        " parameter(s) have ESS < 400\n")
  }

  invisible(x)
}


#' Check MCMC diagnostics and print warnings
#'
#' @description
#' Checks MCMC diagnostics (Rhat, ESS, divergences) and prints actionable
#' warnings with suggestions for fixing issues.
#'
#' @param fit A ratiod_fit object
#' @param quiet Logical; if TRUE, suppress output and just return status.
#'
#' @return Invisibly returns a list with diagnostic status:
#' \describe{
#'   \item{ok}{Logical; TRUE if all diagnostics pass}
#'   \item{n_divergent}{Number of divergent transitions}
#'   \item{n_bad_rhat}{Number of parameters with Rhat > 1.01}
#'   \item{n_low_ess}{Number of parameters with ESS < 400}
#' }
#'
#' @export
check_diagnostics <- function(fit, quiet = FALSE) {
  if (!inherits(fit, "ratiod_fit")) {
    stop("fit must be a ratiod_fit object", call. = FALSE)
  }

  backend <- fit$backend %||% "unknown"
  issues <- list()

  # Check divergences (HMC only)
  n_divergent <- 0
  if (backend == "hmc") {
    diag <- fit$diagnostics
    if (!is.null(diag)) {
      n_divergent <- diag$n_divergent %||% 0
    }
  }

  # Check Rhat and ESS
  n_bad_rhat <- 0
  n_low_ess <- 0

  if (backend %in% c("hmc", "pg")) {
    mcmc_diag <- tryCatch(
      mcmc_diagnostics(fit, pars = NULL),
      error = function(e) NULL
    )

    if (!is.null(mcmc_diag)) {
      # Filter to main parameters only
      main_pars <- select_main_params(mcmc_diag$parameter)
      mcmc_diag <- mcmc_diag[mcmc_diag$parameter %in% main_pars, ]

      bad_rhat <- !is.na(mcmc_diag$rhat) & mcmc_diag$rhat > 1.01
      n_bad_rhat <- sum(bad_rhat)

      low_ess <- (!is.na(mcmc_diag$ess_bulk) & mcmc_diag$ess_bulk < 400) |
                 (!is.na(mcmc_diag$ess_tail) & mcmc_diag$ess_tail < 400)
      n_low_ess <- sum(low_ess)
    }
  }

  # Determine overall status
  ok <- (n_divergent == 0) && (n_bad_rhat == 0) && (n_low_ess == 0)

  # Print warnings if not quiet
  if (!quiet) {
    cat("Diagnostic Check\n")
    cat("================\n\n")

    if (ok) {
      cat("All diagnostics OK.\n")
    } else {
      if (n_divergent > 0) {
        cat("DIVERGENCES: ", n_divergent, " divergent transition(s) detected.\n")
        cat("  This indicates the sampler encountered regions of high curvature.\n")
        cat("  Suggestions:\n")
        cat("    - Increase iter and warmup\n")
        cat("    - Try different initialization (seed)\n")
        cat("    - Consider reparameterization\n")
        cat("    - For spatial models, check adjacency structure\n\n")
      }

      if (n_bad_rhat > 0) {
        cat("RHAT: ", n_bad_rhat, " parameter(s) have Rhat > 1.01.\n")
        cat("  This indicates chains have not converged.\n")
        cat("  Suggestions:\n")
        cat("    - Run more iterations\n")
        cat("    - Run more chains\n")
        cat("    - Check for multimodality with plot(fit)\n\n")
      }

      if (n_low_ess > 0) {
        cat("ESS: ", n_low_ess, " parameter(s) have ESS < 400.\n")
        cat("  This indicates high autocorrelation in samples.\n")
        cat("  Suggestions:\n")
        cat("    - Run more iterations\n")
        cat("    - Increase thinning interval\n")
        cat("    - Consider reparameterization\n\n")
      }
    }

    if (backend == "laplace") {
      cat("Note: Laplace backend uses approximate inference.\n")
      cat("      Consider HMC backend for full uncertainty quantification.\n")
    }
  }

  result <- list(
    ok = ok,
    n_divergent = n_divergent,
    n_bad_rhat = n_bad_rhat,
    n_low_ess = n_low_ess
  )

  invisible(result)
}


#' Get number of divergent transitions
#'
#' @param fit A ratiod_fit object
#' @return Integer count of divergent transitions
#' @export
n_divergent <- function(fit) {
  if (!inherits(fit, "ratiod_fit")) {
    stop("fit must be a ratiod_fit object", call. = FALSE)
  }

  if (fit$backend == "hmc") {
    diag <- fit$diagnostics
    if (!is.null(diag)) {
      return(diag$n_divergent %||% 0)
    }
  }

  return(0)
}
