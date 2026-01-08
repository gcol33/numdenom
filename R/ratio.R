#' Extract ratio posteriors from a quotr model
#'
#' @description
#' Compute posterior distributions of ratios from a fitted quotr model.
#' Ratios are derived quantities: E\[numerator\] / E\[denominator\], computed
#' in Stan's generated quantities block with full uncertainty propagation.
#'
#' @param object A `quotr_fit` object
#' @param newdata Optional data frame for prediction. If NULL, uses
#'   original data.
#' @param type Scale for ratio: "response" (default), "log", or "logit".
#' @param by Optional grouping variable name to aggregate ratios by group.
#' @param summary Logical; if TRUE, return summary statistics instead
#'   of full posterior draws.
#' @param probs Quantiles to compute if `summary = TRUE`.
#' @param ... Ignored
#'
#' @return A `quotr_ratio` object containing posterior draws or summaries.
#'
#' @examples
#' \dontrun{
#' # Extract all observation-level ratios
#' r <- ratio(fit)
#' summary(r)
#'
#' # Log-ratio scale
#' r_log <- ratio(fit, type = "log")
#'
#' # Group-level ratios
#' r_site <- ratio(fit, by = "site")
#' }
#'
#' @export
ratio <- function(object, newdata = NULL, type = c("response", "log", "logit"),
                  by = NULL, summary = FALSE, probs = c(0.025, 0.5, 0.975), ...) {
  UseMethod("ratio")
}

#' @export
ratio.quotr_fit <- function(object, newdata = NULL, type = c("response", "log", "logit"),
                            by = NULL, summary = FALSE, probs = c(0.025, 0.5, 0.975), ...) {

  type <- match.arg(type)

  # Handle newdata via predict()
 if (!is.null(newdata)) {
    pred <- predict(object, newdata = newdata, component = "ratio", summary = FALSE)
    ratio_draws <- pred$ratio
  } else {
    # Get ratio draws from fitted values or stored ratio_draws
    if (!is.null(object$ratio_draws)) {
      # HMC backend stores pre-computed ratio draws
      ratio_draws <- object$ratio_draws
    } else {
      # Compute from fitted values
      fitted_vals <- compute_fitted_values(object)
      ratio_draws <- fitted_vals$ratio
    }
  }

  # Transform to requested scale
  if (type == "log") {
    ratio_draws <- log(ratio_draws)
  } else if (type == "logit") {
    # Logit only makes sense if ratio is a proportion (0-1)
    if (any(ratio_draws > 1, na.rm = TRUE) || any(ratio_draws < 0, na.rm = TRUE)) {
      warning("Logit transform requested but ratios are not in (0, 1). ",
              "Results may include Inf/-Inf.", call. = FALSE)
    }
    ratio_draws <- log(ratio_draws / (1 - ratio_draws))
  }

  # Ensure column names
  if (is.null(colnames(ratio_draws))) {
    colnames(ratio_draws) <- paste0("ratio[", seq_len(ncol(ratio_draws)), "]")
  }

  # Aggregate by group if requested
  if (!is.null(by)) {
    ratio_draws <- aggregate_by_group(ratio_draws, object$data, by)
  }

  # Create result object
  result <- structure(
    list(
      draws = ratio_draws,
      type = type,
      by = by,
      n_obs = if (is.null(by)) nrow(object$data) else length(unique(object$data[[by]])),
      n_draws = nrow(ratio_draws),
      data = object$data
    ),
    class = "quotr_ratio"
  )

  if (summary) {
    return(summary(result, probs = probs))
  }

  result
}

#' Aggregate ratio draws by grouping variable
#'
#' @param draws Matrix of posterior draws
#' @param data Original data frame
#' @param by Grouping variable name
#'
#' @return Matrix with columns for each group level
#' @keywords internal
aggregate_by_group <- function(draws, data, by) {
  if (!(by %in% names(data))) {
    stop(sprintf("Grouping variable '%s' not found in data", by), call. = FALSE)
  }

  groups <- unique(data[[by]])
  n_groups <- length(groups)
  n_draws <- nrow(draws)

  # Average ratios within each group
  result <- matrix(NA_real_, nrow = n_draws, ncol = n_groups)
  colnames(result) <- paste0("ratio[", groups, "]")

  for (g in seq_along(groups)) {
    idx <- which(data[[by]] == groups[g])
    if (length(idx) == 1) {
      result[, g] <- draws[, idx]
    } else {
      # Geometric mean for ratios (arithmetic mean on log scale)
      result[, g] <- exp(rowMeans(log(draws[, idx, drop = FALSE])))
    }
  }

  result
}

#' Summary method for quotr_ratio
#'
#' @param object A quotr_ratio object
#' @param probs Quantiles to compute
#' @param ... Ignored
#'
#' @export
summary.quotr_ratio <- function(object, probs = c(0.025, 0.5, 0.975), ...) {
  draws <- object$draws

  # Compute summaries
  summaries <- data.frame(
    mean = colMeans(draws),
    sd = apply(draws, 2, sd),
    t(apply(draws, 2, quantile, probs = probs))
  )

  # Clean up column names for quantiles
  names(summaries)[3:ncol(summaries)] <- paste0("q", probs * 100)

  # Add observation/group index
  if (is.null(object$by)) {
    summaries$obs <- seq_len(object$n_obs)
    summaries <- summaries[, c("obs", "mean", "sd", names(summaries)[3:(ncol(summaries)-1)])]
  } else {
    summaries$group <- gsub("^ratio\\[|\\]$", "", rownames(summaries))
    summaries <- summaries[, c("group", "mean", "sd", names(summaries)[3:(ncol(summaries)-1)])]
  }

  rownames(summaries) <- NULL

  structure(
    summaries,
    type = object$type,
    n_draws = object$n_draws,
    class = c("quotr_ratio_summary", "data.frame")
  )
}

#' Print method for quotr_ratio
#'
#' @param x A quotr_ratio object
#' @param ... Ignored
#'
#' @export
print.quotr_ratio <- function(x, ...) {
  cat("quotr ratio posterior\n")
  cat("=====================\n\n")
  cat("Scale:", x$type, "\n")
  cat("Observations:", x$n_obs, "\n")
  cat("Posterior draws:", x$n_draws, "\n")
  if (!is.null(x$by)) {
    cat("Grouped by:", x$by, "\n")
  }
  cat("\nUse summary() for posterior summaries\n")
  cat("Use plot() for visualization\n")
  invisible(x)
}

#' Print method for quotr_ratio_summary
#'
#' @param x A quotr_ratio_summary object
#' @param n Number of rows to print
#' @param ... Passed to print.data.frame
#'
#' @export
print.quotr_ratio_summary <- function(x, n = 10, ...) {
  cat("quotr ratio summary (", attr(x, "type"), " scale)\n", sep = "")
  cat("===========================================\n\n")

  if (nrow(x) > n) {
    print.data.frame(head(x, n), row.names = FALSE, ...)
    cat(sprintf("\n... and %d more rows\n", nrow(x) - n))
  } else {
    print.data.frame(x, row.names = FALSE, ...)
  }

  invisible(x)
}

#' Compute contrasts between ratio conditions
#'
#' @description
#' Compute posterior distributions of contrasts (differences or ratios)
#' between conditions for the derived ratio quantity.
#'
#' @param object A `quotr_fit` object
#' @param contrast A formula specifying the contrast, e.g., `~ treatment`
#' @param type Type of contrast: "difference" (default) or "ratio"
#' @param ref Reference level for factor contrasts (default: first level)
#' @param ... Ignored
#'
#' @return A data frame with posterior summaries of contrasts
#'
#' @examples
#' \dontrun{
#' # Compare CPUE between seasons
#' ratio_contrast(fit, ~ season)
#'
#' # Ratio of ratios (multiplicative comparison)
#' ratio_contrast(fit, ~ season, type = "ratio")
#' }
#'
#' @export
ratio_contrast <- function(object, contrast, type = c("difference", "ratio"),
                           ref = NULL, ...) {

  type <- match.arg(type)

  # Extract contrast variable from formula
  if (!inherits(contrast, "formula")) {
    stop("`contrast` must be a formula, e.g., ~ treatment", call. = FALSE)
  }

  contrast_var <- all.vars(contrast)
  if (length(contrast_var) != 1) {
    stop("contrast formula must contain exactly one variable", call. = FALSE)
  }

  if (!(contrast_var %in% names(object$data))) {
    stop(sprintf("Variable '%s' not found in data", contrast_var), call. = FALSE)
  }

  # Get ratio draws on log scale
  ratio_obj <- ratio(object, type = "log")
  draws <- ratio_obj$draws

  # Handle case where draws might be a list (from predict)
  if (is.list(draws) && !is.matrix(draws)) {
    draws <- draws$ratio
  }

  # Get levels
  levels <- unique(object$data[[contrast_var]])
  if (is.null(ref)) {
    ref <- levels[1]
  }
  if (!(ref %in% levels)) {
    stop(sprintf("Reference level '%s' not found", ref), call. = FALSE)
  }

  # Compute contrasts
  ref_idx <- which(object$data[[contrast_var]] == ref)
  ref_mean <- rowMeans(draws[, ref_idx, drop = FALSE])

  results <- list()
  for (lvl in levels) {
    if (lvl == ref) next

    lvl_idx <- which(object$data[[contrast_var]] == lvl)
    lvl_mean <- rowMeans(draws[, lvl_idx, drop = FALSE])

    if (type == "difference") {
      # Difference on log scale = log ratio
      contrast_draws <- lvl_mean - ref_mean
    } else {
      # Ratio = exp(difference on log scale)
      contrast_draws <- exp(lvl_mean - ref_mean)
    }

    results[[as.character(lvl)]] <- data.frame(
      contrast = paste(lvl, "vs", ref),
      mean = mean(contrast_draws),
      sd = sd(contrast_draws),
      q2.5 = quantile(contrast_draws, 0.025),
      q50 = quantile(contrast_draws, 0.5),
      q97.5 = quantile(contrast_draws, 0.975),
      prob_positive = mean(contrast_draws > 0)
    )
  }

  result <- do.call(rbind, results)
  rownames(result) <- NULL

  structure(
    result,
    type = type,
    contrast_var = contrast_var,
    ref = ref,
    class = c("quotr_contrast", "data.frame")
  )
}

#' Print method for quotr_contrast
#'
#' @param x A quotr_contrast object
#' @param ... Passed to print.data.frame
#'
#' @export
print.quotr_contrast <- function(x, ...) {
  cat("quotr ratio contrasts\n")
  cat("=====================\n\n")
  cat("Type:", attr(x, "type"), "\n")
  cat("Variable:", attr(x, "contrast_var"), "\n")
  cat("Reference:", attr(x, "ref"), "\n\n")
  print.data.frame(x, row.names = FALSE, ...)
  invisible(x)
}


#' Extract fitted values from a quotr model
#'
#' @description
#' Compute posterior distributions of fitted values (expected values) for both
#' the numerator and denominator processes from a fitted quotr model.
#'
#' @param object A `quotr_fit` object
#' @param component Which component to return: "numerator", "denominator",
#'   "ratio", or "all" (default).
#' @param summary Logical; if TRUE, return summary statistics instead of
#'   full posterior draws (default TRUE).
#' @param probs Quantiles to compute if `summary = TRUE`.
#' @param ... Ignored
#'
#' @return If `summary = TRUE`, a data frame with posterior summaries.
#'   If `summary = FALSE`, a list with matrices of posterior draws.
#'
#' @examples
#' \dontrun{
#' # Get fitted value summaries
#' fitted(fit)
#'
#' # Get just numerator fitted values
#' fitted(fit, component = "numerator")
#'
#' # Get full posterior draws
#' fitted(fit, summary = FALSE)
#' }
#'
#' @method fitted quotr_fit
#' @export
fitted.quotr_fit <- function(object, component = c("all", "numerator", "denominator", "ratio"),
                              summary = TRUE, probs = c(0.025, 0.5, 0.975), ...) {

  component <- match.arg(component)

  # Compute fitted values based on backend
  fitted_vals <- compute_fitted_values(object)

  # Select requested component
  if (component == "all") {
    draws_list <- fitted_vals
  } else if (component == "ratio") {
    draws_list <- list(ratio = fitted_vals$ratio)
  } else {
    draws_list <- list()
    draws_list[[component]] <- fitted_vals[[component]]
  }

  if (summary) {
    # Compute summaries for each component
    result_list <- list()
    for (comp_name in names(draws_list)) {
      draws <- draws_list[[comp_name]]
      summaries <- data.frame(
        obs = seq_len(ncol(draws)),
        component = comp_name,
        mean = colMeans(draws),
        sd = apply(draws, 2, sd),
        t(apply(draws, 2, quantile, probs = probs))
      )
      names(summaries)[5:ncol(summaries)] <- paste0("q", probs * 100)
      rownames(summaries) <- NULL
      result_list[[comp_name]] <- summaries
    }

    result <- do.call(rbind, result_list)
    rownames(result) <- NULL

    structure(
      result,
      n_draws = nrow(draws_list[[1]]),
      class = c("quotr_fitted", "data.frame")
    )
  } else {
    structure(
      draws_list,
      n_draws = nrow(draws_list[[1]]),
      n_obs = ncol(draws_list[[1]]),
      class = "quotr_fitted_draws"
    )
  }
}


#' Compute fitted values from quotr_fit
#' @keywords internal
compute_fitted_values <- function(object) {
  backend <- object$backend

  if (backend == "hmc") {
    return(compute_fitted_hmc(object))
  } else if (backend == "pg") {
    return(compute_fitted_pg(object))
  } else if (backend == "laplace") {
    return(compute_fitted_laplace(object))
  } else {
    stop("Unknown backend: ", backend, call. = FALSE)
  }
}


#' Compute fitted values for HMC backend
#' @keywords internal
compute_fitted_hmc <- function(object) {
  hmc_data <- object$.internal$hmc_data
  model_type <- object$.internal$model_type
  samples <- object$.internal$samples

  n_samples <- nrow(samples)
  N <- hmc_data$N

  # Extract parameters
  beta_num <- samples[, seq_len(hmc_data$p_num), drop = FALSE]
  beta_denom <- samples[, hmc_data$p_num + seq_len(hmc_data$p_denom), drop = FALSE]

  idx <- hmc_data$p_num + hmc_data$p_denom + 1

  # Random effects
  re <- NULL
  if (hmc_data$n_re_groups > 0) {
    idx <- idx + 1  # Skip log_sigma_re
    re <- samples[, idx:(idx + hmc_data$n_re_groups - 1), drop = FALSE]
    idx <- idx + hmc_data$n_re_groups
  }

  # Skip overdispersion params
  if (model_type == "negbin_negbin") {
    idx <- idx + 2
  } else if (model_type == "poisson_gamma") {
    idx <- idx + 1
  }

  # Spatial
  spatial_info <- object$spatial
  spatial_effect <- NULL
  if (!is.null(spatial_info) && spatial_info$type != "none") {
    if (spatial_info$type == "icar") {
      idx <- idx + 1
      phi_spatial <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]
      spatial_effect <- phi_spatial
    } else if (spatial_info$type == "bym2") {
      sigma_spatial <- exp(samples[, idx])
      idx <- idx + 1
      logit_rho <- samples[, idx]
      rho <- 1 / (1 + exp(-logit_rho))
      idx <- idx + 1

      phi_scaled <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]
      idx <- idx + spatial_info$n_units
      theta <- samples[, idx:(idx + spatial_info$n_units - 1), drop = FALSE]

      spatial_effect <- matrix(0, nrow = n_samples, ncol = spatial_info$n_units)
      for (s in seq_len(n_samples)) {
        spatial_effect[s, ] <- sigma_spatial[s] * (
          sqrt(rho[s]) * phi_scaled[s, ] * spatial_info$bym2_scale +
          sqrt(1 - rho[s]) * theta[s, ]
        )
      }
    }
  }

  # Compute fitted values
  mu_num <- matrix(NA_real_, nrow = n_samples, ncol = N)
  mu_denom <- matrix(NA_real_, nrow = n_samples, ncol = N)

  for (s in seq_len(n_samples)) {
    eta_num <- as.numeric(hmc_data$X_num %*% beta_num[s, ])
    eta_denom <- as.numeric(hmc_data$X_denom %*% beta_denom[s, ])

    if (!is.null(re)) {
      for (i in seq_len(N)) {
        g <- hmc_data$re_group[i]
        if (g > 0) {
          eta_num[i] <- eta_num[i] + re[s, g]
          eta_denom[i] <- eta_denom[i] + re[s, g]
        }
      }
    }

    if (!is.null(spatial_effect) && !is.null(spatial_info)) {
      for (i in seq_len(N)) {
        sp_g <- spatial_info$group[i]
        if (sp_g > 0) {
          eta_num[i] <- eta_num[i] + spatial_effect[s, sp_g]
          eta_denom[i] <- eta_denom[i] + spatial_effect[s, sp_g]
        }
      }
    }

    # Transform to response scale
    if (model_type == "binomial") {
      mu_num[s, ] <- 1 / (1 + exp(-eta_num))
      mu_denom[s, ] <- rep(1, N)  # Trials are data, not fitted
    } else {
      mu_num[s, ] <- exp(eta_num)
      mu_denom[s, ] <- exp(eta_denom)
    }
  }

  # Compute ratio
  if (model_type == "binomial") {
    ratio <- mu_num  # For binomial, ratio is just the probability
  } else {
    ratio <- mu_num / mu_denom
  }

  list(
    numerator = mu_num,
    denominator = mu_denom,
    ratio = ratio
  )
}


#' Compute fitted values for PG backend
#' @keywords internal
compute_fitted_pg <- function(object) {
  # PG backend stores eta (linear predictor) directly
  eta <- object$.internal$eta

  # For PG (binomial), fitted value is logistic(eta)
  p <- 1 / (1 + exp(-eta))

  list(
    numerator = p,
    denominator = matrix(1, nrow = nrow(p), ncol = ncol(p)),
    ratio = p
  )
}


#' Compute fitted values for Laplace backend
#' @keywords internal
compute_fitted_laplace <- function(object) {
  samples <- object$.internal$samples
  X <- object$.internal$X
  re_info <- object$.internal$re_info

  n_samples <- nrow(samples)
  p <- ncol(X)
  N <- nrow(X)

  # Extract beta samples
  beta <- samples[, seq_len(p), drop = FALSE]

  # Compute linear predictor
  eta <- matrix(NA_real_, nrow = n_samples, ncol = N)
  for (s in seq_len(n_samples)) {
    eta[s, ] <- as.numeric(X %*% beta[s, ])
  }

  # Add RE if present
  if (re_info$n_groups > 0) {
    re_start <- p + 1
    re <- samples[, re_start:(re_start + re_info$n_groups - 1), drop = FALSE]
    for (s in seq_len(n_samples)) {
      for (i in seq_len(N)) {
        g <- as.integer(re_info$group_idx[i])
        if (g > 0 && g <= re_info$n_groups) {
          eta[s, i] <- eta[s, i] + re[s, g]
        }
      }
    }
  }

  # Transform (assumes binomial for Laplace - could extend)
  p_fitted <- 1 / (1 + exp(-eta))

  list(
    numerator = p_fitted,
    denominator = matrix(1, nrow = n_samples, ncol = N),
    ratio = p_fitted
  )
}


#' Predict method for quotr models
#'
#' @description
#' Compute posterior predictions for new data from a fitted quotr model.
#'
#' @param object A `quotr_fit` object
#' @param newdata Data frame with new observations. Must contain all
#'   predictor variables used in the original model.
#' @param type Type of prediction: "response" for fitted values on
#'   the response scale, "link" for linear predictor scale.
#' @param component Which component to predict: "ratio" (default),
#'   "numerator", "denominator", or "all".
#' @param summary Logical; if TRUE, return summary statistics instead of
#'   full posterior draws (default TRUE).
#' @param probs Quantiles to compute if `summary = TRUE`.
#' @param re_formula Formula for random effects in prediction.
#'   Use `NULL` to include all random effects (if new levels match),
#'   `NA` or `~ 0` to exclude random effects (population-level prediction).
#' @param allow_new_levels Logical; if TRUE, allows new group levels not

#'   in the original data (predictions use population mean).
#' @param ... Ignored
#'
#' @return If `summary = TRUE`, a data frame with posterior summaries.
#'   If `summary = FALSE`, a matrix of posterior draws (rows = draws, cols = obs).
#'
#' @examples
#' \dontrun{
#' # Predict for new data
#' new_df <- data.frame(depth = c(10, 20, 30), season = "summer")
#' predict(fit, newdata = new_df)
#'
#' # Population-level prediction (no random effects)
#' predict(fit, newdata = new_df, re_formula = NA)
#'
#' # Get full posterior draws
#' predict(fit, newdata = new_df, summary = FALSE)
#' }
#'
#' @method predict quotr_fit
#' @export
predict.quotr_fit <- function(object, newdata, type = c("response", "link"),
                               component = c("ratio", "numerator", "denominator", "all"),
                               summary = TRUE, probs = c(0.025, 0.5, 0.975),
                               re_formula = NULL, allow_new_levels = FALSE, ...) {

  type <- match.arg(type)
  component <- match.arg(component)

  if (missing(newdata) || is.null(newdata)) {
    # Use fitted values for original data
    return(fitted(object, component = component, summary = summary, probs = probs))
  }

  # Check backend
  backend <- object$backend

  # For now, only support HMC backend for predictions
  if (backend != "hmc") {
    stop("predict() with newdata currently only supported for HMC backend.\n",
         "For other backends, use fitted() for in-sample fitted values.",
         call. = FALSE)
  }

  # Build design matrices for new data
  pred_data <- build_prediction_data(object, newdata, re_formula, allow_new_levels)

  # Compute predictions
  pred_draws <- compute_predictions_hmc(object, pred_data, type)

  # Select component
  if (component == "all") {
    draws_list <- pred_draws
  } else {
    draws_list <- list()
    draws_list[[component]] <- pred_draws[[component]]
  }

  if (summary) {
    result_list <- list()
    for (comp_name in names(draws_list)) {
      draws <- draws_list[[comp_name]]
      summaries <- data.frame(
        obs = seq_len(ncol(draws)),
        component = comp_name,
        mean = colMeans(draws),
        sd = apply(draws, 2, sd),
        t(apply(draws, 2, quantile, probs = probs))
      )
      names(summaries)[5:ncol(summaries)] <- paste0("q", probs * 100)
      rownames(summaries) <- NULL
      result_list[[comp_name]] <- summaries
    }

    result <- do.call(rbind, result_list)
    rownames(result) <- NULL

    structure(
      result,
      n_draws = nrow(draws_list[[1]]),
      newdata = newdata,
      class = c("quotr_prediction", "data.frame")
    )
  } else {
    structure(
      draws_list,
      n_draws = nrow(draws_list[[1]]),
      n_obs = ncol(draws_list[[1]]),
      newdata = newdata,
      class = "quotr_prediction_draws"
    )
  }
}


#' Build prediction data structures
#' @keywords internal
build_prediction_data <- function(object, newdata, re_formula, allow_new_levels) {
  formula <- object$formula

  # Build design matrix for numerator
  # Extract fixed effect terms from formula
  num_terms <- formula$numerator$terms
  if (is.null(num_terms)) {
    # Fall back to using X column names
    X_num <- model.matrix(~ 1, data = newdata)
    colnames_orig <- colnames(object$.internal$hmc_data$X_num)
    if (ncol(X_num) != length(colnames_orig)) {
      # Try to match the original predictors
      pred_vars <- setdiff(colnames_orig, "(Intercept)")
      if (length(pred_vars) > 0) {
        fmla <- as.formula(paste("~", paste(pred_vars, collapse = " + ")))
        X_num <- model.matrix(fmla, data = newdata)
      }
    }
  } else {
    X_num <- model.matrix(num_terms, data = newdata)
  }

  # Build design matrix for denominator
  denom_terms <- formula$denominator$terms
  if (is.null(denom_terms)) {
    X_denom <- model.matrix(~ 1, data = newdata)
    colnames_orig <- colnames(object$.internal$hmc_data$X_denom)
    if (ncol(X_denom) != length(colnames_orig)) {
      pred_vars <- setdiff(colnames_orig, "(Intercept)")
      if (length(pred_vars) > 0) {
        fmla <- as.formula(paste("~", paste(pred_vars, collapse = " + ")))
        X_denom <- model.matrix(fmla, data = newdata)
      }
    }
  } else {
    X_denom <- model.matrix(denom_terms, data = newdata)
  }

  # Handle random effects
  include_re <- !isTRUE(is.na(re_formula)) && !identical(re_formula, ~ 0)
  re_group <- rep(0L, nrow(newdata))

  if (include_re && object$.internal$hmc_data$n_re_groups > 0) {
    # Get RE grouping variable
    re_info <- formula$numerator$random_effects
    if (!is.null(re_info) && length(re_info) > 0) {
      re_var <- re_info[[1]]$group_var
      if (re_var %in% names(newdata)) {
        # Map new levels to original levels
        orig_data <- object$data
        orig_levels <- unique(orig_data[[re_var]])

        for (i in seq_len(nrow(newdata))) {
          new_level <- newdata[[re_var]][i]
          match_idx <- match(new_level, orig_levels)
          if (!is.na(match_idx)) {
            re_group[i] <- match_idx
          } else if (!allow_new_levels) {
            stop("New level '", new_level, "' found in '", re_var,
                 "'. Set allow_new_levels = TRUE to predict at population level.",
                 call. = FALSE)
          }
          # If allow_new_levels and no match, re_group stays 0 (no RE contribution)
        }
      }
    }
  }

  list(
    X_num = X_num,
    X_denom = X_denom,
    re_group = re_group,
    N = nrow(newdata),
    include_re = include_re
  )
}


#' Compute predictions for HMC backend
#' @keywords internal
compute_predictions_hmc <- function(object, pred_data, type) {
  samples <- object$.internal$samples
  hmc_data <- object$.internal$hmc_data
  model_type <- object$.internal$model_type

  n_samples <- nrow(samples)
  N <- pred_data$N

  # Extract parameters
  beta_num <- samples[, seq_len(hmc_data$p_num), drop = FALSE]
  beta_denom <- samples[, hmc_data$p_num + seq_len(hmc_data$p_denom), drop = FALSE]

  idx <- hmc_data$p_num + hmc_data$p_denom + 1

  # Random effects
  re <- NULL
  if (hmc_data$n_re_groups > 0 && pred_data$include_re) {
    idx <- idx + 1  # Skip log_sigma_re
    re <- samples[, idx:(idx + hmc_data$n_re_groups - 1), drop = FALSE]
  }

  # Compute predictions
  eta_num <- matrix(NA_real_, nrow = n_samples, ncol = N)
  eta_denom <- matrix(NA_real_, nrow = n_samples, ncol = N)

  for (s in seq_len(n_samples)) {
    eta_num[s, ] <- as.numeric(pred_data$X_num %*% beta_num[s, ])
    eta_denom[s, ] <- as.numeric(pred_data$X_denom %*% beta_denom[s, ])

    if (!is.null(re)) {
      for (i in seq_len(N)) {
        g <- pred_data$re_group[i]
        if (g > 0) {
          eta_num[s, i] <- eta_num[s, i] + re[s, g]
          eta_denom[s, i] <- eta_denom[s, i] + re[s, g]
        }
      }
    }
  }

  if (type == "link") {
    # Return on linear predictor scale
    if (model_type == "binomial") {
      ratio <- eta_num
    } else {
      ratio <- eta_num - eta_denom
    }
    return(list(
      numerator = eta_num,
      denominator = eta_denom,
      ratio = ratio
    ))
  }

  # Transform to response scale
  if (model_type == "binomial") {
    mu_num <- 1 / (1 + exp(-eta_num))
    mu_denom <- matrix(1, nrow = n_samples, ncol = N)
    ratio <- mu_num
  } else {
    mu_num <- exp(eta_num)
    mu_denom <- exp(eta_denom)
    ratio <- mu_num / mu_denom
  }

  list(
    numerator = mu_num,
    denominator = mu_denom,
    ratio = ratio
  )
}


#' Print method for quotr_fitted
#' @export
print.quotr_fitted <- function(x, n = 10, ...) {
  cat("quotr fitted values\n")
  cat("===================\n\n")
  cat("Posterior draws:", attr(x, "n_draws"), "\n\n")

  if (nrow(x) > n) {
    print.data.frame(head(x, n), row.names = FALSE, ...)
    cat(sprintf("\n... and %d more rows\n", nrow(x) - n))
  } else {
    print.data.frame(x, row.names = FALSE, ...)
  }
  invisible(x)
}


#' Print method for quotr_prediction
#' @export
print.quotr_prediction <- function(x, n = 10, ...) {
  cat("quotr predictions\n")
  cat("=================\n\n")
  cat("Posterior draws:", attr(x, "n_draws"), "\n")
  cat("New observations:", nrow(attr(x, "newdata")), "\n\n")

  if (nrow(x) > n) {
    print.data.frame(head(x, n), row.names = FALSE, ...)
    cat(sprintf("\n... and %d more rows\n", nrow(x) - n))
  } else {
    print.data.frame(x, row.names = FALSE, ...)
  }
  invisible(x)
}
