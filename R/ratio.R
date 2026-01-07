#' Extract ratio posteriors from a quotr model
#'
#' @description
#' Compute posterior distributions of ratios from a fitted quotr model.
#' Ratios are derived quantities: E[numerator] / E[denominator], computed
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

  # For now, only support original data (prediction comes later)
  if (!is.null(newdata)) {
    stop("Prediction with newdata not yet implemented", call. = FALSE)
  }

  # Extract ratio draws from generated quantities
  draws <- object$stanfit$draws(variables = c("ratio", "log_ratio"), format = "matrix")

  # Get the appropriate scale
  if (type == "response") {
    ratio_draws <- draws[, grep("^ratio\\[", colnames(draws)), drop = FALSE]
  } else if (type == "log") {
    ratio_draws <- draws[, grep("^log_ratio\\[", colnames(draws)), drop = FALSE]
  } else if (type == "logit") {
    # Logit only makes sense if ratio is a proportion (0-1)
    ratio_draws <- draws[, grep("^ratio\\[", colnames(draws)), drop = FALSE]
    # Check if ratios are in (0, 1)
    if (any(ratio_draws > 1, na.rm = TRUE) || any(ratio_draws < 0, na.rm = TRUE)) {
      warning("Logit transform requested but ratios are not in (0, 1). ",
              "Results may include Inf/-Inf.", call. = FALSE)
    }
    ratio_draws <- log(ratio_draws / (1 - ratio_draws))
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

  # Get ratio draws
  ratio_obj <- ratio(object, type = "log")
  draws <- ratio_obj$draws

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
