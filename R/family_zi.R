#' Zero-inflated model families for quotr
#'
#' @description
#' Distribution families for handling excess zeros in count data.
#' Zero-inflation occurs when there are more zeros than expected
#' under the standard count distribution.
#'
#' Two model types are available:
#' - **Zero-inflated (ZI)**: Mixture of point mass at zero and count distribution
#' - **Hurdle**: Separate processes for zero vs positive counts
#'
#' @name quotr_zi_families
NULL


#' Zero-inflated negative binomial family
#'
#' @description
#' Zero-inflated negative binomial for the numerator process.
#' Models excess zeros as a mixture:
#'
#' \deqn{P(Y = 0) = \pi + (1 - \pi) \cdot P_{NB}(0)}
#' \deqn{P(Y = y) = (1 - \pi) \cdot P_{NB}(y), \quad y > 0}
#'
#' where \eqn{\pi} is the zero-inflation probability and
#' \eqn{P_{NB}} is the negative binomial PMF.
#'
#' @param link_num Link function for count mean (default: "log")
#' @param link_denom Link function for denominator mean (default: "log")
#' @param link_zi Link function for zero-inflation probability (default: "logit")
#' @param denom_family Denominator distribution: "negbin" (default) or "fixed"
#'
#' @return A `quotr_family` object with zero-inflation
#'
#' @details
#' The zero-inflation probability can have its own linear predictor via the
#' `zi` argument in `quotr()`. If not specified, a single intercept is estimated.
#'
#' For ecological data, zero-inflation often represents:
#' - Structural zeros (species truly absent vs not detected)
#' - Sampling zeros (inadequate effort)
#' - False negatives
#'
#' @examples
#' \dontrun{
#' # Species counts with excess zeros
#' fit <- quotr(
#'   count | total ~ habitat + (1 | site),
#'   zi = ~ habitat,  # Zero-inflation varies by habitat
#'   data = df,
#'   family = quotr_zinegbin()
#' )
#' }
#'
#' @seealso [quotr_hurdle_negbin()] for hurdle model alternative
#'
#' @export
quotr_zinegbin <- function(link_num = "log", link_denom = "log",
                           link_zi = "logit", denom_family = "negbin") {

  validate_link(link_num, c("log"))
  validate_link(link_zi, c("logit", "probit", "cloglog"))

  denom_family <- match.arg(denom_family, c("negbin", "fixed"))

  if (denom_family == "negbin") {
    validate_link(link_denom, c("log"))
  }

  structure(
    list(
      name = paste0("zinegbin_", denom_family),
      numerator = list(
        distribution = "zero_inflated_neg_binomial",
        base_distribution = "neg_binomial_2",
        link = link_num,
        link_zi = link_zi
      ),
      denominator = list(
        distribution = if (denom_family == "negbin") "neg_binomial_2" else "fixed",
        link = if (denom_family == "negbin") link_denom else NULL
      ),
      zero_inflated = TRUE,
      zi_type = "mixture",
      description = sprintf(
        "Zero-inflated negative binomial numerator, %s denominator",
        if (denom_family == "negbin") "negative binomial" else "fixed"
      )
    ),
    class = c("quotr_family_zi", "quotr_family", "list")
  )
}


#' Zero-inflated Poisson family
#'
#' @description
#' Zero-inflated Poisson for the numerator process.
#' Useful when overdispersion is primarily due to excess zeros.
#'
#' @inheritParams quotr_zinegbin
#' @param denom_family Denominator distribution: "gamma" (default), "negbin", or "fixed"
#'
#' @return A `quotr_family` object with zero-inflation
#'
#' @details
#' Use `quotr_zipois()` when:
#' - Excess zeros are the main source of overdispersion
#' - The non-zero counts follow Poisson (no additional overdispersion)
#'
#' Use `quotr_zinegbin()` when:
#' - Both excess zeros AND overdispersion in positive counts
#'
#' @examples
#' \dontrun{
#' # CPUE with many zero catches
#' fit <- quotr(
#'   catch | effort ~ depth + (1 | vessel),
#'   zi = ~ 1,  # Constant zero-inflation
#'   data = trawl_data,
#'   family = quotr_zipois(denom_family = "gamma")
#' )
#' }
#'
#' @export
quotr_zipois <- function(link_num = "log", link_denom = "log",
                         link_zi = "logit", denom_family = "gamma") {

  validate_link(link_num, c("log"))
  validate_link(link_zi, c("logit", "probit", "cloglog"))

  denom_family <- match.arg(denom_family, c("gamma", "negbin", "fixed"))

  if (denom_family != "fixed") {
    validate_link(link_denom, c("log"))
  }

  denom_dist <- switch(denom_family,
    gamma = "gamma",
    negbin = "neg_binomial_2",
    fixed = "fixed"
  )

  structure(
    list(
      name = paste0("zipois_", denom_family),
      numerator = list(
        distribution = "zero_inflated_poisson",
        base_distribution = "poisson",
        link = link_num,
        link_zi = link_zi
      ),
      denominator = list(
        distribution = denom_dist,
        link = if (denom_family != "fixed") link_denom else NULL
      ),
      zero_inflated = TRUE,
      zi_type = "mixture",
      description = sprintf(
        "Zero-inflated Poisson numerator, %s denominator",
        denom_family
      )
    ),
    class = c("quotr_family_zi", "quotr_family", "list")
  )
}


#' Hurdle negative binomial family
#'
#' @description
#' Hurdle model where zero vs positive counts are modelled as separate processes.
#'
#' \deqn{P(Y = 0) = 1 - \theta}
#' \deqn{P(Y = y | Y > 0) = P_{NB}^+(y), \quad y > 0}
#'
#' where \eqn{\theta} is the probability of a positive count and
#' \eqn{P_{NB}^+} is the truncated-at-zero negative binomial.
#'
#' @inheritParams quotr_zinegbin
#' @param link_hurdle Link function for hurdle probability (default: "logit")
#'
#' @return A `quotr_family` object with hurdle structure
#'
#' @details
#' The hurdle model differs from zero-inflation in interpretation:
#'
#' **Zero-inflated**: Zeros come from two sources (structural + sampling)
#' - \eqn{\pi}: probability of structural zero
#' - Some zeros also come from the count process
#'
#' **Hurdle**: Zeros are all from one source (the "hurdle" process)
#' - \eqn{1 - \theta}: probability of zero
#' - All positive counts from truncated count process
#'
#' Choose hurdle when zeros represent a distinct biological/sampling state.
#'
#' @examples
#' \dontrun{
#' # Hurdle model: presence/absence + abundance given presence
#' fit <- quotr(
#'   count | total ~ habitat + (1 | site),
#'   zi = ~ habitat,  # Predictors for presence probability
#'   data = df,
#'   family = quotr_hurdle_negbin()
#' )
#' }
#'
#' @export
quotr_hurdle_negbin <- function(link_num = "log", link_denom = "log",
                                link_hurdle = "logit", denom_family = "negbin") {

  validate_link(link_num, c("log"))
  validate_link(link_hurdle, c("logit", "probit", "cloglog"))

  denom_family <- match.arg(denom_family, c("negbin", "fixed"))

  if (denom_family == "negbin") {
    validate_link(link_denom, c("log"))
  }

  structure(
    list(
      name = paste0("hurdle_negbin_", denom_family),
      numerator = list(
        distribution = "hurdle_neg_binomial",
        base_distribution = "neg_binomial_2",
        link = link_num,
        link_hurdle = link_hurdle
      ),
      denominator = list(
        distribution = if (denom_family == "negbin") "neg_binomial_2" else "fixed",
        link = if (denom_family == "negbin") link_denom else NULL
      ),
      zero_inflated = TRUE,
      zi_type = "hurdle",
      description = sprintf(
        "Hurdle negative binomial numerator, %s denominator",
        if (denom_family == "negbin") "negative binomial" else "fixed"
      )
    ),
    class = c("quotr_family_zi", "quotr_family", "list")
  )
}


#' Hurdle Poisson family
#'
#' @description
#' Hurdle model with truncated Poisson for positive counts.
#'
#' @inheritParams quotr_hurdle_negbin
#' @param denom_family Denominator distribution: "gamma" (default), "negbin", or "fixed"
#'
#' @return A `quotr_family` object with hurdle structure
#'
#' @export
quotr_hurdle_pois <- function(link_num = "log", link_denom = "log",
                              link_hurdle = "logit", denom_family = "gamma") {

  validate_link(link_num, c("log"))
  validate_link(link_hurdle, c("logit", "probit", "cloglog"))

  denom_family <- match.arg(denom_family, c("gamma", "negbin", "fixed"))

  if (denom_family != "fixed") {
    validate_link(link_denom, c("log"))
  }

  denom_dist <- switch(denom_family,
    gamma = "gamma",
    negbin = "neg_binomial_2",
    fixed = "fixed"
  )

  structure(
    list(
      name = paste0("hurdle_pois_", denom_family),
      numerator = list(
        distribution = "hurdle_poisson",
        base_distribution = "poisson",
        link = link_num,
        link_hurdle = link_hurdle
      ),
      denominator = list(
        distribution = denom_dist,
        link = if (denom_family != "fixed") link_denom else NULL
      ),
      zero_inflated = TRUE,
      zi_type = "hurdle",
      description = sprintf(
        "Hurdle Poisson numerator, %s denominator",
        denom_family
      )
    ),
    class = c("quotr_family_zi", "quotr_family", "list")
  )
}


#' Check if family is zero-inflated
#'
#' @param family A quotr_family object
#' @return Logical
#' @keywords internal
is_zi_family <- function(family) {
  isTRUE(family$zero_inflated)
}


#' Check if family is hurdle model
#'
#' @param family A quotr_family object
#' @return Logical
#' @keywords internal
is_hurdle_family <- function(family) {
  isTRUE(family$zi_type == "hurdle")
}


#' Print method for zero-inflated quotr_family
#'
#' @param x A quotr_family_zi object
#' @param ... Ignored
#'
#' @export
print.quotr_family_zi <- function(x, ...) {
  zi_type <- if (x$zi_type == "hurdle") "Hurdle" else "Zero-inflated"
  cat("quotr family:", x$name, "\n")
  cat(sprintf("[%s model]\n", zi_type))
  cat(x$description, "\n\n")

  cat("Numerator:  ", x$numerator$base_distribution,
      "(", x$numerator$link, ")\n", sep = "")

  if (x$zi_type == "hurdle") {
    cat("Hurdle:     ", "bernoulli",
        "(", x$numerator$link_hurdle, ")\n", sep = "")
  } else {
    cat("ZI prob:    ", "bernoulli",
        "(", x$numerator$link_zi, ")\n", sep = "")
  }

  cat("Denominator:", x$denominator$distribution,
      if (!is.null(x$denominator$link)) paste0("(", x$denominator$link, ")") else "(fixed)",
      "\n")

  invisible(x)
}
