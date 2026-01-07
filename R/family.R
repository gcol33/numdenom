#' Model families for quotr
#'
#' @description
#' Distribution families for the numerator and denominator processes.
#' Each family specifies the likelihood structure for joint modelling.
#'
#' @name quotr_families
NULL

#' Negative binomial - Negative binomial family
#'
#' @description
#' Two-process model where both numerator and denominator follow negative
#' binomial distributions. This is the core family for count-count ratios
#' where both processes exhibit overdispersion.
#'
#' Use cases:
#' - Species counts / total counts (relative abundance)
#' - Events / opportunities (both counts)
#' - Any ratio where numerator and denominator are overdispersed counts
#'
#' @param link_num Link function for numerator (default: "log")
#' @param link_denom Link function for denominator (default: "log")
#'
#' @return A `quotr_family` object
#'
#' @examples
#' \dontrun{
#' # Relative abundance: species count / total count
#' fit <- quotr(
#'   numerator = sp_count ~ habitat + (1 | site),
#'   denominator = total_count ~ habitat + (1 | site),
#'   shared = ~ (1 | site),
#'   family = quotr_negbin_negbin(),
#'   data = abundance_data
#' )
#' }
#'
#' @export
quotr_negbin_negbin <- function(link_num = "log", link_denom = "log") {
  validate_link(link_num, c("log"))
  validate_link(link_denom, c("log"))

  structure(
    list(
      name = "negbin_negbin",
      numerator = list(
        distribution = "neg_binomial_2",
        link = link_num
      ),
      denominator = list(
        distribution = "neg_binomial_2",
        link = link_denom
      ),
      stan_model = "quotr_negbin",
      description = "Negative binomial numerator, negative binomial denominator"
    ),
    class = c("quotr_family", "list")
  )
}

#' Binomial family for trial-based ratios
#'
#' @description
#' Trial-based model where the numerator represents successes out of
#' denominator trials. The denominator can be treated as known (fixed)
#' or modelled with its own process.
#'
#' Use cases:
#' - Detection / availability (occupancy-style)
#' - Successes / trials (hierarchical binomial)
#' - Proportions with known denominators
#'
#' @param link Link function for probability (default: "logit")
#' @param denominator_known Logical; if TRUE, denominator is treated as
#'   fixed and known. If FALSE, denominator is also modelled.
#'
#' @return A `quotr_family` object
#'
#' @examples
#' \dontrun
#' # Detection given availability
#' fit <- quotr(
#'   numerator = detections ~ effort + (1 | site),
#'   denominator = availability ~ (1 | site),
#'   shared = ~ (1 | site),
#'   family = quotr_binomial(),
#'   data = detection_data
#' )
#' }
#'
#' @export
quotr_binomial <- function(link = "logit", denominator_known = TRUE) {
  validate_link(link, c("logit", "probit", "cloglog"))

  structure(
    list(
      name = if (denominator_known) "binomial_fixed" else "binomial_binomial",
      numerator = list(
        distribution = "binomial",
        link = link
      ),
      denominator = list(
        distribution = if (denominator_known) "fixed" else "binomial",
        link = if (denominator_known) NULL else link
      ),
      denominator_known = denominator_known,
      stan_model = if (denominator_known) "quotr_binomial" else "quotr_binomial_binomial",
      description = if (denominator_known) {
        "Binomial numerator with fixed denominator"
      } else {
        "Binomial numerator, binomial denominator"
      }
    ),
    class = c("quotr_family", "list")
  )
}

#' Poisson-Gamma family for count/effort ratios
#'
#' @description
#' Two-process model where the numerator follows a Poisson distribution
#' and the denominator (effort/exposure) follows a Gamma distribution.
#' This is the natural family for CPUE-type data.
#'
#' Use cases:
#' - Catch per unit effort (CPUE)
#' - Observations per hour
#' - Events per continuous exposure
#'
#' @param link_num Link function for count rate (default: "log")
#' @param link_denom Link function for effort mean (default: "log")
#'
#' @return A `quotr_family` object
#'
#' @examples
#' \dontrun{
#' # CPUE: catch count / fishing effort (hours)
#' fit <- quotr(
#'   numerator = catch ~ depth + season + (1 | vessel),
#'   denominator = effort_hours ~ (1 | vessel),
#'   shared = ~ (1 | vessel),
#'   family = quotr_poisson_gamma(),
#'   data = trawl_data
#' )
#' }
#'
#' @export
quotr_poisson_gamma <- function(link_num = "log", link_denom = "log") {
  validate_link(link_num, c("log"))
  validate_link(link_denom, c("log"))

  structure(
    list(
      name = "poisson_gamma",
      numerator = list(
        distribution = "poisson",
        link = link_num
      ),
      denominator = list(
        distribution = "gamma",
        link = link_denom
      ),
      stan_model = "quotr_poisson_gamma",
      description = "Poisson numerator, Gamma denominator (CPUE-type)"
    ),
    class = c("quotr_family", "list")
  )
}

#' Validate link function
#'
#' @param link Link function name
#' @param allowed Character vector of allowed links
#' @keywords internal
validate_link <- function(link, allowed) {
  if (!link %in% allowed) {
    stop(
      sprintf("Link function '%s' not supported. Use one of: %s",
              link, paste(allowed, collapse = ", ")),
      call. = FALSE
    )
  }
}

#' Print method for quotr_family
#'
#' @param x A quotr_family object
#' @param ... Ignored
#'
#' @export
print.quotr_family <- function(x, ...) {
  cat("quotr family:", x$name, "\n")
  cat(x$description, "\n\n")
  cat("Numerator:  ", x$numerator$distribution,
      "(", x$numerator$link, ")\n", sep = "")
  cat("Denominator:", x$denominator$distribution,
      if (!is.null(x$denominator$link)) paste0("(", x$denominator$link, ")") else "(fixed)",
      "\n")
  invisible(x)
}
