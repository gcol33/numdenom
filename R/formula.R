#' Parse and validate quotr formulas
#'
#' @description
#' Internal functions for parsing the numerator, denominator, and shared
#' formulas into a structured specification for Stan model building.
#'
#' @name quotr_formula
#' @keywords internal
NULL

#' Create a quotr formula specification
#'
#' @param numerator Formula for the numerator process
#' @param denominator Formula for the denominator process
#' @param shared Formula for shared random effects (NULL = infer from formulas)
#' @param data Data frame
#'
#' @return A `quotr_formula` object
#' @keywords internal
quotr_formula <- function(numerator, denominator, shared = NULL, data) {

 # Validate formulas
  validate_formula(numerator, "numerator")
  validate_formula(denominator, "denominator")

  # Check for offset() usage - this is forbidden
  check_no_offset(numerator, "numerator")
  check_no_offset(denominator, "denominator")

  # Parse formulas into components
  num_parsed <- parse_formula(numerator, data, prefix = "num")
  denom_parsed <- parse_formula(denominator, data, prefix = "denom")

  # Handle shared structure
  shared_parsed <- parse_shared(shared, num_parsed, denom_parsed, data)

  structure(
    list(
      numerator = num_parsed,
      denominator = denom_parsed,
      shared = shared_parsed,
      original = list(
        numerator = numerator,
        denominator = denominator,
        shared = shared
      )
    ),
    class = "quotr_formula"
  )
}

#' Validate a formula
#'
#' @param f Formula to validate
#' @param name Name for error messages
#' @keywords internal
validate_formula <- function(f, name) {
  if (!inherits(f, "formula")) {
    stop(sprintf("`%s` must be a formula", name), call. = FALSE)
  }

  if (length(f) < 3) {
    stop(sprintf("`%s` must have a response variable (e.g., y ~ x)", name),
         call. = FALSE)
  }
}

#' Check formula does not contain offset()
#'
#' @description
#' The quotr philosophy explicitly forbids offset terms. Offsets encode
#' the assumption that the ratio is the quantity of interest, which is
#' exactly the modelling approach this package rejects.
#'
#' @param f Formula to check
#' @param name Name for error messages
#' @keywords internal
check_no_offset <- function(f, name) {
  f_text <- deparse(f, width.cutoff = 500)
  f_text <- paste(f_text, collapse = " ")

  if (grepl("\\boffset\\s*\\(", f_text, ignore.case = TRUE)) {
    stop(
      sprintf("offset() is not allowed in the %s formula.\n\n", name),
      "quotr does not model ratios directly. Instead, it jointly models the\n",
      "numerator and denominator processes with shared latent structure.\n",
      "Offsets encode the assumption that the ratio is the quantity of interest,\n",
      "which is exactly the modelling approach this package rejects.\n\n",
      "If you have exposure/effort data, include it as the denominator response.",
      call. = FALSE
    )
  }
}

#' Parse a single formula into components
#'
#' @param f Formula
#' @param data Data frame
#' @param prefix Prefix for naming ("num" or "denom")
#'
#' @return List with response, fixed effects, and random effects
#' @keywords internal
parse_formula <- function(f, data, prefix) {
  # Extract response variable
  response_var <- as.character(f[[2]])

  if (!(response_var %in% names(data))) {
    stop(sprintf("Response variable '%s' not found in data", response_var),
         call. = FALSE)
  }

  response <- data[[response_var]]

  # Parse right-hand side
  rhs <- f[[3]]
  rhs_text <- deparse(rhs, width.cutoff = 500)
  rhs_text <- paste(rhs_text, collapse = " ")

  # Extract random effects (terms with |)
  random_effects <- extract_random_effects(rhs_text, data)

  # Build fixed effects formula (removing random effects)
  fixed_formula <- remove_random_effects(f)
  X <- build_model_matrix(fixed_formula, data)

  list(
    response_var = response_var,
    response = response,
    fixed_formula = fixed_formula,
    X = X,
    random_effects = random_effects,
    prefix = prefix
  )
}

#' Extract random effect specifications from formula text
#'
#' @param rhs_text Right-hand side of formula as text
#' @param data Data frame
#'
#' @return List of random effect specifications
#' @keywords internal
extract_random_effects <- function(rhs_text, data) {
  # Match patterns like (1 | group) or (x | group) or (1 + x | group)
  re_pattern <- "\\(([^|]+)\\|([^)]+)\\)"
  matches <- gregexpr(re_pattern, rhs_text, perl = TRUE)

  if (matches[[1]][1] == -1) {
    return(list())
  }

  re_texts <- regmatches(rhs_text, matches)[[1]]
  random_effects <- list()

  for (i in seq_along(re_texts)) {
    re_text <- re_texts[i]
    # Remove outer parentheses
    re_inner <- gsub("^\\(|\\)$", "", re_text)
    parts <- strsplit(re_inner, "\\|")[[1]]

    lhs <- trimws(parts[1])
    group_var <- trimws(parts[2])

    if (!(group_var %in% names(data))) {
      stop(sprintf("Grouping variable '%s' not found in data", group_var),
           call. = FALSE)
    }

    # Determine if intercept only or slopes
    has_intercept <- grepl("^1$|^1\\s*\\+|\\+\\s*1", lhs) || lhs == "1"
    slope_vars <- NULL

    if (lhs != "1") {
      # Extract slope variables
      slope_terms <- gsub("\\b1\\b", "", lhs)
      slope_terms <- gsub("^\\s*\\+\\s*|\\s*\\+\\s*$", "", slope_terms)
      if (nchar(trimws(slope_terms)) > 0) {
        slope_vars <- trimws(strsplit(slope_terms, "\\+")[[1]])
        slope_vars <- slope_vars[slope_vars != ""]
      }
    }

    random_effects[[i]] <- list(
      group_var = group_var,
      group = as.integer(as.factor(data[[group_var]])),
      n_groups = length(unique(data[[group_var]])),
      has_intercept = has_intercept,
      slope_vars = slope_vars,
      original = re_text
    )
  }

  random_effects
}

#' Remove random effects from formula to get fixed effects only
#'
#' @param f Formula with potential random effects
#'
#' @return Formula with only fixed effects
#' @keywords internal
remove_random_effects <- function(f) {
  rhs_text <- deparse(f[[3]], width.cutoff = 500)
  rhs_text <- paste(rhs_text, collapse = " ")

  # Remove random effect terms
  re_pattern <- "\\([^|]+\\|[^)]+\\)"
  rhs_clean <- gsub(re_pattern, "", rhs_text, perl = TRUE)

  # Clean up leftover + signs

  rhs_clean <- gsub("\\+\\s*\\+", "+", rhs_clean)
  rhs_clean <- gsub("^\\s*\\+\\s*|\\s*\\+\\s*$", "", rhs_clean)
  rhs_clean <- trimws(rhs_clean)

  # Handle empty RHS (intercept only)
  if (rhs_clean == "" || rhs_clean == "1") {
    rhs_clean <- "1"
  }

  # Rebuild formula
  as.formula(paste(deparse(f[[2]]), "~", rhs_clean), env = environment(f))
}

#' Build model matrix from fixed effects formula
#'
#' @param f Formula (response ~ fixed effects)
#' @param data Data frame
#'
#' @return Model matrix X
#' @keywords internal
build_model_matrix <- function(f, data) {
  mf <- model.frame(f, data = data, na.action = na.pass)
  X <- model.matrix(f, data = mf)
  X
}

#' Parse shared formula specification
#'
#' @param shared Shared formula or NULL
#' @param num_parsed Parsed numerator formula
#' @param denom_parsed Parsed denominator formula
#' @param data Data frame
#'
#' @return Shared structure specification
#' @keywords internal
parse_shared <- function(shared, num_parsed, denom_parsed, data) {
  # If shared is NULL, infer from random effects
  if (is.null(shared)) {
    # Default: share all random effects that appear in both formulas
    shared <- infer_shared_structure(num_parsed, denom_parsed)
    return(shared)
  }

  # Check for explicit independence request
  if (inherits(shared, "formula")) {
    shared_text <- deparse(shared, width.cutoff = 500)
    shared_text <- paste(shared_text, collapse = " ")

    # ~ 0 or ~ -1 means no shared structure
    if (grepl("^~\\s*(0|-1)\\s*$", shared_text)) {
      warn_independence()
      return(list(
        type = "independent",
        random_effects = list(),
        warned = TRUE
      ))
    }

    # Parse shared random effects
    if (length(shared) == 2) {
      # One-sided formula ~ (1 | group)
      rhs_text <- deparse(shared[[2]], width.cutoff = 500)
      rhs_text <- paste(rhs_text, collapse = " ")
    } else {
      stop("shared formula should be one-sided: ~ (1 | group)", call. = FALSE)
    }

    shared_re <- extract_random_effects(rhs_text, data)

    return(list(
      type = "shared",
      random_effects = shared_re,
      warned = FALSE
    ))
  }

  stop("`shared` must be a formula or NULL", call. = FALSE)
}

#' Infer shared structure from numerator and denominator formulas
#'
#' @param num_parsed Parsed numerator
#' @param denom_parsed Parsed denominator
#'
#' @return Shared structure specification
#' @keywords internal
infer_shared_structure <- function(num_parsed, denom_parsed) {
  # Find grouping variables that appear in both
  num_groups <- vapply(num_parsed$random_effects, `[[`, character(1), "group_var")
  denom_groups <- vapply(denom_parsed$random_effects, `[[`, character(1), "group_var")

  shared_groups <- intersect(num_groups, denom_groups)

  if (length(shared_groups) == 0) {
    # No shared structure inferred, but don't warn (user didn't explicitly request independence)
    return(list(
      type = "none_inferred",
      random_effects = list(),
      warned = FALSE
    ))
  }

  # Use the random effect specs from numerator for shared groups
  shared_re <- num_parsed$random_effects[num_groups %in% shared_groups]

  list(
    type = "inferred",
    random_effects = shared_re,
    warned = FALSE
  )
}

#' Warn about independence assumption
#'
#' @keywords internal
warn_independence <- function() {
  warning(
    "You have specified `shared = ~ 0`, assuming independence between ",
    "numerator and denominator processes.\n\n",
    "This assumption is often violated in practice and can lead to spurious ",
    "ratio effects when both processes share unmeasured drivers.\n\n",
    "Consider whether shared random effects or latent factors would be more ",
    "appropriate for your data.",
    call. = FALSE,
    immediate. = TRUE
  )
}

#' Print method for quotr_formula
#'
#' @param x A quotr_formula object
#' @param ... Ignored
#'
#' @export
#' @keywords internal
print.quotr_formula <- function(x, ...) {
  cat("quotr formula specification\n")
  cat("===========================\n\n")

  cat("Numerator:\n")
  cat("  ", deparse(x$original$numerator), "\n")
  cat("  Response:", x$numerator$response_var, "\n")
  cat("  Fixed effects:", ncol(x$numerator$X), "coefficients\n")
  cat("  Random effects:", length(x$numerator$random_effects), "terms\n\n")

  cat("Denominator:\n")
  cat("  ", deparse(x$original$denominator), "\n")
  cat("  Response:", x$denominator$response_var, "\n")
  cat("  Fixed effects:", ncol(x$denominator$X), "coefficients\n")
  cat("  Random effects:", length(x$denominator$random_effects), "terms\n\n")

  cat("Shared structure:\n")
  if (x$shared$type == "independent") {
    cat("  INDEPENDENT (no shared effects)\n")
  } else {
    cat("  Type:", x$shared$type, "\n")
    cat("  Shared random effects:", length(x$shared$random_effects), "terms\n")
  }

  invisible(x)
}
