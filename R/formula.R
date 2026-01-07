#' Parse and validate quotr formulas
#'
#' @description
#' Internal functions for parsing quotr formulas into a structured
#' specification for Stan model building.
#'
#' quotr supports a combined formula syntax:
#' - `num | denom ~ x + (1|group)` for shared predictors
#' - Separate `formula_num` and `formula_denom` for different predictors
#'
#' @name quotr_formula
#' @keywords internal
NULL

#' Parse a quotr formula
#'
#' @description
#' Parses the main formula and optional process-specific formulas into
#' a structured specification. Supports both combined syntax
#' (`num | denom ~ x`) and separate formulas.
#'
#' @param formula Main formula. Can be:
#'   - Combined: `num | denom ~ predictors`
#'   - Numerator only: `num ~ predictors` (requires formula_denom)
#' @param formula_num Optional formula for numerator-specific predictors.
#'   Added to those from the main formula.
#' @param formula_denom Optional formula for denominator-specific predictors.
#'   Required if main formula has single response.
#' @param shared Formula for shared random effects. NULL = infer from formulas,
#'   `~ 0` = independence (triggers warning).
#' @param data Data frame containing all variables.
#'
#' @return A `quotr_formula` object
#' @keywords internal
quotr_formula <- function(formula,
                          formula_num = NULL,
                          formula_denom = NULL,
                          shared = NULL,
                          data) {

  # Parse the main formula
  parsed <- parse_main_formula(formula, data)

  # Handle combined vs separate formula specifications

  if (parsed$type == "combined") {
    # Combined syntax: num | denom ~ predictors
    num_response <- parsed$num_var
    denom_response <- parsed$denom_var
    base_rhs <- parsed$rhs

    # Build numerator formula
    num_formula <- build_process_formula(
      response = num_response,
      base_rhs = base_rhs,
      additional = formula_num,
      data = data
    )

    # Build denominator formula
    denom_formula <- build_process_formula(
      response = denom_response,
      base_rhs = base_rhs,
      additional = formula_denom,
      data = data
    )

  } else {
    # Single response: require formula_denom
    if (is.null(formula_denom))

      stop(
        "When using single-response formula, `formula_denom` is required.\n",
        "Either use combined syntax: num | denom ~ x\n",
        "Or provide both formula and formula_denom.",
        call. = FALSE
      )

    num_response <- parsed$response
    base_rhs <- parsed$rhs

    num_formula <- build_process_formula(
      response = num_response,
      base_rhs = base_rhs,
      additional = formula_num,
      data = data
    )

    denom_formula <- parse_denom_formula(formula_denom, data)
    denom_response <- denom_formula$response_var
  }

  # Validate - no offsets allowed
  check_no_offset(num_formula$full_formula, "numerator")
  check_no_offset(denom_formula$full_formula, "denominator")

  # Handle shared structure
  shared_parsed <- parse_shared(shared, num_formula, denom_formula, data)

  structure(
    list(
      numerator = num_formula,
      denominator = denom_formula,
      shared = shared_parsed,
      original_formula = formula
    ),
    class = "quotr_formula"
  )
}

#' Parse main formula to detect combined vs single syntax
#'
#' @param formula The formula to parse
#' @param data Data frame
#' @return List with type, response(s), and RHS
#' @keywords internal
parse_main_formula <- function(formula, data) {
  if (!inherits(formula, "formula")) {
    stop("`formula` must be a formula object", call. = FALSE)
  }

  if (length(formula) < 3) {
    stop("Formula must have response and predictors: response ~ predictors",
         call. = FALSE)
  }

  lhs <- formula[[2]]
  rhs <- formula[[3]]

  # Check for combined syntax: num | denom
  lhs_text <- deparse(lhs, width.cutoff = 500)
  lhs_text <- paste(lhs_text, collapse = " ")

  if (grepl("\\|", lhs_text)) {
    # Combined syntax
    parts <- strsplit(lhs_text, "\\|")[[1]]
    if (length(parts) != 2) {
      stop("Combined formula must have exactly two responses: num | denom ~ x",
           call. = FALSE)
    }

    num_var <- trimws(parts[1])
    denom_var <- trimws(parts[2])

    # Validate variables exist
    if (!num_var %in% names(data)) {
      stop(sprintf("Numerator variable '%s' not found in data", num_var),
           call. = FALSE)
    }
    if (!denom_var %in% names(data)) {
      stop(sprintf("Denominator variable '%s' not found in data", denom_var),
           call. = FALSE)
    }

    list(
      type = "combined",
      num_var = num_var,
      denom_var = denom_var,
      rhs = rhs
    )
  } else {
    # Single response
    response <- as.character(lhs)
    if (!response %in% names(data)) {
      stop(sprintf("Response variable '%s' not found in data", response),
           call. = FALSE)
    }

    list(
      type = "single",
      response = response,
      rhs = rhs
    )
  }
}

#' Build a process-specific formula
#'
#' @param response Response variable name
#' @param base_rhs RHS from main formula
#' @param additional Optional additional formula
#' @param data Data frame
#' @return Parsed formula specification
#' @keywords internal
build_process_formula <- function(response, base_rhs, additional, data) {
  # Start with base formula
  base_rhs_text <- deparse(base_rhs, width.cutoff = 500)
  base_rhs_text <- paste(base_rhs_text, collapse = " ")

  # Add additional terms if provided
  if (!is.null(additional)) {
    if (!inherits(additional, "formula")) {
      stop("Additional formula must be a formula object", call. = FALSE)
    }
    # One-sided formula: ~ extra_terms
    if (length(additional) == 2) {
      add_rhs <- deparse(additional[[2]], width.cutoff = 500)
      add_rhs <- paste(add_rhs, collapse = " ")
      base_rhs_text <- paste(base_rhs_text, "+", add_rhs)
    }
  }

  # Build full formula
  full_formula <- as.formula(
    paste(response, "~", base_rhs_text),
    env = parent.frame(2)
  )

  # Parse into components
  parse_formula_components(full_formula, data)
}

#' Parse denominator-only formula
#'
#' @param formula Formula for denominator (response ~ predictors or ~ predictors)
#' @param data Data frame
#' @return Parsed formula specification
#' @keywords internal
parse_denom_formula <- function(formula, data) {
  if (!inherits(formula, "formula")) {
    stop("`formula_denom` must be a formula", call. = FALSE)
  }

  if (length(formula) == 3) {
    # Two-sided: response ~ predictors
    parse_formula_components(formula, data)
  } else {
    stop("`formula_denom` must include response: denom ~ predictors",
         call. = FALSE)
  }
}

#' Parse formula into fixed and random effect components
#'
#' @param f Formula with response
#' @param data Data frame
#' @return List with response, fixed effects matrix, random effects
#' @keywords internal
parse_formula_components <- function(f, data) {
  response_var <- as.character(f[[2]])

  if (!response_var %in% names(data)) {
    stop(sprintf("Variable '%s' not found in data", response_var),
         call. = FALSE)
  }

  response <- data[[response_var]]

  rhs_text <- deparse(f[[3]], width.cutoff = 500)
  rhs_text <- paste(rhs_text, collapse = " ")

  # Extract random effects
  random_effects <- extract_random_effects(rhs_text, data)

  # Build fixed effects matrix
  fixed_formula <- remove_random_effects(f)
  X <- build_model_matrix(fixed_formula, data)

  list(
    response_var = response_var,
    response = response,
    full_formula = f,
    fixed_formula = fixed_formula,
    X = X,
    random_effects = random_effects
  )
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

#' Extract random effect specifications from formula text
#'
#' @param rhs_text Right-hand side of formula as text
#' @param data Data frame
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

    if (!group_var %in% names(data)) {
      stop(sprintf("Grouping variable '%s' not found in data", group_var),
           call. = FALSE)
    }

    # Determine if intercept only or slopes
    has_intercept <- grepl("^1$|^1\\s*\\+|\\+\\s*1", lhs) || lhs == "1"
    slope_vars <- NULL

    if (lhs != "1") {
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

  if (rhs_clean == "" || rhs_clean == "1") {
    rhs_clean <- "1"
  }

  as.formula(paste(deparse(f[[2]]), "~", rhs_clean), env = environment(f))
}

#' Build model matrix from fixed effects formula
#'
#' @param f Formula (response ~ fixed effects)
#' @param data Data frame
#' @return Model matrix X
#' @keywords internal
build_model_matrix <- function(f, data) {
  mf <- model.frame(f, data = data, na.action = na.pass)
  model.matrix(f, data = mf)
}

#' Parse shared formula specification
#'
#' @param shared Shared formula or NULL
#' @param num_parsed Parsed numerator formula
#' @param denom_parsed Parsed denominator formula
#' @param data Data frame
#' @return Shared structure specification
#' @keywords internal
parse_shared <- function(shared, num_parsed, denom_parsed, data) {
  if (is.null(shared)) {
    # Default: infer from matching random effects
    return(infer_shared_structure(num_parsed, denom_parsed))
  }

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

    # Parse shared random effects (one-sided formula)
    if (length(shared) == 2) {
      rhs_text <- deparse(shared[[2]], width.cutoff = 500)
      rhs_text <- paste(rhs_text, collapse = " ")
    } else {
      stop("shared formula should be one-sided: ~ (1 | group)", call. = FALSE)
    }

    shared_re <- extract_random_effects(rhs_text, data)

    return(list(
      type = "explicit",
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
#' @return Shared structure specification
#' @keywords internal
infer_shared_structure <- function(num_parsed, denom_parsed) {
  num_groups <- vapply(num_parsed$random_effects,
                       function(x) x$group_var, character(1))
  denom_groups <- vapply(denom_parsed$random_effects,
                         function(x) x$group_var, character(1))

  if (length(num_groups) == 0) num_groups <- character(0)
  if (length(denom_groups) == 0) denom_groups <- character(0)

  shared_groups <- intersect(num_groups, denom_groups)

  if (length(shared_groups) == 0) {
    return(list(
      type = "none",
      random_effects = list(),
      warned = FALSE
    ))
  }

  # Use RE specs from numerator for shared groups
  shared_re <- num_parsed$random_effects[num_groups %in% shared_groups]

  list(
    type = "inferred",
    random_effects = shared_re,
    warned = FALSE
  )
}

#' Warn about independence assumption
#' @keywords internal
warn_independence <- function() {
  warning(
    "You have specified `shared = ~ 0`, assuming independence between ",
    "numerator and denominator processes.\n\n",
    "This assumption is often violated in practice and can lead to spurious ",
    "ratio effects when both processes share unmeasured drivers.\n\n",
    "Consider whether shared random effects would be more appropriate.",
    call. = FALSE,
    immediate. = TRUE
  )
}

#' Print method for quotr_formula
#'
#' @param x A quotr_formula object
#' @param ... Ignored
#' @export
#' @keywords internal
print.quotr_formula <- function(x, ...) {
  cat("quotr formula specification\n")
  cat("===========================\n\n")

  cat("Numerator:", x$numerator$response_var, "\n")
  cat("  Fixed effects:", ncol(x$numerator$X), "coefficients\n")
  cat("  Random effects:", length(x$numerator$random_effects), "terms\n")

  cat("\nDenominator:", x$denominator$response_var, "\n")
  cat("  Fixed effects:", ncol(x$denominator$X), "coefficients\n")
  cat("  Random effects:", length(x$denominator$random_effects), "terms\n")

  cat("\nShared structure:", x$shared$type, "\n")
  if (length(x$shared$random_effects) > 0) {
    groups <- vapply(x$shared$random_effects,
                     function(r) r$group_var, character(1))
    cat("  Groups:", paste(groups, collapse = ", "), "\n")
  }

  invisible(x)
}
