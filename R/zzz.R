#' @importFrom instantiate stan_package_model
NULL

.onLoad <- function(libname, pkgname) {

  # Register Stan models with instantiate
}

#' Get path to Stan model
#'
#' @param name Name of the Stan model (without .stan extension)
#' @return Path to the compiled Stan model
#' @keywords internal
quotr_stan_model <- function(name) {
  stan_file <- system.file(
    "stan", paste0(name, ".stan"),
    package = "quotr",
    mustWork = TRUE
  )

  # Use instantiate for pre-compiled models if available
  if (requireNamespace("instantiate", quietly = TRUE)) {
    tryCatch(
      instantiate::stan_package_model(
        name = name,
        package = "quotr"
      ),
      error = function(e) {
        # Fall back to cmdstanr compilation
        cmdstanr::cmdstan_model(stan_file)
      }
    )
  } else {
    cmdstanr::cmdstan_model(stan_file)
  }
}
