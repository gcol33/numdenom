.onLoad <- function(libname, pkgname) {
  # Load tulpa so its R_RegisterCCallable engine symbols are available before
  # any compiled backend resolves them with R_GetCCallable.
  requireNamespace("tulpa")
}
