#!/usr/bin/env Rscript
# B1c parity: print per-variant max-abs-diff between spec path and legacy path,
# and the within-MC noise (legacy seed 42 vs 43). Used to populate the
# commit body for B1c. Mirrors the simulator in tests/testthat/test-specs-zi.R.

setwd("C:/Users/Gilles Colling/Documents/dev/tulpaRatio")
Sys.setenv(NOT_CRAN = "true")
suppressPackageStartupMessages({ library(devtools) })
devtools::load_all(quiet = TRUE)

# Pull simulate_zi_for_variant out of the test file by parsing only the
# function definition (avoiding sys.source which would execute test_that()).
simulate_zi_for_variant <- local({
  src <- readLines("tests/testthat/test-specs-zi.R")
  i_start <- grep("^simulate_zi_for_variant <- function", src)
  # Function body ends at the first standalone "}" at column 1 after the start.
  brace_lines <- grep("^\\}$", src)
  i_end <- brace_lines[brace_lines > i_start][1]
  e <- new.env(parent = globalenv())
  eval(parse(text = paste(src[i_start:i_end], collapse = "\n")), envir = e)
  e$simulate_zi_for_variant
})

fit_one <- function(spec, use_specs, seed_val) {
  op <- options(tulpaRatio.use_specs = use_specs); on.exit(options(op), add = TRUE)
  args <- list(
    formula = spec$formula, data = spec$data, family = spec$family,
    mode = "hmc", iter = 2000L, warmup = 500L, chains = 1L,
    seed = seed_val, verbose = FALSE, gradient_mode = "A_r"
  )
  if (!is.null(spec$zi)) args$zi <- spec$zi
  fit <- do.call(tulpaRatio::ratiod, args)
  colMeans(fit$draws)
}

variants <- c(
  "zi_binomial",
  "hurdle_binomial",
  "oi_binomial",
  "zoib",
  "zi_poisson",
  "hurdle_poisson",
  "zi_negbin",
  "hurdle_negbin"
)

cat(sprintf("%-18s | %-12s | %-12s | %-7s\n",
            "variant", "cross_diff", "within_diff", "ratio"))
cat(strrep("-", 60), "\n")

for (v in variants) {
  spec <- simulate_zi_for_variant(v)
  legacy42 <- tryCatch(fit_one(spec, FALSE, 42L), error = function(e) NULL)
  if (is.null(legacy42)) {
    cat(sprintf("%-18s | LEGACY ERROR\n", v)); next
  }
  specs42  <- fit_one(spec, TRUE, 42L)
  legacy43 <- fit_one(spec, FALSE, 43L)
  cross  <- max(abs(legacy42 - specs42))
  within <- max(abs(legacy42 - legacy43))
  cat(sprintf("%-18s | %-12.4f | %-12.4f | %-7.2f\n",
              v, cross, within, cross / max(within, 1e-9)))
}
