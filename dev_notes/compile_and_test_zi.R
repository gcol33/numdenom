suppressPackageStartupMessages({
  library(Rcpp)
  library(testthat)
})

cat("--- Step 1: Rcpp::compileAttributes() ---\n")
Rcpp::compileAttributes(".", verbose = FALSE)

cat("--- Step 2: devtools::load_all() (will recompile) ---\n")
suppressPackageStartupMessages(devtools::load_all('.', quiet = TRUE))
cat("PKG_LOADED\n")

cat("\n--- Step 3: B1b regression (all 7 base families) ---\n")
testthat::test_file("tests/testthat/test-specs-all-families.R",
                    reporter = testthat::SummaryReporter$new())
